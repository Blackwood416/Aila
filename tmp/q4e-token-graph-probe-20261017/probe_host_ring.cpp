// Productization probe for the Qwen4Exp whole-token host handoff.
//
// The control/doorbell block and bulk payload deliberately use separate
// Level Zero host allocations:
//   - 64-byte control block: uncached, device-pollable
//   - host-to-device payload: cached pinned host memory
//   - device-to-host payload: uncached host memory + uncached device stores
//
// A captured SYCL graph runs 1,000 host->device and 1,000 device->host
// handoffs. Payload bytes are compared bitwise in both directions. Any
// mismatch, sequence error, timeout, or graph failure seals the handoff gate;
// there is no silent fallback.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <level_zero/ze_api.h>

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ex = sycl::ext::oneapi::experimental;
namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint32_t kMagic = 0x51485247u;  // "QHRG"
constexpr uint32_t kVersion = 1u;
constexpr uint64_t kProbeInstance = 0x4b2d52494e472d31ull;

constexpr size_t kControlBytes = 64;
constexpr size_t kPayloadSlots = 4;
constexpr size_t kPayloadBytesPerSlot = 16 * 1024;
constexpr size_t kPayloadArenaBytes =
    2 * kPayloadSlots * kPayloadBytesPerSlot;
constexpr uint32_t kVisibilityRounds = 1000;
constexpr uint32_t kWorkGroup = 256;
constexpr uint32_t kDeviceSpinLimit = 20'000'000u;
constexpr int kHostTimeoutMs = 15'000;

enum : uint32_t {
    kStateSealed = 0,
    kStateProbing = 1,
    kStateReady = 2,
    kStateFailed = 3,
};

enum : uint32_t {
    kFailNone = 0,
    kFailControl = 1,
    kFailHostPayload = 2,
    kFailHostTimeout = 3,
    kFailDevicePayload = 4,
    kFailDeviceHostFlagTimeout = 5,
    kFailDeviceAckTimeout = 6,
    kFailSequence = 7,
    kFailGraph = 8,
};

struct alignas(64) ControlBlock {
    uint32_t magic;
    uint32_t version;
    uint32_t state;
    uint32_t fail_reason;
    uint32_t host_publish_seq;
    uint32_t device_complete_seq;
    uint32_t host_ack_seq;
    uint32_t device_mismatch_index;
    uint32_t host_mismatch_index;
    uint32_t fail_round;
    uint32_t device_timeout_spins;
    uint32_t device_done;
    uint32_t host_visibility_count;
    uint32_t device_visibility_count;
    uint64_t probe_instance;
};

struct alignas(64) HostToDevicePayload {
    alignas(64) uint8_t bytes[kPayloadSlots][kPayloadBytesPerSlot];
};

struct alignas(64) DeviceToHostPayload {
    alignas(64) uint8_t bytes[kPayloadSlots][kPayloadBytesPerSlot];
};

static_assert(sizeof(ControlBlock) == kControlBytes);
static_assert(sizeof(HostToDevicePayload) == kPayloadArenaBytes / 2);
static_assert(sizeof(DeviceToHostPayload) == kPayloadArenaBytes / 2);
static_assert(alignof(ControlBlock) == 64);
static_assert(alignof(HostToDevicePayload) == 64);
static_assert(alignof(DeviceToHostPayload) == 64);

using UncachedRead = decltype(oex::properties(
    iex::read_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));
using UncachedWrite = decltype(oex::properties(
    iex::write_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));

template <class T>
inline T uncached_load(const T* ptr) {
    oex::annotated_ptr<T, UncachedRead> value(const_cast<T*>(ptr));
    return value[0];
}

template <class T>
inline void uncached_store(T* ptr, T value) {
    oex::annotated_ptr<T, UncachedWrite> target(ptr);
    target[0] = value;
}

inline uint32_t device_control_load(const uint32_t* ptr) {
    // Keep the polled load inside the retry loop, then acquire the value that
    // the host released before publishing the corresponding payload.
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);
    const uint32_t value = uncached_load(ptr);
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);
    return value;
}

inline void device_control_store(uint32_t* ptr, uint32_t value) {
    sycl::atomic_fence(
        sycl::memory_order::release, sycl::memory_scope::system);
    uncached_store(ptr, value);
}

inline uint32_t host_load_u32(const uint32_t* ptr) {
    return *reinterpret_cast<const volatile uint32_t*>(ptr);
}

inline void host_release_store_u32(uint32_t* ptr, uint32_t value) {
    std::atomic_thread_fence(std::memory_order_release);
    _mm_sfence();
    *reinterpret_cast<volatile uint32_t*>(ptr) = value;
}

inline void host_acquire_fence() {
    std::atomic_thread_fence(std::memory_order_acquire);
    _mm_lfence();
}

inline void spin_pause() {
    for (int i = 0; i < 32; ++i) {
        _mm_pause();
    }
}

void flush_cpu_cache(const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;
    const uintptr_t begin =
        reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t(63);
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(ptr) + bytes + 63) & ~uintptr_t(63);
    for (uintptr_t line = begin; line < end; line += 64) {
        _mm_clflush(reinterpret_cast<const void*>(line));
    }
    _mm_mfence();
}

inline uint8_t payload_byte(
    uint32_t round, uint32_t index, uint32_t salt) {
    uint32_t x = (round * 0x9e3779b1u) ^
                 (index * 0x85ebca6bu) ^ salt;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<uint8_t>(x >> 24);
}

constexpr uint32_t kHostToDeviceSalt = 0x13579bdfu;
constexpr uint32_t kDeviceToHostSalt = 0x2468ace0u;

void fill_payload(
    uint8_t* dst, size_t bytes, uint32_t round, uint32_t salt) {
    for (size_t i = 0; i < bytes; ++i) {
        dst[i] = payload_byte(round, static_cast<uint32_t>(i), salt);
    }
}

bool verify_payload(
    const uint8_t* src, size_t bytes, uint32_t round, uint32_t salt,
    size_t& mismatch_index) {
    const volatile uint8_t* values =
        reinterpret_cast<const volatile uint8_t*>(src);
    for (size_t i = 0; i < bytes; ++i) {
        if (values[i] !=
            payload_byte(round, static_cast<uint32_t>(i), salt)) {
            mismatch_index = i;
            return false;
        }
    }
    mismatch_index = std::numeric_limits<size_t>::max();
    return true;
}

void* allocate_level_zero_host(
    ze_context_handle_t context, size_t bytes, size_t alignment,
    bool uncached, const char* label) {
    ze_host_mem_alloc_desc_t descriptor{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        nullptr,
        uncached ? ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
                 : ze_host_mem_alloc_flags_t{0}};
    void* ptr = nullptr;
    const ze_result_t result = zeMemAllocHost(
        context, &descriptor, bytes, alignment, &ptr);
    if (result != ZE_RESULT_SUCCESS || !ptr) {
        std::ostringstream os;
        os << label << " allocation failed: ze_result_t="
           << static_cast<int>(result);
        throw std::runtime_error(os.str());
    }
    if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
        (void)zeMemFree(context, ptr);
        throw std::runtime_error(
            std::string(label) + " allocation is not correctly aligned");
    }
    return ptr;
}

bool ranges_overlap(
    const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a);
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b);
    const uintptr_t a1 = a0 + a_bytes;
    const uintptr_t b1 = b0 + b_bytes;
    return a0 < b1 && b0 < a1;
}

const char* state_name(uint32_t state) {
    switch (state) {
    case kStateSealed: return "sealed";
    case kStateProbing: return "probing";
    case kStateReady: return "ready";
    case kStateFailed: return "failed";
    default: return "unknown";
    }
}

const char* fail_name(uint32_t reason) {
    switch (reason) {
    case kFailNone: return "none";
    case kFailControl: return "control";
    case kFailHostPayload: return "host_payload_bitwise";
    case kFailHostTimeout: return "host_timeout";
    case kFailDevicePayload: return "device_payload_bitwise";
    case kFailDeviceHostFlagTimeout: return "device_host_flag_timeout";
    case kFailDeviceAckTimeout: return "device_ack_timeout";
    case kFailSequence: return "sequence";
    case kFailGraph: return "graph";
    default: return "unknown";
    }
}

std::string control_snapshot(const ControlBlock* control) {
    std::ostringstream os;
    os << "state=" << state_name(host_load_u32(&control->state))
       << " reason=" << fail_name(host_load_u32(&control->fail_reason))
       << " round=" << host_load_u32(&control->fail_round)
       << " host_pub=" << host_load_u32(&control->host_publish_seq)
       << " device_done_seq="
       << host_load_u32(&control->device_complete_seq)
       << " host_ack=" << host_load_u32(&control->host_ack_seq)
       << " device_visibility="
       << host_load_u32(&control->device_visibility_count)
       << " host_visibility="
       << host_load_u32(&control->host_visibility_count);
    return os.str();
}

void publish_host_failure(
    ControlBlock* control, uint32_t reason, uint32_t round,
    uint32_t mismatch_index) {
    host_release_store_u32(&control->fail_reason, reason);
    host_release_store_u32(&control->fail_round, round);
    host_release_store_u32(&control->host_mismatch_index, mismatch_index);
    host_release_store_u32(&control->state, kStateFailed);
    // Unblock a device waiting on the current round's ack.
    host_release_store_u32(&control->host_ack_seq, round);
}

inline void publish_device_failure(
    ControlBlock* control, uint32_t reason, uint32_t round,
    uint32_t mismatch_index, uint32_t spins) {
    uncached_store(&control->fail_reason, reason);
    uncached_store(&control->fail_round, round);
    uncached_store(&control->device_mismatch_index, mismatch_index);
    uncached_store(&control->device_timeout_spins, spins);
    device_control_store(&control->state, kStateFailed);
}

void submit_visibility_kernel(
    sycl::queue& queue, ControlBlock* control,
    HostToDevicePayload* h2d_payload,
    DeviceToHostPayload* d2h_payload) {
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<uint32_t, 1> status(
            sycl::range<1>(2), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(kWorkGroup),
                sycl::range<1>(kWorkGroup)),
            [=](sycl::nd_item<1> item) {
                sycl::group<1> group = item.get_group();
                const uint32_t tid =
                    static_cast<uint32_t>(item.get_local_id(0));

                if (tid == 0) {
                    sycl::atomic_fence(
                        sycl::memory_order::acquire,
                        sycl::memory_scope::system);
                    status[0] =
                        device_control_load(&control->magic) == kMagic &&
                        device_control_load(&control->version) == kVersion &&
                        uncached_load(&control->probe_instance) ==
                            kProbeInstance &&
                        device_control_load(&control->state) == kStateProbing
                            ? 0u
                            : kFailControl;
                    status[1] = 0;
                }
                sycl::group_barrier(group);
                if (status[0] != 0) {
                    if (tid == 0) {
                        publish_device_failure(
                            control, status[0], 0, status[1], 0);
                    }
                    return;
                }

                for (uint32_t round = 1;
                     round <= kVisibilityRounds; ++round) {
                    if (tid == 0) {
                        status[0] = 0;
                        status[1] = 0;
                        uint32_t spins = 0;
                        for (;;) {
                            const uint32_t state =
                                device_control_load(&control->state);
                            const uint32_t seq =
                                device_control_load(
                                    &control->host_publish_seq);
                            if (state != kStateProbing) {
                                status[0] = kFailControl;
                                status[1] = state;
                                break;
                            }
                            if (seq == round) break;
                            if (seq > round) {
                                status[0] = kFailSequence;
                                status[1] = seq;
                                break;
                            }
                            if (++spins >= kDeviceSpinLimit) {
                                status[0] =
                                    kFailDeviceHostFlagTimeout;
                                status[1] = spins;
                                break;
                            }
                        }
                    }
                    sycl::group_barrier(
                        group, sycl::memory_scope::system);
                    if (status[0] != 0) {
                        if (tid == 0 && status[0] != kFailControl) {
                            publish_device_failure(
                                control, status[0], round, 0, status[1]);
                        }
                        return;
                    }

                    const size_t slot =
                        static_cast<size_t>((round - 1) % kPayloadSlots);
                    const uint8_t* h2d =
                        h2d_payload->bytes[slot];
                    bool local_mismatch = false;
                    for (size_t i = tid;
                         i < kPayloadBytesPerSlot;
                         i += kWorkGroup) {
                        if (uncached_load(&h2d[i]) != payload_byte(
                                round, static_cast<uint32_t>(i),
                                kHostToDeviceSalt)) {
                            local_mismatch = true;
                        }
                    }
                    if (sycl::any_of_group(group, local_mismatch)) {
                        if (tid == 0) {
                            uint32_t first = 0;
                            while (first < kPayloadBytesPerSlot &&
                                   uncached_load(&h2d[first]) == payload_byte(
                                       round, first,
                                       kHostToDeviceSalt)) {
                                ++first;
                            }
                            status[0] = kFailDevicePayload;
                            status[1] = first;
                        }
                        sycl::group_barrier(group);
                        if (tid == 0) {
                            publish_device_failure(
                                control, status[0], round, status[1], 0);
                        }
                        return;
                    }

                    uint8_t* d2h = d2h_payload->bytes[slot];
                    for (size_t i = tid;
                         i < kPayloadBytesPerSlot;
                         i += kWorkGroup) {
                        uncached_store(&d2h[i], payload_byte(
                            round, static_cast<uint32_t>(i),
                            kDeviceToHostSalt));
                    }
                    sycl::group_barrier(
                        group, sycl::memory_scope::system);
                    if (tid == 0) {
                        device_control_store(
                            &control->device_visibility_count, round);
                        device_control_store(
                            &control->device_complete_seq, round);
                    }

                    if (tid == 0) {
                        status[0] = 0;
                        status[1] = 0;
                        uint32_t spins = 0;
                        for (;;) {
                            const uint32_t state =
                                device_control_load(&control->state);
                            const uint32_t ack =
                                device_control_load(
                                    &control->host_ack_seq);
                            if (state != kStateProbing) {
                                status[0] = kFailControl;
                                status[1] = state;
                                break;
                            }
                            if (ack == round) break;
                            if (ack > round) {
                                status[0] = kFailSequence;
                                status[1] = ack;
                                break;
                            }
                            if (++spins >= kDeviceSpinLimit) {
                                status[0] = kFailDeviceAckTimeout;
                                status[1] = spins;
                                break;
                            }
                        }
                    }
                    sycl::group_barrier(group);
                    if (status[0] != 0) {
                        if (tid == 0 && status[0] != kFailControl) {
                            publish_device_failure(
                                control, status[0], round, 0, status[1]);
                        }
                        return;
                    }
                }

                if (tid == 0) {
                    device_control_store(&control->device_done, 1u);
                    device_control_store(&control->state, kStateReady);
                }
            });
    });
}

bool release_graph_submission(const sycl::event& event) {
    try {
        (void)event.get_info<
            sycl::info::event::command_execution_status>();
        return true;
    } catch (...) {
        return false;
    }
}

bool wait_event_bounded(
    sycl::event event, int timeout_ms, std::string& error) {
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(timeout_ms);
    uint32_t polls = 0;
    for (;;) {
        sycl::info::event_command_status status =
            sycl::info::event_command_status::submitted;
        try {
            status = event.get_info<
                sycl::info::event::command_execution_status>();
        } catch (const sycl::exception& e) {
            error = std::string("event status failed: ") + e.what();
            return false;
        }
        if (status == sycl::info::event_command_status::complete) {
            try {
                event.wait_and_throw();
                return true;
            } catch (const sycl::exception& e) {
                error = std::string("graph failed: ") + e.what();
                return false;
            }
        }
        if (Clock::now() >= deadline) {
            error = "bounded graph wait expired";
            return false;
        }
        if ((++polls & 255u) == 0u) spin_pause();
    }
}

bool wait_for_device_sequence(
    const sycl::event& event, const ControlBlock* control,
    uint32_t want, std::string& error) {
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(kHostTimeoutMs);
    uint32_t polls = 0;
    for (;;) {
        const uint32_t state = host_load_u32(&control->state);
        const uint32_t got =
            host_load_u32(&control->device_complete_seq);
        if (got == want) {
            host_acquire_fence();
            return true;
        }
        if (state == kStateFailed) {
            error = "device sealed handoff: " +
                    control_snapshot(control);
            return false;
        }
        if (got > want) {
            error = "device sequence advanced past expected value: " +
                    control_snapshot(control);
            return false;
        }
        if ((++polls & 1023u) == 0u) {
            try {
                const auto status = event.get_info<
                    sycl::info::event::command_execution_status>();
                if (status ==
                    sycl::info::event_command_status::complete) {
                    error =
                        "graph ended before device sequence: " +
                        control_snapshot(control);
                    return false;
                }
            } catch (const sycl::exception& e) {
                error = std::string("event status failed: ") + e.what();
                return false;
            }
        }
        if (Clock::now() >= deadline) {
            error = "host timeout waiting for device sequence " +
                    std::to_string(want) + ": " +
                    control_snapshot(control);
            return false;
        }
        spin_pause();
    }
}

struct ProbeResult {
    bool ok = false;
    uint64_t host_to_device_bytes = 0;
    uint64_t device_to_host_bytes = 0;
    std::string error;
};

ProbeResult service_visibility_probe(
    sycl::queue& queue, const sycl::event& event,
    ControlBlock* control, HostToDevicePayload* h2d_payload,
    DeviceToHostPayload* d2h_payload) {
    ProbeResult result;
    for (uint32_t round = 1; round <= kVisibilityRounds; ++round) {
        const size_t slot =
            static_cast<size_t>((round - 1) % kPayloadSlots);
        uint8_t* h2d = h2d_payload->bytes[slot];
        fill_payload(
            h2d, kPayloadBytesPerSlot, round, kHostToDeviceSalt);
        flush_cpu_cache(h2d, kPayloadBytesPerSlot);
        host_release_store_u32(&control->host_publish_seq, round);
        result.host_to_device_bytes += kPayloadBytesPerSlot;

        if (!wait_for_device_sequence(
                event, control, round, result.error)) {
            if (host_load_u32(&control->state) != kStateFailed) {
                publish_host_failure(
                    control, kFailHostTimeout, round, 0);
            }
            return result;
        }

        uint8_t* d2h = d2h_payload->bytes[slot];
        flush_cpu_cache(d2h, kPayloadBytesPerSlot);
        size_t mismatch_index = 0;
        const bool bitwise = verify_payload(
            d2h, kPayloadBytesPerSlot, round, kDeviceToHostSalt,
            mismatch_index);
        flush_cpu_cache(d2h, kPayloadBytesPerSlot);
        if (!bitwise) {
            result.error =
                "device->host payload mismatch at round " +
                std::to_string(round) + " byte " +
                std::to_string(mismatch_index);
            publish_host_failure(
                control, kFailHostPayload, round,
                static_cast<uint32_t>(mismatch_index));
            return result;
        }
        result.device_to_host_bytes += kPayloadBytesPerSlot;

        host_release_store_u32(
            &control->host_visibility_count, round);
        host_release_store_u32(&control->host_ack_seq, round);
    }
    result.ok = true;
    (void)queue;
    return result;
}

class HandoffGate {
public:
    bool is_open() const { return open_; }
    void open() { open_ = true; }
    void seal() { open_ = false; }

private:
    bool open_ = false;
};

}  // namespace

int main() {
    std::cout << "[HOST RING] probe=start rounds=" << kVisibilityRounds
              << '/' << kVisibilityRounds
              << " control_bytes=" << kControlBytes
              << " payload_slots=" << kPayloadSlots
              << " payload_bytes_per_slot=" << kPayloadBytesPerSlot
              << '\n';

    HandoffGate handoff;
    bool kernel_in_flight = false;
    bool gpu_complete = false;
    ControlBlock* control = nullptr;
    HostToDevicePayload* h2d_payload = nullptr;
    DeviceToHostPayload* d2h_payload = nullptr;
    ze_context_handle_t ze_context = nullptr;

    try {
        sycl::queue queue(
            sycl::gpu_selector_v, sycl::property::queue::in_order{});
        const sycl::context sycl_context = queue.get_context();
        if (sycl_context.get_backend() !=
            sycl::backend::ext_oneapi_level_zero) {
            throw std::runtime_error(
                "host ring requires the Level Zero SYCL backend");
        }
        ze_context =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                sycl_context);
        if (!ze_context) {
            throw std::runtime_error(
                "Level Zero returned a null SYCL context");
        }

        control = static_cast<ControlBlock*>(allocate_level_zero_host(
            ze_context, sizeof(ControlBlock), alignof(ControlBlock),
            true, "64B uncached control"));
        h2d_payload = static_cast<HostToDevicePayload*>(
            allocate_level_zero_host(
                ze_context, sizeof(HostToDevicePayload),
                alignof(HostToDevicePayload), false,
                "cached pinned host-to-device payload"));
        d2h_payload = static_cast<DeviceToHostPayload*>(
            allocate_level_zero_host(
                ze_context, sizeof(DeviceToHostPayload),
                alignof(DeviceToHostPayload), true,
                "uncached device-to-host payload"));

        const bool overlap =
            ranges_overlap(control, sizeof(ControlBlock),
                           h2d_payload, sizeof(HostToDevicePayload)) ||
            ranges_overlap(control, sizeof(ControlBlock),
                           d2h_payload, sizeof(DeviceToHostPayload)) ||
            ranges_overlap(h2d_payload, sizeof(HostToDevicePayload),
                           d2h_payload, sizeof(DeviceToHostPayload));
        if (overlap) {
            throw std::runtime_error(
                "control and payload allocations overlap");
        }

        std::memset(
            static_cast<void*>(control), 0, sizeof(ControlBlock));
        std::memset(
            static_cast<void*>(h2d_payload), 0,
            sizeof(HostToDevicePayload));
        std::memset(
            static_cast<void*>(d2h_payload), 0,
            sizeof(DeviceToHostPayload));
        flush_cpu_cache(h2d_payload, sizeof(HostToDevicePayload));
        flush_cpu_cache(d2h_payload, sizeof(DeviceToHostPayload));

        host_release_store_u32(&control->magic, kMagic);
        host_release_store_u32(&control->version, kVersion);
        control->probe_instance = kProbeInstance;
        host_release_store_u32(&control->state, kStateProbing);

        std::cout << "[HOST RING SETUP] backend=level_zero"
                  << " control=uncached_64B"
                  << " control_addr=0x" << std::hex
                  << reinterpret_cast<uintptr_t>(control) << std::dec
                  << " h2d_payload=pinned_default"
                  << " h2d_bytes=" << sizeof(HostToDevicePayload)
                  << " h2d_addr=0x" << std::hex
                  << reinterpret_cast<uintptr_t>(h2d_payload) << std::dec
                  << " d2h_payload=uncached"
                  << " d2h_bytes=" << sizeof(DeviceToHostPayload)
                  << " d2h_addr=0x" << std::hex
                  << reinterpret_cast<uintptr_t>(d2h_payload) << std::dec
                  << " overlap=" << (overlap ? 1 : 0)
                  << " handoff=sealed\n";

        if (!queue.get_device().has(
                sycl::aspect::ext_oneapi_limited_graph) &&
            !queue.get_device().has(
                sycl::aspect::ext_oneapi_graph)) {
            throw std::runtime_error(
                "device does not support SYCL graph capture");
        }

        auto executable = [&]() {
            ex::command_graph<ex::graph_state::modifiable> recording(
                queue.get_context(), queue.get_device());
            recording.begin_recording(queue);
            try {
                submit_visibility_kernel(
                    queue, control, h2d_payload, d2h_payload);
                recording.end_recording(queue);
            } catch (...) {
                recording.end_recording(queue);
                throw;
            }
            return recording.finalize();
        }();
        std::cout << "[HOST RING GRAPH] capture=ok graph=1 kernel=1"
                     " host_service=in_place"
                     " handoff=sealed_until_pass\n";

        sycl::event event = queue.ext_oneapi_graph(executable);
        kernel_in_flight = true;
        if (!release_graph_submission(event)) {
            throw std::runtime_error(
                "failed to query graph event after launch");
        }

        ProbeResult probe =
            service_visibility_probe(
                queue, event, control, h2d_payload, d2h_payload);
        std::string wait_error;
        const bool graph_done =
            wait_event_bounded(event, kHostTimeoutMs, wait_error);
        gpu_complete = graph_done;

        if (!probe.ok) {
            handoff.seal();
            std::cerr << "[HOST RING] visibility=NO-GO"
                      << " handoff=disabled"
                      << " error=\"" << probe.error << "\""
                      << " graph_done=" << (graph_done ? 1 : 0)
                      << " snapshot=\"" << control_snapshot(control)
                      << "\"";
            if (!wait_error.empty()) {
                std::cerr << " wait_error=\"" << wait_error << "\"";
            }
            std::cerr << '\n';
        } else if (!graph_done) {
            handoff.seal();
            std::cerr << "[HOST RING] visibility=NO-GO"
                      << " handoff=disabled"
                      << " error=\"graph completion wait failed\""
                      << " wait_error=\"" << wait_error << "\""
                      << " snapshot=\"" << control_snapshot(control)
                      << "\"\n";
        } else {
            host_acquire_fence();
            const bool final_ok =
                host_load_u32(&control->state) == kStateReady &&
                host_load_u32(&control->device_done) == 1u &&
                host_load_u32(&control->host_publish_seq) ==
                    kVisibilityRounds &&
                host_load_u32(&control->device_complete_seq) ==
                    kVisibilityRounds &&
                host_load_u32(&control->host_ack_seq) ==
                    kVisibilityRounds &&
                host_load_u32(&control->host_visibility_count) ==
                    kVisibilityRounds &&
                host_load_u32(&control->device_visibility_count) ==
                    kVisibilityRounds;
            if (final_ok) {
                handoff.open();
                std::cout
                    << "[HOST RING] visibility=pass"
                    << " h2d=" << kVisibilityRounds << '/'
                    << kVisibilityRounds
                    << " d2h=" << kVisibilityRounds << '/'
                    << kVisibilityRounds
                    << " payload_bitwise=pass"
                    << " h2d_bytes=" << probe.host_to_device_bytes
                    << " d2h_bytes=" << probe.device_to_host_bytes
                    << " control_acq_rel=pass"
                    << " handoff=enabled\n";
            } else {
                handoff.seal();
                std::cerr << "[HOST RING] visibility=NO-GO"
                          << " handoff=disabled"
                          << " error=\"final control invariant failed\""
                          << " snapshot=\"" << control_snapshot(control)
                          << "\"\n";
            }
        }

        const int exit_code = handoff.is_open() ? 0 : 2;
        if (gpu_complete || !kernel_in_flight) {
            (void)zeMemFree(ze_context, d2h_payload);
            d2h_payload = nullptr;
            (void)zeMemFree(ze_context, h2d_payload);
            h2d_payload = nullptr;
            (void)zeMemFree(ze_context, control);
            control = nullptr;
        } else {
            std::cerr
                << "[HOST RING] allocations retained until process exit"
                   " because graph completion was not proven\n";
        }
        return exit_code;
    } catch (const std::exception& e) {
        handoff.seal();
        std::cerr << "[HOST RING] visibility=NO-GO"
                  << " handoff=disabled"
                  << " fatal=\"" << e.what() << "\"\n";
        if (gpu_complete || !kernel_in_flight) {
            if (d2h_payload) {
                (void)zeMemFree(ze_context, d2h_payload);
                d2h_payload = nullptr;
            }
            if (h2d_payload) {
                (void)zeMemFree(ze_context, h2d_payload);
                h2d_payload = nullptr;
            }
            if (control) {
                (void)zeMemFree(ze_context, control);
                control = nullptr;
            }
        }
        return 3;
    }
}
