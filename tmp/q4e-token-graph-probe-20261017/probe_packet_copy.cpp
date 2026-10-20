// Packet-copy microprobe for the Qwen4Exp whole-token graph.
//
// The probe keeps the scalar rank-map copy from P3 but separates the small
// uncached host control block from the bulk pinned payload. Four single-launch
// variants copy the same active payload:
//   1. scalar element baseline
//   2. linear vec4
//   3. one work-group per active row, work-group stride
//   4. fixed-grid stride loop
//
// This file only defines the GPU experiment. build_packet_copy.ps1 compiles it
// against AilaLib.lib and never launches it.

#include "core/Context.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <immintrin.h>

namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kHidden = 2560;
constexpr int kModelTopK = 10;
constexpr int kMaxPacketRows = 16;
constexpr int kPayloadVecs = kHidden / 4;
constexpr uint32_t kGateValid = 1u;
constexpr uint32_t kTailSentinel = 0xA5A55A5Au;

using Vec4U32 = sycl::vec<uint32_t, 4>;
static_assert(kHidden % 4 == 0);
static_assert(sizeof(Vec4U32) == 16);

using UncachedRead = decltype(oex::properties(
    iex::read_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));

template <class T>
inline T uncached_load(const T* p) {
    oex::annotated_ptr<T, UncachedRead> a(const_cast<T*>(p));
    return a[0];
}

// Keep metadata and rank_map in one cache line. Qwen4Exp currently has top_k=10.
struct alignas(64) PacketControl {
    uint32_t ready_seq;
    uint32_t rows;
    uint32_t top_k;
    uint32_t reserved0;
    int32_t rank_to_row[kModelTopK];
    uint32_t reserved1[2];
};

struct alignas(64) DeviceGate {
    uint32_t gate_state;
    uint32_t reserved[15];
};

struct alignas(64) HostPayload {
    uint32_t values[kMaxPacketRows * kHidden];
};

struct alignas(64) DevicePacket {
    int32_t rank_to_row[kMaxPacketRows];
    uint32_t values[kMaxPacketRows * kHidden];
};

static_assert(sizeof(PacketControl) == 64);
static_assert(sizeof(DeviceGate) == 64);
static_assert(sizeof(HostPayload) == kMaxPacketRows * kHidden * sizeof(uint32_t));
static_assert(sizeof(DevicePacket) ==
              (kMaxPacketRows + kMaxPacketRows * kHidden) * sizeof(uint32_t));
static_assert(alignof(HostPayload) >= alignof(Vec4U32));
static_assert(offsetof(DevicePacket, values) % alignof(Vec4U32) == 0);

inline uint32_t device_load_u32(const uint32_t* ptr) {
    sycl::atomic_ref<
        uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
        value(*const_cast<uint32_t*>(ptr));
    return value.load(sycl::memory_order::acquire);
}

inline bool gate_valid(const DeviceGate* gate) {
    return device_load_u32(&gate->gate_state) == kGateValid;
}

inline void copy_rank_slot(const PacketControl* control,
                           DevicePacket* device_packet,
                           size_t slot, int top_k, bool valid) {
    device_packet->rank_to_row[slot] =
        valid && slot < static_cast<size_t>(top_k)
            ? uncached_load(&control->rank_to_row[slot])
            : -1;
}

struct ScalarElementKernel;
struct LinearVec4Kernel;
struct WorkGroupStrideVec4Kernel;
struct GridStrideVec4Kernel;

sycl::event submit_scalar_element(sycl::queue& q,
                                  const PacketControl* control,
                                  const HostPayload* payload,
                                  DevicePacket* device_packet,
                                  const DeviceGate* gate,
                                  int top_k, int rows) {
    const size_t payload_count =
        static_cast<size_t>(rows) * static_cast<size_t>(kHidden);
    return q.parallel_for<ScalarElementKernel>(
        sycl::range<1>(kMaxPacketRows + payload_count),
        [=](sycl::id<1> id) {
            const size_t i = id[0];
            const bool valid = gate_valid(gate);
            if (i < kMaxPacketRows) {
                copy_rank_slot(control, device_packet, i, top_k, valid);
                return;
            }
            if (valid) {
                const size_t value_index = i - kMaxPacketRows;
                device_packet->values[value_index] =
                    payload->values[value_index];
            }
        });
}

sycl::event submit_linear_vec4(sycl::queue& q,
                               const PacketControl* control,
                               const HostPayload* payload,
                               DevicePacket* device_packet,
                               const DeviceGate* gate,
                               int top_k, int rows) {
    const size_t vec_count =
        static_cast<size_t>(rows) * static_cast<size_t>(kPayloadVecs);
    return q.parallel_for<LinearVec4Kernel>(
        sycl::range<1>(kMaxPacketRows + vec_count),
        [=](sycl::id<1> id) {
            const size_t i = id[0];
            const bool valid = gate_valid(gate);
            if (i < kMaxPacketRows) {
                copy_rank_slot(control, device_packet, i, top_k, valid);
                return;
            }
            if (valid) {
                const size_t vec_index = i - kMaxPacketRows;
                const auto* src =
                    reinterpret_cast<const Vec4U32*>(payload->values);
                auto* dst =
                    reinterpret_cast<Vec4U32*>(device_packet->values);
                dst[vec_index] = src[vec_index];
            }
        });
}

sycl::event submit_work_group_stride_vec4(
    sycl::queue& q, const PacketControl* control,
    const HostPayload* payload, DevicePacket* device_packet,
    const DeviceGate* gate, int top_k, int rows,
    size_t work_group_size) {
    const size_t groups = std::max<size_t>(1, static_cast<size_t>(rows));
    return q.parallel_for<WorkGroupStrideVec4Kernel>(
        sycl::nd_range<1>(sycl::range<1>(groups * work_group_size),
                          sycl::range<1>(work_group_size)),
        [=](sycl::nd_item<1> item) {
            const size_t global = item.get_global_id(0);
            const size_t group = item.get_group(0);
            const size_t local = item.get_local_id(0);
            const bool valid = gate_valid(gate);
            if (global < kMaxPacketRows) {
                copy_rank_slot(control, device_packet, global, top_k, valid);
            }
            if (!valid || group >= static_cast<size_t>(rows)) return;

            const auto* src =
                reinterpret_cast<const Vec4U32*>(payload->values);
            auto* dst =
                reinterpret_cast<Vec4U32*>(device_packet->values);
            const size_t row_base =
                group * static_cast<size_t>(kPayloadVecs);
            for (size_t vec = local; vec < kPayloadVecs;
                 vec += work_group_size) {
                dst[row_base + vec] = src[row_base + vec];
            }
        });
}

sycl::event submit_grid_stride_vec4(
    sycl::queue& q, const PacketControl* control,
    const HostPayload* payload, DevicePacket* device_packet,
    const DeviceGate* gate, int top_k, int rows,
    size_t global_size, size_t work_group_size) {
    const size_t vec_count =
        static_cast<size_t>(rows) * static_cast<size_t>(kPayloadVecs);
    return q.parallel_for<GridStrideVec4Kernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size),
                          sycl::range<1>(work_group_size)),
        [=](sycl::nd_item<1> item) {
            const size_t global = item.get_global_id(0);
            const bool valid = gate_valid(gate);
            if (global < kMaxPacketRows) {
                copy_rank_slot(control, device_packet, global, top_k, valid);
            }
            if (!valid) return;

            const auto* src =
                reinterpret_cast<const Vec4U32*>(payload->values);
            auto* dst =
                reinterpret_cast<Vec4U32*>(device_packet->values);
            for (size_t vec = global; vec < vec_count;
                 vec += global_size) {
                dst[vec] = src[vec];
            }
        });
}

enum class Arm {
    ScalarElement,
    LinearVec4,
    WorkGroupStrideVec4,
    GridStrideVec4,
};

const char* arm_name(Arm arm) {
    switch (arm) {
    case Arm::ScalarElement: return "scalar_element";
    case Arm::LinearVec4: return "linear_vec4";
    case Arm::WorkGroupStrideVec4: return "wg_stride_vec4";
    case Arm::GridStrideVec4: return "grid_stride_vec4";
    }
    return "unknown";
}

struct Options {
    int top_k = kModelTopK;
    int rows = kModelTopK;
    int warmup = 20;
    int replays = 200;
    size_t work_group_size = 256;
    size_t global_size = 4096;
    std::string payload_alloc = "pinned";
};

void print_help() {
    std::cout
        << "probe_packet_copy [options]\n"
        << "  --top-k N        scalar rank slots, 1..10 (default 10)\n"
        << "  --rows N         active CPU payload rows, 0..top-k (default 10)\n"
        << "  --warmup N       untimed launches per arm (default 20)\n"
        << "  --replays N      timed launches per arm (default 200)\n"
        << "  --wg N           work-group size (default 256)\n"
        << "  --grid N         grid-stride work-items, multiple of --wg "
           "(default 4096)\n"
        << "  --payload-alloc  pinned|uncached (default pinned)\n"
        << "  --help           print this text\n\n"
        << "The executable initializes the GPU and must run under the global "
           "Aila GPU mutex.\n";
}

int parse_int(const char* text, const char* option) {
    try {
        size_t consumed = 0;
        const std::string value(text);
        const int parsed = std::stoi(value, &consumed, 10);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + option +
                                 ": " + text);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        }
        if (arg == "--payload-alloc") {
            if (++i >= argc) {
                throw std::runtime_error("--payload-alloc requires a value");
            }
            options.payload_alloc = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + arg);
        }
        if (arg == "--top-k") {
            options.top_k = parse_int(argv[++i], "--top-k");
        } else if (arg == "--rows") {
            options.rows = parse_int(argv[++i], "--rows");
        } else if (arg == "--warmup") {
            options.warmup = parse_int(argv[++i], "--warmup");
        } else if (arg == "--replays") {
            options.replays = parse_int(argv[++i], "--replays");
        } else if (arg == "--wg") {
            options.work_group_size =
                static_cast<size_t>(parse_int(argv[++i], "--wg"));
        } else if (arg == "--grid") {
            options.global_size =
                static_cast<size_t>(parse_int(argv[++i], "--grid"));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (options.top_k < 1 || options.top_k > kModelTopK) {
        throw std::runtime_error("--top-k must be in 1..10");
    }
    if (options.rows < 0 || options.rows > options.top_k) {
        throw std::runtime_error("--rows must be in 0..top-k");
    }
    if (options.warmup < 0 || options.warmup > 100000) {
        throw std::runtime_error("--warmup must be in 0..100000");
    }
    if (options.replays < 1 || options.replays > 100000) {
        throw std::runtime_error("--replays must be in 1..100000");
    }
    if (options.work_group_size < 32 || options.work_group_size > 1024) {
        throw std::runtime_error("--wg must be in 32..1024");
    }
    if (options.global_size < options.work_group_size ||
        options.global_size > 65536 ||
        options.global_size % options.work_group_size != 0) {
        throw std::runtime_error(
            "--grid must be a multiple of --wg in [--wg, 65536]");
    }
    if (options.payload_alloc != "pinned" &&
        options.payload_alloc != "uncached") {
        throw std::runtime_error("--payload-alloc must be pinned or uncached");
    }
    return options;
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = p * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) return values[lower];
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double median(const std::vector<double>& values) {
    return percentile(values, 0.5);
}

uint32_t test_pattern(int row, int column) {
    if (row == 0) {
        static constexpr std::array<uint32_t, 8> patterns{
            0x00000000u, 0x80000000u, 0x7FC00001u, 0x7F800000u,
            0xFF800000u, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x00000001u};
        if (column < static_cast<int>(patterns.size())) {
            return patterns[static_cast<size_t>(column)];
        }
    }
    uint32_t x = static_cast<uint32_t>(row + 1) * 0x9E3779B9u ^
                 static_cast<uint32_t>(column) * 0x85EBCA6Bu ^
                 0xC2B2AE35u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

std::vector<int32_t> expected_rank_map(int top_k, int rows) {
    std::vector<int32_t> ranks(kMaxPacketRows, -1);
    for (int rank = 0; rank < top_k && rank < rows; ++rank) {
        ranks[static_cast<size_t>(rank)] = rows - 1 - rank;
    }
    return ranks;
}

void fill_control(PacketControl* control, int top_k, int rows,
                  const std::vector<int32_t>& ranks) {
    std::memset(control, 0, sizeof(*control));
    control->ready_seq = 1;
    control->rows = static_cast<uint32_t>(rows);
    control->top_k = static_cast<uint32_t>(top_k);
    for (int rank = 0; rank < kModelTopK; ++rank) {
        control->rank_to_row[rank] = ranks[static_cast<size_t>(rank)];
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
}

void fill_payload(HostPayload* payload, int rows) {
    for (int row = 0; row < kMaxPacketRows; ++row) {
        for (int column = 0; column < kHidden; ++column) {
            const size_t index =
                static_cast<size_t>(row) * kHidden + column;
            payload->values[index] =
                row < rows ? test_pattern(row, column) : kTailSentinel;
        }
    }
}

void reset_device_packet(Context& ctx, DevicePacket* device_packet) {
    static std::vector<uint8_t> storage;
    if (storage.size() != sizeof(DevicePacket)) {
        storage.resize(sizeof(DevicePacket));
        auto* packet = reinterpret_cast<DevicePacket*>(storage.data());
        std::memset(packet, 0, sizeof(*packet));
        std::fill(std::begin(packet->rank_to_row),
                  std::end(packet->rank_to_row), -1);
        std::fill(std::begin(packet->values),
                  std::end(packet->values), kTailSentinel);
    }
    ctx.memcpy_h2d(device_packet, storage.data(), storage.size());
}

struct Verification {
    bool rank_map = false;
    bool payload = false;
    bool tail = false;
    bool all() const { return rank_map && payload && tail; }
};

Verification verify_packet(Context& ctx, const DevicePacket* device_packet,
                           const HostPayload* payload,
                           const std::vector<int32_t>& expected_ranks,
                           int rows) {
    DevicePacket observed{};
    ctx.memcpy_d2h(&observed, device_packet, sizeof(observed));

    Verification result;
    result.rank_map =
        std::memcmp(observed.rank_to_row, expected_ranks.data(),
                    sizeof(observed.rank_to_row)) == 0;
    result.payload = true;
    result.tail = true;
    const size_t active_values =
        static_cast<size_t>(rows) * static_cast<size_t>(kHidden);
    if (active_values != 0) {
        result.payload =
            std::memcmp(observed.values, payload->values,
                        active_values * sizeof(uint32_t)) == 0;
    }
    for (size_t i = active_values;
         i < static_cast<size_t>(kMaxPacketRows) * kHidden; ++i) {
        if (observed.values[i] != kTailSentinel) {
            result.tail = false;
            break;
        }
    }
    return result;
}

struct ArmResult {
    Arm arm;
    Verification verification;
    std::vector<double> kernel_us;
    std::vector<double> launch_to_done_us;
    std::vector<double> submit_us;
};

sycl::event launch_arm(sycl::queue& q, Arm arm,
                       const PacketControl* control,
                       const HostPayload* payload,
                       DevicePacket* device_packet,
                       const DeviceGate* gate,
                       const Options& options) {
    switch (arm) {
    case Arm::ScalarElement:
        return submit_scalar_element(
            q, control, payload, device_packet, gate,
            options.top_k, options.rows);
    case Arm::LinearVec4:
        return submit_linear_vec4(
            q, control, payload, device_packet, gate,
            options.top_k, options.rows);
    case Arm::WorkGroupStrideVec4:
        return submit_work_group_stride_vec4(
            q, control, payload, device_packet, gate,
            options.top_k, options.rows, options.work_group_size);
    case Arm::GridStrideVec4:
        return submit_grid_stride_vec4(
            q, control, payload, device_packet, gate,
            options.top_k, options.rows, options.global_size,
            options.work_group_size);
    }
    throw std::runtime_error("unknown copy arm");
}

ArmResult measure_arm(Context& ctx, Arm arm,
                      const PacketControl* control,
                      const HostPayload* payload,
                      DevicePacket* device_packet,
                      const DeviceGate* gate,
                      const std::vector<int32_t>& expected_ranks,
                      const Options& options) {
    sycl::queue& q = ctx.queue();
    reset_device_packet(ctx, device_packet);

    const int untimed_launches = std::max(1, options.warmup);
    for (int i = 0; i < untimed_launches; ++i) {
        launch_arm(q, arm, control, payload, device_packet, gate, options)
            .wait_and_throw();
    }

    ArmResult result;
    result.arm = arm;
    result.verification = verify_packet(
        ctx, device_packet, payload, expected_ranks, options.rows);
    result.kernel_us.reserve(static_cast<size_t>(options.replays));
    result.launch_to_done_us.reserve(
        static_cast<size_t>(options.replays));
    result.submit_us.reserve(static_cast<size_t>(options.replays));

    for (int i = 0; i < options.replays; ++i) {
        const auto begin = Clock::now();
        sycl::event event =
            launch_arm(q, arm, control, payload, device_packet, gate, options);
        const auto submitted = Clock::now();
        event.wait_and_throw();
        const auto finished = Clock::now();

        const uint64_t start_ns =
            event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        const uint64_t end_ns =
            event.get_profiling_info<
                sycl::info::event_profiling::command_end>();
        result.kernel_us.push_back(
            static_cast<double>(end_ns - start_ns) / 1000.0);
        result.launch_to_done_us.push_back(
            elapsed_us(begin, finished));
        result.submit_us.push_back(elapsed_us(begin, submitted));
    }
    const Verification final_verification = verify_packet(
        ctx, device_packet, payload, expected_ranks, options.rows);
    result.verification.rank_map &= final_verification.rank_map;
    result.verification.payload &= final_verification.payload;
    result.verification.tail &= final_verification.tail;
    return result;
}

void print_arm_result(const ArmResult& result, const Options& options) {
    const size_t payload_bytes =
        static_cast<size_t>(options.rows) * kHidden * sizeof(uint32_t);
    const size_t rank_bytes =
        static_cast<size_t>(kMaxPacketRows) * sizeof(int32_t);
    const double kernel_us = median(result.kernel_us);
    const double launch_us = median(result.launch_to_done_us);
    const double submit_us = median(result.submit_us);
    const double traffic_gib = static_cast<double>(payload_bytes + rank_bytes) /
                               (1024.0 * 1024.0 * 1024.0);
    const double kernel_gib_s =
        kernel_us > 0.0 ? traffic_gib / (kernel_us / 1.0e6) : 0.0;

    std::cout << "[PACKET COPY ARM]"
              << " name=" << arm_name(result.arm)
              << " bit_exact=" << (result.verification.all() ? 1 : 0)
              << " rank_map=" << (result.verification.rank_map ? 1 : 0)
              << " payload=" << (result.verification.payload ? 1 : 0)
              << " tail_untouched=" << (result.verification.tail ? 1 : 0)
              << " samples=" << result.kernel_us.size()
              << " kernel_median_us=" << std::fixed << std::setprecision(3)
              << kernel_us
              << " kernel_p10_us=" << percentile(result.kernel_us, 0.10)
              << " kernel_p90_us=" << percentile(result.kernel_us, 0.90)
              << " launch_done_median_us=" << launch_us
              << " submit_median_us=" << submit_us
              << " payload_bytes=" << payload_bytes
              << " rank_bytes=" << rank_bytes
              << " kernel_GiB_s=" << kernel_gib_s
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        Context ctx(nullptr, true);
        sycl::queue& q = ctx.queue();
        if (!q.get_device().is_gpu()) {
            throw std::runtime_error(
                "packet copy probe requires a GPU device");
        }

        auto* control = static_cast<PacketControl*>(
            ctx.alloc_host_uncached(sizeof(PacketControl)));
        void* payload_storage =
            options.payload_alloc == "uncached"
                ? ctx.alloc_host_uncached(sizeof(HostPayload))
                : ctx.alloc_host(sizeof(HostPayload));
        auto* payload = static_cast<HostPayload*>(payload_storage);
        auto* device_packet = static_cast<DevicePacket*>(
            ctx.alloc_device(sizeof(DevicePacket)));
        auto* gate =
            static_cast<DeviceGate*>(ctx.alloc_device(sizeof(DeviceGate)));
        if (reinterpret_cast<uintptr_t>(control) %
                alignof(PacketControl) != 0 ||
            reinterpret_cast<uintptr_t>(payload) %
                alignof(Vec4U32) != 0 ||
            reinterpret_cast<uintptr_t>(device_packet->values) %
                alignof(Vec4U32) != 0 ||
            reinterpret_cast<uintptr_t>(gate) %
                alignof(DeviceGate) != 0) {
            throw std::runtime_error(
                "packet copy probe allocation alignment is invalid");
        }

        const std::vector<int32_t> expected_ranks =
            expected_rank_map(options.top_k, options.rows);
        fill_control(control, options.top_k, options.rows, expected_ranks);
        fill_payload(payload, options.rows);

        DeviceGate gate_host{};
        gate_host.gate_state = kGateValid;
        ctx.memcpy_h2d(gate, &gate_host, sizeof(gate_host));

        std::cout << "[PACKET COPY CONFIG]"
                  << " device=\"" << q.get_device().get_info<
                         sycl::info::device::name>() << "\""
                  << " top_k=" << options.top_k
                  << " rows=" << options.rows
                  << " warmup=" << options.warmup
                  << " replays=" << options.replays
                  << " wg=" << options.work_group_size
                  << " grid=" << options.global_size
                  << " payload_alloc=" << options.payload_alloc
                  << " control_bytes=" << sizeof(PacketControl)
                  << " control_alloc=uncached"
                  << " payload_bytes_full=" << sizeof(HostPayload)
                  << " host_mapping=usm_host"
                  << " rank_map=scalar"
                  << " queue=in_order_profiling\n";

        const std::array<Arm, 4> arms{
            Arm::ScalarElement,
            Arm::LinearVec4,
            Arm::WorkGroupStrideVec4,
            Arm::GridStrideVec4,
        };
        std::vector<ArmResult> results;
        results.reserve(arms.size());
        bool all_ok = true;
        for (Arm arm : arms) {
            ArmResult result = measure_arm(
                ctx, arm, control, payload, device_packet, gate,
                expected_ranks, options);
            all_ok &= result.verification.all();
            print_arm_result(result, options);
            results.push_back(std::move(result));
        }

        const ArmResult* best = nullptr;
        for (const ArmResult& result : results) {
            if (result.verification.all() &&
                (!best ||
                 median(result.kernel_us) < median(best->kernel_us))) {
                best = &result;
            }
        }
        std::cout << "[PACKET COPY VERDICT]"
                  << " all_ok=" << (all_ok ? 1 : 0)
                  << " bit_exact=" << (all_ok ? 1 : 0);
        if (all_ok && best) {
            const double baseline_us = median(results.front().kernel_us);
            const double best_us = median(best->kernel_us);
            const double speedup =
                best_us > 0.0 ? baseline_us / best_us : 0.0;
            std::cout
                << " best_arm=" << arm_name(best->arm)
                << " baseline_kernel_median_us=" << std::fixed
                << std::setprecision(3) << baseline_us
                << " best_kernel_median_us=" << best_us
                << " speedup_vs_scalar=" << speedup;
        } else {
            std::cout << " best_arm=none"
                      << " speedup_vs_scalar=0.000";
        }
        std::cout << '\n';

        ctx.synchronize();
        ctx.free_device(gate);
        ctx.free_device(device_packet);
        ctx.free_host(payload);
        ctx.free_host(control);
        return all_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[PACKET COPY ERROR] " << error.what() << '\n';
        return 1;
    }
}
