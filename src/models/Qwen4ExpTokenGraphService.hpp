#pragma once

// Host/director side service for the Qwen4Exp whole-token handoff.
//
// This is intentionally one fixed ring, not a general graph framework:
//   - 64-byte control block, allocated with Level Zero uncached host USM.
//   - one independent bulk CPU packet, also allocated uncached.
//   - the graph body publishes route_sequence and consumes cpu_flag.
//   - host publishes the packet only after its sequence, gate, and shape
//     checks pass; a failure is sticky and never publishes a ready packet.
//
// The packet and control allocations are host USM because device control
// kernels poll them while the host services CPU experts. They must not be
// replaced with cached pinned allocations: the P3 experiment demonstrated
// that cached reusable payload can become stale after slot reuse.

#include "models/Qwen4ExpTokenGraphRuntime.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

#if defined(AILA_HAS_LEVEL_ZERO)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif

#if defined(_MSC_VER)
#include <immintrin.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace aila {
namespace token_graph_service {

inline constexpr std::uint32_t kAbortSequence = 0xffffffffu;
inline constexpr std::uint32_t kPacketMagic = 0x51475356u;  // "QGSV"
inline constexpr std::uint32_t kPacketTailSentinel = 0xcdcdcdcdu;
inline constexpr std::uint32_t kPacketVectorElements = 16u;
inline constexpr std::uint32_t kPacketRowVectors =
    kTokenGraphHiddenElements / kPacketVectorElements;
inline constexpr std::uint32_t kDefaultDeviceSpinLimit = 20'000'000u;

// Reserved packet words used by this service protocol. The existing P3 wire
// layout has fourteen reserved words; these indices keep the packet layout
// unchanged and leave the rest available for future handoff metadata.
inline constexpr std::uint32_t kPacketWordMagic = 0u;
inline constexpr std::uint32_t kPacketWordGate = 1u;
inline constexpr std::uint32_t kPacketWordTopK = 2u;
inline constexpr std::uint32_t kPacketWordHidden = 3u;
inline constexpr std::uint32_t kPacketWordSequence = 4u;
inline constexpr std::uint32_t kEpochPacketRowsWord = 0u;
inline constexpr std::uint32_t kEpochSequenceWord = 1u;
// Device-side diagnostic accumulator: total packet-wait spin iterations for the
// current replay. Written by submit_packet_wait (one atomic add per layer) and
// relayed to the host ring by submit_publish_done as control->reserved[0].
inline constexpr std::uint32_t kEpochSpinItersWord = 2u;
inline constexpr std::uint32_t kEpochReservedWords = 13u;

static_assert(kTokenGraphHiddenElements % kPacketVectorElements == 0u);
static_assert(kTokenGraphPacketSlots <= 32u,
              "packet shape validation uses a 32-bit row bitset");
static_assert(kPacketWordSequence < 14u);
static_assert(kEpochPacketRowsWord < kEpochReservedWords);
static_assert(kEpochSequenceWord < kEpochReservedWords);
static_assert(sizeof(TokenGraphControlRing) == 64u);
static_assert(sizeof(TokenGraphRouteRing) == 128u);
static_assert(sizeof(TokenGraphCpuPacketRing) == 169088u);
static_assert(sizeof(TokenGraphDeviceEpoch) == 64u);
static_assert(std::is_trivially_copyable<TokenGraphCpuPacketRing>::value);
static_assert(std::is_standard_layout<TokenGraphCpuPacketRing>::value);

using VecPacketU32 =
    sycl::vec<std::uint32_t, kPacketVectorElements>;
static_assert(sizeof(VecPacketU32) == 64u);

struct RouteView {
    std::int32_t ids[kTokenGraphRouteRanks]{};
    float weights[kTokenGraphRouteRanks]{};
};

using UncachedRead = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::read_hint<
        sycl::ext::intel::experimental::cache_control<
            sycl::ext::intel::experimental::cache_mode::uncached,
            sycl::ext::oneapi::experimental::cache_level::L1,
            sycl::ext::oneapi::experimental::cache_level::L3>>));
using UncachedWrite = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::write_hint<
        sycl::ext::intel::experimental::cache_control<
            sycl::ext::intel::experimental::cache_mode::uncached,
            sycl::ext::oneapi::experimental::cache_level::L1,
            sycl::ext::oneapi::experimental::cache_level::L3>>));

template <class T>
inline T uncached_load(const T* ptr) {
    sycl::ext::oneapi::experimental::annotated_ptr<T, UncachedRead> value(
        const_cast<T*>(ptr));
    return value[0];
}

template <class T>
inline void uncached_store(T* ptr, T value) {
    sycl::ext::oneapi::experimental::annotated_ptr<T, UncachedWrite> target(
        ptr);
    target[0] = value;
}

inline std::uint32_t host_load_acquire_u32(
    const std::uint32_t* ptr) noexcept {
    const std::uint32_t value =
        *reinterpret_cast<const volatile std::uint32_t*>(ptr);
    std::atomic_thread_fence(std::memory_order_acquire);
#if defined(_MSC_VER)
    _mm_lfence();
#else
    std::atomic_signal_fence(std::memory_order_acquire);
#endif
    return value;
}

inline void host_store_release_u32(
    std::uint32_t* ptr, std::uint32_t value) noexcept {
    std::atomic_thread_fence(std::memory_order_release);
#if defined(_MSC_VER)
    _mm_sfence();
#else
    std::atomic_signal_fence(std::memory_order_release);
#endif
    *reinterpret_cast<volatile std::uint32_t*>(ptr) = value;
}

inline void host_pause() noexcept {
#if defined(_MSC_VER)
    for (int i = 0; i < 32; ++i) {
        _mm_pause();
    }
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline void host_invalidate_lines(
    const void* ptr, std::size_t bytes) noexcept {
#if defined(_MSC_VER)
    if (!ptr || bytes == 0u) return;
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(ptr) & ~std::uintptr_t(63u);
    const std::uintptr_t end =
        (reinterpret_cast<std::uintptr_t>(ptr) + bytes + 63u) &
        ~std::uintptr_t(63u);
    for (std::uintptr_t line = begin; line < end; line += 64u) {
        _mm_clflush(reinterpret_cast<const void*>(line));
    }
    _mm_mfence();
#else
    (void)ptr;
    (void)bytes;
#endif
}

// Device side loads/stores for the host-visible doorbell words. The uncached
// access bypasses L1/L3; the acquire after the load and release before the
// store keep the payload ordered at system scope.
inline std::uint32_t device_control_load(
    const std::uint32_t* ptr) {
    const std::uint32_t value = uncached_load(ptr);
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);
    return value;
}

inline void device_control_store(
    std::uint32_t* ptr, std::uint32_t value) {
    sycl::atomic_fence(
        sycl::memory_order::release, sycl::memory_scope::system);
    uncached_store(ptr, value);
}

// Device-only epoch state never crosses the PCIe root complex. Device scope
// avoids IGC's system-wide cache-eviction fence while the release/acquire pair
// still publishes packet results to the following graph kernels.
inline std::uint32_t device_u32_load(const std::uint32_t* ptr) {
    sycl::atomic_ref<
        std::uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        value(*const_cast<std::uint32_t*>(ptr));
    return value.load(sycl::memory_order::acquire);
}

inline void device_u32_store(
    std::uint32_t* ptr, std::uint32_t value) {
    sycl::atomic_ref<
        std::uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        target(*ptr);
    target.store(value, sycl::memory_order::release);
}

inline bool device_epoch_valid(
    const TokenGraphDeviceEpoch* epoch) {
    return device_u32_load(&epoch->gate_state) ==
               static_cast<std::uint32_t>(TokenGraphGateState::Valid) &&
           device_u32_load(&epoch->sticky_abort) == 0u;
}

inline bool device_epoch_sequence_matches(
    const TokenGraphDeviceEpoch* epoch, std::uint32_t sequence) {
    return device_epoch_valid(epoch) &&
           device_u32_load(&epoch->reserved[kEpochSequenceWord]) ==
               sequence;
}

inline void device_increment_counter(std::uint32_t* counter) {
    const std::uint32_t old = uncached_load(counter);
    uncached_store(counter, old + 1u);
}

inline void device_publish_failure(
    TokenGraphControlRing* control, TokenGraphDeviceEpoch* epoch,
    TokenGraphGateState state, std::uint32_t detail,
    std::uint32_t spins) {
    if (!control || !epoch) return;

    switch (state) {
        case TokenGraphGateState::Timeout:
            device_increment_counter(&control->timeout_count);
            device_increment_counter(&control->gate_abort_count);
            uncached_store(&control->timeout_spins,
                           static_cast<std::uint64_t>(spins));
            break;
        case TokenGraphGateState::Stale:
            device_increment_counter(&control->stale_count);
            device_increment_counter(&control->gate_abort_count);
            break;
        case TokenGraphGateState::ChannelMismatch:
            device_increment_counter(&control->channel_mismatch_count);
            device_increment_counter(&control->gate_abort_count);
            break;
        case TokenGraphGateState::PacketMismatch:
            device_increment_counter(&control->mismatch_count);
            device_increment_counter(
                &control->packet_sequence_mismatch_count);
            device_increment_counter(&control->gate_abort_count);
            break;
        case TokenGraphGateState::PacketShape:
            device_increment_counter(&control->packet_shape_error_count);
            device_increment_counter(&control->gate_abort_count);
            break;
        default:
            device_increment_counter(&control->gate_abort_count);
            break;
    }

    device_u32_store(&epoch->sticky_abort,
                     static_cast<std::uint32_t>(state));
    device_u32_store(
        &epoch->reserved[kEpochPacketRowsWord], detail);
    device_u32_store(
        &epoch->gate_state, static_cast<std::uint32_t>(state));
    device_control_store(&control->done_sequence, kAbortSequence);
}

inline bool device_packet_shape_ok(
    const TokenGraphCpuPacketRing* packet,
    std::uint32_t expected_sequence,
    std::uint32_t expected_top_k, std::uint32_t& rows,
    std::uint32_t& detail) {
    rows = 0;
    detail = 0;
    if (!packet) {
        detail = 1;
        return false;
    }

    const std::uint32_t magic =
        uncached_load(&packet->reserved[kPacketWordMagic]);
    const std::uint32_t gate =
        uncached_load(&packet->reserved[kPacketWordGate]);
    const std::uint32_t top_k =
        uncached_load(&packet->reserved[kPacketWordTopK]);
    const std::uint32_t hidden =
        uncached_load(&packet->reserved[kPacketWordHidden]);
    const std::uint32_t packet_sequence =
        uncached_load(&packet->reserved[kPacketWordSequence]);
    rows = uncached_load(&packet->rows);

    if (magic != kPacketMagic || gate !=
            static_cast<std::uint32_t>(TokenGraphGateState::Valid) ||
        hidden != kTokenGraphHiddenElements ||
        packet_sequence != expected_sequence ||
        top_k != expected_top_k || top_k == 0u ||
        top_k > kTokenGraphPacketSlots ||
        rows > top_k || rows > kTokenGraphPacketSlots) {
        detail = 2;
        return false;
    }

    std::uint32_t seen_rows = 0u;
    std::uint32_t mapped_rows = 0u;
    for (std::uint32_t rank = 0u; rank < top_k; ++rank) {
        const std::int32_t row =
            uncached_load(&packet->rank_to_row[rank]);
        if (row == -1) continue;
        if (row < 0 || static_cast<std::uint32_t>(row) >= rows) {
            detail = 3u + rank;
            return false;
        }
        const std::uint32_t bit =
            1u << static_cast<std::uint32_t>(row);
        if ((seen_rows & bit) != 0u) {
            detail = 3u + rank;
            return false;
        }
        seen_rows |= bit;
        ++mapped_rows;
    }
    if (mapped_rows != rows) {
        detail = 3u + top_k;
        return false;
    }
    return true;
}

// Capture exactly one of these at the start of each graph replay, before the
// first packet wait. It clears the previous replay's first-failure marker;
// subsequent waits must never clear it.
inline void submit_begin_replay(
    sycl::queue& queue, TokenGraphDeviceEpoch* epoch) {
    if (!epoch) {
        throw std::invalid_argument(
            "token graph replay begin requires a non-null epoch");
    }
    queue.single_task([=] {
        device_u32_store(
            &epoch->gate_state,
            static_cast<std::uint32_t>(TokenGraphGateState::Pending));
        device_u32_store(
            &epoch->reserved[kEpochPacketRowsWord], 0u);
        device_u32_store(
            &epoch->reserved[kEpochSequenceWord], 0u);
        device_u32_store(
            &epoch->reserved[kEpochSpinItersWord], 0u);
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::device);
        device_u32_store(&epoch->sticky_abort, 0u);
    });
}

// Publish the CPU cold-expert request as one work-group. The activation copy
// uses 16-byte vectors, the route is emitted by the first ten work-items, and
// only lid zero advances the device epoch and rings the host doorbell.
inline void submit_input_and_route_publish(
    sycl::queue& queue, const std::uint16_t* input_bf16,
    TokenGraphControlRing* control, TokenGraphRouteRing* route,
    TokenGraphCpuPacketRing* packet, TokenGraphDeviceEpoch* epoch,
    const std::int32_t* ids, const float* weights,
    std::uint32_t top_k, bool clear_sticky,
    std::size_t work_group = 256u) {
    if (!input_bf16 || !control || !route || !packet || !epoch ||
        !ids || !weights) {
        throw std::invalid_argument(
            "token graph input/route publish requires non-null pointers");
    }
    if (top_k == 0u || top_k > kTokenGraphRouteRanks) {
        throw std::invalid_argument(
            "token graph input/route publish top-k is out of range");
    }
    if (work_group == 0u) {
        throw std::invalid_argument(
            "token graph input/route publish work-group must be non-zero");
    }
    using Vec8U16 = sycl::vec<std::uint16_t, 8>;
    constexpr std::uint32_t kInputVectors =
        kTokenGraphHiddenElements / 8u;
    static_assert(sizeof(Vec8U16) == 16u);
    static_assert(kTokenGraphHiddenElements % 8u == 0u);

    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint32_t, 1> status(
            sycl::range<1>(2), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(work_group),
                sycl::range<1>(work_group)),
            [=](sycl::nd_item<1> item) {
                const std::size_t tid = item.get_local_id(0);
                const sycl::group<1> group = item.get_group();
                std::uint32_t next = 0u;
                if (tid == 0u) {
                    next = device_u32_load(&epoch->sequence) + 1u;
                    status[0] =
                        next != 0u && next != kAbortSequence ? 1u : 0u;
                    status[1] = next;
                    if (status[0] == 0u) {
                        device_publish_failure(
                            control, epoch,
                            TokenGraphGateState::ChannelMismatch, next, 0u);
                    } else {
                        device_u32_store(&epoch->sequence, next);
                        if (clear_sticky) {
                            device_u32_store(&epoch->sticky_abort, 0u);
                        }
                    }
                }
                sycl::group_barrier(
                    group, sycl::memory_scope::work_group);
                if (status[0] == 0u) return;
                next = status[1];

                const auto* source =
                    reinterpret_cast<const Vec8U16*>(input_bf16);
                auto* destination =
                    reinterpret_cast<Vec8U16*>(packet->input_bf16);
                for (std::uint32_t vector =
                         static_cast<std::uint32_t>(tid);
                     vector < kInputVectors;
                     vector += static_cast<std::uint32_t>(work_group)) {
                    uncached_store(destination + vector, source[vector]);
                }
                for (std::uint32_t rank =
                         static_cast<std::uint32_t>(tid);
                     rank < kTokenGraphRouteRanks;
                     rank += static_cast<std::uint32_t>(work_group)) {
                    if (rank < top_k) {
                        uncached_store(
                            &route->ids[rank],
                            static_cast<std::int32_t>(ids[rank]));
                        uncached_store(
                            &route->weights[rank], weights[rank]);
                    } else {
                        uncached_store(&route->ids[rank], -1);
                        uncached_store(&route->weights[rank], 0.0f);
                    }
                }
                // The barrier gives lid zero a happens-before edge from every
                // payload store; device_control_store then publishes the
                // uncached host-visible doorbell at system scope.
                sycl::group_barrier(
                    group, sycl::memory_scope::work_group);
                if (tid == 0u) {
                    device_control_store(
                        &control->route_sequence, next);
                }
            });
    });
}

// Wait for the graph's route request, then validate and publish the packet
// gate. This kernel deliberately does not copy the bulk payload; the copy is
// a separate captured node so its active-row shape is explicit. A sticky
// failure from an earlier handoff in the same replay is never overwritten.
inline void submit_packet_wait(
    sycl::queue& queue, TokenGraphControlRing* control,
    TokenGraphCpuPacketRing* packet, TokenGraphDeviceEpoch* epoch,
    std::int32_t* rank_dev, std::uint32_t expected_top_k,
    std::uint32_t spin_limit = kDefaultDeviceSpinLimit,
    std::size_t work_group = 256u) {
    if (!control || !packet || !epoch || !rank_dev) {
        throw std::invalid_argument(
            "token graph packet wait requires non-null pointers");
    }
    if (work_group == 0u) {
        throw std::invalid_argument(
            "token graph packet wait work-group must be non-zero");
    }
    if (spin_limit == 0u) {
        throw std::invalid_argument(
            "token graph packet wait spin limit must be non-zero");
    }

    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint32_t, 1> status(
            sycl::range<1>(4), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(work_group),
                sycl::range<1>(work_group)),
            [=](sycl::nd_item<1> item) {
                const std::size_t tid = item.get_local_id(0);
                const sycl::group<1> group = item.get_group();

                if (tid == 0u) {
                    status[0] = 0u;
                    status[1] = 0u;
                    status[2] = 0u;
                    status[3] = 0u;
                    device_u32_store(
                        &epoch->gate_state,
                        static_cast<std::uint32_t>(
                            TokenGraphGateState::Pending));
                    const std::uint32_t sticky =
                        device_u32_load(&epoch->sticky_abort);
                    if (sticky != 0u) {
                        status[0] = sticky;
                        status[1] = sticky;
                    } else {
                        const std::uint32_t request =
                            device_control_load(
                                &control->route_sequence);
                        if (request == 0u ||
                            request == kAbortSequence) {
                            status[0] = static_cast<std::uint32_t>(
                                TokenGraphGateState::ChannelMismatch);
                            status[1] = request;
                        } else {
                            std::uint32_t spins = 0u;
                            for (;;) {
                                const std::uint32_t flag =
                                    device_control_load(
                                        &control->cpu_flag);
                                if (flag == request) break;
                                if (flag == kAbortSequence) {
                                    status[0] =
                                        static_cast<std::uint32_t>(
                                            TokenGraphGateState::Stale);
                                    status[1] = flag;
                                    break;
                                }
                                if (token_graph_sequence_status(
                                        request, flag) ==
                                    TokenGraphSequenceStatus::Ahead) {
                                    status[0] =
                                        static_cast<std::uint32_t>(
                                            TokenGraphGateState::
                                                ChannelMismatch);
                                    status[1] = flag;
                                    break;
                                }
                                if (spins >= spin_limit) {
                                    status[0] =
                                        static_cast<std::uint32_t>(
                                            TokenGraphGateState::Timeout);
                                    status[1] = spins;
                                    break;
                                }
                                ++spins;
                            }
                            // Diagnostic: accumulate how long the device spun
                            // waiting for the CPU packet. A near-zero total
                            // proves the device never waits; a large total
                            // proves it does.
                            sycl::atomic_ref<
                                std::uint32_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                spin_total(
                                    epoch->reserved[kEpochSpinItersWord]);
                            (void)spin_total.fetch_add(spins);
                        }

                        if (status[0] == 0u) {
                            const std::uint32_t ready =
                                device_control_load(
                                    &packet->ready_sequence);
                            std::uint32_t rows = 0u;
                            std::uint32_t detail = 0u;
                            if (ready != request) {
                                status[0] = static_cast<std::uint32_t>(
                                    TokenGraphGateState::PacketMismatch);
                                status[1] = ready;
                            } else if (!device_packet_shape_ok(
                                           packet, request,
                                           expected_top_k, rows,
                                           detail)) {
                                status[0] = static_cast<std::uint32_t>(
                                    TokenGraphGateState::PacketShape);
                                status[1] = detail;
                            } else {
                                status[0] = static_cast<std::uint32_t>(
                                    TokenGraphGateState::Valid);
                                status[1] = rows;
                                status[2] = expected_top_k;
                                status[3] = request;
                            }
                        }
                    }
                }

                sycl::group_barrier(
                    group, sycl::memory_scope::work_group);

                const std::uint32_t state = status[0];
                if (state != static_cast<std::uint32_t>(
                                 TokenGraphGateState::Valid)) {
                    for (std::size_t slot = tid;
                         slot < kTokenGraphPacketSlots;
                         slot += work_group) {
                        rank_dev[slot] = -1;
                    }
                    if (tid == 0u) {
                        const std::uint32_t sticky =
                            device_u32_load(&epoch->sticky_abort);
                        if (sticky == 0u) {
                            device_publish_failure(
                                control, epoch,
                                static_cast<TokenGraphGateState>(state),
                                status[1], status[1]);
                        } else {
                            device_u32_store(&epoch->gate_state, sticky);
                            device_control_store(
                                &control->done_sequence,
                                kAbortSequence);
                        }
                    }
                    return;
                }

                const std::uint32_t rows = status[1];
                const std::uint32_t top_k = status[2];
                for (std::size_t slot = tid;
                     slot < kTokenGraphPacketSlots;
                     slot += work_group) {
                    rank_dev[slot] =
                        slot < top_k
                            ? uncached_load(&packet->rank_to_row[slot])
                            : -1;
                }

                if (tid == 0u) {
                    device_u32_store(
                        &epoch->reserved[kEpochPacketRowsWord], rows);
                    device_u32_store(
                        &epoch->reserved[kEpochSequenceWord],
                        status[3]);
                }
                sycl::group_barrier(
                    group, sycl::memory_scope::work_group);
                if (tid == 0u) {
                    device_u32_store(
                        &epoch->gate_state,
                        static_cast<std::uint32_t>(
                            TokenGraphGateState::Valid));
                }
            });
    });
}

// One work-group owns one packet row. The rank map above stays scalar; only
// the active output rows are copied as 16-byte vectors. Inactive rows are
// intentionally written with a device-side sentinel so stale packet data can
// never be mistaken for a valid row.
inline void submit_packet_copy(
    sycl::queue& queue, const TokenGraphCpuPacketRing* packet,
    const TokenGraphDeviceEpoch* epoch, std::int32_t* packet_dev,
    std::size_t work_group = 256u) {
    if (!packet || !epoch || !packet_dev) {
        throw std::invalid_argument(
            "token graph packet copy requires non-null pointers");
    }
    if (work_group == 0u) {
        throw std::invalid_argument(
            "token graph packet copy work-group must be non-zero");
    }
    if (reinterpret_cast<std::uintptr_t>(packet_dev) %
            alignof(VecPacketU32) != 0u) {
        throw std::invalid_argument(
            "token graph packet destination must be 64-byte aligned");
    }

    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint32_t, 1> status(
            sycl::range<1>(2), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<std::size_t>(kTokenGraphPacketSlots) *
                    work_group),
                sycl::range<1>(work_group)),
            [=](sycl::nd_item<1> item) {
                const std::size_t group = item.get_group(0);
                const std::size_t local = item.get_local_id(0);
                if (group >= kTokenGraphPacketSlots) return;
                if (local == 0u) {
                    status[0] = device_epoch_valid(epoch) ? 1u : 0u;
                    status[1] =
                        status[0] != 0u
                            ? device_u32_load(
                                  &epoch->reserved[
                                      kEpochPacketRowsWord])
                            : 0u;
                }
                sycl::group_barrier(
                    item.get_group(), sycl::memory_scope::work_group);
                if (status[0] == 0u) return;

                const std::uint32_t rows = status[1];
                const auto* source =
                    reinterpret_cast<const VecPacketU32*>(packet->values);
                auto* destination = reinterpret_cast<VecPacketU32*>(
                    packet_dev + kTokenGraphPacketSlots);
                const std::size_t row_base =
                    group * static_cast<std::size_t>(
                                kPacketRowVectors);

                if (group < rows) {
                    for (std::size_t vector = local;
                         vector < kPacketRowVectors;
                         vector += work_group) {
                        destination[row_base + vector] =
                            uncached_load(source + row_base + vector);
                    }
                } else {
                    VecPacketU32 sentinel;
#pragma unroll
                    for (std::uint32_t lane = 0u;
                         lane < kPacketVectorElements; ++lane) {
                        sentinel[lane] = kPacketTailSentinel;
                    }
                    for (std::size_t vector = local;
                         vector < kPacketRowVectors;
                         vector += work_group) {
                        destination[row_base + vector] = sentinel;
                    }
                }
            });
    });
}

inline void submit_packet_handoff(
    sycl::queue& queue, TokenGraphControlRing* control,
    TokenGraphCpuPacketRing* packet, TokenGraphDeviceEpoch* epoch,
    std::int32_t* packet_dev, std::uint32_t expected_top_k,
    std::uint32_t spin_limit = kDefaultDeviceSpinLimit,
    std::size_t wait_work_group = 256u,
    std::size_t copy_work_group = 128u) {
    submit_packet_wait(
        queue, control, packet, epoch, packet_dev, expected_top_k,
        spin_limit, wait_work_group);
    submit_packet_copy(
        queue, packet, epoch, packet_dev, copy_work_group);
}

// Call after all consumers of the packet have run. It publishes the request
// sequence only after the epoch gate is valid; a failed gate publishes the
// abort sentinel so the host does not need to wait for its timeout.
inline void submit_publish_done(
    sycl::queue& queue, TokenGraphControlRing* control,
    const TokenGraphDeviceEpoch* epoch) {
    if (!control || !epoch) {
        throw std::invalid_argument(
            "token graph done publication requires non-null pointers");
    }
    queue.single_task([=] {
        const std::uint32_t request =
            device_control_load(&control->route_sequence);
        const bool valid =
            device_epoch_sequence_matches(epoch, request) &&
            request != 0u && request != kAbortSequence;
        device_control_store(
            &control->done_sequence,
            valid ? request : kAbortSequence);
        device_control_store(
            reinterpret_cast<std::uint32_t*>(&control->reserved[0]),
            device_u32_load(&epoch->reserved[kEpochSpinItersWord]));
    });
}

}  // namespace token_graph_service

class Qwen4ExpTokenGraphService {
public:
    enum class Status : std::uint32_t {
        Unallocated = 0,
        Ready = 1,
        Failed = 2,
    };

    enum class WaitResult : std::uint32_t {
        Reached = 0,
        Timeout = 1,
        Failed = 2,
    };

    enum class ErrorCode : std::uint32_t {
        None = 0,
        AlreadyAllocated,
        AllocationFailed,
        NotLevelZero,
        MissingLevelZeroSupport,
        SequenceExhausted,
        SequenceMismatch,
        SequenceAhead,
        GateClosed,
        TopKOutOfRange,
        RowsOutOfRange,
        RankMapNull,
        ValuesNull,
        RankOutOfRange,
        DuplicateRank,
        RowCountMismatch,
        InvalidRouteId,
        InvalidRouteWeight,
        RequestTimeout,
        DoneTimeout,
        DeviceAbort,
    };

    struct PacketView {
        TokenGraphGateState gate = TokenGraphGateState::Valid;
        std::uint32_t top_k = kTokenGraphRouteRanks;
        std::uint32_t rows = 0u;
        const std::int32_t* rank_to_row = nullptr;
        const float* values = nullptr;
    };

    Qwen4ExpTokenGraphService() = default;
    ~Qwen4ExpTokenGraphService() { release(); }

    Qwen4ExpTokenGraphService(
        const Qwen4ExpTokenGraphService&) = delete;
    Qwen4ExpTokenGraphService& operator=(
        const Qwen4ExpTokenGraphService&) = delete;
    Qwen4ExpTokenGraphService(
        Qwen4ExpTokenGraphService&&) = delete;
    Qwen4ExpTokenGraphService& operator=(
        Qwen4ExpTokenGraphService&&) = delete;

    // Allocates exactly two Level Zero host USM blocks. A non-Level-Zero
    // context, a build without Level Zero support, or any allocation failure
    // throws; there is no cached-host fallback.
    void allocate(const sycl::context& context) {
        if (control_ || route_ || packet_) {
            throw std::logic_error(
                "Qwen4Exp token graph service is already allocated");
        }

#if defined(AILA_HAS_LEVEL_ZERO)
        if (context.get_backend() !=
            sycl::backend::ext_oneapi_level_zero) {
            throw std::runtime_error(
                "Qwen4Exp token graph service requires Level Zero");
        }
        const ze_context_handle_t ze_context =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                context);
        if (!ze_context) {
            throw std::runtime_error(
                "Qwen4Exp token graph service received a null Level Zero "
                "context");
        }

        void* control_storage = nullptr;
        void* route_storage = nullptr;
        void* packet_storage = nullptr;
        try {
            control_storage = allocate_uncached(
                ze_context, sizeof(TokenGraphControlRing),
                alignof(TokenGraphControlRing), "control");
            route_storage = allocate_uncached(
                ze_context, sizeof(TokenGraphRouteRing),
                alignof(TokenGraphRouteRing), "route");
            packet_storage = allocate_uncached(
                ze_context, sizeof(TokenGraphCpuPacketRing),
                alignof(TokenGraphCpuPacketRing), "packet");
        } catch (...) {
            free_uncached(ze_context, packet_storage);
            free_uncached(ze_context, route_storage);
            free_uncached(ze_context, control_storage);
            throw;
        }

        native_context_ = reinterpret_cast<void*>(ze_context);
        control_ = static_cast<TokenGraphControlRing*>(control_storage);
        route_ = static_cast<TokenGraphRouteRing*>(route_storage);
        packet_ = static_cast<TokenGraphCpuPacketRing*>(packet_storage);
        reset();
#else
        (void)context;
        throw std::runtime_error(
            "Qwen4Exp token graph service requires a build with "
            "AILA_HAS_LEVEL_ZERO");
#endif
    }

    // Must only be called after the previous graph has drained. It resets the
    // wire state and returns the service to Ready; a failed service can be
    // reset explicitly after the caller has confirmed device idle.
    bool reset() noexcept {
        if (!control_ || !route_ || !packet_) return false;
        std::memset(control_, 0, sizeof(TokenGraphControlRing));
        std::memset(route_, 0, sizeof(TokenGraphRouteRing));
        std::memset(packet_, 0, sizeof(TokenGraphCpuPacketRing));
        std::atomic_thread_fence(std::memory_order_release);
#if defined(_MSC_VER)
        _mm_sfence();
#endif
        status_ = Status::Ready;
        failed_ = false;
        last_error_ = ErrorCode::None;
        last_requested_sequence_ = 0u;
        last_published_sequence_ = 0u;
        last_done_sequence_ = 0u;
        return true;
    }

    // Polls route_sequence without sleeping. The first request is sequence 1;
    // every later request must advance exactly once.
    WaitResult wait_request(
        std::uint32_t timeout_ms, std::uint32_t& sequence) noexcept {
        sequence = 0u;
        if (!control_ || !route_ || !packet_) {
            fail(ErrorCode::MissingLevelZeroSupport);
            return WaitResult::Failed;
        }
        if (failed_) return WaitResult::Failed;

        const std::uint32_t expected =
            last_requested_sequence_ + 1u;
        if (expected == 0u ||
            expected == token_graph_service::kAbortSequence) {
            fail(ErrorCode::SequenceExhausted);
            return WaitResult::Failed;
        }

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(timeout_ms);
        std::uint32_t polls = 0u;
        for (;;) {
            const std::uint32_t observed =
                token_graph_service::host_load_acquire_u32(
                    &control_->route_sequence);
            if (observed == expected) {
                last_requested_sequence_ = expected;
                sequence = expected;
                return WaitResult::Reached;
            }
            if (observed == token_graph_service::kAbortSequence) {
                fail(ErrorCode::DeviceAbort);
                return WaitResult::Failed;
            }
            if (token_graph_sequence_status(expected, observed) ==
                TokenGraphSequenceStatus::Ahead) {
                fail(ErrorCode::SequenceAhead);
                return WaitResult::Failed;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                fail(ErrorCode::RequestTimeout);
                return WaitResult::Timeout;
            }
            ++polls;
            if ((polls & 255u) == 0u) {
                token_graph_service::host_pause();
            }
        }
    }

    // The host publishes the packet only after validating the request
    // sequence, gate value, and rank-map shape. On failure it publishes the
    // abort sentinel in control.cpu_flag and leaves packet.ready_sequence
    // unpublished.
    bool publish_packet(
        std::uint32_t sequence, const PacketView& packet_view) noexcept {
        if (!control_ || !route_ || !packet_) {
            fail(ErrorCode::MissingLevelZeroSupport);
            return false;
        }
        if (failed_) return false;
        if (sequence != last_requested_sequence_ ||
            sequence == 0u ||
            sequence == token_graph_service::kAbortSequence) {
            fail(ErrorCode::SequenceMismatch);
            return false;
        }
        if (token_graph_service::host_load_acquire_u32(
                &control_->route_sequence) != sequence) {
            fail(ErrorCode::SequenceMismatch);
            return false;
        }
        if (packet_view.gate != TokenGraphGateState::Valid) {
            fail(ErrorCode::GateClosed);
            return false;
        }
        if (packet_view.top_k == 0u ||
            packet_view.top_k > kTokenGraphPacketSlots) {
            fail(ErrorCode::TopKOutOfRange);
            return false;
        }
        if (packet_view.rows > packet_view.top_k) {
            fail(ErrorCode::RowsOutOfRange);
            return false;
        }
        if (!packet_view.rank_to_row) {
            fail(ErrorCode::RankMapNull);
            return false;
        }
        if (packet_view.rows != 0u && !packet_view.values) {
            fail(ErrorCode::ValuesNull);
            return false;
        }

        std::uint32_t seen_rows = 0u;
        std::uint32_t mapped_rows = 0u;
        for (std::uint32_t rank = 0u;
             rank < packet_view.top_k; ++rank) {
            const std::int32_t row = packet_view.rank_to_row[rank];
            if (row == -1) continue;
            if (row < 0 ||
                static_cast<std::uint32_t>(row) >= packet_view.rows) {
                fail(ErrorCode::RankOutOfRange);
                return false;
            }
            const std::uint32_t bit =
                1u << static_cast<std::uint32_t>(row);
            if ((seen_rows & bit) != 0u) {
                fail(ErrorCode::DuplicateRank);
                return false;
            }
            seen_rows |= bit;
            ++mapped_rows;
        }
        if (mapped_rows != packet_view.rows) {
            fail(ErrorCode::RowCountMismatch);
            return false;
        }

        packet_->reserved[token_graph_service::kPacketWordMagic] =
            token_graph_service::kPacketMagic;
        packet_->reserved[token_graph_service::kPacketWordGate] =
            static_cast<std::uint32_t>(packet_view.gate);
        packet_->reserved[token_graph_service::kPacketWordTopK] =
            packet_view.top_k;
        packet_->reserved[token_graph_service::kPacketWordHidden] =
            kTokenGraphHiddenElements;
        packet_->reserved[token_graph_service::kPacketWordSequence] =
            sequence;
        packet_->rows = packet_view.rows;

        // Rank map is deliberately scalar even when payload copies are
        // vectorized. Ranks beyond top_k are cleared to make stale entries
        // impossible to consume.
        for (std::uint32_t slot = 0u;
             slot < kTokenGraphPacketSlots; ++slot) {
            packet_->rank_to_row[slot] =
                slot < packet_view.top_k
                    ? packet_view.rank_to_row[slot]
                    : -1;
        }
        if (packet_view.rows != 0u &&
            packet_view.values != packet_->values) {
            std::memcpy(
                packet_->values, packet_view.values,
                static_cast<std::size_t>(packet_view.rows) *
                    kTokenGraphHiddenElements * sizeof(float));
        }

        // Clear the old ready marker before making the new packet visible.
        // The final cpu_flag store is the only release-publication event.
        token_graph_service::host_store_release_u32(
            &packet_->ready_sequence, 0u);
        token_graph_service::host_store_release_u32(
            &packet_->ready_sequence, sequence);
        token_graph_service::host_store_release_u32(
            &control_->cpu_flag, sequence);
        last_published_sequence_ = sequence;
        return true;
    }

    // Reads and validates the device-published route for one handoff. This is
    // called after wait_request() and before the CPU cold-expert work.
    bool read_route(std::uint32_t sequence, std::uint32_t top_k,
                    std::uint32_t experts,
                    token_graph_service::RouteView& route) noexcept {
        if (!control_ || !route_ || !packet_) {
            fail(ErrorCode::MissingLevelZeroSupport);
            return false;
        }
        if (failed_) return false;
        if (sequence != last_requested_sequence_ || sequence == 0u ||
            sequence == token_graph_service::kAbortSequence) {
            fail(ErrorCode::SequenceMismatch);
            return false;
        }
        if (top_k == 0u || top_k > kTokenGraphRouteRanks) {
            fail(ErrorCode::TopKOutOfRange);
            return false;
        }
        if (experts == 0u) {
            fail(ErrorCode::InvalidRouteId);
            return false;
        }

        token_graph_service::host_invalidate_lines(
            route_->ids, sizeof(route_->ids));
        token_graph_service::host_invalidate_lines(
            route_->weights, sizeof(route_->weights));
        for (std::uint32_t rank = 0u; rank < top_k; ++rank) {
            const std::int32_t id =
                route_->ids[rank];
            const float weight =
                route_->weights[rank];
            if (id < 0 || static_cast<std::uint32_t>(id) >= experts) {
                fail(ErrorCode::InvalidRouteId);
                return false;
            }
            if (!std::isfinite(weight) || weight < 0.0f ||
                weight > 1.0f) {
                fail(ErrorCode::InvalidRouteWeight);
                return false;
            }
            route.ids[rank] = id;
            route.weights[rank] = weight;
        }
        for (std::uint32_t rank = top_k;
             rank < kTokenGraphRouteRanks; ++rank) {
            route.ids[rank] = -1;
            route.weights[rank] = 0.0f;
        }
        return true;
    }

    // Polls done_sequence. Old done values are allowed while a new graph is
    // running; only a future sequence or the abort sentinel fail early.
    bool wait_done(
        std::uint32_t sequence,
        std::uint32_t timeout_ms) noexcept {
        if (!control_ || !route_ || !packet_) {
            fail(ErrorCode::MissingLevelZeroSupport);
            return false;
        }
        if (failed_ || sequence == 0u ||
            sequence != last_published_sequence_) {
            fail(ErrorCode::SequenceMismatch);
            return false;
        }

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(timeout_ms);
        std::uint32_t polls = 0u;
        for (;;) {
            const std::uint32_t observed =
                token_graph_service::host_load_acquire_u32(
                    &control_->done_sequence);
            if (observed == sequence) {
                last_done_sequence_ = sequence;
                return true;
            }
            if (observed == token_graph_service::kAbortSequence) {
                fail(ErrorCode::DeviceAbort);
                return false;
            }
            if (token_graph_sequence_status(sequence, observed) ==
                TokenGraphSequenceStatus::Ahead) {
                fail(ErrorCode::SequenceAhead);
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                fail(ErrorCode::DoneTimeout);
                return false;
            }
            ++polls;
            if ((polls & 255u) == 0u) {
                token_graph_service::host_pause();
            }
        }
    }

    Status status() const noexcept { return status_; }
    bool allocated() const noexcept {
        return control_ != nullptr && route_ != nullptr &&
               packet_ != nullptr;
    }
    bool failed() const noexcept { return failed_; }
    ErrorCode last_error() const noexcept { return last_error_; }
    std::uint32_t last_requested_sequence() const noexcept {
        return last_requested_sequence_;
    }
    std::uint32_t last_published_sequence() const noexcept {
        return last_published_sequence_;
    }
    std::uint32_t last_done_sequence() const noexcept {
        return last_done_sequence_;
    }

    TokenGraphControlRing* control() noexcept { return control_; }
    const TokenGraphControlRing* control() const noexcept {
        return control_;
    }
    TokenGraphCpuPacketRing* packet() noexcept { return packet_; }
    const TokenGraphCpuPacketRing* packet() const noexcept {
        return packet_;
    }
    std::uint16_t* packet_input_bf16() noexcept {
        return packet_ ? packet_->input_bf16 : nullptr;
    }

    // Terminal unblock for a service-aware drain. The caller must still wait
    // for the graph event and may reset() only after the queue is idle.
    void abort() noexcept {
        if (!control_ || !route_ || !packet_) return;
        fail(ErrorCode::DeviceAbort);
    }
    TokenGraphRouteRing* route() noexcept { return route_; }
    const TokenGraphRouteRing* route() const noexcept {
        return route_;
    }
    std::size_t control_bytes() const noexcept {
        return sizeof(TokenGraphControlRing);
    }
    std::size_t packet_bytes() const noexcept {
        return sizeof(TokenGraphCpuPacketRing);
    }

    static const char* error_name(ErrorCode error) noexcept {
        switch (error) {
            case ErrorCode::None: return "none";
            case ErrorCode::AlreadyAllocated: return "already_allocated";
            case ErrorCode::AllocationFailed: return "allocation_failed";
            case ErrorCode::NotLevelZero: return "not_level_zero";
            case ErrorCode::MissingLevelZeroSupport:
                return "missing_level_zero_support";
            case ErrorCode::SequenceExhausted: return "sequence_exhausted";
            case ErrorCode::SequenceMismatch: return "sequence_mismatch";
            case ErrorCode::SequenceAhead: return "sequence_ahead";
            case ErrorCode::GateClosed: return "gate_closed";
            case ErrorCode::TopKOutOfRange: return "top_k_out_of_range";
            case ErrorCode::RowsOutOfRange: return "rows_out_of_range";
            case ErrorCode::RankMapNull: return "rank_map_null";
            case ErrorCode::ValuesNull: return "values_null";
            case ErrorCode::RankOutOfRange: return "rank_out_of_range";
            case ErrorCode::DuplicateRank: return "duplicate_rank";
            case ErrorCode::RowCountMismatch: return "row_count_mismatch";
            case ErrorCode::InvalidRouteId: return "invalid_route_id";
            case ErrorCode::InvalidRouteWeight: return "invalid_route_weight";
            case ErrorCode::RequestTimeout: return "request_timeout";
            case ErrorCode::DoneTimeout: return "done_timeout";
            case ErrorCode::DeviceAbort: return "device_abort";
        }
        return "unknown";
    }

private:
#if defined(AILA_HAS_LEVEL_ZERO)
    static void* allocate_uncached(
        ze_context_handle_t context, std::size_t bytes,
        std::size_t alignment, const char* label) {
        ze_host_mem_alloc_desc_t descriptor{
            ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
            ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* ptr = nullptr;
        const ze_result_t result =
            zeMemAllocHost(context, &descriptor, bytes, alignment, &ptr);
        if (result != ZE_RESULT_SUCCESS || !ptr) {
            throw std::runtime_error(
                std::string("Level Zero uncached ") + label +
                " allocation failed (ze_result_t=" +
                std::to_string(static_cast<int>(result)) + ")");
        }
        if (reinterpret_cast<std::uintptr_t>(ptr) % alignment != 0u) {
            (void)zeMemFree(context, ptr);
            throw std::runtime_error(
                std::string("Level Zero ") + label +
                " allocation is not correctly aligned");
        }
        return ptr;
    }

    static void free_uncached(
        ze_context_handle_t context, void* ptr) noexcept {
        if (context && ptr) {
            (void)zeMemFree(context, ptr);
        }
    }
#else
    static void* allocate_uncached(
        void*, std::size_t, std::size_t, const char*) {
        throw std::runtime_error(
            "Qwen4Exp token graph service requires Level Zero");
    }
    static void free_uncached(void*, void*) noexcept {}
#endif

    void release() noexcept {
#if defined(AILA_HAS_LEVEL_ZERO)
        const auto context =
            reinterpret_cast<ze_context_handle_t>(native_context_);
        free_uncached(context, packet_);
        free_uncached(context, route_);
        free_uncached(context, control_);
#else
        (void)native_context_;
#endif
        packet_ = nullptr;
        route_ = nullptr;
        control_ = nullptr;
        native_context_ = nullptr;
    }

    void fail(ErrorCode error) noexcept {
        if (failed_) return;
        failed_ = true;
        status_ = Status::Failed;
        last_error_ = error;
        if (!control_ || !route_ || !packet_) return;

        switch (error) {
            case ErrorCode::GateClosed:
                ++control_->gate_abort_count;
                break;
            case ErrorCode::TopKOutOfRange:
            case ErrorCode::RowsOutOfRange:
            case ErrorCode::RankMapNull:
            case ErrorCode::ValuesNull:
            case ErrorCode::RankOutOfRange:
            case ErrorCode::DuplicateRank:
            case ErrorCode::RowCountMismatch:
                ++control_->packet_shape_error_count;
                ++control_->gate_abort_count;
                break;
            case ErrorCode::InvalidRouteId:
            case ErrorCode::InvalidRouteWeight:
                ++control_->channel_mismatch_count;
                ++control_->gate_abort_count;
                break;
            case ErrorCode::RequestTimeout:
            case ErrorCode::DoneTimeout:
                ++control_->timeout_count;
                ++control_->gate_abort_count;
                break;
            case ErrorCode::DeviceAbort:
                ++control_->gate_abort_count;
                break;
            case ErrorCode::SequenceMismatch:
            case ErrorCode::SequenceAhead:
                ++control_->mismatch_count;
                ++control_->gate_abort_count;
                break;
            default:
                break;
        }
        token_graph_service::host_store_release_u32(
            &packet_->ready_sequence, 0u);
        token_graph_service::host_store_release_u32(
            &control_->cpu_flag, token_graph_service::kAbortSequence);
    }

    void* native_context_ = nullptr;
    TokenGraphControlRing* control_ = nullptr;
    TokenGraphRouteRing* route_ = nullptr;
    TokenGraphCpuPacketRing* packet_ = nullptr;
    Status status_ = Status::Unallocated;
    ErrorCode last_error_ = ErrorCode::None;
    bool failed_ = false;
    std::uint32_t last_requested_sequence_ = 0u;
    std::uint32_t last_published_sequence_ = 0u;
    std::uint32_t last_done_sequence_ = 0u;
};

static_assert(std::is_standard_layout<Qwen4ExpTokenGraphService>::value);
static_assert(!std::is_copy_constructible<Qwen4ExpTokenGraphService>::value);
static_assert(!std::is_move_constructible<Qwen4ExpTokenGraphService>::value);

}  // namespace aila
