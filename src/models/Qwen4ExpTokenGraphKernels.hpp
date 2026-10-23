#pragma once

#include "models/Qwen4ExpTokenGraphService.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace aila {
namespace token_graph_kernels {

using bf16 = sycl::ext::oneapi::bfloat16;
using Vec8 = sycl::vec<bf16, 8>;

inline constexpr std::uint32_t kHiddenGroups = 4u;
inline constexpr std::uint32_t kVec8Elements = 8u;
inline constexpr std::uint32_t kHiddenVectors =
    kTokenGraphHiddenElements / kVec8Elements;
inline constexpr std::uint32_t kFinalBridgeWorkGroup = 64u;
inline constexpr std::uint32_t kMergeWorkGroup = 256u;

static_assert(kTokenGraphHiddenElements % kVec8Elements == 0u);
static_assert(kTokenGraphRouteRanks <= kTokenGraphPacketSlots);
static_assert(sizeof(Vec8) == 16u);

// These loads/stores require an aligned hidden row.
inline Vec8 load_vec8(const bf16* ptr) {
    return *reinterpret_cast<const Vec8*>(ptr);
}

inline void store_vec8(bf16* ptr, Vec8 value) {
    *reinterpret_cast<Vec8*>(ptr) = value;
}

// packet_dev is the service-produced device mirror: sixteen rank slots,
// followed by sixteen hidden-row payloads. A negative rank is resident;
// only a non-negative rank reads packet_dev payload.
inline void submit_gated_ordered_merge(
    sycl::queue& queue, const float* resident_down,
    const float* route_weights, const std::int32_t* selected,
    const std::int32_t* packet_dev, bf16* y,
    TokenGraphControlRing* control,
    const TokenGraphDeviceEpoch* epoch,
    std::uint32_t top_k = kTokenGraphRouteRanks) {
    if (top_k == 0u || top_k > kTokenGraphRouteRanks) {
        throw std::invalid_argument(
            "token graph merge top-k is out of range");
    }
    const std::size_t groups =
        (static_cast<std::size_t>(kTokenGraphHiddenElements) +
         kMergeWorkGroup - 1u) /
        kMergeWorkGroup;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint32_t, 1> status(
            sycl::range<1>(2), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(groups * kMergeWorkGroup),
            sycl::range<1>(kMergeWorkGroup)),
            [=](sycl::nd_item<1> item) {
            if (item.get_local_id(0) == 0u) {
                status[0] =
                    epoch != nullptr &&
                            token_graph_service::device_epoch_valid(epoch)
                        ? 1u
                        : 0u;
                status[1] = 1u;
                if (status[0] != 0u) {
                    for (std::uint32_t rank = 0u; rank < top_k; ++rank) {
                        const bool device_resident = selected[rank] >= 0;
                        const bool host_cold = packet_dev[rank] >= 0;
                        if (device_resident == host_cold) {
                            status[1] = 0u;
                            break;
                        }
                    }
                }
            }
            sycl::group_barrier(
                item.get_group(), sycl::memory_scope::work_group);
            if (status[0] == 0u) return;
            if (status[1] == 0u) {
                if (item.get_group(0) == 0u &&
                    item.get_local_id(0) == 0u) {
                    token_graph_service::device_publish_failure(
                        control,
                        const_cast<TokenGraphDeviceEpoch*>(epoch),
                        TokenGraphGateState::PacketShape, 0xE0000000u,
                        0u);
                }
                return;
            }
            const std::size_t row = item.get_global_id(0);
            if (row >= kTokenGraphHiddenElements) return;

            const auto* packet_values =
                reinterpret_cast<const float*>(
                    packet_dev + kTokenGraphPacketSlots);
            float total = 0.0f;
            for (std::uint32_t rank = 0u; rank < top_k; ++rank) {
                const std::int32_t cpu_row = packet_dev[rank];
                float value;
                if (cpu_row >= 0) {
                    value = packet_values[
                        static_cast<std::size_t>(cpu_row) *
                            kTokenGraphHiddenElements +
                        row];
                } else {
                    value = resident_down[
                        static_cast<std::size_t>(rank) *
                            kTokenGraphHiddenElements +
                        row];
                }
                total += route_weights[rank] * value;
            }
                y[row] = bf16(total);
            });
    });
}

// One token: y and shared_out are hidden rows, shared_gate is one BF16
// scalar. The gate computation and store order mirror the direct path.
inline void submit_add_shared_expert_to_y(
    sycl::queue& queue, bf16* y, const bf16* shared_out,
    const bf16* shared_gate, const TokenGraphDeviceEpoch* epoch) {
    constexpr std::uint32_t work_group = 64u;
    constexpr std::uint32_t vectors = kHiddenVectors;
    constexpr std::uint32_t groups =
        (vectors + work_group - 1u) / work_group;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint32_t, 1> gate_valid(
            sycl::range<1>(1), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<std::size_t>(groups) * work_group),
                sycl::range<1>(work_group)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(16)]] {
                    if (item.get_local_id(0) == 0u) {
                        gate_valid[0] =
                            epoch != nullptr &&
                                    token_graph_service::device_epoch_valid(
                                        epoch)
                                ? 1u
                                : 0u;
                    }
                    sycl::group_barrier(
                        item.get_group(),
                        sycl::memory_scope::work_group);
                    if (gate_valid[0] == 0u) return;
                    const std::size_t vector = item.get_global_id(0);
                    if (vector >= vectors) return;
                    const float gate_value =
                        static_cast<float>(shared_gate[0]);
                    const float gate_act =
                        1.0f /
                        (1.0f + sycl::native::exp(-gate_value));
                    const std::size_t element =
                        vector * kVec8Elements;
                    const Vec8 shared =
                        load_vec8(shared_out + element);
                    Vec8 accumulated = load_vec8(y + element);
#pragma unroll
                    for (std::uint32_t lane = 0u;
                         lane < kVec8Elements; ++lane) {
                        accumulated[lane] = bf16(
                            static_cast<float>(accumulated[lane]) +
                            gate_act *
                                static_cast<float>(shared[lane]));
                    }
                    store_vec8(y + element, accumulated);
                });
    });
}

// One token: hidden4 holds four hidden groups, y is one hidden row, and
// inject holds four FP32 group weights.
inline void submit_gated_final_bridge(
    sycl::queue& queue, bf16* hidden4, const bf16* y,
    const float* inject, const TokenGraphDeviceEpoch* epoch) {
    constexpr std::size_t work_group = kFinalBridgeWorkGroup;
    constexpr std::size_t vectors = kHiddenVectors;
    constexpr std::size_t groups_per_group =
        (vectors + work_group - 1u) / work_group;
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint32_t, 1> gate_valid(
            sycl::range<1>(1), handler);
        handler.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(
                    kHiddenGroups, groups_per_group * work_group),
                sycl::range<2>(1u, work_group)),
            [=](sycl::nd_item<2> item)
                [[sycl::reqd_sub_group_size(16)]] {
                    if (item.get_local_id(1) == 0u) {
                        gate_valid[0] =
                            epoch != nullptr &&
                                    token_graph_service::device_epoch_valid(
                                        epoch)
                                ? 1u
                                : 0u;
                    }
                    sycl::group_barrier(
                        item.get_group(),
                        sycl::memory_scope::work_group);
                    if (gate_valid[0] == 0u) return;
                    const std::size_t group = item.get_group(0);
                    const std::size_t vector = item.get_global_id(1);
                    if (vector >= vectors) return;

                    const float inject_value =
                        static_cast<float>(inject[group]);
                    const std::size_t offset =
                        group * kTokenGraphHiddenElements +
                        vector * kVec8Elements;
                    Vec8 residual = load_vec8(hidden4 + offset);
                    const Vec8 routed =
                        load_vec8(y + vector * kVec8Elements);
#pragma unroll
                    for (std::uint32_t lane = 0u;
                         lane < kVec8Elements; ++lane) {
                        residual[lane] = bf16(
                            static_cast<float>(residual[lane]) +
                            inject_value *
                                static_cast<float>(routed[lane]));
                    }
                    store_vec8(hidden4 + offset, residual);
                });
    });
}

}  // namespace token_graph_kernels
}  // namespace aila
