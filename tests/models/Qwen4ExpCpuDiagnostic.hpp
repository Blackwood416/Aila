#pragma once

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

namespace q4e_cpu_diagnostic {
using bf16 = sycl::ext::oneapi::bfloat16;
struct Complete {};
struct State { int samples = 0; int limit = 16; };
inline constexpr float kQmap[16] = {-1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f};
#if defined(__clang__) && (defined(__x86_64__) || defined(_M_X64)) && !defined(__SYCL_DEVICE_ONLY__)
__attribute__((target("fma")))
#endif
inline float host_fma(float a, float b, float c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

template<class T>
void compare(const char* stage, const T* reference, const T* actual, size_t count) {
    size_t different = 0;
    double max_abs = 0, sum_sq = 0, ref_sq = 0;
    size_t first = count;
    for (size_t i = 0; i < count; ++i) {
        const double a = static_cast<float>(reference[i]);
        const double b = static_cast<float>(actual[i]);
        const float af = static_cast<float>(reference[i]), bf = static_cast<float>(actual[i]);
        uint32_t abits, bbits;
        std::memcpy(&abits, &af, sizeof(abits));
        std::memcpy(&bbits, &bf, sizeof(bbits));
        if ((abits & 0x7f800000u) == 0x7f800000u || (bbits & 0x7f800000u) == 0x7f800000u)
            throw std::runtime_error("Nonfinite diagnostic value");
        if (std::memcmp(reference + i, actual + i, sizeof(T))) {
            ++different;
            if (first == count) first = i;
        }
        max_abs = std::max(max_abs, std::abs(a - b));
        sum_sq += (a - b) * (a - b);
        ref_sq += a * a;
    }
    std::cout << std::setprecision(9) << "[CPU Diagnostic] stage=" << stage
              << " bits_different=" << different << '/' << count << " max_abs=" << max_abs
              << " relative_l2=" << std::sqrt(sum_sq / std::max(ref_sq, 1e-300));
    if (first < count) std::cout << " first=" << first << " ref=" << static_cast<float>(reference[first])
                                 << " actual=" << static_cast<float>(actual[first]);
    std::cout << '\n';
}

// Force two f32 roundings independently of host contraction flags.
inline std::vector<float> expand(const ops::MoeNestedScaleHost& nested, size_t count) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        volatile float product = nested.nested_quant_map[nested.codes[i]] * nested.nested_absmax[i / nested.blocksize];
        result[i] = product + nested.offset;
    }
    return result;
}

// Diagnostic only: reproduce the production kernel's lane-strided FMA order.
inline void lane_dot(const uint8_t* packed, const float* scales, const bf16* x,
                     int rows, int columns, float* halves, float* adjacent, const float* reference = nullptr,
                     float* partials = nullptr, const float* override_partials = nullptr, int arithmetic = 0) {
    size_t tree_errors[24] = {};
    for (int row = 0; row < rows; ++row) {
        volatile float lanes[16] = {};
        for (int lane = 0; lane < 16; ++lane) {
            for (int column = lane * 8; column < columns; column += 128) {
                const float scale = scales[row * (columns / 64) + column / 64];
                for (int j = 0; j < 8; ++j) {
                    const uint8_t code = packed[row * (columns / 2) + (column + j) / 2];
                    volatile float weight = kQmap[(j & 1) ? code & 15 : code >> 4] * scale;
                    if (arithmetic == 1) {
                        volatile float product = static_cast<float>(x[column + j]) * weight;
                        lanes[lane] = lanes[lane] + product;
                    } else if (arithmetic == 2) {
                        volatile float input_q = static_cast<float>(x[column + j]) * kQmap[(j & 1) ? code & 15 : code >> 4];
                        lanes[lane] = host_fma(input_q, scale, lanes[lane]);
                    } else {
                        lanes[lane] = host_fma(static_cast<float>(x[column + j]), weight, lanes[lane]);
                    }
                }
            }
        }
        for (int lane = 0; lane < 16; ++lane) {
            if (partials) partials[row * 16 + lane] = lanes[lane];
            if (override_partials) lanes[lane] = override_partials[row * 16 + lane];
        }
        if (reference) {
            int order[4] = {1, 2, 4, 8};
            int candidate = 0;
            do {
                volatile float values[16];
                for (int i = 0; i < 16; ++i) values[i] = lanes[i];
                for (int stride : order)
                    for (int i = 0; i < 16; ++i) if (!(i & stride)) values[i] = values[i] + values[i | stride];
                const float sum = values[0];
                if (std::memcmp(&sum, reference + row, sizeof(float))) ++tree_errors[candidate];
                ++candidate;
            } while (std::next_permutation(order, order + 4));
        }
        volatile float tree[16];
        for (int i = 0; i < 16; ++i) tree[i] = lanes[i];
        for (int stride = 8; stride; stride /= 2)
            for (int i = 0; i < stride; ++i) tree[i] = tree[i] + tree[i + stride];
        halves[row] = tree[0];
        for (int stride = 1; stride < 16; stride *= 2)
            for (int i = 0; i < 16; i += stride * 2) lanes[i] = lanes[i] + lanes[i + stride];
        adjacent[row] = lanes[0];
    }
    if (reference) {
        int order[4] = {1, 2, 4, 8};
        int candidate = 0;
        do {
            std::cout << "[CPU Diagnostic] tree=" << order[0] << ',' << order[1] << ',' << order[2] << ',' << order[3]
                      << " bits_different=" << tree_errors[candidate++] << '/' << rows << '\n';
        } while (std::next_permutation(order, order + 4));
    }
}

inline void observe(void* opaque, Context& ctx, int layer, const aila::RouteRankEntry& entry,
                    const ops::MoeExpertSlotHost& host, const bf16* x, const bf16* cpu_h, const float* cpu_out) {
    auto& state = *static_cast<State*>(opaque);
    constexpr int H = 2560, I = 640, B = 64;
    if (!host.gate_up_nested.codes || !host.down_nested.codes) throw std::runtime_error("Nested fixture required");
    ctx.synchronize();
    std::cout << "[CPU Diagnostic] sample=" << state.samples << " layer=" << layer
              << " expert=" << entry.expert_id << " token=" << entry.token_idx << " rank=" << entry.rank << '\n';
    std::vector<Tensor> storage;
    storage.reserve(12);
    auto upload = [&](const void* source, size_t bytes) -> void* {
        storage.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(bytes)}, dnnl::memory::data_type::u8));
        ctx.memcpy_h2d(storage.back().data(), source, bytes);
        return storage.back().data();
    };
    ops::MoeExpertSlotDev dev;
    dev.gate_up_packed = static_cast<uint8_t*>(upload(host.gate_up_packed, H * I));
    dev.down_packed = static_cast<uint8_t*>(upload(host.down_packed, H * I / 2));
    auto upload_nested = [&](const ops::MoeNestedScaleHost& src, size_t count) {
        ops::MoeNestedScaleDev dst;
        dst.codes = static_cast<uint8_t*>(upload(src.codes, count));
        dst.nested_absmax = static_cast<float*>(upload(src.nested_absmax, ((count + src.blocksize - 1) / src.blocksize) * sizeof(float)));
        dst.nested_quant_map = static_cast<float*>(upload(src.nested_quant_map, 256 * sizeof(float)));
        dst.offset = src.offset;
        dst.blocksize = src.blocksize;
        return dst;
    };
    dev.gate_up_nested = upload_nested(host.gate_up_nested, 2 * H * I / B);
    dev.down_nested = upload_nested(host.down_nested, H * I / B);
    auto gu_scales = expand(host.gate_up_nested, 2 * H * I / B);
    auto dn_scales = expand(host.down_nested, H * I / B);
    auto unnested = host;
    unnested.gate_up_nested = {}; unnested.down_nested = {};
    unnested.gate_up_absmax = gu_scales.data(); unnested.down_absmax = dn_scales.data();
    auto dev_unnested = dev;
    dev_unnested.gate_up_nested = {}; dev_unnested.down_nested = {};
    dev_unnested.gate_up_absmax = static_cast<float*>(upload(gu_scales.data(), gu_scales.size() * sizeof(float)));
    dev_unnested.down_absmax = static_cast<float*>(upload(dn_scales.data(), dn_scales.size() * sizeof(float)));
    auto* xd = static_cast<bf16*>(upload(x, H * sizeof(bf16)));
    Tensor hd = Tensor::allocate(ctx, {I}, dnnl::memory::data_type::bf16);
    Tensor od = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::f32);
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    std::vector<bf16> gpu_h(I), gpu_h_expanded(I), cpu_h_expanded(I);
    std::vector<float> gpu_out(H), gpu_out_expanded(H), same_h_out(H), expanded_out(H);
    auto gpu = [&](const ops::MoeExpertSlotDev& slot, bool nested, std::vector<bf16>& h, std::vector<float>& out) {
        auto gu = ops::moe_decode_gate_up_swiglu_grouped(ctx, &slot, nullptr, xd, hd.data_as<bf16>(), 1, H, I, B, nullptr, nullptr, nested);
        const int rank = 0;
        ops::moe_decode_down_to_rank_f32(ctx, &slot, &rank, hd.data_as<bf16>(), od.data_as<float>(), 1, 1, I, H, B, nullptr, nullptr, &gu, nested).wait_and_throw();
        ctx.memcpy_d2h(h.data(), hd.data(), I * sizeof(bf16));
        ctx.memcpy_d2h(out.data(), od.data(), H * sizeof(float));
    };
    gpu(dev, true, gpu_h, gpu_out);
    gpu(dev_unnested, false, gpu_h_expanded, gpu_out_expanded);
    compare("gpu_nested_vs_expanded_h", gpu_h_expanded.data(), gpu_h.data(), I);
    compare("gpu_nested_vs_expanded_down", gpu_out_expanded.data(), gpu_out.data(), H);
    ops::cpu::cpu_expert_gate_up_swiglu(unnested, x, cpu_h_expanded.data(), 1);
    compare("cpu_nested_vs_expanded_h", cpu_h_expanded.data(), cpu_h, I);
    compare("gpu_vs_cpu_h", gpu_h.data(), cpu_h, I);
    ops::cpu::cpu_expert_down(host, gpu_h.data(), same_h_out.data(), 1);
    ops::cpu::cpu_expert_down(unnested, gpu_h.data(), expanded_out.data(), 1);
    compare("cpu_nested_vs_expanded_down_same_h", expanded_out.data(), same_h_out.data(), H);
    compare("gpu_vs_cpu_down_same_h", gpu_out.data(), same_h_out.data(), H);
    compare("gpu_vs_cpu_full", gpu_out.data(), cpu_out, H);
    std::vector<float> down_halves(H), down_adjacent(H), gu_halves(2 * I), gu_adjacent(2 * I);
    lane_dot(host.down_packed, dn_scales.data(), gpu_h.data(), H, I, down_halves.data(), down_adjacent.data(), gpu_out.data());
    compare("gpu_vs_lane_halves_down_same_h", gpu_out.data(), down_halves.data(), H);
    compare("gpu_vs_lane_adjacent_down_same_h", gpu_out.data(), down_adjacent.data(), H);
    lane_dot(host.gate_up_packed, gu_scales.data(), x, 2 * I, H, gu_halves.data(), gu_adjacent.data());
    std::vector<bf16> lane_h(I), adjacent_h(I);
    for (int i = 0; i < I; ++i) {
        lane_h[i] = bf16((gu_halves[i] / (1.0f + std::exp(-gu_halves[i]))) * gu_halves[I + i]);
        adjacent_h[i] = bf16((gu_adjacent[i] / (1.0f + std::exp(-gu_adjacent[i]))) * gu_adjacent[I + i]);
    }
    compare("gpu_vs_lane_halves_h", gpu_h.data(), lane_h.data(), I);
    compare("gpu_vs_lane_adjacent_h", gpu_h.data(), adjacent_h.data(), I);
    // Captured lane partials separate local dot arithmetic from subgroup reduction.
    Tensor lane_dev = Tensor::allocate(ctx, {H, 16}, dnnl::memory::data_type::f32);
    Tensor sum_dev = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::f32);
    Tensor map_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(map_dev.data(), kQmap, sizeof(kQmap));
    ctx.memcpy_h2d(hd.data(), gpu_h.data(), I * sizeof(bf16));
    auto* lane_ptr = lane_dev.data_as<float>();
    auto* sum_ptr = sum_dev.data_as<float>();
    const auto* map_ptr = map_dev.data_as<float>();
    const auto* h_ptr = hd.data_as<bf16>();
    const auto* packed_ptr = dev.down_packed;
    const auto* scales_ptr = dev_unnested.down_absmax;
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(H * 16, 256), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int row = item.get_global_id(0) / 16;
            const int lane = item.get_local_id(0) % 16;
            float acc = 0;
            for (int column = lane * 8; column < I; column += 128) {
                const float scale = scales_ptr[row * (I / 64) + column / 64];
                for (int j = 0; j < 8; ++j) {
                    const uint8_t code = packed_ptr[row * (I / 2) + (column + j) / 2];
                    acc = sycl::fma(static_cast<float>(h_ptr[column + j]),
                        map_ptr[(j & 1) ? code & 15 : code >> 4] * scale, acc);
                }
            }
            lane_ptr[row * 16 + lane] = acc;
            const float sum = sycl::reduce_over_group(item.get_sub_group(), acc, sycl::plus<>());
            if (lane == 0) sum_ptr[row] = sum;
        });
    }).wait_and_throw();
    std::vector<float> device_lanes(H * 16), host_lanes(H * 16), device_sum(H);
    ctx.memcpy_d2h(device_lanes.data(), lane_ptr, device_lanes.size() * sizeof(float));
    ctx.memcpy_d2h(device_sum.data(), sum_ptr, H * sizeof(float));
    compare("production_vs_instrumented_down", gpu_out.data(), device_sum.data(), H);
    lane_dot(host.down_packed, dn_scales.data(), gpu_h.data(), H, I, down_halves.data(), down_adjacent.data(),
             nullptr, host_lanes.data(), device_lanes.data());
    compare("gpu_vs_cpu_lane_partials", device_lanes.data(), host_lanes.data(), host_lanes.size());
    compare("device_lanes_halves_reduce", device_sum.data(), down_halves.data(), H);
    compare("device_lanes_adjacent_reduce", device_sum.data(), down_adjacent.data(), H);
    lane_dot(host.down_packed, dn_scales.data(), gpu_h.data(), H, I, down_halves.data(), down_adjacent.data(),
             nullptr, host_lanes.data(), nullptr, 1);
    compare("gpu_vs_cpu_nonfused_partials", device_lanes.data(), host_lanes.data(), host_lanes.size());
    lane_dot(host.down_packed, dn_scales.data(), gpu_h.data(), H, I, down_halves.data(), down_adjacent.data(),
             nullptr, host_lanes.data(), nullptr, 2);
    compare("gpu_vs_cpu_input_q_fma_partials", device_lanes.data(), host_lanes.data(), host_lanes.size());
    std::vector<bf16> candidate_h(I);
    std::vector<float> candidate_down(H);
    ops::cpu::cpu_expert_gate_up_swiglu(host, x, candidate_h.data(), 1, nullptr, H, I, 64, ops::cpu::CpuKernelBackend::GpuOrderAvx2);
    ops::cpu::cpu_expert_down(host, gpu_h.data(), candidate_down.data(), 1, nullptr, 1.0f, false, I, H, 64, ops::cpu::CpuKernelBackend::GpuOrderAvx2);
    compare("gpu_vs_candidate_h", gpu_h.data(), candidate_h.data(), I);
    compare("gpu_vs_candidate_down_same_h", gpu_out.data(), candidate_down.data(), H);
    ++state.samples;
    if (state.samples == state.limit) throw Complete{};
}
} // namespace q4e_cpu_diagnostic
