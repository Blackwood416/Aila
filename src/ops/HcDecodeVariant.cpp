#include "ops/HcDecodeVariant.hpp"

#include <stdexcept>

namespace ops {
namespace {

using vec8 = sycl::vec<bf16, 8>;

// Explicit load-ahead pipeline over the production lane-strided vec8 loop.
// FMA order per lane: i = lane, lane + sg_size, ... with the v = 0..7 inner
// order inside each vec8. Only the loads move earlier.
template <int Depth>
struct Pipeline {
    static_assert(Depth >= 1 && Depth <= 4, "prefetch depth must be 1..4");

    template <typename LoadFn, typename AccFn>
    static void run(int lane, int limit, int stride, LoadFn&& load, AccFn&& accumulate) {
        vec8 w_buf[Depth];
        vec8 x_buf[Depth];
        int next = lane;
        for (int d = 0; d < Depth; ++d) {
            if (next < limit) load(next, w_buf[d], x_buf[d]);
            next += stride;
        }
        int slot = 0;
        for (int i = lane; i < limit; i += stride) {
            const vec8 w = w_buf[slot];
            const vec8 x = x_buf[slot];
            if (next < limit) load(next, w_buf[slot], x_buf[slot]);
            next += stride;
            slot = (slot + 1 == Depth) ? 0 : slot + 1;
            accumulate(w, x);
        }
    }
};

template <int Depth, int WorkGroup>
void hc_decode_down_inject_pl_impl(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups) {
    constexpr int sg_size = 16;
    constexpr int rows_per_wg = WorkGroup / sg_size;
    const int k_dim = groups * group_dim;
    const int k8 = k_dim / 8;
    const int total_rows = lowrank + (w_inject ? groups : 0);
    const size_t num_wgs = (static_cast<size_t>(total_rows) + rows_per_wg - 1) / rows_per_wg;
    const float inv_groups = 1.0f / static_cast<float>(groups);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * WorkGroup), sycl::range<1>(WorkGroup)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int sg_id = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int row = static_cast<int>(item.get_group(0)) * rows_per_wg + sg_id;
                if (row >= total_rows) return;

                const bool is_inject = row >= lowrank;
                const bf16* w_row = is_inject
                    ? w_inject + static_cast<size_t>(row - lowrank) * k_dim
                    : w_down + static_cast<size_t>(row) * k_dim;
                float acc = 0.0f;
                const bool is_aligned = (reinterpret_cast<uintptr_t>(w_row) % 16 == 0) &&
                                        (reinterpret_cast<uintptr_t>(N) % 16 == 0);
                if (is_aligned) {
                    Pipeline<Depth>::run(
                        lane, k8, sg_size,
                        [&](int i, vec8& w, vec8& x) {
                            w = *reinterpret_cast<const vec8*>(w_row + static_cast<size_t>(i) * 8);
                            x = *reinterpret_cast<const vec8*>(N + static_cast<size_t>(i) * 8);
                        },
                        [&](const vec8& w, const vec8& x) {
#pragma unroll
                            for (int v = 0; v < 8; ++v)
                                acc = sycl::fma(static_cast<float>(w[v]), static_cast<float>(x[v]), acc);
                        });
                    const int tail_start = k8 * 8;
                    for (int i = tail_start + lane; i < k_dim; i += sg_size)
                        acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(N[i]), acc);
                } else {
                    for (int i = lane; i < k_dim; i += sg_size)
                        acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(N[i]), acc);
                }

                const float total = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                if (lane == 0) {
                    const float v = static_cast<float>(bf16(total)) * inv_groups;
                    const float s = 1.0f / (1.0f + sycl::native::exp(-v));
                    if (is_inject) {
                        inj[row - lowrank] = 2.0f * s;
                    } else {
                        act[row] = bf16(v * s);
                    }
                }
            });
    });
}

template <int Depth>
void hc_decode_up_mix_pl_impl(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups) {
    constexpr int wg = 256;
    constexpr int sg_size = 16;
    constexpr int dims_per_wg = wg / sg_size;
    const int k8 = lowrank / 8;
    const size_t num_wgs = (static_cast<size_t>(group_dim) + dims_per_wg - 1) / dims_per_wg;
    const float inv_groups = 1.0f / static_cast<float>(groups);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int d = static_cast<int>(item.get_group(0)) * dims_per_wg +
                              static_cast<int>(sg.get_group_linear_id());
                if (d >= group_dim) return;

                float mixed = 0.0f;
                for (int g = 0; g < groups; ++g) {
                    const size_t row = static_cast<size_t>(g) * group_dim + d;
                    const bf16* w_row = w_up + row * lowrank;
                    float acc = 0.0f;
                    const bool is_aligned = (reinterpret_cast<uintptr_t>(w_row) % 16 == 0) &&
                                            (reinterpret_cast<uintptr_t>(act) % 16 == 0);
                    if (is_aligned) {
                        Pipeline<Depth>::run(
                            lane, k8, sg_size,
                            [&](int i, vec8& w, vec8& a) {
                                w = *reinterpret_cast<const vec8*>(w_row + static_cast<size_t>(i) * 8);
                                a = *reinterpret_cast<const vec8*>(act + static_cast<size_t>(i) * 8);
                            },
                            [&](const vec8& w, const vec8& a) {
#pragma unroll
                                for (int v = 0; v < 8; ++v)
                                    acc = sycl::fma(static_cast<float>(w[v]), static_cast<float>(a[v]), acc);
                            });
                        const int tail_start = k8 * 8;
                        for (int i = tail_start + lane; i < lowrank; i += sg_size)
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(act[i]), acc);
                    } else {
                        for (int i = lane; i < lowrank; i += sg_size)
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(act[i]), acc);
                    }
                    const float dot = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                    const float u = static_cast<float>(bf16(dot));
                    const float m = static_cast<float>(bf16(1.0f / (1.0f + sycl::native::exp(-u))));
                    mixed += m * static_cast<float>(N[row]);
                }
                if (lane == 0) x[d] = bf16(mixed * inv_groups);
            });
    });
}

template <int Depth>
void down_pl_dispatch(Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
                      bf16* act, float* inj, int lowrank, int group_dim, int groups, int work_group) {
    switch (work_group) {
    case 64: hc_decode_down_inject_pl_impl<Depth, 64>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups); break;
    case 128: hc_decode_down_inject_pl_impl<Depth, 128>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups); break;
    case 256: hc_decode_down_inject_pl_impl<Depth, 256>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups); break;
    default: throw std::invalid_argument("hc_decode_down_inject_pl: work-group must be 64, 128 or 256");
    }
}

} // namespace

void hc_decode_down_inject_pl(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups,
    int prefetch_depth, int work_group) {
    switch (prefetch_depth) {
    case 1: down_pl_dispatch<1>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, work_group); break;
    case 2: down_pl_dispatch<2>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, work_group); break;
    case 3: down_pl_dispatch<3>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, work_group); break;
    case 4: down_pl_dispatch<4>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, work_group); break;
    default: throw std::invalid_argument("hc_decode_down_inject_pl: prefetch_depth must be 1..4");
    }
}

void hc_decode_up_mix_pl(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups, int prefetch_depth) {
    switch (prefetch_depth) {
    case 1: hc_decode_up_mix_pl_impl<1>(ctx, act, w_up, N, x, lowrank, group_dim, groups); break;
    case 2: hc_decode_up_mix_pl_impl<2>(ctx, act, w_up, N, x, lowrank, group_dim, groups); break;
    case 3: hc_decode_up_mix_pl_impl<3>(ctx, act, w_up, N, x, lowrank, group_dim, groups); break;
    case 4: hc_decode_up_mix_pl_impl<4>(ctx, act, w_up, N, x, lowrank, group_dim, groups); break;
    default: throw std::invalid_argument("hc_decode_up_mix_pl: prefetch_depth must be 1..4");
    }
}

} // namespace ops
