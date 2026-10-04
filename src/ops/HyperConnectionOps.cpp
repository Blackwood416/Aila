#include "HyperConnectionOps.hpp"

namespace ops {

void grouped_rms_norm(
    Context& ctx, const bf16* x, const float* w, float eps, bf16* out,
    int rows, int groups, int group_dim, bool plus_one) {
    constexpr int wg = 256;
    const int total_groups = rows * groups;


    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(wg), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(total_groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int group_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int r = group_idx / groups;
                const int g = group_idx % groups;
                const size_t row_base = static_cast<size_t>(r) * (groups * group_dim);
                const size_t stream_base = row_base + static_cast<size_t>(g) * group_dim;
                const size_t w_base = static_cast<size_t>(g) * group_dim;

                float sum_sq = 0.0f;
                for (int i = lid; i < group_dim; i += wg) {
                    float v = static_cast<float>(x[stream_base + i]);
                    sum_sq += v * v;
                }
                red_slm[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        red_slm[lid] += red_slm[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float inv_rms = 1.0f / sycl::sqrt(red_slm[0] / static_cast<float>(group_dim) + eps);

                for (int i = lid; i < group_dim; i += wg) {
                    float v = static_cast<float>(x[stream_base + i]);
                    float scale = inv_rms;
                    if (w) {
                        float w_val = w[w_base + i];
                        scale *= (plus_one ? (1.0f + w_val) : w_val);
                    }
                    out[stream_base + i] = bf16(v * scale);
                }
            });
    });
}

void grouped_rms_norm(
    Context& ctx, Tensor& x, Tensor& w, float eps, Tensor& out,
    int rows, int groups, int group_dim, bool plus_one) {
    grouped_rms_norm(
        ctx, static_cast<const bf16*>(x.data()),
        w.valid() ? static_cast<const float*>(w.data()) : nullptr,
        eps, static_cast<bf16*>(out.data()),
        rows, groups, group_dim, plus_one);
}

void hc_mix_reduce(
    Context& ctx, const bf16* N, const bf16* mix, bf16* out,
    int rows, int group_dim, int groups, bool apply_sigmoid_to_mix) {
    if ((group_dim % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = group_dim / 8;
        constexpr int wg = 64;
        const size_t num_wgs_per_row = (dim8 + wg - 1) / wg;
        const float inv_groups = 1.0f / static_cast<float>(groups);

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(rows), num_wgs_per_row * wg),
                                  sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t r = item.get_group(0);
                    const size_t d8 = item.get_global_id(1);
                    if (d8 < static_cast<size_t>(dim8)) {
                        const size_t row_stride = static_cast<size_t>(groups) * group_dim;
                        const size_t row_base = r * row_stride;

                        float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                        for (int g = 0; g < groups; ++g) {
                            const size_t off = row_base + static_cast<size_t>(g) * group_dim + d8 * 8;
                            const vec8 vm = *reinterpret_cast<const vec8*>(mix + off);
                            const vec8 vn = *reinterpret_cast<const vec8*>(N + off);
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                float m = static_cast<float>(vm[v]);
                                if (apply_sigmoid_to_mix) {
                                    m = static_cast<float>(bf16(1.0f / (1.0f + sycl::native::exp(-m))));
                                }
                                sum[v] += m * static_cast<float>(vn[v]);
                            }
                        }
                        vec8 vout;
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            vout[v] = bf16(sum[v] * inv_groups);
                        }
                        *reinterpret_cast<vec8*>(out + r * group_dim + d8 * 8) = vout;
                    }
                });
        });
        return;
    }

    constexpr int wg = 256;
    const size_t total_elements = static_cast<size_t>(rows) * group_dim;
    const size_t num_wgs = (total_elements + wg - 1) / wg;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t idx = item.get_global_id(0);
                if (idx < total_elements) {
                    const int r = static_cast<int>(idx / group_dim);
                    const int d = static_cast<int>(idx % group_dim);
                    const size_t row_stride = static_cast<size_t>(groups) * group_dim;
                    const size_t row_base = static_cast<size_t>(r) * row_stride;

                    float sum = 0.0f;
                    for (int g = 0; g < groups; ++g) {
                        size_t off = row_base + static_cast<size_t>(g) * group_dim + d;
                        float m = static_cast<float>(mix[off]);
                        if (apply_sigmoid_to_mix) {
                            m = static_cast<float>(bf16(1.0f / (1.0f + sycl::native::exp(-m))));
                        }
                        sum += m * static_cast<float>(N[off]);
                    }
                    out[idx] = bf16(sum / static_cast<float>(groups));
                }
            });
    });
}

template <typename TInj>
static void hc_inject_add_impl(
    Context& ctx, bf16* H, const bf16* y, const TInj* inj,
    int rows, int group_dim, int groups) {
    if (group_dim % 8 == 0) {
        constexpr size_t wg = 64;
        using vec8 = sycl::vec<bf16, 8>;
        const size_t dim8 = static_cast<size_t>(group_dim / 8);
        const size_t num_wgs_per_group = (dim8 + wg - 1) / wg;
        const size_t total_groups = static_cast<size_t>(rows) * groups;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(total_groups, num_wgs_per_group * wg),
                                  sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t rg = item.get_group(0);
                    const size_t i = item.get_global_id(1);

                    if (i < dim8) {
                        size_t r;
                        if (groups == 4) {
                            r = rg >> 2;
                        } else {
                            r = (rows == 1) ? 0 : (rg / groups);
                        }

                        const float inj_val = static_cast<float>(inj[rg]);
                        const size_t g_offset = rg * group_dim + i * 8;
                        const size_t y_offset = r * group_dim + i * 8;

                        vec8 vh = *reinterpret_cast<const vec8*>(H + g_offset);
                        const vec8 vy = *reinterpret_cast<const vec8*>(y + y_offset);
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            vh[v] = bf16(static_cast<float>(vh[v]) + inj_val * static_cast<float>(vy[v]));
                        }
                        *reinterpret_cast<vec8*>(H + g_offset) = vh;
                    }
                });
        });
        return;
    }

    constexpr int wg = 256;
    const size_t total = static_cast<size_t>(rows) * groups * group_dim;
    const size_t num_wgs = (total + wg - 1) / wg;
    const size_t group_stride = static_cast<size_t>(groups) * group_dim;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t idx = item.get_global_id(0);
                if (idx < total) {
                    const int r = static_cast<int>(idx / group_stride);
                    const size_t rem = idx % group_stride;
                    const int g = static_cast<int>(rem / group_dim);
                    const int d = static_cast<int>(rem % group_dim);

                    float h_val = static_cast<float>(H[idx]);
                    float inj_val = static_cast<float>(inj[static_cast<size_t>(r) * groups + g]);
                    float y_val = static_cast<float>(y[static_cast<size_t>(r) * group_dim + d]);
                    H[idx] = bf16(h_val + inj_val * y_val);
                }
            });
    });
}

void hc_inject_add(
    Context& ctx, bf16* H, const bf16* y, const float* inj,
    int rows, int group_dim, int groups) {
    hc_inject_add_impl(ctx, H, y, inj, rows, group_dim, groups);
}

void hc_inject_add(
    Context& ctx, bf16* H, const bf16* y, const bf16* inj,
    int rows, int group_dim, int groups) {
    hc_inject_add_impl(ctx, H, y, inj, rows, group_dim, groups);
}

void hc_inject_add_and_grouped_rms_norm(
    Context& ctx,
    bf16* H,
    const bf16* y,
    const float* inj,
    const float* norm_w,
    float eps,
    bf16* out_N,
    int rows,
    int groups,
    int group_dim,
    bool plus_one) {
    if (rows <= 0 || groups <= 0 || group_dim <= 0) return;

    constexpr int wg = 256;
    const int total_groups = rows * groups;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(wg), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(total_groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int group_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int r = (groups == 4) ? (group_idx >> 2) : (group_idx / groups);
                const int g = (groups == 4) ? (group_idx & 3) : (group_idx % groups);
                const size_t stream_base = static_cast<size_t>(group_idx) * group_dim;
                const size_t y_base = static_cast<size_t>(r) * group_dim;
                const size_t w_base = static_cast<size_t>(g) * group_dim;
                const float inj_val = inj ? static_cast<float>(inj[group_idx]) : 0.0f;

                constexpr int kMaxElemsPerThread = 16;
                bf16 v_reg[kMaxElemsPerThread];
                int reg_idx = 0;
                float sum_sq = 0.0f;

                for (int i = lid; i < group_dim; i += wg) {
                    float h_val = static_cast<float>(H[stream_base + i]);
                    float y_val = y ? static_cast<float>(y[y_base + i]) : 0.0f;
                    bf16 new_h = bf16(h_val + inj_val * y_val);
                    H[stream_base + i] = new_h;
                    if (reg_idx < kMaxElemsPerThread) {
                        v_reg[reg_idx++] = new_h;
                    }
                    float v = static_cast<float>(new_h);
                    sum_sq += v * v;
                }
                red_slm[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        red_slm[lid] += red_slm[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float inv_rms = 1.0f / sycl::sqrt(red_slm[0] / static_cast<float>(group_dim) + eps);

                reg_idx = 0;
                for (int i = lid; i < group_dim; i += wg) {
                    float v = (reg_idx < kMaxElemsPerThread)
                        ? static_cast<float>(v_reg[reg_idx++])
                        : static_cast<float>(H[stream_base + i]);
                    float scale = inv_rms;
                    if (norm_w) {
                        float w_val = norm_w[w_base + i];
                        scale *= (plus_one ? (1.0f + w_val) : w_val);
                    }
                    out_N[stream_base + i] = bf16(v * scale);
                }
            });
    });
}

bool try_hc_decode_inject_add_and_grouped_rms_norm(
    Context& ctx, bf16* H, const bf16* y, const float* inj,
    const float* norm_w, float eps, bf16* out_N, int group_dim, bool plus_one) {
    using vec8 = sycl::vec<bf16, 8>;
    if (group_dim <= 0 || group_dim % 8 != 0 || !H || !y || !inj || !out_N ||
        reinterpret_cast<uintptr_t>(H) % alignof(vec8) != 0 ||
        reinterpret_cast<uintptr_t>(y) % alignof(vec8) != 0) return false;

    constexpr int wg = 256;
    constexpr int groups = 4;
    const int dim8 = group_dim / 8;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(wg), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int group_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const size_t stream_base = static_cast<size_t>(group_idx) * group_dim;
                const size_t w_base = static_cast<size_t>(group_idx) * group_dim;
                const float inj_val = static_cast<float>(inj[group_idx]);
                // Keep the original standalone vec8 arithmetic and BF16 store.
                // Each stream has one WG, so its entire update precedes its norm.
                for (int i = lid; i < dim8; i += wg) {
                    const size_t g_offset = stream_base + static_cast<size_t>(i) * 8;
                    const size_t y_offset = static_cast<size_t>(i) * 8;
                    vec8 vh = *reinterpret_cast<const vec8*>(H + g_offset);
                    const vec8 vy = *reinterpret_cast<const vec8*>(y + y_offset);
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        vh[v] = bf16(static_cast<float>(vh[v]) + inj_val * static_cast<float>(vy[v]));
                    }
                    *reinterpret_cast<vec8*>(H + g_offset) = vh;
                }
                item.barrier(sycl::access::fence_space::global_and_local);

                // These loads and the lid+256 accumulation/reduction/output
                // order intentionally match the original grouped_rms_norm.
                float sum_sq = 0.0f;
                for (int i = lid; i < group_dim; i += wg) {
                    float v = static_cast<float>(H[stream_base + i]);
                    sum_sq += v * v;
                }
                red_slm[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) red_slm[lid] += red_slm[lid + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float inv_rms = 1.0f / sycl::sqrt(red_slm[0] / static_cast<float>(group_dim) + eps);
                for (int i = lid; i < group_dim; i += wg) {
                    float v = static_cast<float>(H[stream_base + i]);
                    float scale = inv_rms;
                    if (norm_w) {
                        float w_val = norm_w[w_base + i];
                        scale *= (plus_one ? (1.0f + w_val) : w_val);
                    }
                    out_N[stream_base + i] = bf16(v * scale);
                }
            });
    });
    return true;
}

namespace {
template<bool Batched, int WorkGroup = 256>
void hc_decode_down_inject_impl(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups, int rows) {
    constexpr int wg = WorkGroup;
    constexpr int sg_size = 16;
    constexpr int rows_per_wg = wg / sg_size;
    using vec8 = sycl::vec<bf16, 8>;
    const int k_dim = groups * group_dim;
    const int k8 = k_dim / 8;
    const int total_rows = lowrank + (w_inject ? groups : 0);
    const size_t num_wgs = (static_cast<size_t>(total_rows) + rows_per_wg - 1) / rows_per_wg;
    const float inv_groups = 1.0f / static_cast<float>(groups);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * (Batched ? rows : 1) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int sg_id = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const size_t batch = Batched ? item.get_group(0) / num_wgs : 0;
                const size_t group = Batched ? item.get_group(0) % num_wgs : item.get_group(0);
                const bf16* n_row = N + batch * k_dim;
                const int row = static_cast<int>(group) * rows_per_wg + sg_id;

                if (row >= total_rows) return;

                const bool is_inject = row >= lowrank;
                const bf16* w_row = is_inject
                    ? w_inject + static_cast<size_t>(row - lowrank) * k_dim
                    : w_down + static_cast<size_t>(row) * k_dim;

                float acc = 0.0f;
                const bool is_aligned = (reinterpret_cast<uintptr_t>(w_row) % 16 == 0) &&
                                        (reinterpret_cast<uintptr_t>(n_row) % 16 == 0);
                if (is_aligned) {
                    for (int i = lane; i < k8; i += sg_size) {
                        const vec8 w = *reinterpret_cast<const vec8*>(w_row + i * 8);
                        const vec8 x = *reinterpret_cast<const vec8*>(n_row + i * 8);
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            acc = sycl::fma(static_cast<float>(w[v]), static_cast<float>(x[v]), acc);
                        }
                    }
                    const int tail_start = k8 * 8;
                    for (int i = tail_start + lane; i < k_dim; i += sg_size) {
                        acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(n_row[i]), acc);
                    }
                } else {
                    for (int i = lane; i < k_dim; i += sg_size) {
                        acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(n_row[i]), acc);
                    }
                }

                const float total = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                if (lane == 0) {
                    const float v = static_cast<float>(bf16(total)) * inv_groups;
                    const float s = 1.0f / (1.0f + sycl::native::exp(-v));
                    if (is_inject) {
                        inj[batch * groups + row - lowrank] = 2.0f * s;
                    } else {
                        act[batch * lowrank + row] = bf16(v * s);
                    }
                }
            });
    });
}

template<bool Batched>
void hc_decode_down_inject_dispatch(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups, int rows, int work_group) {
    switch (work_group) {
    case 64: hc_decode_down_inject_impl<Batched, 64>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, rows); break;
    case 128: hc_decode_down_inject_impl<Batched, 128>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, rows); break;
    case 256: hc_decode_down_inject_impl<Batched, 256>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, rows); break;
    default: throw std::invalid_argument("hc_decode_down_inject: work-group must be 64, 128 or 256");
    }
}

template<bool Batched>
void hc_decode_up_mix_impl(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups, int rows) {
    constexpr int wg = 256;
    constexpr int sg_size = 16;
    constexpr int dims_per_wg = wg / sg_size;
    constexpr int kMaxGroups = 8;
    using vec8 = sycl::vec<bf16, 8>;
    const int k8 = lowrank / 8;
    const size_t num_wgs = (static_cast<size_t>(group_dim) + dims_per_wg - 1) / dims_per_wg;
    const float inv_groups = 1.0f / static_cast<float>(groups);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * (Batched ? rows : 1) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const size_t batch = Batched ? item.get_group(0) / num_wgs : 0;
                const size_t group = Batched ? item.get_group(0) % num_wgs : item.get_group(0);
                const bf16* a_row = act + batch * lowrank;
                const bf16* n_row = N + batch * groups * group_dim;
                const int d = static_cast<int>(group) * dims_per_wg +
                              static_cast<int>(sg.get_group_linear_id());
                if (d >= group_dim) return;

                float mixed = 0.0f;
                for (int g = 0; g < groups && g < kMaxGroups; ++g) {
                    const size_t row = static_cast<size_t>(g) * group_dim + d;
                    const bf16* w_row = w_up + row * lowrank;
                    float acc = 0.0f;
                    const bool is_aligned = (reinterpret_cast<uintptr_t>(w_row) % 16 == 0) &&
                                            (reinterpret_cast<uintptr_t>(a_row) % 16 == 0);
                    if (is_aligned) {
                        for (int i = lane; i < k8; i += sg_size) {
                            const vec8 w = *reinterpret_cast<const vec8*>(w_row + i * 8);
                            const vec8 a = *reinterpret_cast<const vec8*>(a_row + i * 8);
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                acc = sycl::fma(static_cast<float>(w[v]), static_cast<float>(a[v]), acc);
                            }
                        }
                        const int tail_start = k8 * 8;
                        for (int i = tail_start + lane; i < lowrank; i += sg_size) {
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(a_row[i]), acc);
                        }
                    } else {
                        for (int i = lane; i < lowrank; i += sg_size) {
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(a_row[i]), acc);
                        }
                    }
                    const float dot = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                    const float u = static_cast<float>(bf16(dot));
                    const float m = static_cast<float>(bf16(1.0f / (1.0f + sycl::native::exp(-u))));
                    mixed += m * static_cast<float>(n_row[row]);
                }
                if (lane == 0) {
                    x[batch * group_dim + d] = bf16(mixed * inv_groups);
                }
            });
    });
}

template<bool Batched>
void hc_decode_up_mix_streams_impl(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int rows) {
    constexpr int wg = 256, sg_size = 16, groups = 4;
    constexpr int dims_per_wg = wg / (sg_size * groups);
    using vec8 = sycl::vec<bf16, 8>;
    const int k8 = lowrank / 8;
    const size_t num_wgs = (static_cast<size_t>(group_dim) + dims_per_wg - 1) / dims_per_wg;
    const float inv_groups = 1.0f / static_cast<float>(groups);
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> gates(sycl::range<1>(dims_per_wg * groups), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * (Batched ? rows : 1) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sg_id = static_cast<int>(sg.get_group_linear_id());
                const int local_d = sg_id / groups;
                const int g = sg_id % groups;
                const size_t batch = Batched ? item.get_group(0) / num_wgs : 0;
                const size_t group = Batched ? item.get_group(0) % num_wgs : item.get_group(0);
                const int d = static_cast<int>(group) * dims_per_wg + local_d;
                const bf16* a_row = act + batch * lowrank;
                const bf16* n_row = N + batch * groups * group_dim;
                // Tail subgroups must reach the WG barrier, even without a row.
                if (d < group_dim) {
                    const size_t row = static_cast<size_t>(g) * group_dim + d;
                    const bf16* w_row = w_up + row * lowrank;
                    float acc = 0.0f;
                    const bool is_aligned = (reinterpret_cast<uintptr_t>(w_row) % 16 == 0) &&
                                            (reinterpret_cast<uintptr_t>(a_row) % 16 == 0);
                    if (is_aligned) {
                        for (int i = lane; i < k8; i += sg_size) {
                            const vec8 w = *reinterpret_cast<const vec8*>(w_row + i * 8);
                            const vec8 a = *reinterpret_cast<const vec8*>(a_row + i * 8);
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                acc = sycl::fma(static_cast<float>(w[v]), static_cast<float>(a[v]), acc);
                            }
                        }
                        const int tail_start = k8 * 8;
                        for (int i = tail_start + lane; i < lowrank; i += sg_size) {
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(a_row[i]), acc);
                        }
                    } else {
                        for (int i = lane; i < lowrank; i += sg_size) {
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(a_row[i]), acc);
                        }
                    }
                    const float dot = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                    const float u = static_cast<float>(bf16(dot));
                    const bf16 m = bf16(1.0f / (1.0f + sycl::native::exp(-u)));
                    if (lane == 0) gates[local_d * groups + g] = m;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (g == 0 && d < group_dim) {
                    float mixed = 0.0f;
                    // Store only the BF16 gate, not its product. Preserve the
                    // original ordered multiply-add and final BF16 conversion.
                    for (int stream = 0; stream < groups; ++stream) {
                        const size_t row = static_cast<size_t>(stream) * group_dim + d;
                        const float m = static_cast<float>(gates[local_d * groups + stream]);
                        mixed += m * static_cast<float>(n_row[row]);
                    }
                    if (lane == 0) x[batch * group_dim + d] = bf16(mixed * inv_groups);
                }
            });
    });
}

} // namespace

void hc_decode_down_inject(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups, int down_work_group) {
    hc_decode_down_inject_dispatch<false>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, 1, down_work_group);
}

void hc_decode_up_mix(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups, bool parallel_streams) {
    if (parallel_streams && groups == 4) hc_decode_up_mix_streams_impl<false>(ctx, act, w_up, N, x, lowrank, group_dim, 1);
    else hc_decode_up_mix_impl<false>(ctx, act, w_up, N, x, lowrank, group_dim, groups, 1);
}

void hc_decode_down_inject_rows(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups, int rows, int down_work_group) {
    if (rows < 1 || rows > 4 || groups < 1 || groups > 8 || lowrank < 1 || group_dim < 1)
        throw std::invalid_argument("hc_decode_down_inject_rows: unsupported shape");
    hc_decode_down_inject_dispatch<true>(ctx, N, w_down, w_inject, act, inj, lowrank, group_dim, groups, rows, down_work_group);
}

void hc_decode_up_mix_rows(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups, int rows, bool parallel_streams) {
    if (rows < 1 || rows > 4 || groups < 1 || groups > 8 || lowrank < 1 || group_dim < 1)
        throw std::invalid_argument("hc_decode_up_mix_rows: unsupported shape");
    if (parallel_streams && groups == 4) hc_decode_up_mix_streams_impl<true>(ctx, act, w_up, N, x, lowrank, group_dim, rows);
    else hc_decode_up_mix_impl<true>(ctx, act, w_up, N, x, lowrank, group_dim, groups, rows);
}

void ple_gate_value(
    Context& ctx, const bf16* K, const bf16* Q, const bf16* V, bf16* GV,
    int rows, int hc_count, int hidden_size) {
    constexpr int wg = 256;
    const int total_pairs = rows * hc_count;
    const float inv_sqrt_d = 1.0f / sycl::sqrt(static_cast<float>(hidden_size));

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(wg), cgh);
        sycl::local_accessor<float, 1> gate_slm(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(total_pairs) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int pair_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int r = pair_idx / hc_count;
                const int s = pair_idx % hc_count;
                const size_t stream_stride = static_cast<size_t>(hc_count) * hidden_size;
                const size_t base = static_cast<size_t>(r) * stream_stride + static_cast<size_t>(s) * hidden_size;

                float dot = 0.0f;
                for (int d = lid; d < hidden_size; d += wg) {
                    dot += static_cast<float>(K[base + d]) * static_cast<float>(Q[base + d]);
                }
                red_slm[lid] = dot;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        red_slm[lid] += red_slm[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0) {
                    float g_raw = red_slm[0] * inv_sqrt_d;
                    float sign = (g_raw >= 0.0f ? 1.0f : -1.0f);
                    float g_clamped = sycl::max(sycl::fabs(g_raw), 1.0e-6f);
                    float g = sign * sycl::sqrt(g_clamped);
                    gate_slm[0] = 1.0f / (1.0f + sycl::native::exp(-g));
                }
                item.barrier(sycl::access::fence_space::local_space);

                const float gate = gate_slm[0];
                const size_t v_base = static_cast<size_t>(r) * hidden_size;
                for (int d = lid; d < hidden_size; d += wg) {
                    GV[base + d] = bf16(gate * static_cast<float>(V[v_base + d]));
                }
            });
    });
}

void ple_dilated_conv_silu_add(
    Context& ctx, const bf16* gv_normed, const bf16* gv,
    float* conv_state, int& conv_head, const bf16* conv_w, bf16* out,
    int rows, int channels, int kernel_size, int dilation,
    bf16* residual) {
    const int state_len = (kernel_size - 1) * dilation;
    constexpr int wg = 256;
    const size_t num_wgs = (static_cast<size_t>(channels) + wg - 1) / wg;

    auto silu_dev = [](float x) {
        return x / (1.0f + sycl::native::exp(-x));
    };

    if (rows == 1) {
        const int row0 = conv_head;
        const int row1 = (conv_head + 3) % state_len;
        const int row2 = (conv_head + 6) % state_len;

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    const int c = static_cast<int>(item.get_global_id(0));
                    if (c < channels) {
                        const bf16* w = conv_w + static_cast<size_t>(c) * kernel_size;
                        float tap0 = conv_state[static_cast<size_t>(row0) * channels + c];
                        float tap1 = conv_state[static_cast<size_t>(row1) * channels + c];
                        float tap2 = conv_state[static_cast<size_t>(row2) * channels + c];
                        float tap3 = static_cast<float>(gv_normed[c]);
                        float conv_val = tap0 * static_cast<float>(w[0]) +
                                         tap1 * static_cast<float>(w[1]) +
                                         tap2 * static_cast<float>(w[2]) +
                                         tap3 * static_cast<float>(w[3]);
                        float silu_val = silu_dev(conv_val);
                        float res_val = static_cast<float>(gv[c]) + silu_val;
                        bf16 ple_elem = bf16(res_val);
                        if (out) out[c] = ple_elem;
                        if (residual) {
                            residual[c] = bf16(static_cast<float>(residual[c]) + static_cast<float>(ple_elem));
                        }
                        conv_state[static_cast<size_t>(conv_head) * channels + c] = tap3;
                    }
                });
        });
        conv_head = (conv_head + 1) % state_len;
        return;
    }

    const int initial_conv_head = conv_head;
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int c = static_cast<int>(item.get_global_id(0));
                if (c < channels) {
                    const bf16* w = conv_w + static_cast<size_t>(c) * kernel_size;
                    auto history_val = [&](int t, int tap_idx) {
                        int prior = t - (3 - tap_idx) * 3;
                        if (prior >= 0) {
                            return static_cast<float>(gv_normed[static_cast<size_t>(prior) * channels + c]);
                        }
                        int hist_row = (initial_conv_head + prior + state_len * 100) % state_len;
                        return conv_state[static_cast<size_t>(hist_row) * channels + c];
                    };

                    for (int t = 0; t < rows; ++t) {
                        float tap0 = history_val(t, 0);
                        float tap1 = history_val(t, 1);
                        float tap2 = history_val(t, 2);
                        float tap3 = static_cast<float>(gv_normed[static_cast<size_t>(t) * channels + c]);
                        float conv_val = tap0 * static_cast<float>(w[0]) +
                                         tap1 * static_cast<float>(w[1]) +
                                         tap2 * static_cast<float>(w[2]) +
                                         tap3 * static_cast<float>(w[3]);
                        float silu_val = silu_dev(conv_val);
                        size_t out_idx = static_cast<size_t>(t) * channels + c;
                        float res_val = static_cast<float>(gv[out_idx]) + silu_val;
                        bf16 ple_elem = bf16(res_val);
                        if (out) out[out_idx] = ple_elem;
                        if (residual) {
                            residual[out_idx] = bf16(static_cast<float>(residual[out_idx]) + static_cast<float>(ple_elem));
                        }
                    }
                }
            });
    });

    const int rows_to_copy = std::min(rows, state_len);
    const int first_token = rows - rows_to_copy;
    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(rows_to_copy) * channels),
        [=](sycl::id<1> index) {
            const size_t c = index[0] % static_cast<size_t>(channels);
            const int token = first_token + static_cast<int>(index[0] / channels);
            const int row = (initial_conv_head + token % state_len) % state_len;
            conv_state[static_cast<size_t>(row) * channels + c] =
                static_cast<float>(gv_normed[static_cast<size_t>(token) * channels + c]);
        });

    conv_head = (initial_conv_head + rows) % state_len;
}

} // namespace ops
