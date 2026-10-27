#include "Qwen4ExpDecodeFusion.hpp"

#include <cstdint>
#include <stdexcept>

namespace q4e_fusion {
namespace {

using vec8 = sycl::vec<bf16, 8>;

constexpr int kWorkGroup = 256;
constexpr int kSubGroup = 16;
constexpr int kMaxGroups = 8;
constexpr int kMaxSlmBytes = 48 * 1024;

// Mirrors the alignment predicate used by hc_decode_down_inject_impl: the
// vector path needs 16-byte aligned weight rows (and, there, an aligned N).
inline bool row_is_aligned(const bf16* row) {
    return reinterpret_cast<uintptr_t>(row) % 16 == 0;
}

} // namespace

bool run_hc_norm_down_fused(
    Context& ctx,
    const bf16* H,
    const bf16* y,
    const float* inj,
    const float* norm_w,
    float eps,
    const bf16* w_down,
    const bf16* w_inject,
    bf16* n_out,
    bf16* act_out,
    float* inj_out,
    int lowrank,
    int group_dim,
    int groups) {
    if (!H || !norm_w || !w_down || !n_out || !act_out) return false;
    if (groups < 1 || groups > kMaxGroups || group_dim <= 0 || lowrank < 1) return false;
    if (group_dim % 8 != 0) return false;
    if ((y == nullptr) != (inj == nullptr)) return false;
    if (w_inject != nullptr && inj_out == nullptr) return false;

    const int k_dim = groups * group_dim;
    if (k_dim % 8 != 0) return false;
    const int slm_bytes = k_dim * static_cast<int>(sizeof(bf16)) + groups * kWorkGroup * static_cast<int>(sizeof(float));
    if (slm_bytes > kMaxSlmBytes) return false;

    const bool do_update = (y != nullptr && inj != nullptr);
    if (do_update && inj == inj_out) return false;
    if (reinterpret_cast<uintptr_t>(H) % alignof(vec8) != 0) return false;
    if (do_update) {
        if (reinterpret_cast<uintptr_t>(y) % alignof(vec8) != 0) return false;
        if (reinterpret_cast<uintptr_t>(n_out) % alignof(vec8) != 0) return false;
    }

    const int k8 = k_dim / 8;
    const int total_rows = lowrank + (w_inject ? groups : 0);
    const int subgroups_per_wg = kWorkGroup / kSubGroup;
    const size_t num_wgs = static_cast<size_t>((total_rows + subgroups_per_wg - 1) / subgroups_per_wg);
    const float inv_groups = 1.0f / static_cast<float>(groups);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> nh_slm(sycl::range<1>(static_cast<size_t>(k_dim)), cgh);
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(static_cast<size_t>(groups) * kWorkGroup), cgh);
        sycl::local_accessor<float, 1> inv_slm(sycl::range<1>(static_cast<size_t>(groups)), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * kWorkGroup), sycl::range<1>(kWorkGroup)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubGroup)]] {
                auto sg = item.get_sub_group();
                const int wg_id = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));

                // Phase A1: materialize H' (updated or identity) into SLM with
                // the exact vec8 arithmetic of try_hc_decode_inject_add_and_
                // grouped_rms_norm.  Local copies only: no work-group ever
                // writes the caller's H, so no cross-work-group race exists.
                for (int b = lid; b < k8; b += kWorkGroup) {
                    const int base = b * 8;
                    if (do_update) {
                        const int g = base / group_dim;
                        const int d0 = base - g * group_dim;
                        const vec8 vh = *reinterpret_cast<const vec8*>(H + base);
                        const vec8 vy = *reinterpret_cast<const vec8*>(y + d0);
                        const float inj_val = static_cast<float>(inj[g]);
                        vec8 vout;
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            vout[v] = bf16(static_cast<float>(vh[v]) + inj_val * static_cast<float>(vy[v]));
                        }
                        *reinterpret_cast<vec8*>(&nh_slm[base]) = vout;
                    } else {
                        *reinterpret_cast<vec8*>(&nh_slm[base]) =
                            *reinterpret_cast<const vec8*>(H + base);
                    }
                }

                // Phase A2: grouped leaf sums and SLM tree, replicating the
                // lid + 256 accumulation and stride=(wg/2..1) reduction order
                // of grouped_rms_norm for every group independently.
                item.barrier(sycl::access::fence_space::local_space);
                for (int g = 0; g < groups; ++g) {
                    float sum_sq = 0.0f;
                    for (int i = lid; i < group_dim; i += kWorkGroup) {
                        const float v = static_cast<float>(nh_slm[g * group_dim + i]);
                        sum_sq += v * v;
                    }
                    red_slm[g * kWorkGroup + lid] = sum_sq;
                }
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = kWorkGroup / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        for (int g = 0; g < groups; ++g) {
                            red_slm[g * kWorkGroup + lid] += red_slm[g * kWorkGroup + lid + stride];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (lid < groups) {
                    inv_slm[lid] = 1.0f /
                        sycl::sqrt(red_slm[lid * kWorkGroup] / static_cast<float>(group_dim) + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Phase A3: exactly one work-group writes N; other work-groups
                // recompute the same values on the fly in phase B.  Redundant
                // writers would be benign but wasteful.
                if (wg_id == 0) {
                    for (int i = lid; i < k_dim; i += kWorkGroup) {
                        const int g = i / group_dim;
                        const int d = i - g * group_dim;
                        float scale = inv_slm[g];
                        const float w_val = norm_w[g * group_dim + d];
                        scale *= (1.0f + w_val);
                        n_out[i] = bf16(static_cast<float>(nh_slm[i]) * scale);
                    }
                }

                // Phase B: hc_decode_down_inject, one sub-group per output
                // row, with N recomputed from SLM in the identical expression
                // order so the FMA inputs are bit-identical to a global load.
                const int sg_id = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const bool n_aligned =
                    reinterpret_cast<uintptr_t>(n_out) % alignof(vec8) == 0;
                for (int row = wg_id * subgroups_per_wg + sg_id; row < total_rows;
                     row += static_cast<int>(num_wgs) * subgroups_per_wg) {
                    const bool is_inject = row >= lowrank;
                    const bf16* w_row = is_inject
                        ? w_inject + static_cast<size_t>(row - lowrank) * k_dim
                        : w_down + static_cast<size_t>(row) * k_dim;

                    float acc = 0.0f;
                    if (n_aligned && row_is_aligned(w_row)) {
                        for (int i = lane; i < k8; i += kSubGroup) {
                            const int base = i * 8;
                            const int g = base / group_dim;
                            const int d0 = base - g * group_dim;
                            const vec8 wv = *reinterpret_cast<const vec8*>(w_row + base);
                            const float inv = inv_slm[g];
                            vec8 nv;
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                float scale = inv;
                                const float w_val = norm_w[g * group_dim + d0 + v];
                                scale *= (1.0f + w_val);
                                nv[v] = bf16(static_cast<float>(nh_slm[base + v]) * scale);
                                acc = sycl::fma(static_cast<float>(wv[v]), static_cast<float>(nv[v]), acc);
                            }
                        }
                    } else {
                        for (int i = lane; i < k_dim; i += kSubGroup) {
                            const int g = i / group_dim;
                            const int d = i - g * group_dim;
                            float scale = inv_slm[g];
                            const float w_val = norm_w[g * group_dim + d];
                            scale *= (1.0f + w_val);
                            const bf16 n = bf16(static_cast<float>(nh_slm[i]) * scale);
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(n), acc);
                        }
                    }

                    const float total = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                    if (lane == 0) {
                        const float v = static_cast<float>(bf16(total)) * inv_groups;
                        const float s = 1.0f / (1.0f + sycl::native::exp(-v));
                        if (is_inject) {
                            inj_out[row - lowrank] = 2.0f * s;
                        } else {
                            act_out[row] = bf16(v * s);
                        }
                    }
                }
            });
    });
    return true;
}

sycl::event run_hc_up_mix_with_hidden_update(
    Context& ctx,
    const bf16* act,
    const bf16* w_up,
    const bf16* N,
    bf16* x,
    bf16* H,
    const bf16* y,
    const float* inj,
    int lowrank,
    int group_dim,
    int groups) {
    if (!act || !w_up || !N || !x) {
        throw std::invalid_argument("run_hc_up_mix_with_hidden_update: null operand");
    }
    if (lowrank < 1 || group_dim < 1 || groups < 1 || groups > kMaxGroups) {
        throw std::invalid_argument("run_hc_up_mix_with_hidden_update: unsupported shape");
    }
    const bool do_update = (H != nullptr && y != nullptr && inj != nullptr);
    const int k8 = lowrank / 8;
    const int dims_per_wg = kWorkGroup / kSubGroup;
    const size_t num_wgs =
        (static_cast<size_t>(group_dim) + dims_per_wg - 1) / dims_per_wg;
    const float inv_groups = 1.0f / static_cast<float>(groups);

    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * kWorkGroup), sycl::range<1>(kWorkGroup)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubGroup)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const size_t group = item.get_group(0);
                const int d = static_cast<int>(group) * dims_per_wg +
                              static_cast<int>(sg.get_group_linear_id());
                if (d >= group_dim) return;

                float mixed = 0.0f;
                for (int g = 0; g < groups; ++g) {
                    const size_t row = static_cast<size_t>(g) * group_dim + d;
                    const bf16* w_row = w_up + row * lowrank;
                    float acc = 0.0f;
                    const bool is_aligned =
                        (reinterpret_cast<uintptr_t>(w_row) % 16 == 0) &&
                        (reinterpret_cast<uintptr_t>(act) % 16 == 0);
                    if (is_aligned) {
                        for (int i = lane; i < k8; i += kSubGroup) {
                            const vec8 w = *reinterpret_cast<const vec8*>(w_row + i * 8);
                            const vec8 a = *reinterpret_cast<const vec8*>(act + i * 8);
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                acc = sycl::fma(static_cast<float>(w[v]), static_cast<float>(a[v]), acc);
                            }
                        }
                        const int tail_start = k8 * 8;
                        for (int i = tail_start + lane; i < lowrank; i += kSubGroup) {
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(act[i]), acc);
                        }
                    } else {
                        for (int i = lane; i < lowrank; i += kSubGroup) {
                            acc = sycl::fma(static_cast<float>(w_row[i]), static_cast<float>(act[i]), acc);
                        }
                    }
                    const float dot = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                    const float u = static_cast<float>(bf16(dot));
                    const float m = static_cast<float>(bf16(1.0f / (1.0f + sycl::native::exp(-u))));
                    mixed += m * static_cast<float>(N[row]);

                    // Relocated in-place residual update, identical expression
                    // to try_hc_decode_inject_add_and_grouped_rms_norm: read
                    // before write, exactly one sub-group owns (g, d).
                    if (do_update) {
                        const float inj_val = static_cast<float>(inj[g]);
                        H[row] = bf16(static_cast<float>(H[row]) + inj_val * static_cast<float>(y[d]));
                    }
                }
                if (lane == 0) {
                    x[d] = bf16(mixed * inv_groups);
                }
            });
    });
}

} // namespace q4e_fusion
