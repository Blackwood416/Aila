// SLM-warm-row variant of the GDN prefill recurrent kernel.
//
// The production kernel (src/ops/GatedDeltaNetOps.cpp,
// run_linear_delta_prefill_gpu_batched_impl) walks the fp32 state column of
// every (head, dv) lane twice per token and writes it once: 2R + 1W of
// global traffic over a 64 KiB per-head state. K4 measured 74.8172 ms/layer
// at vh=48 n=4096, i.e. 516.7 GB/s of effective 2R+1W traffic at 9.4%
// thread occupancy (docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md, M1/D4).
//
// This variant stages the first slm_rows rows of each head's state in local
// memory for the whole window. Global state traffic per token drops from
// 3*64 KiB to 3*(128-rows)/128*64 KiB at the cost of one stage + one unstage
// pass per kernel call. Rows [slm_rows, 128) keep the exact production
// expressions, and the staged rows use the same expressions with the SLM
// array as the load/store target, so the fp32 op sequence per lane is
// unchanged and the result is expected to be bit-identical.
//
// The column split proposed by K4 section 6 direction (i) is NOT used here:
// the per-head output RMS norm reduces over all 128 columns of the head
// (GatedDeltaNetOps.cpp:730-740), so splitting columns across work-groups
// would force a per-token cross-work-group reduction. Rows do not have that
// coupling.
#include "GdnPrefillVariant.hpp"

#include "utils/EnvUtils.hpp"

namespace ops {
namespace {

// Device-side copies of the two head-mapping helpers. The production
// definitions (GatedDeltaNetOps.cpp:45-63) live in another translation unit,
// and DPC++ requires device-callee definitions to be visible in the TU that
// compiles the kernel, so the variant keeps its own identical copies. The
// fastpath predicate guarantees kv_heads % q_heads == 0, i.e. group_size > 0.
int slm_key_head_for_value_head(int value_head, int q_heads, int kv_heads) {
    const int group_size = (q_heads > 0 && kv_heads > 0 && (kv_heads % q_heads) == 0)
                               ? kv_heads / q_heads
                               : 0;
    if (group_size <= 0) {
        return std::clamp(value_head, 0, std::max(0, q_heads - 1));
    }
    return std::min(value_head / group_size, q_heads - 1);
}

bool slm_is_group_leader(int value_head, int q_heads, int kv_heads) {
    const int group_size = (q_heads > 0 && kv_heads > 0 && (kv_heads % q_heads) == 0)
                               ? kv_heads / q_heads
                               : 0;
    if (group_size <= 0) {
        return value_head < q_heads;
    }
    return (value_head % group_size) == 0;
}

// Same eligibility predicate the production kernel evaluates
// (GatedDeltaNetOps.cpp:6-30 and :477-490).
bool gdn_prefill_slm_allowed(const DeltaNetGeometry& geom) {
    static const bool s_gqa_fastpath_cached =
        aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    const bool default_grouped = supports_default_grouped_linear_delta_fastpath(
        geom.q_heads, geom.v_heads, geom.head_k_dim, geom.head_v_dim,
        geom.kernel, geom.conv_rows);
    return supports_linear_delta_gqa_fastpath(
        geom.q_heads, geom.v_heads, geom.head_k_dim, geom.head_v_dim,
        geom.kernel, geom.conv_rows, s_gqa_fastpath_cached || default_grouped);
}

// The state walk visits four rows per iteration, so a non-multiple of four
// would leave a partial group straddling the SLM/global boundary.
int clamp_slm_rows(int rows) {
    if (rows < 0) rows = 0;
    if (rows > kGdnSlmRowsMax) rows = kGdnSlmRowsMax;
    return rows - (rows % 4);
}

} // namespace

bool supports_gdn_prefill_slm(const DeltaNetGeometry& geom) {
    return gdn_prefill_slm_allowed(geom);
}

bool run_linear_delta_prefill_gpu_batched_slm(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* fused_ptr,
    bf16* out_ptr,
    int seq_len,
    int slm_rows) {
    if (!gdn_prefill_slm_allowed(geom) || seq_len <= 0) {
        return false;
    }
    slm_rows = clamp_slm_rows(slm_rows);

    const int qkv_dim = geom.qkv_dim;
    const int z_dim = geom.z_dim;
    const int q_dim = geom.q_dim;
    const int kv_dim = geom.kv_dim;
    const int num_heads = geom.v_heads;
    const int q_heads = geom.q_heads;
    const int kernel = geom.kernel;
    const int conv_rows = geom.conv_rows;
    const int conv_head = (conv_rows > 0) ? (state.conv_head % conv_rows) : 0;
    const float eps = geom.eps;
    const float scale = geom.scale;
    const int all_dim = geom.all_dim;
    const GateActivation gate_act = geom.gate_act;

    float* state_ptr = state.state;
    float* conv_state_ptr = state.conv_state;
    const bf16* conv_w_ptr = weights.conv_weight;
    const float* norm_w_ptr = weights.norm_weight;
    const float* a_log_ptr = weights.a_log;
    const bf16* dt_bias_ptr = weights.dt_bias;

    constexpr int head_dim = 128;
    constexpr int head_wg = 128;
    const bool grouped_heads = num_heads > q_heads;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
        sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
        sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);
        // At least one element so the accessor stays valid when slm_rows is 0
        // (the bench isolation arm: SLM allocated, zero staged rows).
        sycl::local_accessor<float, 1> slm_state(
            sycl::range<1>(static_cast<size_t>(std::max(slm_rows, 1)) * head_dim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                auto softplus_dev = [](float x) {
                    if (x > 20.0f) return x;
                    if (x < -20.0f) return sycl::native::exp(x);
                    return sycl::log1p(sycl::native::exp(x));
                };
                auto silu_dev = [](float x) {
                    return x / (1.0f + sycl::native::exp(-x));
                };
                auto sigmoid_dev = [](float x) {
                    return 1.0f / (1.0f + sycl::native::exp(-x));
                };

                int hv = static_cast<int>(item.get_group(0));
                int hk = slm_key_head_for_value_head(hv, q_heads, num_heads);
                bool write_shared_qk = slm_is_group_leader(hv, q_heads, num_heads);
                int lid = static_cast<int>(item.get_local_id(0));
                int q_base = hk * head_dim;
                int k_base = q_dim + hk * head_dim;
                int v_base = q_dim + q_dim + hv * head_dim;
                int z_base_offset = qkv_dim + hv * head_dim;
                int a_offset = qkv_dim + z_dim;
                int b_offset = qkv_dim + z_dim + num_heads;
                float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;

                // Stage rows [0, slm_rows) of this head's state. Every lane
                // later touches only the entries it staged itself, so no
                // cross-lane exchange is needed inside the token loop.
                for (int j = 0; j < slm_rows; ++j) {
                    slm_state[static_cast<size_t>(j) * head_dim + lid] =
                        S[static_cast<size_t>(j) * head_dim + lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                int local_conv_head = conv_head;

                for (int t = 0; t < seq_len; ++t) {
                    const bf16* row = fused_ptr + static_cast<size_t>(t) * all_dim;
                    bf16* out_row = out_ptr + static_cast<size_t>(t) * kv_dim;

                    float v_reg = 0.0f;
                    float z_reg = 0.0f;

                    const int row0 = local_conv_head;
                    const int row1 = (local_conv_head + 1) % conv_rows;
                    const int row2 = (local_conv_head + 2) % conv_rows;

                    if (lid < head_dim) {
                        auto conv_channel = [&](int channel) {
                            if (grouped_heads) {
                                auto history_value = [&](int offset) {
                                    const int prior_token = t - conv_rows + offset;
                                    if (prior_token >= 0) {
                                        return static_cast<float>(fused_ptr[
                                            static_cast<size_t>(prior_token) * all_dim + channel]);
                                    }
                                    const int history_row =
                                        (conv_head + prior_token + conv_rows) % conv_rows;
                                    return conv_state_ptr[
                                        static_cast<size_t>(history_row) * qkv_dim + channel];
                                };
                                const bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                                const float v = history_value(0) * static_cast<float>(w[0]) +
                                                history_value(1) * static_cast<float>(w[1]) +
                                                history_value(2) * static_cast<float>(w[2]) +
                                                static_cast<float>(row[channel]) * static_cast<float>(w[3]);
                                return silu_dev(v);
                            }
                            const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                            const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                            const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                            const bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                            float v = conv_state_ptr[state0] * static_cast<float>(w[0]) +
                                      conv_state_ptr[state1] * static_cast<float>(w[1]) +
                                      conv_state_ptr[state2] * static_cast<float>(w[2]) +
                                      static_cast<float>(row[channel]) * static_cast<float>(w[3]);
                            return silu_dev(v);
                        };

                        q_local[lid] = conv_channel(q_base + lid);
                        k_local[lid] = conv_channel(k_base + lid);
                        v_reg = conv_channel(v_base + lid);
                        z_reg = static_cast<float>(row[z_base_offset + lid]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (!grouped_heads && lid < head_dim) {
                        size_t dst = static_cast<size_t>(local_conv_head) * qkv_dim;
                        if (write_shared_qk) {
                            conv_state_ptr[dst + q_base + lid] = static_cast<float>(row[q_base + lid]);
                            conv_state_ptr[dst + k_base + lid] = static_cast<float>(row[k_base + lid]);
                        }
                        conv_state_ptr[dst + v_base + lid] = static_cast<float>(row[v_base + lid]);
                    }

                    float q_sq = (lid < head_dim) ? (q_local[lid] * q_local[lid]) : 0.0f;
                    float k_sq = (lid < head_dim) ? (k_local[lid] * k_local[lid]) : 0.0f;
                    q_sq = sycl::reduce_over_group(item.get_group(), q_sq, sycl::plus<float>());
                    k_sq = sycl::reduce_over_group(item.get_group(), k_sq, sycl::plus<float>());
                    float q_scale = 1.0f / sycl::sqrt(q_sq + eps);
                    float k_scale = 1.0f / sycl::sqrt(k_sq + eps);

                    if (lid < head_dim) {
                        q_local[lid] *= q_scale;
                        k_local[lid] *= k_scale;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        float beta = 1.0f / (1.0f + sycl::native::exp(-static_cast<float>(row[b_offset + hv])));
                        float alpha = static_cast<float>(row[a_offset + hv]);
                        float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                    * (-sycl::native::exp(a_log_ptr[hv]));
                        head_params[0] = beta;
                        head_params[1] = sycl::native::exp(g_pre);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float beta = head_params[0];
                    float decay = head_params[1];

                    if (lid < head_dim) {
                        const int dv = lid;
                        float acc = 0.0f;
                        // Warm rows: same expressions as the production pass,
                        // with the staged SLM array as the load target.
                        for (int j = 0; j < slm_rows; j += 4) {
                            const float sv0 = slm_state[static_cast<size_t>(j + 0) * head_dim + dv] * decay;
                            const float sv1 = slm_state[static_cast<size_t>(j + 1) * head_dim + dv] * decay;
                            const float sv2 = slm_state[static_cast<size_t>(j + 2) * head_dim + dv] * decay;
                            const float sv3 = slm_state[static_cast<size_t>(j + 3) * head_dim + dv] * decay;
                            acc += sv0 * k_local[j + 0] +
                                   sv1 * k_local[j + 1] +
                                   sv2 * k_local[j + 2] +
                                   sv3 * k_local[j + 3];
                        }
                        // Cold rows: production code unchanged (1R + 1W).
                        for (int j = slm_rows; j < head_dim; j += 4) {
                            const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                            const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                            const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                            const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                            float sv0 = S[off0] * decay;
                            float sv1 = S[off1] * decay;
                            float sv2 = S[off2] * decay;
                            float sv3 = S[off3] * decay;
                            acc += sv0 * k_local[j + 0] +
                                   sv1 * k_local[j + 1] +
                                   sv2 * k_local[j + 2] +
                                   sv3 * k_local[j + 3];
                        }

                        float dval = (v_reg - acc) * beta;
                        float out = 0.0f;
                        for (int j = 0; j < slm_rows; j += 4) {
                            const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                            const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                            const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                            const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                            float sv0 = slm_state[off0] * decay + k_local[j + 0] * dval;
                            float sv1 = slm_state[off1] * decay + k_local[j + 1] * dval;
                            float sv2 = slm_state[off2] * decay + k_local[j + 2] * dval;
                            float sv3 = slm_state[off3] * decay + k_local[j + 3] * dval;
                            slm_state[off0] = sv0;
                            slm_state[off1] = sv1;
                            slm_state[off2] = sv2;
                            slm_state[off3] = sv3;
                            out += sv0 * q_local[j + 0] +
                                   sv1 * q_local[j + 1] +
                                   sv2 * q_local[j + 2] +
                                   sv3 * q_local[j + 3];
                        }
                        for (int j = slm_rows; j < head_dim; j += 4) {
                            const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                            const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                            const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                            const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                            float sv0 = S[off0] * decay + k_local[j + 0] * dval;
                            float sv1 = S[off1] * decay + k_local[j + 1] * dval;
                            float sv2 = S[off2] * decay + k_local[j + 2] * dval;
                            float sv3 = S[off3] * decay + k_local[j + 3] * dval;
                            S[off0] = sv0;
                            S[off1] = sv1;
                            S[off2] = sv2;
                            S[off3] = sv3;
                            out += sv0 * q_local[j + 0] +
                                   sv1 * q_local[j + 1] +
                                   sv2 * q_local[j + 2] +
                                   sv3 * q_local[j + 3];
                        }
                        out_local[dv] = out * scale;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float out_sq = (lid < head_dim) ? (out_local[lid] * out_local[lid]) : 0.0f;
                    out_sq = sycl::reduce_over_group(item.get_group(), out_sq, sycl::plus<float>());
                    float out_scale = 1.0f / sycl::sqrt(out_sq / static_cast<float>(head_dim) + eps);

                    if (lid < head_dim) {
                        float n = out_local[lid] * out_scale * norm_w_ptr[lid];
                        float gate_val = (gate_act == GateActivation::Silu)
                                             ? silu_dev(z_reg)
                                             : sigmoid_dev(z_reg);
                        out_row[hv * head_dim + lid] = bf16(n * gate_val);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    local_conv_head = (local_conv_head + 1) % conv_rows;
                }

                // Write the staged rows back; the global rows
                // [slm_rows, 128) are already current.
                item.barrier(sycl::access::fence_space::local_space);
                for (int j = 0; j < slm_rows; ++j) {
                    S[static_cast<size_t>(j) * head_dim + lid] =
                        slm_state[static_cast<size_t>(j) * head_dim + lid];
                }
            });
    });

    if (grouped_heads) {
        commit_linear_conv_history(ctx, fused_ptr, all_dim, conv_state_ptr,
                                   qkv_dim, conv_rows, conv_head, seq_len);
    }
    state.device_state_dirty = true;
    state.conv_head = (conv_head + seq_len) % conv_rows;
    return true;
}

} // namespace ops
