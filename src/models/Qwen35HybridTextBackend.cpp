#include "Qwen35HybridTextBackend.hpp"
#include "HybridLayerExecutor.hpp"
#include "HybridDecodeBatchExecutor.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "memory/SessionMemory.hpp"
#include "memory/AllocationBudget.hpp"
#include <chrono>
#include <limits>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>

using bf16 = sycl::ext::oneapi::bfloat16;
using aila::models::ProfileStage;
using aila::models::StageProfileTotals;
using aila::models::StageProfiler;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace {
int round_up_seq(int v, int granularity) {
    return ((v + granularity - 1) / granularity) * granularity;
}

bool supports_jm_bf16_f32(const sycl::device& dev, int m, int n, int k) {
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }

    using matrix_type = sycl::ext::oneapi::experimental::matrix::matrix_type;
    using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
    auto combos = dev.get_info<matrix_combinations>();
    for (const auto& c : combos) {
        bool m_ok = (static_cast<int>(c.msize) == 0) || (static_cast<int>(c.msize) == m);
        if (m_ok && static_cast<int>(c.nsize) == n &&
            static_cast<int>(c.ksize) == k &&
            c.atype == matrix_type::bf16 &&
            c.btype == matrix_type::bf16 &&
            c.ctype == matrix_type::fp32 &&
            c.dtype == matrix_type::fp32) {
            return true;
        }
    }
    return false;
}

constexpr int kDecodeFfnHidden = 1024;
constexpr int kDecodeFfnIntermediate = 3584;
constexpr int kDecodeFfnTile = 8;
constexpr int kDecodeFfnK = 16;
constexpr int kDecodeFfnSubgroup = 8;

bool supports_linear_delta_gqa_fastpath(int q_heads, int kv_heads,
                                        int head_k_dim, int head_v_dim,
                                        int kernel, int conv_rows,
                                        bool allow_grouped_fastpath = false) {
    return q_heads > 0 &&
           kv_heads > 0 &&
           (kv_heads % q_heads) == 0 &&
           (allow_grouped_fastpath || q_heads == kv_heads) &&
           head_k_dim == head_v_dim &&
           head_k_dim == 128 &&
           kernel == 4 &&
           conv_rows == 3;
}

bool supports_default_grouped_linear_delta_fastpath(int q_heads, int kv_heads,
                                                    int head_k_dim, int head_v_dim,
                                                    int kernel, int conv_rows) {
    return kv_heads > q_heads &&
           supports_linear_delta_gqa_fastpath(q_heads, kv_heads,
                                              head_k_dim, head_v_dim,
                                              kernel, conv_rows,
                                              true);
}

// Grouped value heads share Q/K history across work-groups. Keep the history
// read-only during computation, then commit it on the in-order queue once
// every reader has finished.
void commit_linear_conv_history(Context& ctx, const bf16* input, int input_stride,
                                float* history, int channels, int history_rows,
                                int initial_head, int token_count) {
    const int rows_to_copy = std::min(token_count, history_rows);
    const int first_token = token_count - rows_to_copy;
    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(rows_to_copy) * channels),
        [=](sycl::id<1> index) {
            const size_t channel = index[0] % static_cast<size_t>(channels);
            const int token = first_token + static_cast<int>(index[0] / channels);
            const int row = (initial_head + token % history_rows) % history_rows;
            history[static_cast<size_t>(row) * channels + channel] =
                static_cast<float>(input[static_cast<size_t>(token) * input_stride + channel]);
        });
}

int linear_delta_gqa_group_size(int q_heads, int kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || (kv_heads % q_heads) != 0) {
        return 0;
    }
    return kv_heads / q_heads;
}

int linear_delta_key_head_for_value_head(int value_head, int q_heads, int kv_heads) {
    int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return std::clamp(value_head, 0, std::max(0, q_heads - 1));
    }
    return std::min(value_head / group_size, q_heads - 1);
}

bool linear_delta_is_group_leader(int value_head, int q_heads, int kv_heads) {
    int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return value_head < q_heads;
    }
    return (value_head % group_size) == 0;
}
}

struct Qwen35HybridTextBackend::ProfileState {
    StageProfileTotals decode;
    StageProfileTotals prefill;
};

float Qwen35HybridTextBackend::softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float Qwen35HybridTextBackend::silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void Qwen35HybridTextBackend::head_l2_norm_inplace(std::vector<float>& x,
                                                    int seq_len, int num_heads,
                                                    int head_dim, float eps) {
    for (int s = 0; s < seq_len; ++s) {
        float* row = x.data() + static_cast<size_t>(s) * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            float* v = row + static_cast<size_t>(h) * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) sum_sq += v[d] * v[d];
            float inv = 1.0f / std::sqrt(sum_sq + eps);
            for (int d = 0; d < head_dim; ++d) v[d] *= inv;
        }
    }
}

void Qwen35HybridTextBackend::head_rms_norm_and_silu_gate(std::vector<float>& x,
                                                           const std::vector<float>& norm_weight,
                                                           const std::vector<float>& z,
                                                           int seq_len, int num_heads,
                                                           int head_dim, float eps) {
    for (int s = 0; s < seq_len; ++s) {
        float* row = x.data() + static_cast<size_t>(s) * num_heads * head_dim;
        const float* z_row = z.data() + static_cast<size_t>(s) * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            float* v = row + static_cast<size_t>(h) * head_dim;
            const float* zg = z_row + static_cast<size_t>(h) * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) sum_sq += v[d] * v[d];
            float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + eps);
            for (int d = 0; d < head_dim; ++d) {
                float n = v[d] * inv_rms * norm_weight[d];
                v[d] = n * silu(zg[d]);
            }
        }
    }
}

void Qwen35HybridTextBackend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;
    if (runtime_seq_capacity_ > 0) ctx.synchronize();

    int new_cap = std::max(seq_len, std::min(max_seq_len_, round_up_seq(seq_len, 64)));
    buf_.hidden = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)hidden_size_});
    buf_.normed = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)hidden_size_});
    buf_.qkv = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_qkv_dim_});
    buf_.linear_all = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)linear_all_dim_});
    buf_.q = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.k = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.v = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.z = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.a = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)linear_kv_heads_});
    buf_.b = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)linear_kv_heads_});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.full_qkv = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)full_fused_qkv_dim_});
    buf_.gate_up = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(2 * ff_dim_)});
    // Attention/down output uses hidden width; FFN activation uses ff_dim_.
    // Keep physical capacity sufficient for either logical row layout.
    const int projection_capacity = std::max(ff_dim_, hidden_size_);
    buf_.gate = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)projection_capacity});
    buf_.up = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)projection_capacity});

    runtime_seq_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3.5] Runtime buffers resized: seq_cap=%d", runtime_seq_capacity_);
}

void Qwen35HybridTextBackend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;
    if (buf_.scores.valid()) ctx.synchronize();

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(
        ctx,
        {(int64_t)max_attn_heads_, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);
    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3.5] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen35HybridTextBackend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= incr_prefill_seq_cap_ && total_len <= incr_prefill_total_cap_) return;
    if (buf_.incr_scores.valid()) ctx.synchronize();

    int new_seq_cap = std::max(seq_len, incr_prefill_seq_cap_);
    int new_total_cap = std::max(total_len, incr_prefill_total_cap_);
    new_seq_cap = round_up_seq(new_seq_cap, 16);
    new_total_cap = round_up_seq(new_total_cap, 64);

    buf_.incr_scores = Tensor::allocate(
        ctx,
        {(int64_t)max_attn_heads_, (int64_t)new_seq_cap, (int64_t)new_total_cap},
        dnnl::memory::data_type::f32);

    incr_prefill_seq_cap_ = new_seq_cap;
    incr_prefill_total_cap_ = new_total_cap;
    AILA_LOG_INFO("[Qwen3.5] Incremental prefill score buffer resized: seq_cap=%d total_cap=%d",
                  incr_prefill_seq_cap_, incr_prefill_total_cap_);
}

bool Qwen35HybridTextBackend::use_decode_ffn_custom_path(int seq_len) const {
    return use_decode_jm_custom_path(seq_len);
}

bool Qwen35HybridTextBackend::use_decode_jm_custom_path(int seq_len) const {
    return decode_ffn_custom_enabled_ && seq_len == 1;
}

void Qwen35HybridTextBackend::run_decode_ffn_gate_up_swiglu_custom(
    Context& ctx, const Layer& layer, Tensor& input, Tensor& output) {
    using vec8 = sycl::vec<bf16, 8>;
    const bf16* input_ptr = static_cast<const bf16*>(input.data());
    const bf16* weight_ptr = static_cast<const bf16*>(layer.gate_up_weight_jm->data());
    bf16* output_ptr = static_cast<bf16*>(output.data());

    constexpr int sg_size = kDecodeFfnSubgroup;
    constexpr int ff_dim = kDecodeFfnIntermediate;
    constexpr int hidden = kDecodeFfnHidden;
    constexpr int vec_width = 8;
    const int vec_chunks = hidden / vec_width;
    const int tail_start = vec_chunks * vec_width;
    const vec8* input_vec = reinterpret_cast<const vec8*>(input_ptr);
    const vec8* weight_vec = reinterpret_cast<const vec8*>(weight_ptr);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(ff_dim * sg_size), sycl::range<1>(sg_size)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int row = static_cast<int>(item.get_group(0));
                const int lane = static_cast<int>(item.get_local_id(0));
                const vec8* gate_row = weight_vec + row * vec_chunks;
                const vec8* up_row = weight_vec + (ff_dim + row) * vec_chunks;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;

                for (int chunk = lane; chunk < vec_chunks; chunk += sg_size) {
                    vec8 in_v = input_vec[chunk];
                    vec8 gate_v = gate_row[chunk];
                    vec8 up_v = up_row[chunk];
                    for (int i = 0; i < vec_width; ++i) {
                        gate_acc = sycl::fma(static_cast<float>(in_v[i]), static_cast<float>(gate_v[i]), gate_acc);
                        up_acc = sycl::fma(static_cast<float>(in_v[i]), static_cast<float>(up_v[i]), up_acc);
                    }
                }

                for (int k = tail_start + lane; k < hidden; k += sg_size) {
                    const float in_v = static_cast<float>(input_ptr[k]);
                    gate_acc = sycl::fma(in_v, static_cast<float>(weight_ptr[row * hidden + k]), gate_acc);
                    up_acc = sycl::fma(in_v, static_cast<float>(weight_ptr[(ff_dim + row) * hidden + k]), up_acc);
                }

                gate_acc = sycl::reduce_over_group(sg, gate_acc, sycl::plus<float>());
                up_acc = sycl::reduce_over_group(sg, up_acc, sycl::plus<float>());

                if (lane == 0) {
                    const float silu_gate = gate_acc / (1.0f + sycl::native::exp(-gate_acc));
                    output_ptr[row] = bf16(silu_gate * up_acc);
                }
            });
    });
}

void Qwen35HybridTextBackend::run_decode_ffn_down_custom(
    Context& ctx, const Layer& layer, Tensor& input, Tensor& output) {
    run_decode_jm_matvec_custom(ctx, input, *layer.down_weight_jm,
                                kDecodeFfnIntermediate, kDecodeFfnHidden, output);
}

void Qwen35HybridTextBackend::run_decode_jm_matvec_custom(
    Context& ctx, Tensor& input, const Tensor& weight_jm,
    int in_dim, int out_dim, Tensor& output) {
    using vec8 = sycl::vec<bf16, 8>;
    const bf16* input_ptr = static_cast<const bf16*>(input.data());
    const bf16* weight_ptr = static_cast<const bf16*>(weight_jm.data());
    bf16* output_ptr = static_cast<bf16*>(output.data());

    constexpr int sg_size = kDecodeFfnSubgroup;
    constexpr int vec_width = 8;
    const int vec_chunks = in_dim / vec_width;
    const int tail_start = vec_chunks * vec_width;
    const vec8* input_vec = reinterpret_cast<const vec8*>(input_ptr);
    const vec8* weight_vec = reinterpret_cast<const vec8*>(weight_ptr);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(out_dim * sg_size), sycl::range<1>(sg_size)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const int row = static_cast<int>(item.get_group(0));
                const int lane = static_cast<int>(item.get_local_id(0));
                const vec8* weight_row = weight_vec + row * vec_chunks;
                float acc = 0.0f;

                for (int chunk = lane; chunk < vec_chunks; chunk += sg_size) {
                    vec8 in_v = input_vec[chunk];
                    vec8 w_v = weight_row[chunk];
                    for (int i = 0; i < vec_width; ++i) {
                        acc = sycl::fma(static_cast<float>(in_v[i]), static_cast<float>(w_v[i]), acc);
                    }
                }

                for (int k = tail_start + lane; k < in_dim; k += sg_size) {
                    acc = sycl::fma(static_cast<float>(input_ptr[k]),
                                    static_cast<float>(weight_ptr[row * in_dim + k]),
                                    acc);
                }

                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());

                if (lane == 0) {
                    output_ptr[row] = bf16(acc);
                }
            });
    });
}

// Multi-row decode variants: one launch for 2..4 independent sequences. Each
// (sequence, output row) pair still uses the original subgroup size, vec8/FMA
// order and reduce_over_group, so a batched row is numerically identical to the
// single-row kernel. Groups are ordered so all sequences of the same weight row
// are adjacent, letting L2 serve one weight row to every sequence.
void Qwen35HybridTextBackend::run_decode_jm_matvec_rows(
    Context& ctx, Tensor& input, const Tensor& weight_jm,
    int in_dim, int out_dim, Tensor& output, int rows) {
    if (rows < 1 || rows > 4) {
        throw std::invalid_argument("run_decode_jm_matvec_rows: rows must be in [1, 4]");
    }
    // A sequence row offset must stay vec8-aligned, so an in_dim that is not a
    // multiple of the vector width keeps the original per-row kernels.
    if (rows == 1 || (in_dim % 8) != 0) {
        for (int r = 0; r < rows; ++r) {
            auto row_input = Tensor::view(ctx, input.data_as<bf16>() + static_cast<size_t>(r) * in_dim,
                                          {1, static_cast<int64_t>(in_dim)}, input.dtype());
            auto row_output = Tensor::view(ctx, output.data_as<bf16>() + static_cast<size_t>(r) * out_dim,
                                           {1, static_cast<int64_t>(out_dim)}, output.dtype());
            run_decode_jm_matvec_custom(ctx, row_input, weight_jm, in_dim, out_dim, row_output);
        }
        return;
    }
    using vec8 = sycl::vec<bf16, 8>;
    const bf16* input_base = static_cast<const bf16*>(input.data());
    const bf16* weight_ptr = static_cast<const bf16*>(weight_jm.data());
    bf16* output_base = static_cast<bf16*>(output.data());

    constexpr int sg_size = kDecodeFfnSubgroup;
    constexpr int vec_width = 8;
    const int vec_chunks = in_dim / vec_width;
    const int tail_start = vec_chunks * vec_width;
    const vec8* weight_vec = reinterpret_cast<const vec8*>(weight_ptr);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(out_dim) * rows * sg_size),
                              sycl::range<1>(sg_size)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const size_t group = item.get_group(0);
                // row varies slowest so every sequence reads the same weights together.
                const int row = static_cast<int>(group / static_cast<size_t>(rows));
                const int sequence = static_cast<int>(group % static_cast<size_t>(rows));
                const int lane = static_cast<int>(item.get_local_id(0));
                const bf16* input_ptr = input_base + static_cast<size_t>(sequence) * in_dim;
                const bf16* input_row_bytes = input_ptr;
                const vec8* input_vec = reinterpret_cast<const vec8*>(input_row_bytes);
                const vec8* weight_row = weight_vec + row * vec_chunks;
                float acc = 0.0f;

                for (int chunk = lane; chunk < vec_chunks; chunk += sg_size) {
                    vec8 in_v = input_vec[chunk];
                    vec8 w_v = weight_row[chunk];
                    for (int i = 0; i < vec_width; ++i) {
                        acc = sycl::fma(static_cast<float>(in_v[i]), static_cast<float>(w_v[i]), acc);
                    }
                }

                for (int k = tail_start + lane; k < in_dim; k += sg_size) {
                    acc = sycl::fma(static_cast<float>(input_ptr[k]),
                                    static_cast<float>(weight_ptr[row * in_dim + k]),
                                    acc);
                }

                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());

                if (lane == 0) {
                    (output_base + static_cast<size_t>(sequence) * out_dim)[row] = bf16(acc);
                }
            });
    });
}

void Qwen35HybridTextBackend::run_decode_ffn_gate_up_swiglu_rows(
    Context& ctx, const Layer& layer, Tensor& input, Tensor& output, int rows) {
    if (rows < 1 || rows > 4) {
        throw std::invalid_argument("run_decode_ffn_gate_up_swiglu_rows: rows must be in [1, 4]");
    }
    if (rows == 1) {
        run_decode_ffn_gate_up_swiglu_custom(ctx, layer, input, output);
        return;
    }
    using vec8 = sycl::vec<bf16, 8>;
    const bf16* input_base = static_cast<const bf16*>(input.data());
    const bf16* weight_ptr = static_cast<const bf16*>(layer.gate_up_weight_jm->data());
    bf16* output_base = static_cast<bf16*>(output.data());

    constexpr int sg_size = kDecodeFfnSubgroup;
    constexpr int ff_dim = kDecodeFfnIntermediate;
    constexpr int hidden = kDecodeFfnHidden;
    constexpr int vec_width = 8;
    const int vec_chunks = hidden / vec_width;
    const int tail_start = vec_chunks * vec_width;
    const vec8* weight_vec = reinterpret_cast<const vec8*>(weight_ptr);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(ff_dim) * rows * sg_size),
                              sycl::range<1>(sg_size)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
                auto sg = item.get_sub_group();
                const size_t group = item.get_group(0);
                const int row = static_cast<int>(group / static_cast<size_t>(rows));
                const int sequence = static_cast<int>(group % static_cast<size_t>(rows));
                const int lane = static_cast<int>(item.get_local_id(0));
                const bf16* input_ptr = input_base + static_cast<size_t>(sequence) * hidden;
                const vec8* input_vec = reinterpret_cast<const vec8*>(input_ptr);
                const vec8* gate_row = weight_vec + row * vec_chunks;
                const vec8* up_row = weight_vec + (ff_dim + row) * vec_chunks;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;

                for (int chunk = lane; chunk < vec_chunks; chunk += sg_size) {
                    vec8 in_v = input_vec[chunk];
                    vec8 gate_v = gate_row[chunk];
                    vec8 up_v = up_row[chunk];
                    for (int i = 0; i < vec_width; ++i) {
                        gate_acc = sycl::fma(static_cast<float>(in_v[i]), static_cast<float>(gate_v[i]), gate_acc);
                        up_acc = sycl::fma(static_cast<float>(in_v[i]), static_cast<float>(up_v[i]), up_acc);
                    }
                }

                for (int k = tail_start + lane; k < hidden; k += sg_size) {
                    const float in_v = static_cast<float>(input_ptr[k]);
                    gate_acc = sycl::fma(in_v, static_cast<float>(weight_ptr[row * hidden + k]), gate_acc);
                    up_acc = sycl::fma(in_v, static_cast<float>(weight_ptr[(ff_dim + row) * hidden + k]), up_acc);
                }

                gate_acc = sycl::reduce_over_group(sg, gate_acc, sycl::plus<float>());
                up_acc = sycl::reduce_over_group(sg, up_acc, sycl::plus<float>());

                if (lane == 0) {
                    const float silu_gate = gate_acc / (1.0f + sycl::native::exp(-gate_acc));
                    (output_base + static_cast<size_t>(sequence) * ff_dim)[row] = bf16(silu_gate * up_acc);
                }
            });
    });
}

Qwen35HybridTextBackend::Qwen35HybridTextBackend()
    : profile_(std::make_unique<ProfileState>()) {}

Qwen35HybridTextBackend::~Qwen35HybridTextBackend() {
    // The runtime also drains its execution slot. Keep direct legacy users safe
    // before shared scratch, bindings and the default cache are released.
    if (context_) {
        try { context_->synchronize(); } catch (...) {}
    }
    defaultSession_.reset();
}

Qwen35HybridTextBackend::Session::Session(
    Context& ctx, int capacity, std::shared_ptr<const SessionOwner> owner,
    bool boundedSnapshots, std::size_t maxSnapshots)
    : context_(&ctx), owner_(std::move(owner)), capacity_(capacity),
      boundedSnapshots_(boundedSnapshots), maxSnapshots_(maxSnapshots) {}

Qwen35HybridTextBackend::Session::~Session() {
    // Covers asynchronous forward/reset as well as partially allocated sessions.
    // Context must outlive the Session, including when construction fails.
    if (context_) {
        try { context_->synchronize(); } catch (...) {}
    }
}

std::unique_ptr<Qwen35HybridTextBackend::Session>
Qwen35HybridTextBackend::CreateSessionImpl(
    Context& ctx, bool quantizedKv, bool boundedSnapshots, std::size_t maxSnapshots) {
    if (!sessionOwner_ || !decoderPlan_) {
        throw std::logic_error("Qwen35HybridTextBackend::CreateSession: model is not loaded");
    }
    if (&ctx != context_) {
        throw std::invalid_argument(
            "Qwen35HybridTextBackend::CreateSession: context does not match the loaded model");
    }

    const auto reservation = aila::memory::qwen35_session_memory(
        cfg_, max_seq_len_, boundedSnapshots ? maxSnapshots : 0);
    // The shared helper checked these dimensions and products before any
    // allocation; use the same reservation as model-level admission.
    const size_t recurrentElements = static_cast<size_t>(linear_kv_heads_) *
        static_cast<size_t>(linear_head_dim_) * static_cast<size_t>(cfg_.linear_value_head_dim);
    const size_t convElements = static_cast<size_t>(std::max(0, linear_conv_kernel_dim_ - 1)) *
        static_cast<size_t>(linear_conv_channels_);
    size_t actualCacheBytes = 0;

    // Own the Session before allocating any tensors: its destructor drains
    // queued initialization writes if a later allocation fails.
    auto session = std::unique_ptr<Session>(
        new Session(ctx, max_seq_len_, sessionOwner_, boundedSnapshots, maxSnapshots));
    session->layerCaches_.resize(layers_.size());
    const auto kvType = quantizedKv
        ? dnnl::memory::data_type::f8_e4m3 : dnnl::memory::data_type::bf16;
    for (const auto& plan : decoderPlan_->layers) {
        auto& cache = session->layerCaches_[static_cast<size_t>(plan.cacheIndex)];
        if (plan.kind == LayerKind::DeltaNet) {
            cache.linear_state = Tensor::allocate(
                ctx, {static_cast<int64_t>(linear_kv_heads_),
                      static_cast<int64_t>(linear_head_dim_),
                      static_cast<int64_t>(cfg_.linear_value_head_dim)},
                dnnl::memory::data_type::f32);
            if (convElements != 0) {
                cache.linear_conv_state = Tensor::allocate(
                    ctx, {static_cast<int64_t>(linear_conv_kernel_dim_ - 1),
                          static_cast<int64_t>(linear_conv_channels_)},
                    dnnl::memory::data_type::f32);
            }
            cache.host_linear_state.assign(recurrentElements, 0.0f);
            cache.host_linear_conv_state.assign(convElements, 0.0f);
            ctx.queue().memset(cache.linear_state.data(), 0, cache.linear_state.size_bytes());
            if (cache.linear_conv_state.valid()) {
                ctx.queue().memset(cache.linear_conv_state.data(), 0,
                                   cache.linear_conv_state.size_bytes());
            }
        } else {
            const bool legacyLinear = plan.kind == LayerKind::LegacyLinearAttention;
            const int keyHeads = legacyLinear ? linear_q_heads_ : full_kv_heads_;
            const int valueHeads = legacyLinear ? linear_kv_heads_ : full_kv_heads_;
            const int keyDim = legacyLinear ? linear_head_dim_ : full_head_dim_;
            const int valueDim = legacyLinear ? cfg_.linear_value_head_dim : full_head_dim_;
            cache.k = Tensor::allocate(
                ctx, {static_cast<int64_t>(keyHeads), static_cast<int64_t>(max_seq_len_),
                      static_cast<int64_t>(keyDim)}, kvType);
            cache.v = Tensor::allocate(
                ctx, {static_cast<int64_t>(valueHeads), static_cast<int64_t>(max_seq_len_),
                      static_cast<int64_t>(valueDim)}, kvType);
        }
        for (const auto* tensor : {&cache.k, &cache.v, &cache.linear_state, &cache.linear_conv_state}) {
            actualCacheBytes = aila::memory::checked_memory_sum(actualCacheBytes, tensor->size_bytes());
        }
    }
    session->cacheBytes_ = actualCacheBytes;
    session->hostReservationBytes_ = decoderPlan_->hasRecurrentLayers ? reservation.hostBytes : 0;
    return session;
}

std::unique_ptr<Qwen35HybridTextBackend::Session>
Qwen35HybridTextBackend::CreateSession(Context& ctx, std::size_t maxSnapshots) {
    if (!use_delta_linear_) {
        throw std::invalid_argument(
            "Qwen3.5 shared sessions require the DeltaNet path; set AILA_Q35_LINEAR_DELTA=1");
    }
    if (aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT")) {
        throw std::invalid_argument(
            "Qwen3.5 shared sessions require BF16 KV; set AILA_VLM_KV_QUANT=0 "
            "(or AILA_KV_QUANT=0 when no scoped override is set)");
    }
    auto session = CreateSessionImpl(ctx, false, true, maxSnapshots);
    explicitSessionsCreated_ = true;
    return session;
}

Qwen35HybridTextBackend::Session&
Qwen35HybridTextBackend::DefaultSession(Context& ctx) {
    if (&ctx != context_ || !sessionOwner_) {
        throw std::invalid_argument(
            "Qwen35HybridTextBackend: default session requires the loaded model's context");
    }
    if (!defaultSession_) {
        defaultSession_ = CreateSessionImpl(ctx, legacyQuantizedKv_, false, 0);
    }
    return *defaultSession_;
}

void Qwen35HybridTextBackend::ValidateSession(const Session& session) const {
    if (!sessionOwner_ || !decoderPlan_) {
        throw std::logic_error("Qwen35HybridTextBackend: model is not loaded");
    }
    if (session.owner_ != sessionOwner_ || session.context_ != context_) {
        throw std::invalid_argument(
            "Qwen35HybridTextBackend: session does not belong to this loaded model");
    }
}

bool Qwen35HybridTextBackend::load(Context& ctx,
                                   ModelWeights& weights,
                                   const ModelSpec& spec,
                                   int max_seq_len,
                                   std::string* error_message) {
    if (explicitSessionsCreated_) {
        if (error_message) {
            *error_message = "Qwen35HybridTextBackend: cannot reload after explicit sessions have been created";
        }
        return false;
    }
    if (spec.family != ModelFamily::Qwen35Hybrid) {
        if (error_message) *error_message = "Qwen35HybridTextBackend: invalid model family";
        return false;
    }
    if (spec.is_bitsandbytes_4bit()) {
        if (error_message) *error_message = "Qwen35HybridTextBackend: BF16 weights are required";
        return false;
    }
    if (max_seq_len <= 0 || spec.qwen35_text.num_hidden_layers <= 0 ||
        spec.qwen35_text.hidden_size <= 0 || spec.qwen35_text.intermediate_size <= 0) {
        if (error_message) *error_message = "Qwen35HybridTextBackend: context capacity and layer count must be positive";
        return false;
    }
    try {
        // Validate dimensions and layer types before replacing a loaded model.
        (void)aila::memory::qwen35_session_memory(spec.qwen35_text, max_seq_len, 0);
    } catch (const aila::memory::AllocationLimitExceeded&) {
        throw;
    } catch (const std::exception& ex) {
        if (error_message) *error_message = ex.what();
        return false;
    }

    if (context_) context_->synchronize();
    defaultSession_.reset();
    decoderPlan_.reset();
    sessionOwner_.reset();
    context_ = &ctx;
    profile_->decode.reset();
    profile_->prefill.reset();

    spec_ = spec;
    cfg_ = spec.qwen35_text;
    legacyQuantizedKv_ = aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT");

    hidden_size_ = cfg_.hidden_size;
    ff_dim_ = cfg_.intermediate_size;
    max_seq_len_ = max_seq_len;

    full_q_heads_ = cfg_.num_attention_heads;
    full_kv_heads_ = cfg_.num_key_value_heads;
    full_head_dim_ = cfg_.head_dim;
    full_q_dim_ = full_q_heads_ * full_head_dim_;
    full_kv_dim_ = full_kv_heads_ * full_head_dim_;
    full_q_proj_dim_ = cfg_.attn_output_gate ? (2 * full_q_dim_) : full_q_dim_;
    full_fused_qkv_dim_ = full_q_proj_dim_ + 2 * full_kv_dim_;

    linear_q_heads_ = cfg_.linear_num_key_heads;
    linear_kv_heads_ = cfg_.linear_num_value_heads;
    linear_head_dim_ = cfg_.linear_key_head_dim;
    linear_q_dim_ = linear_q_heads_ * linear_head_dim_;
    linear_kv_dim_ = linear_kv_heads_ * cfg_.linear_value_head_dim;
    linear_qkv_dim_ = linear_q_dim_ + linear_q_dim_ + linear_kv_dim_;
    linear_z_dim_ = linear_kv_dim_;
    linear_all_dim_ = linear_qkv_dim_ + linear_z_dim_ + 2 * linear_kv_heads_;
    linear_conv_kernel_dim_ = std::max(1, cfg_.linear_conv_kernel_dim);
    linear_conv_channels_ = linear_qkv_dim_;
    use_delta_linear_ = aila::env::read_flag("AILA_Q35_LINEAR_DELTA", true);
    bool force_direct_tied_lm_head = aila::env::read_flag("AILA_Q35_DIRECT_TIED_LM_HEAD", false);
    bool allow_unsupported_legacy_linear = aila::env::read_flag("AILA_Q35_ALLOW_UNSUPPORTED_LEGACY_LINEAR", false);
    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool experimental_grouped_linear_gpu = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU", false);
    bool force_host_grouped_linear = aila::env::read_flag("AILA_Q35_FORCE_HOST_GROUPED_LINEAR", false);
    decode_ffn_custom_enabled_ = (hidden_size_ == kDecodeFfnHidden &&
                                  ff_dim_ == kDecodeFfnIntermediate &&
                                  supports_jm_bf16_f32(ctx.queue().get_device(), 1, kDecodeFfnTile, kDecodeFfnK));
    bool is_exact_q35_0p8b_spec = is_exact_qwen35_hybrid_0p8b_spec(cfg_);
    if (!use_delta_linear_ && linear_kv_heads_ > linear_q_heads_ && !allow_unsupported_legacy_linear) {
        AILA_LOG_WARN("[Qwen3.5] Legacy linear-attn fallback is unsupported when value heads exceed key heads (q_heads=%d kv_heads=%d), forcing delta path. Set AILA_Q35_ALLOW_UNSUPPORTED_LEGACY_LINEAR=1 to override.",
                      linear_q_heads_, linear_kv_heads_);
        use_delta_linear_ = true;
    }

    int linear_conv_rows = std::max(0, linear_conv_kernel_dim_ - 1);
    bool default_grouped_linear_fastpath = supports_default_grouped_linear_delta_fastpath(
        linear_q_heads_,
        linear_kv_heads_,
        linear_head_dim_,
        cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_,
        linear_conv_rows);
    bool linear_fastpath_eligible = supports_linear_delta_gqa_fastpath(
        linear_q_heads_,
        linear_kv_heads_,
        linear_head_dim_,
        cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_,
        linear_conv_rows,
        experimental_gqa_fastpath || default_grouped_linear_fastpath);
    bool use_grouped_linear_gpu_default = !force_host_grouped_linear && default_grouped_linear_fastpath;
    bool use_grouped_linear_gpu_experimental =
        !use_grouped_linear_gpu_default && !force_host_grouped_linear && experimental_grouped_linear_gpu;
    int linear_gqa_group_size = linear_delta_gqa_group_size(linear_q_heads_, linear_kv_heads_);

    max_attn_heads_ = std::max(full_q_heads_, linear_q_heads_);
    max_qkv_dim_ = std::max(full_q_proj_dim_, linear_qkv_dim_);
    max_attn_dim_ = std::max(full_q_dim_, linear_kv_dim_);

    auto transpose_weight = [&](const std::string& name) -> Tensor* {
        Tensor& src = weights.get(name);
        int64_t out_f = src.shape(0);
        int64_t in_f = src.shape(1);
        Tensor dst = Tensor::allocate(ctx, {in_f, out_f}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };
    auto plus_one_norm_weight = [&](const std::string& name) -> Tensor* {
        Tensor& src = weights.get(name);
        if (src.dtype() != dnnl::memory::data_type::bf16) {
            return &src;
        }
        size_t n = (size_t)src.numel();
        std::vector<bf16> host(n);
        ctx.memcpy_d2h(host.data(), src.data(), n * sizeof(bf16));
        for (size_t i = 0; i < n; ++i) {
            host[i] = bf16(static_cast<float>(host[i]) + 1.0f);
        }
        Tensor dst = Tensor::allocate(ctx, src.shape(), src.dtype());
        ctx.memcpy_h2d(dst.data(), host.data(), n * sizeof(bf16));
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };
    auto erase_language_weight = [&](const std::string& name) {
        if (name.rfind("model.language_model.", 0) == 0 && weights.has(name)) {
            weights.erase(name);
        }
    };

    embed_weight_ = &weights.get("model.language_model.embed_tokens.weight");

    layers_.clear();
    fused_weights_.clear();
    layers_.resize(cfg_.num_hidden_layers);
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 6 + 1);

    auto fuse_three_cols = [&](Tensor& a, Tensor& b, Tensor& c) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t c_cols = c.shape(1);
        int64_t total_cols = a_cols + b_cols + c_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* c_ptr = static_cast<bf16*>(c.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else if (cidx < a_cols + b_cols) {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                } else {
                    int cc = cidx - static_cast<int>(a_cols + b_cols);
                    o_ptr[out_idx] = c_ptr[static_cast<int64_t>(r) * c_cols + cc];
                }
            });
        return out;
    };

    auto fuse_four_cols = [&](Tensor& a, Tensor& b, Tensor& c, Tensor& d) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t c_cols = c.shape(1);
        int64_t d_cols = d.shape(1);
        int64_t total_cols = a_cols + b_cols + c_cols + d_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* c_ptr = static_cast<bf16*>(c.data());
        bf16* d_ptr = static_cast<bf16*>(d.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else if (cidx < a_cols + b_cols) {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                } else if (cidx < a_cols + b_cols + c_cols) {
                    int cc = cidx - static_cast<int>(a_cols + b_cols);
                    o_ptr[out_idx] = c_ptr[static_cast<int64_t>(r) * c_cols + cc];
                } else {
                    int dc = cidx - static_cast<int>(a_cols + b_cols + c_cols);
                    o_ptr[out_idx] = d_ptr[static_cast<int64_t>(r) * d_cols + dc];
                }
            });
        return out;
    };

    auto fuse_two_cols = [&](Tensor& a, Tensor& b) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t total_cols = a_cols + b_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                }
            });
        return out;
    };

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "model.language_model.layers." + std::to_string(i) + ".";
        std::vector<std::string> weights_to_erase;

        // Qwen3.5 RMSNorm uses (1 + weight) instead of direct weight multiply.
        layer.input_ln_weight = plus_one_norm_weight(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = plus_one_norm_weight(prefix + "post_attention_layernorm.weight");

        bool is_linear = true;
        if (i < static_cast<int>(cfg_.layer_types.size())) {
            is_linear = (cfg_.layer_types[(size_t)i] == "linear_attention");
        } else {
            is_linear = ((i + 1) % 4 != 0);
        }
        layer.is_linear = is_linear;

        if (is_linear) {
            Tensor* o_w = transpose_weight(prefix + "linear_attn.out_proj.weight");
            layer.linear_o_proj.init(ctx, *o_w, linear_kv_dim_, hidden_size_, true);

            if (use_delta_linear_) {
                Tensor* qkv_w = transpose_weight(prefix + "linear_attn.in_proj_qkv.weight");
                Tensor* z_w = transpose_weight(prefix + "linear_attn.in_proj_z.weight");
                Tensor* a_w = transpose_weight(prefix + "linear_attn.in_proj_a.weight");
                Tensor* b_w = transpose_weight(prefix + "linear_attn.in_proj_b.weight");
                fused_weights_.push_back(fuse_four_cols(*qkv_w, *z_w, *a_w, *b_w));
                layer.linear_all_proj.init(ctx, fused_weights_.back(), hidden_size_, linear_all_dim_, true);
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_qkv.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_z.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_a.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_b.weight");

                if (decode_ffn_custom_enabled_) {
                    Tensor linear_all_jm = Tensor::allocate(ctx,
                                                            {(int64_t)linear_all_dim_, (int64_t)hidden_size_},
                                                            fused_weights_.back().dtype());
                    ops::transpose(ctx, fused_weights_.back(), linear_all_jm);
                    fused_weights_.push_back(std::move(linear_all_jm));
                    layer.linear_all_weight_jm = &fused_weights_.back();
                }

                layer.linear_norm_weight = &weights.get(prefix + "linear_attn.norm.weight");
                layer.linear_A_log = &weights.get(prefix + "linear_attn.A_log");
                layer.linear_dt_bias = &weights.get(prefix + "linear_attn.dt_bias");
                layer.linear_conv1d_weight = &weights.get(prefix + "linear_attn.conv1d.weight");

                layer.host_linear_A_negexp.resize((size_t)linear_kv_heads_);
                layer.host_linear_dt_bias.resize((size_t)linear_kv_heads_);
                layer.host_linear_norm.resize((size_t)cfg_.linear_value_head_dim);
                layer.host_linear_conv.resize((size_t)linear_conv_channels_ * linear_conv_kernel_dim_);

                std::vector<float> host_a((size_t)linear_kv_heads_);
                ctx.memcpy_d2h(host_a.data(), layer.linear_A_log->data(),
                               host_a.size() * sizeof(float));
                for (int h = 0; h < linear_kv_heads_; ++h) {
                    layer.host_linear_A_negexp[(size_t)h] = -std::exp(host_a[(size_t)h]);
                }

                {
                    std::vector<bf16> host_dt((size_t)linear_kv_heads_);
                    ctx.memcpy_d2h(host_dt.data(), layer.linear_dt_bias->data(),
                                   host_dt.size() * sizeof(bf16));
                    for (int h = 0; h < linear_kv_heads_; ++h) {
                        layer.host_linear_dt_bias[(size_t)h] = static_cast<float>(host_dt[(size_t)h]);
                    }
                }

                {
                    std::vector<float> host_norm((size_t)cfg_.linear_value_head_dim);
                    ctx.memcpy_d2h(host_norm.data(), layer.linear_norm_weight->data(),
                                   host_norm.size() * sizeof(float));
                    layer.host_linear_norm = std::move(host_norm);
                }

                {
                    size_t n_conv = (size_t)linear_conv_channels_ * (size_t)linear_conv_kernel_dim_;
                    std::vector<bf16> conv_raw(n_conv);
                    ctx.memcpy_d2h(conv_raw.data(), layer.linear_conv1d_weight->data(),
                                   n_conv * sizeof(bf16));
                    for (size_t idx = 0; idx < n_conv; ++idx) {
                        layer.host_linear_conv[idx] = static_cast<float>(conv_raw[idx]);
                    }
                }
            } else {
                Tensor* qkv_w = transpose_weight(prefix + "linear_attn.in_proj_qkv.weight");
                Tensor* z_w = transpose_weight(prefix + "linear_attn.in_proj_z.weight");
                layer.linear_qkv_proj.init(ctx, *qkv_w, hidden_size_, linear_qkv_dim_, true);
                layer.linear_z_proj.init(ctx, *z_w, hidden_size_, linear_z_dim_, true);
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_a.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_b.weight");
                layer.linear_norm_weight = nullptr;
                layer.linear_A_log = nullptr;
                layer.linear_dt_bias = nullptr;
                layer.linear_conv1d_weight = nullptr;
                layer.host_linear_A_negexp.clear();
                layer.host_linear_dt_bias.clear();
                layer.host_linear_norm.clear();
                layer.host_linear_conv.clear();
            }

            if (decode_ffn_custom_enabled_) {
                Tensor linear_o_jm = Tensor::allocate(ctx,
                                                      {(int64_t)hidden_size_, (int64_t)linear_kv_dim_},
                                                      o_w->dtype());
                ops::transpose(ctx, *o_w, linear_o_jm);
                fused_weights_.push_back(std::move(linear_o_jm));
                layer.linear_o_weight_jm = &fused_weights_.back();
            }

        } else {
            Tensor* q_w = transpose_weight(prefix + "self_attn.q_proj.weight");
            Tensor* k_w = transpose_weight(prefix + "self_attn.k_proj.weight");
            Tensor* v_w = transpose_weight(prefix + "self_attn.v_proj.weight");
            Tensor* o_w = transpose_weight(prefix + "self_attn.o_proj.weight");
            fused_weights_.push_back(fuse_three_cols(*q_w, *k_w, *v_w));
            layer.qkv_proj.init(ctx, fused_weights_.back(), hidden_size_, full_fused_qkv_dim_, true);
            layer.o_proj.init(ctx, *o_w, full_q_dim_, hidden_size_, true);
            weights_to_erase.push_back(prefix + "self_attn.q_proj.weight");
            weights_to_erase.push_back(prefix + "self_attn.k_proj.weight");
            weights_to_erase.push_back(prefix + "self_attn.v_proj.weight");
            if (decode_ffn_custom_enabled_) {
                Tensor qkv_jm = Tensor::allocate(ctx,
                                                 {(int64_t)full_fused_qkv_dim_, (int64_t)hidden_size_},
                                                 fused_weights_.back().dtype());
                ops::transpose(ctx, fused_weights_.back(), qkv_jm);
                fused_weights_.push_back(std::move(qkv_jm));
                layer.qkv_weight_jm = &fused_weights_.back();

                Tensor o_jm = Tensor::allocate(ctx,
                                               {(int64_t)hidden_size_, (int64_t)full_q_dim_},
                                               o_w->dtype());
                ops::transpose(ctx, *o_w, o_jm);
                fused_weights_.push_back(std::move(o_jm));
                layer.o_weight_jm = &fused_weights_.back();
            }
            layer.q_norm_weight = plus_one_norm_weight(prefix + "self_attn.q_norm.weight");
            layer.k_norm_weight = plus_one_norm_weight(prefix + "self_attn.k_norm.weight");

        }

        Tensor* gate_w = transpose_weight(prefix + "mlp.gate_proj.weight");
        Tensor* up_w = transpose_weight(prefix + "mlp.up_proj.weight");
        Tensor* down_w = transpose_weight(prefix + "mlp.down_proj.weight");
        fused_weights_.push_back(fuse_two_cols(*gate_w, *up_w));
        layer.gate_up_weight = &fused_weights_.back();
        layer.gate_up_proj.init(ctx, fused_weights_.back(), hidden_size_, 2 * ff_dim_, true);
        layer.down_weight = down_w;
        layer.down_proj.init(ctx, *down_w, ff_dim_, hidden_size_, true);
        weights_to_erase.push_back(prefix + "mlp.gate_proj.weight");
        weights_to_erase.push_back(prefix + "mlp.up_proj.weight");

        if (decode_ffn_custom_enabled_) {
            Tensor gate_up_jm = Tensor::allocate(ctx,
                                                 {(int64_t)(2 * ff_dim_), (int64_t)hidden_size_},
                                                 layer.gate_up_weight->dtype());
            ops::transpose(ctx, *layer.gate_up_weight, gate_up_jm);
            fused_weights_.push_back(std::move(gate_up_jm));
            layer.gate_up_weight_jm = &fused_weights_.back();

            Tensor down_jm = Tensor::allocate(ctx,
                                              {(int64_t)hidden_size_, (int64_t)ff_dim_},
                                              layer.down_weight->dtype());
            ops::transpose(ctx, *layer.down_weight, down_jm);
            fused_weights_.push_back(std::move(down_jm));
            layer.down_weight_jm = &fused_weights_.back();
        }

        if (!weights_to_erase.empty()) {
            ctx.synchronize();
            for (const auto& name : weights_to_erase) {
                erase_language_weight(name);
            }
        }
    }

    final_norm_weight_ = plus_one_norm_weight("model.language_model.norm.weight");

    if (weights.has("lm_head.weight")) {
        Tensor* lm_w = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lm_w, hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3.5] lm_head loaded (standalone)");
    } else {
        bool tied_lm_head_ready = false;
        bool prefer_preprocessed_tied_lm_head = is_exact_q35_0p8b_spec && !force_direct_tied_lm_head;
        if (prefer_preprocessed_tied_lm_head) {
            try {
                Tensor& src = *embed_weight_;
                Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
                ops::transpose(ctx, src, dst);
                ctx.synchronize();
                weights.put("model.language_model.lm_head.weight_preprocessed", std::move(dst));
                lm_head_.init(ctx, weights.get("model.language_model.lm_head.weight_preprocessed"),
                              hidden_size_, cfg_.vocab_size, true);
                tied_lm_head_ready = true;
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, preprocessed copy)");
            } catch (const aila::memory::AllocationLimitExceeded&) {
                throw;
            } catch (const sycl::exception&) {
                throw;
            } catch (const dnnl::error&) {
                throw;
            } catch (const std::exception& e) {
                AILA_LOG_WARN("[Qwen3.5] Failed to preprocess tied lm_head, falling back to direct embed weight: %s",
                              e.what());
            }
        }
        if (!tied_lm_head_ready) {
            lm_head_.init(ctx, *embed_weight_, hidden_size_, cfg_.vocab_size, false);
            if (force_direct_tied_lm_head) {
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, direct embed weight forced via AILA_Q35_DIRECT_TIED_LM_HEAD=1)");
            } else if (prefer_preprocessed_tied_lm_head) {
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, direct embed weight fallback)");
            } else {
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, direct embed weight for hybrid spec)");
            }
        }
    }

    ctx.synchronize();

    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    incr_prefill_seq_cap_ = 0;
    incr_prefill_total_cap_ = 0;
    clear_embedding_overrides();
    ensure_runtime_buffers(ctx, 1);

    buf_.logits = Tensor::allocate(ctx, {1, (int64_t)cfg_.vocab_size});
    buf_.decode_scores = Tensor::allocate(
        ctx, {(int64_t)max_attn_heads_, (int64_t)max_seq_len_},
        dnnl::memory::data_type::f32);
    if (full_head_dim_ == 256) {
        constexpr int kDecodeExactTile = 128;
        constexpr int kDecodeExactPartialStride = 272;
        const int max_tiles = (max_seq_len_ + kDecodeExactTile - 1) / kDecodeExactTile;
        // Four sequences is the decode batch bound, so each batched row owns a
        // private [head, tile] block and the multi-row exact merge needs no
        // cross-row arbitration. The size is independent of context length.
        buf_.decode_attn_partials = Tensor::allocate(
            ctx,
            {4, (int64_t)max_attn_heads_, (int64_t)max_tiles,
             (int64_t)kDecodeExactPartialStride},
            dnnl::memory::data_type::f32);
    }

    AILA_LOG_INFO("[Qwen3.5] Hybrid text backend loaded: layers=%d hidden=%d vocab=%d",
                  cfg_.num_hidden_layers, hidden_size_, cfg_.vocab_size);
    AILA_LOG_INFO("[Qwen3.5] Full attention: q_heads=%d kv_heads=%d head_dim=%d",
                  full_q_heads_, full_kv_heads_, full_head_dim_);
    AILA_LOG_INFO("[Qwen3.5] Linear attention: q_heads=%d kv_heads=%d key_dim=%d value_dim=%d conv_kernel=%d exact_0p8b=%d",
                  linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
                  linear_conv_kernel_dim_, is_exact_q35_0p8b_spec ? 1 : 0);
    AILA_LOG_INFO("[Qwen3.5] Linear fast path: eligible=%d gqa_group=%d head_dims_match=%d key_dim=%d value_dim=%d kernel=%d conv_rows=%d experimental_gqa=%d default_grouped=%d force_host=%d",
                  linear_fastpath_eligible ? 1 : 0,
                  linear_gqa_group_size,
                  (linear_head_dim_ == cfg_.linear_value_head_dim) ? 1 : 0,
                  linear_head_dim_,
                  cfg_.linear_value_head_dim,
                  linear_conv_kernel_dim_,
                  linear_conv_rows,
                  experimental_gqa_fastpath ? 1 : 0,
                  default_grouped_linear_fastpath ? 1 : 0,
                  force_host_grouped_linear ? 1 : 0);
    if (linear_kv_heads_ > linear_q_heads_) {
        const char* grouped_backend = "host-reference";
        if (use_grouped_linear_gpu_default) {
            grouped_backend = "gpu-fastpath-default";
        } else if (use_grouped_linear_gpu_experimental) {
            grouped_backend = "gpu-experimental";
        }
        AILA_LOG_INFO("[Qwen3.5] Grouped linear delta backend: %s", grouped_backend);
        if (force_host_grouped_linear) {
            AILA_LOG_INFO("[Qwen3.5] Grouped linear host fallback forced by AILA_Q35_FORCE_HOST_GROUPED_LINEAR=1");
        } else if (!default_grouped_linear_fastpath) {
            AILA_LOG_INFO("[Qwen3.5] Set AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU=1 to try the generic grouped GPU path");
        }
    }
    AILA_LOG_INFO("[Qwen3.5] Decode FFN custom path: enabled=%d (hidden=%d ff=%d)",
                  decode_ffn_custom_enabled_ ? 1 : 0,
                  hidden_size_,
                  ff_dim_);
    const char* linear_mode = "legacy-attn";
    if (use_delta_linear_) {
        linear_mode = "delta-prefill-gpu-iter/decode-gpu-ring";
    }
    AILA_LOG_INFO("[Qwen3.5] Linear mode: %s", linear_mode);
    if (linear_kv_heads_ > linear_q_heads_) {
        AILA_LOG_INFO("[Qwen3.5] Legacy linear-attn fallback is unsupported for grouped linear heads unless AILA_Q35_ALLOW_UNSUPPORTED_LEGACY_LINEAR=1 is set");
    } else {
        AILA_LOG_INFO("[Qwen3.5] Set AILA_Q35_LINEAR_DELTA=0 to force legacy-attn fallback");
    }
    AILA_LOG_INFO("[Qwen3.5] KV Cache config: quantized=%s", aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT") ? "true" : "false");
    BuildDecoderPlan();
    sessionOwner_ = std::make_shared<const SessionOwner>();
    return true;
}

void Qwen35HybridTextBackend::BuildDecoderPlan() {
    auto plan = std::make_unique<DecoderPlan>();
    plan->initialNormWeight = layers_.front().input_ln_weight;
    plan->layers.reserve(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto kind = !layers_[i].is_linear ? LayerKind::FullAttention
            : use_delta_linear_ ? LayerKind::DeltaNet : LayerKind::LegacyLinearAttention;
        plan->hasRecurrentLayers |= kind == LayerKind::DeltaNet;
        Tensor* postMlpNorm = i + 1 < layers_.size()
            ? layers_[i + 1].input_ln_weight : final_norm_weight_;
        // Projection addresses remain stable after loading; bind the existing
        // cross-layer residual + norm fusion rather than splitting its kernels.
        plan->layers.push_back({&layers_[i], postMlpNorm, static_cast<int>(i), kind});
    }
    decoderPlan_ = std::move(plan);
}

void Qwen35HybridTextBackend::set_embedding_overrides(
    const std::vector<int>& positions,
    const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
    int hidden_size) {
    if (positions.empty() || embeddings.empty() || hidden_size <= 0 ||
        positions.size() > std::numeric_limits<size_t>::max() / static_cast<size_t>(hidden_size) ||
        embeddings.size() != positions.size() * static_cast<size_t>(hidden_size)) {
        clear_embedding_overrides();
        return;
    }
    if (!context_ || !sessionOwner_) {
        throw std::logic_error("Qwen35HybridTextBackend: embedding overrides require a loaded model");
    }
    auto& session = DefaultSession(*context_);
    auto newPositions = positions;
    auto newValues = embeddings;
    // Forward borrows the host vectors for asynchronous copies.
    context_->synchronize();
    session.embeddingOverridePositions_.swap(newPositions);
    session.embeddingOverrideValues_.swap(newValues);
    session.embeddingOverrideHiddenSize_ = hidden_size;
}

void Qwen35HybridTextBackend::clear_embedding_overrides() {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    if (!session.embeddingOverrideValues_.empty()) session.context_->synchronize();
    session.embeddingOverridePositions_.clear();
    session.embeddingOverrideValues_.clear();
    session.embeddingOverrideHiddenSize_ = 0;
}

void Qwen35HybridTextBackend::set_mrope_positions(Context& ctx,
                                                  const std::vector<int>& pos_t,
                                                  const std::vector<int>& pos_h,
                                                  const std::vector<int>& pos_w,
                                                  int text_pos_delta) {
    if (pos_t.empty() || pos_h.empty() || pos_w.empty() ||
        pos_t.size() != pos_h.size() || pos_t.size() != pos_w.size()) {
        clear_mrope_positions();
        return;
    }
    if (pos_t.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Qwen35HybridTextBackend: too many MRoPE positions");
    }
    auto& session = DefaultSession(ctx);
    const size_t bytes = aila::memory::checked_memory_product(pos_t.size(), sizeof(int));
    const auto shape = std::vector<int64_t>{static_cast<int64_t>(pos_t.size())};
    auto newT = Tensor::allocate(ctx, shape, dnnl::memory::data_type::s32);
    auto newH = Tensor::allocate(ctx, shape, dnnl::memory::data_type::s32);
    auto newW = Tensor::allocate(ctx, shape, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(newT.data(), pos_t.data(), bytes);
    ctx.memcpy_h2d(newH.data(), pos_h.data(), bytes);
    ctx.memcpy_h2d(newW.data(), pos_w.data(), bytes);
    // These blocking copies also retire earlier work borrowing the old tensors.
    session.mropePosT_ = std::move(newT);
    session.mropePosH_ = std::move(newH);
    session.mropePosW_ = std::move(newW);
    session.mropePromptLength_ = static_cast<int>(pos_t.size());
    session.mropeTextPositionDelta_ = text_pos_delta;
}

void Qwen35HybridTextBackend::clear_mrope_positions() {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    if (session.mropePosT_.valid() || session.mropePosH_.valid() || session.mropePosW_.valid()) {
        session.context_->synchronize();
    }
    session.mropePosT_ = Tensor();
    session.mropePosH_ = Tensor();
    session.mropePosW_ = Tensor();
    session.mropePromptLength_ = 0;
    session.mropeTextPositionDelta_ = 0;
}

void Qwen35HybridTextBackend::run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                                          Tensor& qkv_src, Tensor& z_src,
                                                          Tensor& a_src, Tensor& b_src) {
    run_linear_delta_decode_gpu(ctx, layer, cache, qkv_src, z_src, a_src, b_src, buf_.attn_out);
}

void Qwen35HybridTextBackend::run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                                          Tensor& qkv_src, Tensor& z_src,
                                                          Tensor& a_src, Tensor& b_src,
                                                          Tensor& out_dst) {
    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int q_dim = linear_q_dim_;
    const int kv_dim = linear_kv_dim_;
    const int num_heads = linear_kv_heads_;
    const int q_heads = std::max(1, linear_q_heads_);
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;
    const int conv_rows = std::max(0, kernel - 1);
    const int conv_head = (conv_rows > 0) ? (cache.linear_conv_head % conv_rows) : 0;
    const float eps = cfg_.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    constexpr int wg = 256;

    bf16* qkv_ptr = static_cast<bf16*>(qkv_src.data());
    bf16* z_ptr = static_cast<bf16*>(z_src.data());
    bf16* a_ptr = static_cast<bf16*>(a_src.data());
    bf16* b_ptr = static_cast<bf16*>(b_src.data());
    bf16* out_ptr = static_cast<bf16*>(out_dst.data());
    float* state_ptr = static_cast<float*>(cache.linear_state.data());
    float* conv_state_ptr = cache.linear_conv_state.valid()
        ? static_cast<float*>(cache.linear_conv_state.data())
        : nullptr;
    bf16* conv_w_ptr = static_cast<bf16*>(layer.linear_conv1d_weight->data());
    float* norm_w_ptr = static_cast<float*>(layer.linear_norm_weight->data());
    float* a_log_ptr = static_cast<float*>(layer.linear_A_log->data());
    bf16* dt_bias_ptr = static_cast<bf16*>(layer.linear_dt_bias->data());
    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool default_grouped_fastpath = supports_default_grouped_linear_delta_fastpath(
        q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows);
    bool allow_grouped_fastpath = experimental_gqa_fastpath || default_grouped_fastpath;

    if (supports_linear_delta_gqa_fastpath(q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows,
                                           allow_grouped_fastpath)) {
        constexpr int head_dim = 128;
        constexpr int head_wg = 128;
        const bool grouped_heads = num_heads > q_heads;
        const int row0 = conv_head;
        const int row1 = (conv_head + 1) % conv_rows;
        const int row2 = (conv_head + 2) % conv_rows;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
                [=](sycl::nd_item<1> item) {
                    auto softplus_dev = [](float x) {
                        if (x > 20.0f) return x;
                        if (x < -20.0f) return sycl::native::exp(x);
                        return sycl::log1p(sycl::native::exp(x));
                    };
                    auto silu_dev = [](float x) {
                        return x / (1.0f + sycl::native::exp(-x));
                    };

                    int hv = static_cast<int>(item.get_group(0));
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    bool write_shared_qk = linear_delta_is_group_leader(hv, q_heads, num_heads);
                    int lid = static_cast<int>(item.get_local_id(0));
                    int q_base = hk * head_dim;
                    int k_base = q_dim + hk * head_dim;
                    int v_base = q_dim + q_dim + hv * head_dim;
                    int z_base = hv * head_dim;
                    float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;
                    float v_reg = 0.0f;
                    float z_reg = 0.0f;

                    if (lid < head_dim) {
                        auto conv_channel = [&](int channel) {
                            const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                            const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                            const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                            bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                            float v = conv_state_ptr[state0] * static_cast<float>(w[0]) +
                                      conv_state_ptr[state1] * static_cast<float>(w[1]) +
                                      conv_state_ptr[state2] * static_cast<float>(w[2]) +
                                      static_cast<float>(qkv_ptr[channel]) * static_cast<float>(w[3]);
                            return silu_dev(v);
                        };

                        q_local[lid] = conv_channel(q_base + lid);
                        k_local[lid] = conv_channel(k_base + lid);
                        v_reg = conv_channel(v_base + lid);
                        z_reg = static_cast<float>(z_ptr[z_base + lid]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (!grouped_heads && lid < head_dim) {
                        size_t dst = static_cast<size_t>(conv_head) * qkv_dim;
                        if (write_shared_qk) {
                            conv_state_ptr[dst + q_base + lid] = static_cast<float>(qkv_ptr[q_base + lid]);
                            conv_state_ptr[dst + k_base + lid] = static_cast<float>(qkv_ptr[k_base + lid]);
                        }
                        conv_state_ptr[dst + v_base + lid] = static_cast<float>(qkv_ptr[v_base + lid]);
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
                        float beta = 1.0f / (1.0f + sycl::native::exp(-static_cast<float>(b_ptr[hv])));
                        float alpha = static_cast<float>(a_ptr[hv]);
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
                        for (int j = 0; j < head_dim; j += 4) {
                            const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                            const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                            const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                            const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                            float sv0 = S[off0] * decay;
                            float sv1 = S[off1] * decay;
                            float sv2 = S[off2] * decay;
                            float sv3 = S[off3] * decay;
                            S[off0] = sv0;
                            S[off1] = sv1;
                            S[off2] = sv2;
                            S[off3] = sv3;
                            acc += sv0 * k_local[j + 0] +
                                   sv1 * k_local[j + 1] +
                                   sv2 * k_local[j + 2] +
                                   sv3 * k_local[j + 3];
                        }

                        float dval = (v_reg - acc) * beta;
                        float out = 0.0f;
                        for (int j = 0; j < head_dim; j += 4) {
                            const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                            const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                            const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                            const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                            float sv0 = S[off0] + k_local[j + 0] * dval;
                            float sv1 = S[off1] + k_local[j + 1] * dval;
                            float sv2 = S[off2] + k_local[j + 2] * dval;
                            float sv3 = S[off3] + k_local[j + 3] * dval;
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
                        out_ptr[hv * head_dim + lid] = bf16(n * silu_dev(z_reg));
                    }
                });
        });
        if (grouped_heads) {
            commit_linear_conv_history(ctx, qkv_ptr, qkv_dim, conv_state_ptr,
                                       qkv_dim, conv_rows, conv_head, 1);
        }
        cache.device_state_dirty = true;
        if (conv_rows > 0) {
            cache.linear_conv_head = (conv_head + 1) % conv_rows;
        }
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> q_local(sycl::range<1>(q_dim), cgh);
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(q_dim), cgh);
        sycl::local_accessor<float, 1> v_local(sycl::range<1>(kv_dim), cgh);
        sycl::local_accessor<float, 1> z_local(sycl::range<1>(z_dim), cgh);
        sycl::local_accessor<float, 1> out_local(sycl::range<1>(kv_dim), cgh);
        sycl::local_accessor<float, 1> q_scale_local(sycl::range<1>(q_heads), cgh);
        sycl::local_accessor<float, 1> k_scale_local(sycl::range<1>(q_heads), cgh);
        sycl::local_accessor<float, 1> out_scale_local(sycl::range<1>(num_heads), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                auto softplus_dev = [](float x) {
                    if (x > 20.0f) return x;
                    if (x < -20.0f) return sycl::exp(x);
                    return sycl::log1p(sycl::exp(x));
                };
                auto silu_dev = [](float x) {
                    return x / (1.0f + sycl::exp(-x));
                };

                int lid = static_cast<int>(item.get_local_id(0));

                for (int i = lid; i < z_dim; i += wg) {
                    z_local[i] = static_cast<float>(z_ptr[i]);
                }

                for (int c = lid; c < qkv_dim; c += wg) {
                    float v = 0.0f;
                    bf16* w = conv_w_ptr + static_cast<size_t>(c) * kernel;
                    if (conv_rows > 0 && conv_state_ptr) {
                        for (int j = 0; j < kernel - 1; ++j) {
                            int row = (conv_head + j) % conv_rows;
                            v += conv_state_ptr[static_cast<size_t>(row) * qkv_dim + c]
                                * static_cast<float>(w[j]);
                        }
                    }
                    v += static_cast<float>(qkv_ptr[c]) * static_cast<float>(w[kernel - 1]);
                    float y = silu_dev(v);
                    if (c < q_dim) {
                        q_local[c] = y;
                    } else if (c < q_dim + q_dim) {
                        k_local[c - q_dim] = y;
                    } else {
                        v_local[c - q_dim - q_dim] = y;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (conv_rows > 0 && conv_state_ptr) {
                    size_t dst_base = static_cast<size_t>(conv_head) * qkv_dim;
                    for (int idx = lid; idx < qkv_dim; idx += wg) {
                        conv_state_ptr[dst_base + idx] = static_cast<float>(qkv_ptr[idx]);
                    }
                }

                for (int h = lid; h < q_heads; h += wg) {
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    int base = h * head_k_dim;
                    for (int d = 0; d < head_k_dim; ++d) {
                        float qv = q_local[base + d];
                        float kv = k_local[base + d];
                        q_sum += qv * qv;
                        k_sum += kv * kv;
                    }
                    q_scale_local[h] = 1.0f / sycl::sqrt(q_sum + eps);
                    k_scale_local[h] = 1.0f / sycl::sqrt(k_sum + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < q_dim; idx += wg) {
                    int h = idx / head_k_dim;
                    q_local[idx] *= q_scale_local[h];
                    k_local[idx] *= k_scale_local[h];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < kv_dim; idx += wg) {
                    int hv = idx / head_v_dim;
                    int dv = idx % head_v_dim;
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    float* S = state_ptr + static_cast<size_t>(hv) * head_k_dim * head_v_dim;

                    float beta = 1.0f / (1.0f + sycl::exp(-static_cast<float>(b_ptr[hv])));
                    float alpha = static_cast<float>(a_ptr[hv]);
                    float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                * (-sycl::exp(a_log_ptr[hv]));
                    float decay = sycl::exp(g_pre);

                    float acc = 0.0f;
                    for (int j = 0; j < head_k_dim; ++j) {
                        size_t off = static_cast<size_t>(j) * head_v_dim + dv;
                        float sv = S[off] * decay;
                        S[off] = sv;
                        acc += sv * k_local[hk * head_k_dim + j];
                    }

                    float d = (v_local[idx] - acc) * beta;
                    float out = 0.0f;
                    for (int j = 0; j < head_k_dim; ++j) {
                        size_t off = static_cast<size_t>(j) * head_v_dim + dv;
                        float sv = S[off] + k_local[hk * head_k_dim + j] * d;
                        S[off] = sv;
                        out += sv * q_local[hk * head_k_dim + j];
                    }
                    out_local[idx] = out * scale;
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int h = lid; h < num_heads; h += wg) {
                    float sum_sq = 0.0f;
                    int base = h * head_v_dim;
                    for (int d = 0; d < head_v_dim; ++d) {
                        float v = out_local[base + d];
                        sum_sq += v * v;
                    }
                    out_scale_local[h] = 1.0f / sycl::sqrt(
                        sum_sq / static_cast<float>(head_v_dim) + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < kv_dim; idx += wg) {
                    int hv = idx / head_v_dim;
                    int dv = idx % head_v_dim;
                    float n = out_local[idx] * out_scale_local[hv] * norm_w_ptr[dv];
                    float zg = z_local[idx];
                    out_ptr[idx] = bf16(n * silu_dev(zg));
                }
            });
    });

    cache.device_state_dirty = true;
    if (conv_rows > 0) {
        cache.linear_conv_head = (conv_head + 1) % conv_rows;
    }
}

void Qwen35HybridTextBackend::run_linear_delta_prefill_gpu_batched(
    Context& ctx, Layer& layer, LayerCache& cache,
    Tensor& fused_all, Tensor& out_dst, int seq_len) {

    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int q_dim = linear_q_dim_;
    const int kv_dim = linear_kv_dim_;
    const int num_heads = linear_kv_heads_;
    const int q_heads = std::max(1, linear_q_heads_);
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;
    const int conv_rows = std::max(0, kernel - 1);
    int conv_head = (conv_rows > 0) ? (cache.linear_conv_head % conv_rows) : 0;
    const float eps = cfg_.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    const int all_dim = linear_all_dim_;

    bf16* fused_ptr = static_cast<bf16*>(fused_all.data());
    bf16* out_ptr = static_cast<bf16*>(out_dst.data());
    float* state_ptr = static_cast<float*>(cache.linear_state.data());
    float* conv_state_ptr = cache.linear_conv_state.valid()
        ? static_cast<float*>(cache.linear_conv_state.data())
        : nullptr;
    bf16* conv_w_ptr = static_cast<bf16*>(layer.linear_conv1d_weight->data());
    float* norm_w_ptr = static_cast<float*>(layer.linear_norm_weight->data());
    float* a_log_ptr = static_cast<float*>(layer.linear_A_log->data());
    bf16* dt_bias_ptr = static_cast<bf16*>(layer.linear_dt_bias->data());
    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool default_grouped_fastpath = supports_default_grouped_linear_delta_fastpath(
        q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows);
    bool allow_grouped_fastpath = experimental_gqa_fastpath || default_grouped_fastpath;

    if (supports_linear_delta_gqa_fastpath(q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows,
                                           allow_grouped_fastpath)) {
        constexpr int head_dim = 128;
        constexpr int head_wg = 128;
        const bool grouped_heads = num_heads > q_heads;

        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
                [=](sycl::nd_item<1> item) {
                    auto softplus_dev = [](float x) {
                        if (x > 20.0f) return x;
                        if (x < -20.0f) return sycl::native::exp(x);
                        return sycl::log1p(sycl::native::exp(x));
                    };
                    auto silu_dev = [](float x) {
                        return x / (1.0f + sycl::native::exp(-x));
                    };

                    int hv = static_cast<int>(item.get_group(0));
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    bool write_shared_qk = linear_delta_is_group_leader(hv, q_heads, num_heads);
                    int lid = static_cast<int>(item.get_local_id(0));
                    int q_base = hk * head_dim;
                    int k_base = q_dim + hk * head_dim;
                    int v_base = q_dim + q_dim + hv * head_dim;
                    int z_base_offset = qkv_dim + hv * head_dim;
                    int a_offset = qkv_dim + z_dim;
                    int b_offset = qkv_dim + z_dim + num_heads;
                    float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;

                    int local_conv_head = conv_head;

                    for (int t = 0; t < seq_len; ++t) {
                        bf16* row = fused_ptr + static_cast<size_t>(t) * all_dim;
                        bf16* out_row = out_ptr + static_cast<size_t>(t) * kv_dim;

                        float v_reg = 0.0f;
                        float z_reg = 0.0f;

                        const int row0 = local_conv_head;
                        const int row1 = (local_conv_head + 1) % conv_rows;
                        const int row2 = (local_conv_head + 2) % conv_rows;

                        if (lid < head_dim) {
                            auto conv_channel = [&](int channel) {
                                if (grouped_heads) {
                                    // Previous rows in this prefill are already
                                    // available in the immutable projection input.
                                    // Only negative positions use the initial ring;
                                    // no work-group updates it until this kernel ends.
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
                                    bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                                    const float v = history_value(0) * static_cast<float>(w[0]) +
                                                    history_value(1) * static_cast<float>(w[1]) +
                                                    history_value(2) * static_cast<float>(w[2]) +
                                                    static_cast<float>(row[channel]) * static_cast<float>(w[3]);
                                    return silu_dev(v);
                                }
                                const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                                const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                                const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                                bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
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
                            for (int j = 0; j < head_dim; j += 4) {
                                const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                                const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                                const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                                const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                                float sv0 = S[off0] * decay;
                                float sv1 = S[off1] * decay;
                                float sv2 = S[off2] * decay;
                                float sv3 = S[off3] * decay;
                                S[off0] = sv0;
                                S[off1] = sv1;
                                S[off2] = sv2;
                                S[off3] = sv3;
                                acc += sv0 * k_local[j + 0] +
                                       sv1 * k_local[j + 1] +
                                       sv2 * k_local[j + 2] +
                                       sv3 * k_local[j + 3];
                            }

                            float dval = (v_reg - acc) * beta;
                            float out = 0.0f;
                            for (int j = 0; j < head_dim; j += 4) {
                                const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                                const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                                const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                                const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                                float sv0 = S[off0] + k_local[j + 0] * dval;
                                float sv1 = S[off1] + k_local[j + 1] * dval;
                                float sv2 = S[off2] + k_local[j + 2] * dval;
                                float sv3 = S[off3] + k_local[j + 3] * dval;
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
                            out_row[hv * head_dim + lid] = bf16(n * silu_dev(z_reg));
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        local_conv_head = (local_conv_head + 1) % conv_rows;
                    }
                });
        });

        if (grouped_heads) {
            commit_linear_conv_history(ctx, fused_ptr, all_dim, conv_state_ptr,
                                       qkv_dim, conv_rows, conv_head, seq_len);
        }
        cache.device_state_dirty = true;
        cache.linear_conv_head = (conv_head + seq_len) % conv_rows;
        return;
    }

    // Fallback: per-token iteration for non-standard shapes
    bf16* out_base = static_cast<bf16*>(out_dst.data());
    const int64_t fused_stride = fused_all.shape(1);
    const int64_t out_stride = out_dst.shape(1);
    for (int t = 0; t < seq_len; ++t) {
        bf16* row_ptr = fused_ptr + static_cast<size_t>(t) * fused_stride;
        Tensor qkv_view = Tensor::view(ctx, row_ptr, {1, (int64_t)qkv_dim});
        Tensor z_view = Tensor::view(ctx, row_ptr + qkv_dim, {1, (int64_t)z_dim});
        Tensor a_view = Tensor::view(ctx, row_ptr + qkv_dim + z_dim,
                                     {1, (int64_t)num_heads});
        Tensor b_view = Tensor::view(ctx, row_ptr + qkv_dim + z_dim + num_heads,
                                     {1, (int64_t)num_heads});
        Tensor out_view = Tensor::view(ctx,
                                       out_base + static_cast<size_t>(t) * out_stride,
                                       {1, (int64_t)kv_dim});
        run_linear_delta_decode_gpu(ctx, layer, cache,
                                    qkv_view, z_view, a_view, b_view, out_view);
    }
}

void Qwen35HybridTextBackend::debug_compare_linear_delta_decode(Context& ctx, int layer_idx,
                                                                Layer& layer, LayerCache& cache,
                                                                Tensor& qkv_src, Tensor& z_src,
                                                                Tensor& a_src, Tensor& b_src) {
    std::vector<float> orig_state(cache.host_linear_state.size(), 0.0f);
    std::vector<float> orig_conv(cache.host_linear_conv_state.size(), 0.0f);
    if (cache.linear_state.valid() && !orig_state.empty()) {
        ctx.memcpy_d2h(orig_state.data(), cache.linear_state.data(),
                       orig_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !orig_conv.empty()) {
        ctx.memcpy_d2h(orig_conv.data(), cache.linear_conv_state.data(),
                       orig_conv.size() * sizeof(float));
    }

    run_linear_delta_decode_gpu(ctx, layer, cache, qkv_src, z_src, a_src, b_src);

    std::vector<bf16> gpu_out((size_t)linear_kv_dim_);
    std::vector<float> gpu_state(orig_state.size(), 0.0f);
    std::vector<float> gpu_conv(orig_conv.size(), 0.0f);
    ctx.memcpy_d2h(gpu_out.data(), buf_.attn_out.data(), gpu_out.size() * sizeof(bf16));
    if (cache.linear_state.valid() && !gpu_state.empty()) {
        ctx.memcpy_d2h(gpu_state.data(), cache.linear_state.data(),
                       gpu_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !gpu_conv.empty()) {
        ctx.memcpy_d2h(gpu_conv.data(), cache.linear_conv_state.data(),
                       gpu_conv.size() * sizeof(float));
    }

    if (cache.linear_state.valid() && !orig_state.empty()) {
        ctx.memcpy_h2d(cache.linear_state.data(), orig_state.data(),
                       orig_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !orig_conv.empty()) {
        ctx.memcpy_h2d(cache.linear_conv_state.data(), orig_conv.data(),
                       orig_conv.size() * sizeof(float));
    }
    cache.host_linear_state = orig_state;
    cache.host_linear_conv_state = orig_conv;
    cache.device_state_dirty = false;

    run_linear_delta_host(ctx, layer, cache, qkv_src, z_src, a_src, b_src, 1);

    std::vector<bf16> host_out((size_t)linear_kv_dim_);
    ctx.memcpy_d2h(host_out.data(), buf_.attn_out.data(), host_out.size() * sizeof(bf16));

    auto max_abs_bf16 = [](const std::vector<bf16>& a, const std::vector<bf16>& b,
                           size_t* idx_out, float* a_val_out, float* b_val_out) {
        float best = 0.0f;
        size_t best_idx = 0;
        float best_a = 0.0f;
        float best_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float av = static_cast<float>(a[i]);
            float bv = static_cast<float>(b[i]);
            float diff = std::abs(av - bv);
            if (diff > best) {
                best = diff;
                best_idx = i;
                best_a = av;
                best_b = bv;
            }
        }
        if (idx_out) *idx_out = best_idx;
        if (a_val_out) *a_val_out = best_a;
        if (b_val_out) *b_val_out = best_b;
        return best;
    };
    auto max_abs_f32 = [](const std::vector<float>& a, const std::vector<float>& b,
                          size_t* idx_out, float* a_val_out, float* b_val_out) {
        float best = 0.0f;
        size_t best_idx = 0;
        float best_a = 0.0f;
        float best_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = std::abs(a[i] - b[i]);
            if (diff > best) {
                best = diff;
                best_idx = i;
                best_a = a[i];
                best_b = b[i];
            }
        }
        if (idx_out) *idx_out = best_idx;
        if (a_val_out) *a_val_out = best_a;
        if (b_val_out) *b_val_out = best_b;
        return best;
    };

    size_t out_idx = 0, state_idx = 0, conv_idx = 0;
    float gpu_out_val = 0.0f, host_out_val = 0.0f;
    float gpu_state_val = 0.0f, host_state_val = 0.0f;
    float gpu_conv_val = 0.0f, host_conv_val = 0.0f;
    float out_diff = max_abs_bf16(gpu_out, host_out, &out_idx, &gpu_out_val, &host_out_val);
    float state_diff = gpu_state.empty() ? 0.0f
        : max_abs_f32(gpu_state, cache.host_linear_state,
                      &state_idx, &gpu_state_val, &host_state_val);
    float conv_diff = gpu_conv.empty() ? 0.0f
        : max_abs_f32(gpu_conv, cache.host_linear_conv_state,
                      &conv_idx, &gpu_conv_val, &host_conv_val);

    AILA_LOG_INFO(
        "[Q35LinearCompare] layer=%d out_max=%.6g idx=%zu gpu=%.6g host=%.6g "
        "state_max=%.6g idx=%zu gpu=%.6g host=%.6g conv_max=%.6g idx=%zu gpu=%.6g host=%.6g",
        layer_idx, out_diff, out_idx, gpu_out_val, host_out_val,
        state_diff, state_idx, gpu_state_val, host_state_val,
        conv_diff, conv_idx, gpu_conv_val, host_conv_val);
}

void Qwen35HybridTextBackend::run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                                                    Tensor& qkv_src, Tensor& z_src,
                                                    Tensor& a_src, Tensor& b_src,
                                                    int seq_len) {
    run_linear_delta_host(ctx, layer, cache, qkv_src, z_src, a_src, b_src, seq_len, buf_.attn_out);
}

void Qwen35HybridTextBackend::run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                                                    Tensor& qkv_src, Tensor& z_src,
                                                    Tensor& a_src, Tensor& b_src,
                                                    int seq_len, Tensor& out_dst) {
    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int num_heads = linear_kv_heads_;
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;

    if (cache.device_state_dirty) {
        if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
            ctx.memcpy_d2h(cache.host_linear_state.data(), cache.linear_state.data(),
                           cache.host_linear_state.size() * sizeof(float));
        }
        if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
            ctx.memcpy_d2h(cache.host_linear_conv_state.data(), cache.linear_conv_state.data(),
                           cache.host_linear_conv_state.size() * sizeof(float));
        }
        cache.device_state_dirty = false;
    }

    auto& scratch = linear_delta_scratch_;
    scratch.h_qkv.resize((size_t)seq_len * qkv_dim);
    scratch.h_z.resize((size_t)seq_len * z_dim);
    scratch.h_a.resize((size_t)seq_len * num_heads);
    scratch.h_b.resize((size_t)seq_len * num_heads);
    ctx.memcpy_d2h(scratch.h_qkv.data(), qkv_src.data(), scratch.h_qkv.size() * sizeof(bf16));
    ctx.memcpy_d2h(scratch.h_z.data(), z_src.data(), scratch.h_z.size() * sizeof(bf16));
    ctx.memcpy_d2h(scratch.h_a.data(), a_src.data(), scratch.h_a.size() * sizeof(bf16));
    ctx.memcpy_d2h(scratch.h_b.data(), b_src.data(), scratch.h_b.size() * sizeof(bf16));

    std::vector<float>& h_conv_state = cache.host_linear_conv_state;
    if (h_conv_state.size() != (size_t)std::max(0, kernel - 1) * (size_t)linear_conv_channels_) {
        h_conv_state.assign((size_t)std::max(0, kernel - 1) * (size_t)linear_conv_channels_, 0.0f);
    }

    std::vector<float>& state = cache.host_linear_state;
    if (state.size() != (size_t)num_heads * (size_t)head_k_dim * (size_t)head_v_dim) {
        state.assign((size_t)num_heads * (size_t)head_k_dim * (size_t)head_v_dim, 0.0f);
    }

    scratch.conv_in.resize((size_t)seq_len * qkv_dim);
    scratch.z.resize((size_t)seq_len * z_dim);
    scratch.a.resize((size_t)seq_len * num_heads);
    scratch.b.resize((size_t)seq_len * num_heads);
    for (size_t i = 0; i < scratch.conv_in.size(); ++i) scratch.conv_in[i] = static_cast<float>(scratch.h_qkv[i]);
    for (size_t i = 0; i < scratch.z.size(); ++i) scratch.z[i] = static_cast<float>(scratch.h_z[i]);
    for (size_t i = 0; i < scratch.a.size(); ++i) scratch.a[i] = static_cast<float>(scratch.h_a[i]);
    for (size_t i = 0; i < scratch.b.size(); ++i) scratch.b[i] = static_cast<float>(scratch.h_b[i]);

    scratch.conv_out.assign((size_t)seq_len * qkv_dim, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = scratch.conv_in.data() + (size_t)t * qkv_dim;
        float* y_t = scratch.conv_out.data() + (size_t)t * qkv_dim;
        for (int c = 0; c < qkv_dim; ++c) {
            const float* w = layer.host_linear_conv.data() + (size_t)c * kernel;
            float v = 0.0f;
            for (int j = 0; j < kernel - 1; ++j) {
                if (kernel > 1) {
                    v += h_conv_state[(size_t)j * qkv_dim + c] * w[j];
                }
            }
            v += x_t[c] * w[kernel - 1];
            y_t[c] = silu(v);
        }
        if (kernel > 1) {
            if (kernel - 2 > 0) {
                std::memmove(h_conv_state.data(),
                             h_conv_state.data() + qkv_dim,
                             (size_t)(kernel - 2) * qkv_dim * sizeof(float));
            }
            std::memcpy(h_conv_state.data() + (size_t)(kernel - 2) * qkv_dim,
                        x_t, (size_t)qkv_dim * sizeof(float));
        }
    }

    scratch.q.resize((size_t)seq_len * linear_q_dim_);
    scratch.k.resize((size_t)seq_len * linear_q_dim_);
    scratch.v.resize((size_t)seq_len * linear_kv_dim_);
    for (int t = 0; t < seq_len; ++t) {
        const float* src = scratch.conv_out.data() + (size_t)t * qkv_dim;
        std::memcpy(scratch.q.data() + (size_t)t * linear_q_dim_, src, (size_t)linear_q_dim_ * sizeof(float));
        std::memcpy(scratch.k.data() + (size_t)t * linear_q_dim_, src + linear_q_dim_, (size_t)linear_q_dim_ * sizeof(float));
        std::memcpy(scratch.v.data() + (size_t)t * linear_kv_dim_, src + linear_q_dim_ + linear_q_dim_,
                    (size_t)linear_kv_dim_ * sizeof(float));
    }

    head_l2_norm_inplace(scratch.q, seq_len, linear_q_heads_, head_k_dim, cfg_.rms_norm_eps);
    head_l2_norm_inplace(scratch.k, seq_len, linear_q_heads_, head_k_dim, cfg_.rms_norm_eps);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    scratch.out.assign((size_t)seq_len * linear_kv_dim_, 0.0f);
    scratch.sk.resize((size_t)head_v_dim);
    scratch.d.resize((size_t)head_v_dim);

    for (int t = 0; t < seq_len; ++t) {
        for (int hv = 0; hv < num_heads; ++hv) {
            int hk = linear_delta_key_head_for_value_head(hv, linear_q_heads_, num_heads);
            const float* q_h = scratch.q.data() + (size_t)t * linear_q_dim_ + (size_t)hk * head_k_dim;
            const float* k_h = scratch.k.data() + (size_t)t * linear_q_dim_ + (size_t)hk * head_k_dim;
            const float* v_h = scratch.v.data() + (size_t)t * linear_kv_dim_ + (size_t)hv * head_v_dim;

            float beta = 1.0f / (1.0f + std::exp(-scratch.b[(size_t)t * num_heads + hv]));
            float alpha = scratch.a[(size_t)t * num_heads + hv];
            float g_pre = softplus(alpha + layer.host_linear_dt_bias[(size_t)hv]) *
                          layer.host_linear_A_negexp[(size_t)hv];
            float decay = std::exp(g_pre);

            // S: [key_dim, value_dim] per head (row-major)
            float* S = state.data() + ((size_t)hv * head_k_dim * head_v_dim);
            for (int idx = 0; idx < head_k_dim * head_v_dim; ++idx) S[idx] *= decay;

            for (int dv = 0; dv < head_v_dim; ++dv) {
                float acc = 0.0f;
                for (int j = 0; j < head_k_dim; ++j) {
                    acc += S[(size_t)j * head_v_dim + dv] * k_h[j];
                }
                scratch.sk[(size_t)dv] = acc;
                scratch.d[(size_t)dv] = (v_h[dv] - acc) * beta;
            }

            for (int j = 0; j < head_k_dim; ++j) {
                float kj = k_h[j];
                float* row = S + (size_t)j * head_v_dim;
                for (int dv = 0; dv < head_v_dim; ++dv) {
                    row[dv] += kj * scratch.d[(size_t)dv];
                }
            }

            float* o_h = scratch.out.data() + (size_t)t * linear_kv_dim_ + (size_t)hv * head_v_dim;
            for (int dv = 0; dv < head_v_dim; ++dv) {
                float acc = 0.0f;
                for (int j = 0; j < head_k_dim; ++j) {
                    acc += S[(size_t)j * head_v_dim + dv] * q_h[j];
                }
                o_h[dv] = acc * scale;
            }
        }
    }

    head_rms_norm_and_silu_gate(scratch.out, layer.host_linear_norm, scratch.z, seq_len, num_heads,
                                head_v_dim, cfg_.rms_norm_eps);

    scratch.out_bf16.resize(scratch.out.size());
    for (size_t i = 0; i < scratch.out.size(); ++i) scratch.out_bf16[i] = bf16(scratch.out[i]);
    ctx.memcpy_h2d(out_dst.data(), scratch.out_bf16.data(), scratch.out_bf16.size() * sizeof(bf16));

    if (cache.linear_state.valid() && !cache.host_linear_state.empty()) {
        ctx.memcpy_h2d(cache.linear_state.data(), cache.host_linear_state.data(),
                       cache.host_linear_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !cache.host_linear_conv_state.empty()) {
        ctx.memcpy_h2d(cache.linear_conv_state.data(), cache.host_linear_conv_state.data(),
                       cache.host_linear_conv_state.size() * sizeof(float));
    }
    cache.device_state_dirty = false;
}

struct Qwen35HybridTextBackend::ProjectionPolicy {
    Qwen35HybridTextBackend& model;

    bf16* ResidualHidden() const { return nullptr; }

    template <aila::models::HybridProjectionSlot Slot>
    void Forward(Context& ctx, Layer& layer, Tensor& input, Tensor& output, int count) const {
        using SlotKind = aila::models::HybridProjectionSlot;
        if constexpr (Slot == SlotKind::FfnDown) {
            if (model.use_decode_ffn_custom_path(count)) {
                model.run_decode_ffn_down_custom(ctx, layer, input, output);
                return;
            }
        }
        // Existing transposed JM weights belong only to the BF16 storage
        // backend. The fallback continues to use the original Linear object.
        Tensor* jmWeight = nullptr;
        int inputWidth = 0, outputWidth = 0;
        if constexpr (Slot == SlotKind::LinearAll) {
            jmWeight = layer.linear_all_weight_jm;
            inputWidth = model.hidden_size_;
            outputWidth = model.linear_all_dim_;
        } else if constexpr (Slot == SlotKind::LinearOutput) {
            jmWeight = layer.linear_o_weight_jm;
            inputWidth = model.linear_kv_dim_;
            outputWidth = model.hidden_size_;
        } else if constexpr (Slot == SlotKind::FullQkv) {
            jmWeight = layer.qkv_weight_jm;
            inputWidth = model.hidden_size_;
            outputWidth = model.full_fused_qkv_dim_;
        } else if constexpr (Slot == SlotKind::FullOutput) {
            jmWeight = layer.o_weight_jm;
            inputWidth = model.full_q_dim_;
            outputWidth = model.hidden_size_;
        }
        if (jmWeight && model.use_decode_jm_custom_path(count)) {
            model.run_decode_jm_matvec_custom(ctx, input, *jmWeight, inputWidth, outputWidth, output);
            return;
        }
        aila::models::ProjectionFor<Slot>(layer).forward(ctx, input, output, count);
    }

    template <aila::models::HybridProjectionSlot Slot>
    void ForwardDecodeRows(Context& ctx, Layer& layer, Tensor& input, Tensor& output, int count) const {
        using SlotKind = aila::models::HybridProjectionSlot;
        int inputWidth = model.hidden_size_, outputWidth = model.hidden_size_;
        if constexpr (Slot == SlotKind::LinearAll) outputWidth = model.linear_all_dim_;
        else if constexpr (Slot == SlotKind::LinearOutput) inputWidth = model.linear_kv_dim_;
        else if constexpr (Slot == SlotKind::FullQkv) outputWidth = model.full_fused_qkv_dim_;
        else if constexpr (Slot == SlotKind::FullOutput) inputWidth = model.full_q_dim_;
        else {
            static_assert(Slot == SlotKind::FfnDown, "Unsupported batched decode projection slot");
            inputWidth = model.ff_dim_;
        }
        // Preserve the BF16 backend's decode JM/custom selection. A batch of
        // independent decode rows is not a multi-token prefill operation, so a
        // row still runs the original count=1 kernel body and reduction order.
        if (count > 1) {
            Tensor* jmWeight = nullptr;
            if constexpr (Slot == SlotKind::LinearAll) jmWeight = layer.linear_all_weight_jm;
            else if constexpr (Slot == SlotKind::LinearOutput) jmWeight = layer.linear_o_weight_jm;
            else if constexpr (Slot == SlotKind::FullQkv) jmWeight = layer.qkv_weight_jm;
            else if constexpr (Slot == SlotKind::FullOutput) jmWeight = layer.o_weight_jm;
            else jmWeight = layer.down_weight_jm;
            if (jmWeight && model.use_decode_jm_custom_path(1)) {
                model.run_decode_jm_matvec_rows(ctx, input, *jmWeight, inputWidth, outputWidth, output, count);
                return;
            }
        }
        for (int row = 0; row < count; ++row) {
            auto source = Tensor::view(ctx, input.data_as<bf16>() + static_cast<size_t>(row) * inputWidth,
                                       {1, static_cast<int64_t>(inputWidth)}, input.dtype());
            auto target = Tensor::view(ctx, output.data_as<bf16>() + static_cast<size_t>(row) * outputWidth,
                                       {1, static_cast<int64_t>(outputWidth)}, output.dtype());
            Forward<Slot>(ctx, layer, source, target, 1);
        }
    }

    // Called once per row with a single-row view, so this stays the original
    // fused helper. The width argument is the FFN dimension, not a row count.
    bool TryFusedGateUp(Context& ctx, Layer& layer, Tensor& input, Tensor& output, int) const {
        if (!model.use_decode_ffn_custom_path(1)) return false;
        model.run_decode_ffn_gate_up_swiglu_custom(ctx, layer, input, output);
        return true;
    }

    // Multi-row gate/up. The rows kernel applies each row's own SiLU, so the
    // caller must not run the single-row fused helper afterwards. Layout matches
    // the per-row path: row stride is hidden_size_ for input and ff_dim_ for the
    // gate output.
    bool TryFusedGateUpRows(Context& ctx, Layer& layer, Tensor& input, Tensor& output, int rows) const {
        if (rows < 2 || !layer.gate_up_weight_jm) return false;
        if (!model.use_decode_ffn_custom_path(1)) return false;
        model.run_decode_ffn_gate_up_swiglu_rows(ctx, layer, input, output, rows);
        return true;
    }
};

void Qwen35HybridTextBackend::ExecuteLayer(
    const LayerPlan& plan, const LayerExecution& execution) {
    aila::models::HybridLayerExecutor::Execute(*this, plan, execution, ProjectionPolicy{*this});
}

bool Qwen35HybridTextBackend::SupportsDecodeBatch() const noexcept {
    return aila::models::HybridDecodeBatchExecutor::Supported(*this);
}

Tensor& Qwen35HybridTextBackend::ForwardDecodeBatch(
    Context& ctx, const std::vector<Session*>& sessions, const int* tokenIdsDevice) {
    const bool grouped = linear_kv_heads_ > linear_q_heads_;
    const bool defaultFastpath = supports_default_grouped_linear_delta_fastpath(
        linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_, std::max(0, linear_conv_kernel_dim_ - 1));
    static const bool experimental = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU", false);
    static const bool forceHost = aila::env::read_flag("AILA_Q35_FORCE_HOST_GROUPED_LINEAR", false);
    const bool useHost = grouped && (forceHost || (!defaultFastpath && !experimental));
    return aila::models::HybridDecodeBatchExecutor::Execute(
        *this, ctx, sessions, tokenIdsDevice, ProjectionPolicy{*this}, useHost);
}

Tensor& Qwen35HybridTextBackend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    return ForwardImpl(ctx, DefaultSession(ctx), token_ids_device, seq_len, false);
}

Tensor& Qwen35HybridTextBackend::Forward(
    Context& ctx, Session& session, const int* token_ids_device, int seq_len) {
    return ForwardImpl(ctx, session, token_ids_device, seq_len, true);
}

Tensor& Qwen35HybridTextBackend::ForwardLegacyLogits(
    Context& ctx, Session& session, const int* token_ids_device, int seq_len) {
    return ForwardImpl(ctx, session, token_ids_device, seq_len, false);
}

Tensor& Qwen35HybridTextBackend::ForwardImpl(
    Context& ctx, Session& session, const int* token_ids_device, int seq_len, bool lastTokenOnly) {
    ValidateSession(session);
    if (&ctx != context_) {
        throw std::invalid_argument(
            "Qwen35HybridTextBackend::Forward: context does not match the loaded model");
    }
    if (seq_len <= 0 || !token_ids_device) {
        throw std::invalid_argument(
            "Qwen35HybridTextBackend::Forward: token IDs must be non-null and count positive");
    }
    const int start_pos = session.currentLength_;
    if (seq_len > session.Capacity() - start_pos) {
        throw std::invalid_argument("Qwen35HybridTextBackend::Forward: context window exceeded");
    }
    const int cached_len = start_pos + seq_len;
    static const bool s_profile_decode = aila::env::read_flag("AILA_PROFILE_Q35_DECODE", false);
    static const bool s_profile_prefill = aila::env::read_flag("AILA_PROFILE_Q35_PREFILL", false);
    static const int s_decode_every = std::max(1, aila::env::read_int_raw("AILA_PROFILE_Q35_DECODE_EVERY", 32));
    static const int s_prefill_every = std::max(1, aila::env::read_int_raw("AILA_PROFILE_Q35_PREFILL_EVERY", 1));
    static const bool s_profile_host_only = aila::env::read_flag("AILA_PROFILE_Q35_HOST_ONLY", false);
    bool profile_decode = (seq_len == 1) && s_profile_decode;
    bool profile_prefill = (seq_len > 1) && s_profile_prefill;
    int profile_every = profile_decode ? s_decode_every : s_prefill_every;
    bool profile_enabled = profile_decode || profile_prefill;
    std::array<double, static_cast<size_t>(ProfileStage::Count)> stage_ms{};
    bool profile_host_only = profile_enabled && s_profile_host_only;
    const StageProfiler time_stage{ctx, stage_ms, profile_enabled, profile_host_only};

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) {
            ensure_prefill_scores(ctx, seq_len);
        } else {
            ensure_incr_prefill_scores(ctx, seq_len, cached_len);
        }
    }

    time_stage(ProfileStage::EmbedNorm, [&] {
        ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, hidden_size_);
        if (!session.embeddingOverridePositions_.empty() && session.embeddingOverrideHiddenSize_ == hidden_size_) {
            bf16* hidden_ptr = static_cast<bf16*>(buf_.hidden.data());
            for (size_t i = 0; i < session.embeddingOverridePositions_.size(); ++i) {
                int pos = session.embeddingOverridePositions_[i];
                if (pos < start_pos || pos >= start_pos + seq_len) continue;
                int local_row = pos - start_pos;
                const bf16* src = session.embeddingOverrideValues_.data() + i * static_cast<size_t>(hidden_size_);
                bf16* dst = hidden_ptr + static_cast<size_t>(local_row) * static_cast<size_t>(hidden_size_);
                ctx.queue().memcpy(dst, src, static_cast<size_t>(hidden_size_) * sizeof(bf16));
            }
        }
        ops::rms_norm(ctx, buf_.hidden, *decoderPlan_->initialNormWeight, cfg_.rms_norm_eps,
                      buf_.normed, seq_len, hidden_size_);
    });

    int full_rotary_dim = std::max(2, (int)std::floor(full_head_dim_ * cfg_.rope.partial_rotary_factor));
    full_rotary_dim = std::min(full_head_dim_, full_rotary_dim);
    if (full_rotary_dim & 1) --full_rotary_dim;
    if (full_rotary_dim <= 0) full_rotary_dim = std::min(2, full_head_dim_);
    static const bool s_debug_layer_stats = aila::env::read_flag("AILA_DEBUG_Q35_LAYER_STATS", false);
    static const int s_debug_layer_detail = aila::env::read_int_raw("AILA_DEBUG_Q35_LAYER_DETAIL", -1);
    static const bool s_debug_linear_compare = aila::env::read_flag("AILA_DEBUG_Q35_LINEAR_COMPARE", false);
    static const int s_debug_linear_compare_layer = aila::env::read_int_raw("AILA_DEBUG_Q35_LINEAR_COMPARE_LAYER", -1);
    static const bool s_experimental_grouped_linear_gpu = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU", false);
    static const bool s_force_host_grouped_linear = aila::env::read_flag("AILA_Q35_FORCE_HOST_GROUPED_LINEAR", false);
    bool debug_layer_stats = s_debug_layer_stats;
    int debug_layer_detail_idx = s_debug_layer_detail;
    bool debug_linear_compare = s_debug_linear_compare;
    int debug_linear_compare_layer = s_debug_linear_compare_layer;
    bool grouped_linear_heads = linear_kv_heads_ > linear_q_heads_;
    int linear_conv_rows = std::max(0, linear_conv_kernel_dim_ - 1);
    bool default_grouped_linear_fastpath = supports_default_grouped_linear_delta_fastpath(
        linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_, linear_conv_rows);
    bool experimental_grouped_linear_gpu = s_experimental_grouped_linear_gpu;
    bool force_host_grouped_linear = s_force_host_grouped_linear;
    bool use_grouped_linear_gpu = grouped_linear_heads && !force_host_grouped_linear &&
                                  (default_grouped_linear_fastpath || experimental_grouped_linear_gpu);
    bool use_host_grouped_linear = grouped_linear_heads && !use_grouped_linear_gpu;
    const auto phase = seq_len == 1 ? ExecutionPhase::Decode : ExecutionPhase::Prefill;
    for (const auto& plan : decoderPlan_->layers) {
        const LayerExecution execution{
            ctx, buf_, session.layerCaches_[static_cast<size_t>(plan.cacheIndex)],
            phase, start_pos, seq_len, cached_len, session.Capacity(), full_rotary_dim,
            static_cast<const int*>(session.mropePosT_.data()),
            static_cast<const int*>(session.mropePosH_.data()),
            static_cast<const int*>(session.mropePosW_.data()),
            session.mropePromptLength_, session.mropeTextPositionDelta_,
            debug_layer_stats, debug_layer_detail_idx, debug_linear_compare,
            debug_linear_compare_layer, use_host_grouped_linear, time_stage, {}
        };
        ExecuteLayer(plan, execution);
    }

    int output_rows = 1;
    time_stage(ProfileStage::LmHead, [&] {
        if (seq_len > 1) {
            if (start_pos > 0 && seq_len <= 16) {
                output_rows = seq_len;
                if (!buf_.logits.valid() || buf_.logits.shape(0) < seq_len) {
                    if (buf_.logits.valid()) ctx.synchronize();
                    buf_.logits = Tensor::allocate(ctx, {seq_len, (int64_t)cfg_.vocab_size});
                }
                lm_head_.forward(ctx, buf_.normed, buf_.logits, seq_len);
            } else {
                bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * hidden_size_;
                Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)hidden_size_});
                lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
            }
        } else {
            if (!buf_.logits.valid() || buf_.logits.shape(0) < 1) {
                buf_.logits = Tensor::allocate(ctx, {1, (int64_t)cfg_.vocab_size});
            }
            lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
        }
    });

    if (profile_enabled) {
        StageProfileTotals& totals = profile_decode ? profile_->decode : profile_->prefill;
        for (size_t i = 0; i < stage_ms.size(); ++i) {
            totals.stage_ms[i] += stage_ms[i];
        }
        totals.calls += 1;
        totals.tokens += profile_decode ? 1 : seq_len;

        if (totals.calls >= profile_every) {
            auto avg = [&](ProfileStage stage) {
                return totals.stage_ms[static_cast<size_t>(stage)] /
                       static_cast<double>(std::max(1, totals.calls));
            };
            auto per_token = [&](ProfileStage stage) {
                return totals.stage_ms[static_cast<size_t>(stage)] /
                       static_cast<double>(std::max(1, totals.tokens));
            };
            double total_ms = 0.0;
            for (double v : totals.stage_ms) total_ms += v;
            if (profile_decode) {
                AILA_LOG_INFO(
                    "[Q35DecodeProfile] tokens=%d total=%.3f embed=%.3f linear_proj=%.3f linear_delta=%.3f linear_o=%.3f "
                    "full_qkv=%.3f full_split=%.3f qk_rope=%.3f kv_copy=%.3f attn=%.3f attn_gate=%.3f full_o=%.3f "
                    "post_attn=%.3f ffn_proj=%.3f ffn_act=%.3f down=%.3f post_mlp=%.3f lm_head=%.3f",
                    totals.tokens,
                    total_ms / static_cast<double>(std::max(1, totals.tokens)),
                    avg(ProfileStage::EmbedNorm),
                    avg(ProfileStage::LinearProj),
                    avg(ProfileStage::LinearDelta),
                    avg(ProfileStage::LinearOProj),
                    avg(ProfileStage::FullQkvProj),
                    avg(ProfileStage::FullSplit),
                    avg(ProfileStage::QkNormRope),
                    avg(ProfileStage::KvCache),
                    avg(ProfileStage::Attention),
                    avg(ProfileStage::AttnGate),
                    avg(ProfileStage::FullOProj),
                    avg(ProfileStage::PostAttnNorm),
                    avg(ProfileStage::FfnProj),
                    avg(ProfileStage::FfnAct),
                    avg(ProfileStage::DownProj),
                    avg(ProfileStage::PostMlpNorm),
                    avg(ProfileStage::LmHead));
            } else {
                AILA_LOG_INFO(
                    "[Q35PrefillProfile] calls=%d tokens=%d total=%.3f total_tok=%.6f "
                    "embed=%.6f linear_proj=%.6f linear_delta=%.6f linear_o=%.6f "
                    "full_qkv=%.6f full_split=%.6f qk_rope=%.6f kv_copy=%.6f attn=%.6f attn_gate=%.6f full_o=%.6f "
                    "post_attn=%.6f ffn_proj=%.6f ffn_act=%.6f down=%.6f post_mlp=%.6f lm_head=%.6f",
                    totals.calls,
                    totals.tokens,
                    total_ms / static_cast<double>(std::max(1, totals.calls)),
                    total_ms / static_cast<double>(std::max(1, totals.tokens)),
                    per_token(ProfileStage::EmbedNorm),
                    per_token(ProfileStage::LinearProj),
                    per_token(ProfileStage::LinearDelta),
                    per_token(ProfileStage::LinearOProj),
                    per_token(ProfileStage::FullQkvProj),
                    per_token(ProfileStage::FullSplit),
                    per_token(ProfileStage::QkNormRope),
                    per_token(ProfileStage::KvCache),
                    per_token(ProfileStage::Attention),
                    per_token(ProfileStage::AttnGate),
                    per_token(ProfileStage::FullOProj),
                    per_token(ProfileStage::PostAttnNorm),
                    per_token(ProfileStage::FfnProj),
                    per_token(ProfileStage::FfnAct),
                    per_token(ProfileStage::DownProj),
                    per_token(ProfileStage::PostMlpNorm),
                    per_token(ProfileStage::LmHead));
            }
            totals.reset();
        }
    }

    CaptureSnapshot(ctx, session, cached_len, seq_len);
    session.currentLength_ = cached_len;
    if (lastTokenOnly) {
        auto* last = static_cast<bf16*>(buf_.logits.data()) +
            static_cast<size_t>(output_rows - 1) * static_cast<size_t>(cfg_.vocab_size);
        lastTokenLogits_ = Tensor::view(
            ctx, last, {1, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
        return lastTokenLogits_;
    }
    return buf_.logits;
}

void Qwen35HybridTextBackend::CaptureSnapshot(
    Context& ctx, Session& session, int nextLength, int tokenCount) {
    if (!decoderPlan_->hasRecurrentLayers ||
        (session.boundedSnapshots_ && session.maxSnapshots_ == 0)) {
        return;
    }
    int step = aila::env::g_q35_prefill_step_override;
    if (step <= 0) step = aila::env::read_int("AILA_Q35_PREFILL_STEP", 64);
    if (step <= 0) step = 64;
    if (tokenCount <= 1 && nextLength % step != 0) return;

    // Preserve the existing checkpoint boundary. No per-layer synchronization
    // is introduced into normal execution.
    ctx.synchronize();
    auto& snapshots = session.snapshots_;
    if (session.boundedSnapshots_ && snapshots.find(nextLength) == snapshots.end() &&
        snapshots.size() >= session.maxSnapshots_) {
        const auto oldest = std::min_element(
            snapshots.begin(), snapshots.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
        snapshots.erase(oldest);
    }
    // Evict before copying so snapshot payload never briefly exceeds its quota.
    auto& snapshot = snapshots[nextLength];
    try {
        snapshot.layers.resize(session.layerCaches_.size());
        for (const auto& plan : decoderPlan_->layers) {
            if (plan.kind != LayerKind::DeltaNet) continue;
            const auto index = static_cast<size_t>(plan.cacheIndex);
            auto& cache = session.layerCaches_[index];
            if (cache.device_state_dirty) {
                if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
                    ctx.memcpy_d2h(cache.host_linear_state.data(), cache.linear_state.data(),
                                   cache.host_linear_state.size() * sizeof(float));
                }
                if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
                    ctx.memcpy_d2h(cache.host_linear_conv_state.data(), cache.linear_conv_state.data(),
                                   cache.host_linear_conv_state.size() * sizeof(float));
                }
                cache.device_state_dirty = false;
            }
            snapshot.layers[index].linear_state_host = cache.host_linear_state;
            snapshot.layers[index].linear_conv_state_host = cache.host_linear_conv_state;
            snapshot.layers[index].linear_conv_head = cache.linear_conv_head;
        }
    } catch (...) {
        // A partial snapshot must never become a valid rollback checkpoint.
        snapshots.erase(nextLength);
        throw;
    }
}

void Qwen35HybridTextBackend::reset() {
    if (defaultSession_) Reset(*defaultSession_);
}

bool Qwen35HybridTextBackend::truncate_kv_cache(int new_len) {
    // Preserve the legacy bool API: a larger requested length is a no-op;
    // resetting/failing to find a recurrent checkpoint reports false.
    if (new_len >= get_current_context_len()) return true;
    if (new_len <= 0) {
        reset();
        return false;
    }
    return Truncate(*defaultSession_, new_len) > 0;
}

void Qwen35HybridTextBackend::Reset(Session& session) {
    ValidateSession(session);
    session.context_->synchronize();
    session.snapshots_.clear();
    for (auto& cache : session.layerCaches_) {
        if (cache.linear_state.valid()) {
            session.context_->queue().memset(
                cache.linear_state.data(), 0, cache.linear_state.size_bytes());
        }
        if (cache.linear_conv_state.valid()) {
            session.context_->queue().memset(
                cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
        }
        std::fill(cache.host_linear_state.begin(), cache.host_linear_state.end(), 0.0f);
        std::fill(cache.host_linear_conv_state.begin(), cache.host_linear_conv_state.end(), 0.0f);
        cache.device_state_dirty = false;
        cache.linear_conv_head = 0;
    }
    session.currentLength_ = 0;
}

int Qwen35HybridTextBackend::Truncate(Session& session, int newLength) {
    ValidateSession(session);
    if (newLength < 0 || newLength > session.CurrentLength()) {
        throw std::invalid_argument(
            "Qwen35HybridTextBackend::Truncate: length must be between zero and the current length");
    }
    if (newLength == session.CurrentLength()) return newLength;
    if (newLength == 0) {
        Reset(session);
        return 0;
    }
    if (!decoderPlan_->hasRecurrentLayers) {
        session.currentLength_ = newLength;
        return newLength;
    }

    int bestCheckpoint = -1;
    for (const auto& [length, snapshot] : session.snapshots_) {
        if (length <= newLength && length > bestCheckpoint) bestCheckpoint = length;
    }
    if (bestCheckpoint < 0) {
        AILA_LOG_WARN(
            "[Qwen35HybridTextBackend] No suitable checkpoint found for truncate to %d, falling back to full reset",
            newLength);
        Reset(session);
        return 0;
    }

    // Host mirrors may be borrowed by earlier asynchronous operations.
    session.context_->synchronize();
    const auto& snapshot = session.snapshots_.at(bestCheckpoint);
    for (const auto& plan : decoderPlan_->layers) {
        if (plan.kind != LayerKind::DeltaNet) continue;
        const auto index = static_cast<size_t>(plan.cacheIndex);
        auto& cache = session.layerCaches_[index];
        const auto& saved = snapshot.layers[index];
        std::copy(saved.linear_state_host.begin(), saved.linear_state_host.end(),
                  cache.host_linear_state.begin());
        std::copy(saved.linear_conv_state_host.begin(), saved.linear_conv_state_host.end(),
                  cache.host_linear_conv_state.begin());
        cache.linear_conv_head = saved.linear_conv_head;
        if (cache.linear_state.valid() && !cache.host_linear_state.empty()) {
            session.context_->queue().memcpy(
                cache.linear_state.data(), cache.host_linear_state.data(),
                cache.host_linear_state.size() * sizeof(float));
        }
        if (cache.linear_conv_state.valid() && !cache.host_linear_conv_state.empty()) {
            session.context_->queue().memcpy(
                cache.linear_conv_state.data(), cache.host_linear_conv_state.data(),
                cache.host_linear_conv_state.size() * sizeof(float));
        }
        cache.device_state_dirty = false;
    }
    // Drain once after all uploads, even if layer zero is a full-attention layer.
    // Returning only after the host borrows end makes immediate reset/destruction safe.
    session.context_->synchronize();
    session.currentLength_ = bestCheckpoint;
    for (auto it = session.snapshots_.begin(); it != session.snapshots_.end();) {
        if (it->first > bestCheckpoint) it = session.snapshots_.erase(it);
        else ++it;
    }
    AILA_LOG_INFO(
        "[Qwen35HybridTextBackend] Restored Mamba recurrent state from checkpoint %d (requested truncate to %d)",
        bestCheckpoint, newLength);
    return bestCheckpoint;
}
