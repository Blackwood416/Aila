#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: RMSNorm
// output[s][h] = input[s][h] / sqrt(mean(input[s][:]^2) + eps) * weight[h]
// ============================================================

void rms_norm(Context& ctx, Tensor& input, Tensor& weight,
               float eps, Tensor& output, int seq_len, int hidden_size) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* w_ptr = static_cast<bf16*>(weight.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    int wg_size = 256; // Intel A770 倾向于较大的 WG 以掩盖延迟

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(seq_len * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                auto sg = item.get_sub_group();
                int wg = item.get_local_range(0);

                float local_sum_sq = 0.0f;
                // 每个线程负责一部分数据的累加
                for (int h = lid; h < hidden_size; h += wg) {
                    float val = static_cast<float>(in_ptr[row * hidden_size + h]);
                    local_sum_sq += val * val;
                }

                // 使用 Sub-group 原语快速规约
                float group_sum = sycl::reduce_over_group(item.get_group(), local_sum_sq, sycl::plus<>());
                float rms = sycl::sqrt(group_sum / static_cast<float>(hidden_size) + eps);

                // 应用归一化
                for (int h = lid; h < hidden_size; h += wg) {
                    float val = static_cast<float>(in_ptr[row * hidden_size + h]);
                    float w = static_cast<float>(w_ptr[h]);
                    out_ptr[row * hidden_size + h] = bf16(val / rms * w);
                }
            });
    });
}

void fused_add_rms_norm(Context& ctx, Tensor& input, Tensor& residual,
                         Tensor& weight, float eps, Tensor& output,
                         int seq_len, int hidden_size) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* res_ptr = static_cast<bf16*>(residual.data());
    bf16* w_ptr = static_cast<bf16*>(weight.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    int wg_size = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(seq_len * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                int wg = item.get_local_range(0);

                float local_sum_sq = 0.0f;
                for (int h = lid; h < hidden_size; h += wg) {
                    float val = static_cast<float>(in_ptr[row * hidden_size + h]) +
                                static_cast<float>(res_ptr[row * hidden_size + h]);
                    in_ptr[row * hidden_size + h] = bf16(val);
                    local_sum_sq += val * val;
                }

                float group_sum = sycl::reduce_over_group(item.get_group(), local_sum_sq, sycl::plus<>());
                float rms = sycl::sqrt(group_sum / static_cast<float>(hidden_size) + eps);

                for (int h = lid; h < hidden_size; h += wg) {
                    float val = static_cast<float>(in_ptr[row * hidden_size + h]);
                    float w = static_cast<float>(w_ptr[h]);
                    out_ptr[row * hidden_size + h] = bf16(val / rms * w);
                }
            });
    });
}

// ============================================================
// SYCL Kernel: Per-head RMSNorm (Qwen3 QK-norm)
// ============================================================

void head_rms_norm(Context& ctx, Tensor& x, Tensor& weight,
                    float eps, int seq_len, int num_heads, int head_dim) {
    bf16* x_ptr = static_cast<bf16*>(x.data());
    bf16* w_bf16_ptr = nullptr;
    float* w_f32_ptr = nullptr;
    if (weight.dtype() == dnnl::memory::data_type::f32) {
        w_f32_ptr = static_cast<float*>(weight.data());
    } else {
        w_bf16_ptr = static_cast<bf16*>(weight.data());
    }

    int wg_size = (head_dim > 128 ? 256 : 128);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(seq_len * num_heads * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int group_idx = item.get_group(0);
                int lid = item.get_local_id(0);
                int base = group_idx * head_dim;

                float sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float val = static_cast<float>(x_ptr[base + d]);
                    sum_sq += val * val;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) local_sum[lid] += local_sum[lid + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                for (int d = lid; d < head_dim; d += wg_size) {
                    float val = static_cast<float>(x_ptr[base + d]);
                    float w = w_f32_ptr ? w_f32_ptr[d] : static_cast<float>(w_bf16_ptr[d]);
                    x_ptr[base + d] = bf16(val / rms * w);
                }
            });
    });
}

} // namespace ops
