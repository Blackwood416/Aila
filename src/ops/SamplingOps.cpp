#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_map>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: Argmax (GPU Side)
// ============================================================

void argmax(Context& ctx, Tensor& logits, int vocab_size, int* d_result) {
    bf16* l_ptr = static_cast<bf16*>(logits.data());
    
    int wg_size = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(wg_size), cgh);
        sycl::local_accessor<int, 1> local_idx(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
            int lid = item.get_local_id(0);
            float max_val = -1e30f;
            int max_idx = 0;

            for (int i = lid; i < vocab_size; i += wg_size) {
                float v = static_cast<float>(l_ptr[i]);
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            local_max[lid] = max_val;
            local_idx[lid] = max_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    if (local_max[lid + stride] > local_max[lid]) {
                        local_max[lid] = local_max[lid + stride];
                        local_idx[lid] = local_idx[lid + stride];
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) *d_result = local_idx[0];
        });
    }); 
}

int argmax(Context& ctx, Tensor& logits, int vocab_size) {
    bf16* l_ptr = static_cast<bf16*>(logits.data());
    
    int wg_size = 256;
    int* d_result = static_cast<int*>(ctx.alloc_device(sizeof(int)));
    int h_result = 0;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(wg_size), cgh);
        sycl::local_accessor<int, 1> local_idx(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
            int lid = item.get_local_id(0);
            float max_val = -1e30f;
            int max_idx = 0;

            for (int i = lid; i < vocab_size; i += wg_size) {
                float v = static_cast<float>(l_ptr[i]);
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            local_max[lid] = max_val;
            local_idx[lid] = max_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    if (local_max[lid + stride] > local_max[lid]) {
                        local_max[lid] = local_max[lid + stride];
                        local_idx[lid] = local_idx[lid + stride];
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) *d_result = local_idx[0];
        });
    }).wait();

    ctx.memcpy_d2h(&h_result, d_result, sizeof(int));
    ctx.free_device(d_result);
    return h_result;
}

// ============================================================
// Top-k Sampling (CPU 端)
// ============================================================

int topk_sample(Context& ctx, Tensor& logits, int vocab_size,
                 float temperature, int top_k) {
    std::vector<bf16> host_logits_bf16(vocab_size);
    ctx.memcpy_d2h(host_logits_bf16.data(), logits.data(), vocab_size * sizeof(bf16));

    std::vector<float> logits_f(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logits_f[i] = static_cast<float>(host_logits_bf16[i]) / temperature;
    }

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int a, int b) { return logits_f[a] > logits_f[b]; });

    float max_val = logits_f[indices[0]];
    float sum = 0.0f;
    std::vector<float> probs(top_k);
    for (int i = 0; i < top_k; i++) {
        probs[i] = std::exp(logits_f[indices[i]] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) {
        probs[i] /= sum;
    }

    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return indices[i];
        }
    }
    return indices[top_k - 1];
}

// ============================================================
// Repetition / Presence / Frequency Penalty (CPU Side)
// ============================================================

void apply_penalties(float* logits_f, int vocab_size,
                     const std::vector<int>& generated_ids,
                     float repetition_penalty,
                     float presence_penalty,
                     float frequency_penalty) {

    if (generated_ids.empty()) return;
    if (repetition_penalty == 1.0f && presence_penalty == 0.0f && frequency_penalty == 0.0f) return;

    // Count token frequencies
    std::unordered_map<int, int> freq_map;
    for (int id : generated_ids) {
        if (id >= 0 && id < vocab_size) {
            freq_map[id]++;
        }
    }

    for (auto& [token_id, count] : freq_map) {
        // Repetition penalty (multiplicative)
        if (repetition_penalty != 1.0f) {
            if (logits_f[token_id] > 0.0f) {
                logits_f[token_id] /= repetition_penalty;
            } else {
                logits_f[token_id] *= repetition_penalty;
            }
        }

        // Presence penalty (additive, flat)
        logits_f[token_id] -= presence_penalty;

        // Frequency penalty (additive, proportional to count)
        logits_f[token_id] -= frequency_penalty * static_cast<float>(count);
    }
}

// ============================================================
// Unified Sampling with Penalties
// ============================================================

int sample_with_config(Context& ctx, Tensor& logits, int vocab_size,
                       const GenerationConfig& gen_config,
                       const std::vector<int>& generated_ids) {

    using bf16 = sycl::ext::oneapi::bfloat16;

    // D2H: copy logits to CPU
    std::vector<bf16> host_logits_bf16(vocab_size);
    ctx.memcpy_d2h(host_logits_bf16.data(), logits.data(), vocab_size * sizeof(bf16));

    std::vector<float> logits_f(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logits_f[i] = static_cast<float>(host_logits_bf16[i]);
    }

    // Apply penalties
    if (gen_config.has_penalties()) {
        apply_penalties(logits_f.data(), vocab_size, generated_ids,
                        gen_config.repetition_penalty,
                        gen_config.presence_penalty,
                        gen_config.frequency_penalty);
    }

    if (!gen_config.do_sample) {
        // Greedy: CPU argmax
        int best_idx = 0;
        float best_val = logits_f[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits_f[i] > best_val) {
                best_val = logits_f[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Temperature scaling
    float temperature = std::max(gen_config.temperature, 1e-6f);
    for (int i = 0; i < vocab_size; i++) {
        logits_f[i] /= temperature;
    }

    // Top-k sampling
    int top_k = std::min(gen_config.top_k, vocab_size);
    if (top_k <= 0) top_k = vocab_size;

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int a, int b) { return logits_f[a] > logits_f[b]; });

    float max_val = logits_f[indices[0]];
    float sum = 0.0f;
    std::vector<float> probs(top_k);
    for (int i = 0; i < top_k; i++) {
        probs[i] = std::exp(logits_f[indices[i]] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) {
        probs[i] /= sum;
    }

    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return indices[i];
        }
    }
    return indices[top_k - 1];
}

} // namespace ops
