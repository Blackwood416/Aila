#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_map>
#include <functional>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

namespace {

std::mt19937& sampling_rng() {
    static thread_local std::mt19937 rng(42);
    return rng;
}

struct TopCandidate {
    float logit;
    int id;
};

struct MinHeapByLogit {
    bool operator()(const TopCandidate& a, const TopCandidate& b) const {
        // std::make_heap with this comparator forms a min-heap (front = smallest logit)
        return a.logit > b.logit;
    }
};

template <typename LogitGetter>
int sample_topk_topp(int vocab_size, int top_k, float top_p, LogitGetter get_logit) {
    if (vocab_size <= 0) return 0;

    int k = top_k;
    if (k <= 0 || k > vocab_size) k = vocab_size;

    static thread_local std::vector<TopCandidate> heap;
    static thread_local std::vector<TopCandidate> sorted;
    static thread_local std::vector<float> probs;
    heap.clear();
    heap.reserve(static_cast<size_t>(k));

    MinHeapByLogit cmp;
    bool heap_ready = false;

    for (int i = 0; i < vocab_size; ++i) {
        float v = get_logit(i);
        if (static_cast<int>(heap.size()) < k) {
            heap.push_back({v, i});
            if (static_cast<int>(heap.size()) == k) {
                std::make_heap(heap.begin(), heap.end(), cmp);
                heap_ready = true;
            }
        } else if (v > heap.front().logit) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {v, i};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }

    if (!heap_ready && !heap.empty()) {
        std::make_heap(heap.begin(), heap.end(), cmp);
        heap_ready = true;
    }
    if (heap.empty()) return 0;

    sorted = heap;
    std::sort(sorted.begin(), sorted.end(),
              [](const TopCandidate& a, const TopCandidate& b) {
                  return a.logit > b.logit;
              });

    float max_val = sorted[0].logit;
    probs.resize(sorted.size());
    float sum = 0.0f;
    for (size_t i = 0; i < sorted.size(); ++i) {
        float p = std::exp(sorted[i].logit - max_val);
        probs[i] = p;
        sum += p;
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        return sorted[0].id;
    }
    for (float& p : probs) {
        p /= sum;
    }

    float p_cut = std::clamp(top_p, 1e-6f, 1.0f);
    size_t keep = probs.size();
    if (p_cut < 0.999999f) {
        float cdf = 0.0f;
        keep = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cdf += probs[i];
            keep = i + 1;
            if (cdf >= p_cut) {
                break;
            }
        }
        if (keep == 0) keep = 1;

        float kept_sum = 0.0f;
        for (size_t i = 0; i < keep; ++i) {
            kept_sum += probs[i];
        }
        if (kept_sum > 0.0f) {
            for (size_t i = 0; i < keep; ++i) {
                probs[i] /= kept_sum;
            }
        }
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(sampling_rng());

    float cumsum = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return sorted[i].id;
        }
    }
    return sorted[keep - 1].id;
}

} // namespace

void set_sampling_seed(uint64_t seed) {
    sampling_rng().seed(static_cast<uint32_t>(seed));
}

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
    static thread_local std::vector<bf16> host_logits_bf16;
    host_logits_bf16.resize(static_cast<size_t>(vocab_size));
    ctx.memcpy_d2h(host_logits_bf16.data(), logits.data(), vocab_size * sizeof(bf16));
    float inv_temperature = 1.0f / std::max(temperature, 1e-6f);
    return sample_topk_topp(vocab_size, top_k, 1.0f,
                            [&](int i) { return static_cast<float>(host_logits_bf16[i]) * inv_temperature; });
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
    float inv_temperature = 1.0f / std::max(gen_config.temperature, 1e-6f);
    int top_k = std::min(gen_config.top_k, vocab_size);

    static thread_local std::vector<bf16> host_logits_bf16;
    host_logits_bf16.resize(static_cast<size_t>(vocab_size));
    ctx.memcpy_d2h(host_logits_bf16.data(), logits.data(), vocab_size * sizeof(bf16));

    if (!gen_config.do_sample) {
        // Greedy: CPU argmax
        int best_idx = 0;
        float best_val = static_cast<float>(host_logits_bf16[0]);
        for (int i = 1; i < vocab_size; i++) {
            float v = static_cast<float>(host_logits_bf16[i]);
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Fast path: sampling without penalties (CPU fallback)
    if (!gen_config.has_penalties()) {
        return sample_topk_topp(vocab_size, top_k, gen_config.top_p,
            [&](int i) { return static_cast<float>(host_logits_bf16[i]) * inv_temperature; });
    }

    // Generic path: penalties on CPU float logits
    static thread_local std::vector<float> logits_f;
    logits_f.resize(static_cast<size_t>(vocab_size));
    for (int i = 0; i < vocab_size; i++) {
        logits_f[i] = static_cast<float>(host_logits_bf16[i]);
    }
    apply_penalties(logits_f.data(), vocab_size, generated_ids,
                    gen_config.repetition_penalty,
                    gen_config.presence_penalty,
                    gen_config.frequency_penalty);
    for (int i = 0; i < vocab_size; ++i) {
        logits_f[i] *= inv_temperature;
    }
    return sample_topk_topp(vocab_size, top_k, gen_config.top_p,
        [&](int i) { return logits_f[i]; });
}

} // namespace ops
