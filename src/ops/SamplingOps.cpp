#include "Ops.hpp"
#include "utils/EnvUtils.hpp"
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

constexpr int kDeviceSampleFastMaxTopK = 64;
constexpr int kDeviceSampleFastWG = 64;

struct HostSamplingWorkspace {
    sycl::context sycl_ctx;
    bool has_ctx = false;
    bf16* bf16_logits = nullptr;
    size_t bf16_capacity = 0;

    ~HostSamplingWorkspace() {
        if (bf16_logits && has_ctx) {
            sycl::free(bf16_logits, sycl_ctx);
        }
    }

    bf16* ensure_bf16(Context& ctx, size_t count) {
        if (bf16_logits != nullptr && has_ctx && count <= bf16_capacity &&
            sycl_ctx == ctx.queue().get_context()) {
            return bf16_logits;
        }
        if (bf16_logits && has_ctx) {
            sycl::free(bf16_logits, sycl_ctx);
            bf16_logits = nullptr;
            bf16_capacity = 0;
        }
        sycl_ctx = ctx.queue().get_context();
        has_ctx = true;
        bf16_logits = sycl::malloc_host<bf16>(count, ctx.queue());
        if (!bf16_logits) {
            throw std::runtime_error("Pinned host allocation failed for sampling logits");
        }
        bf16_capacity = count;
        return bf16_logits;
    }
};

struct DeviceSamplingWorkspace {
    sycl::context sycl_ctx;
    bool has_ctx = false;
    float* partial_logits = nullptr;
    int* partial_ids = nullptr;
    size_t partial_capacity = 0;

    ~DeviceSamplingWorkspace() {
        if (partial_logits && has_ctx) {
            sycl::free(partial_logits, sycl_ctx);
        }
        if (partial_ids && has_ctx) {
            sycl::free(partial_ids, sycl_ctx);
        }
    }

    void ensure(Context& ctx, size_t count) {
        if (partial_logits != nullptr && partial_ids != nullptr && has_ctx &&
            count <= partial_capacity && sycl_ctx == ctx.queue().get_context()) {
            return;
        }
        if (partial_logits && has_ctx) {
            sycl::free(partial_logits, sycl_ctx);
            partial_logits = nullptr;
        }
        if (partial_ids && has_ctx) {
            sycl::free(partial_ids, sycl_ctx);
            partial_ids = nullptr;
        }
        sycl_ctx = ctx.queue().get_context();
        has_ctx = true;
        partial_logits = sycl::malloc_device<float>(count, ctx.queue());
        partial_ids = sycl::malloc_device<int>(count, ctx.queue());
        if (!partial_logits || !partial_ids) {
            throw std::runtime_error("Device allocation failed for sampling partial buffers");
        }
        partial_capacity = count;
    }
};

template <typename LogitGetter>
int sample_topk_topp(int vocab_size, int top_k, float top_p, LogitGetter get_logit) {
    if (vocab_size <= 0) return 0;

    int k = top_k;
    if (k <= 0 || k > vocab_size) k = vocab_size;

    if (k <= 64) {
        constexpr float neg_inf = -1.0e30f;
        float sorted_logits[64];
        int sorted_ids[64];
        float probs[64];

        for (int i = 0; i < 64; ++i) {
            sorted_logits[i] = neg_inf;
            sorted_ids[i] = -1;
            probs[i] = 0.0f;
        }

        auto insert_sorted = [&](float logit, int id) {
            if (logit <= sorted_logits[k - 1]) return;
            int pos = k - 1;
            while (pos > 0 && logit > sorted_logits[pos - 1]) {
                sorted_logits[pos] = sorted_logits[pos - 1];
                sorted_ids[pos] = sorted_ids[pos - 1];
                --pos;
            }
            sorted_logits[pos] = logit;
            sorted_ids[pos] = id;
        };

        for (int i = 0; i < vocab_size; ++i) {
            insert_sorted(get_logit(i), i);
        }

        float max_val = sorted_logits[0];
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            float p = std::exp(sorted_logits[i] - max_val);
            probs[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !std::isfinite(sum)) {
            return sorted_ids[0];
        }
        for (int i = 0; i < k; ++i) {
            probs[i] /= sum;
        }

        float p_cut = std::clamp(top_p, 1e-6f, 1.0f);
        int keep = k;
        if (p_cut < 0.999999f) {
            float cdf = 0.0f;
            keep = 0;
            for (int i = 0; i < k; ++i) {
                cdf += probs[i];
                keep = i + 1;
                if (cdf >= p_cut) {
                    break;
                }
            }
            if (keep == 0) keep = 1;

            float kept_sum = 0.0f;
            for (int i = 0; i < keep; ++i) {
                kept_sum += probs[i];
            }
            if (kept_sum > 0.0f) {
                for (int i = 0; i < keep; ++i) {
                    probs[i] /= kept_sum;
                }
            }
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(sampling_rng());

        float cumsum = 0.0f;
        for (int i = 0; i < keep; ++i) {
            cumsum += probs[i];
            if (r <= cumsum) {
                return sorted_ids[i];
            }
        }
        return sorted_ids[keep - 1];
    }

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

inline void insert_desc_candidate(float* logits, int* ids, int k, float logit, int id) {
    if (k <= 0) return;
    if (logit <= logits[k - 1]) return;
    int pos = k - 1;
    while (pos > 0 && logit > logits[pos - 1]) {
        logits[pos] = logits[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
    }
    logits[pos] = logit;
    ids[pos] = id;
}

inline int choose_sampling_groups(int vocab_size) {
    int groups = (vocab_size + 4095) / 4096;
    groups = std::clamp(groups, 1, 64);
    return groups;
}

} // namespace

void set_sampling_seed(uint64_t seed) {
    sampling_rng().seed(static_cast<uint32_t>(seed));
}

float next_sampling_uniform() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(sampling_rng());
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
    });

    ctx.memcpy_d2h(&h_result, d_result, sizeof(int));
    ctx.free_device(d_result);
    return h_result;
}

// ============================================================
// Top-k Sampling (CPU 端)
// ============================================================

int topk_sample(Context& ctx, Tensor& logits, int vocab_size,
                 float temperature, int top_k) {
    static thread_local HostSamplingWorkspace host_ws;
    bf16* host_logits_bf16 = host_ws.ensure_bf16(ctx, static_cast<size_t>(vocab_size));
    ctx.memcpy_d2h(host_logits_bf16, logits.data(), vocab_size * sizeof(bf16));
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
    if (!gen_config.do_sample) {
        return argmax(ctx, logits, vocab_size);
    }

    float inv_temperature = 1.0f / std::max(gen_config.temperature, 1e-6f);
    int top_k = std::min(gen_config.top_k, vocab_size);

    static thread_local HostSamplingWorkspace host_ws;
    bf16* host_logits_bf16 = host_ws.ensure_bf16(ctx, static_cast<size_t>(vocab_size));
    ctx.memcpy_d2h(host_logits_bf16, logits.data(), vocab_size * sizeof(bf16));

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

bool can_use_device_sampling(int vocab_size, const GenerationConfig& gen_config) {
    if (!gen_config.do_sample) {
        return false;
    }

    static int device_sampling_mode = -1;
    if (device_sampling_mode < 0) {
        device_sampling_mode = aila::env::read_int_raw("AILA_DEVICE_SAMPLING", 1);
    }
    if (device_sampling_mode == 0) {
        return false;
    }

    int top_k = gen_config.top_k;
    if (top_k <= 0 || top_k > vocab_size) {
        top_k = vocab_size;
    }

    return !gen_config.has_penalties() &&
           top_k > 0 &&
           top_k <= kDeviceSampleFastMaxTopK &&
           gen_config.temperature > 0.0f &&
           gen_config.top_p > 0.0f &&
           vocab_size > 0;
}

void sample_with_config_device(Context& ctx, Tensor& logits, int vocab_size,
                               const GenerationConfig& gen_config,
                               float random_u, int* d_result) {
    if (d_result == nullptr) {
        return;
    }
    if (!can_use_device_sampling(vocab_size, gen_config)) {
        ops::argmax(ctx, logits, vocab_size, d_result);
        return;
    }

    constexpr float neg_inf = -1.0e30f;
    bf16* l_ptr = static_cast<bf16*>(logits.data());
    float inv_temperature = 1.0f / std::max(gen_config.temperature, 1e-6f);
    int top_k = std::min(gen_config.top_k, vocab_size);
    float top_p = std::clamp(gen_config.top_p, 1e-6f, 1.0f);
    int num_groups = choose_sampling_groups(vocab_size);

    static thread_local DeviceSamplingWorkspace device_ws;
    device_ws.ensure(ctx, static_cast<size_t>(num_groups) * kDeviceSampleFastMaxTopK);
    float* partial_logits_ptr = device_ws.partial_logits;
    int* partial_ids_ptr = device_ws.partial_ids;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_candidate_logits(
            sycl::range<1>(kDeviceSampleFastWG * kDeviceSampleFastMaxTopK), cgh);
        sycl::local_accessor<int, 1> local_candidate_ids(
            sycl::range<1>(kDeviceSampleFastWG * kDeviceSampleFastMaxTopK), cgh);

        cgh.parallel_for(sycl::nd_range<1>(num_groups * kDeviceSampleFastWG, kDeviceSampleFastWG),
                         [=](sycl::nd_item<1> item) {
            int group = static_cast<int>(item.get_group(0));
            int lid = static_cast<int>(item.get_local_id(0));
            float private_logits[kDeviceSampleFastMaxTopK];
            int private_ids[kDeviceSampleFastMaxTopK];
            for (int i = 0; i < kDeviceSampleFastMaxTopK; ++i) {
                private_logits[i] = neg_inf;
                private_ids[i] = -1;
            }

            for (int token = group * kDeviceSampleFastWG + lid;
                 token < vocab_size;
                 token += kDeviceSampleFastWG * num_groups) {
                float scaled_logit = static_cast<float>(l_ptr[token]) * inv_temperature;
                insert_desc_candidate(private_logits, private_ids, top_k, scaled_logit, token);
            }

            int base = lid * kDeviceSampleFastMaxTopK;
            for (int i = 0; i < top_k; ++i) {
                local_candidate_logits[base + i] = private_logits[i];
                local_candidate_ids[base + i] = private_ids[i];
            }
            item.barrier(sycl::access::fence_space::local_space);

            if (lid == 0) {
                float final_logits[kDeviceSampleFastMaxTopK];
                int final_ids[kDeviceSampleFastMaxTopK];
                float probs[kDeviceSampleFastMaxTopK];
                for (int i = 0; i < kDeviceSampleFastMaxTopK; ++i) {
                    final_logits[i] = neg_inf;
                    final_ids[i] = -1;
                    probs[i] = 0.0f;
                }

                for (int thread = 0; thread < kDeviceSampleFastWG; ++thread) {
                    int thread_base = thread * kDeviceSampleFastMaxTopK;
                    for (int i = 0; i < top_k; ++i) {
                        insert_desc_candidate(final_logits, final_ids, top_k,
                                              local_candidate_logits[thread_base + i],
                                              local_candidate_ids[thread_base + i]);
                    }
                }
                int out_base = group * kDeviceSampleFastMaxTopK;
                for (int i = 0; i < top_k; ++i) {
                    partial_logits_ptr[out_base + i] = final_logits[i];
                    partial_ids_ptr[out_base + i] = final_ids[i];
                }
            }
        });
    });

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(kDeviceSampleFastWG, kDeviceSampleFastWG),
                         [=](sycl::nd_item<1> item) {
            int lid = static_cast<int>(item.get_local_id(0));
            if (lid != 0) {
                return;
            }

            float final_logits[kDeviceSampleFastMaxTopK];
            int final_ids[kDeviceSampleFastMaxTopK];
            float probs[kDeviceSampleFastMaxTopK];
            for (int i = 0; i < kDeviceSampleFastMaxTopK; ++i) {
                final_logits[i] = neg_inf;
                final_ids[i] = -1;
                probs[i] = 0.0f;
            }

            for (int group = 0; group < num_groups; ++group) {
                int base = group * kDeviceSampleFastMaxTopK;
                for (int i = 0; i < top_k; ++i) {
                    insert_desc_candidate(final_logits, final_ids, top_k,
                                          partial_logits_ptr[base + i],
                                          partial_ids_ptr[base + i]);
                }
            }

            int result = (final_ids[0] >= 0) ? final_ids[0] : 0;
            float max_val = final_logits[0];
            float sum = 0.0f;
            int keep = top_k;
            for (int i = 0; i < top_k; ++i) {
                if (final_ids[i] < 0) {
                    keep = i;
                    break;
                }
                float p = sycl::exp(final_logits[i] - max_val);
                probs[i] = p;
                sum += p;
            }

            if (keep == 0) {
                keep = 1;
            }

            if (sum > 0.0f && sycl::isfinite(sum)) {
                for (int i = 0; i < keep; ++i) {
                    probs[i] /= sum;
                }

                if (top_p < 0.999999f) {
                    float cdf = 0.0f;
                    int clipped_keep = 0;
                    for (int i = 0; i < keep; ++i) {
                        cdf += probs[i];
                        clipped_keep = i + 1;
                        if (cdf >= top_p) {
                            break;
                        }
                    }
                    keep = sycl::max(clipped_keep, 1);

                    float kept_sum = 0.0f;
                    for (int i = 0; i < keep; ++i) {
                        kept_sum += probs[i];
                    }
                    if (kept_sum > 0.0f) {
                        for (int i = 0; i < keep; ++i) {
                            probs[i] /= kept_sum;
                        }
                    }
                }

                float cumsum = 0.0f;
                for (int i = 0; i < keep; ++i) {
                    cumsum += probs[i];
                    if (random_u <= cumsum) {
                        result = final_ids[i];
                        break;
                    }
                }
                if (result < 0 && keep > 0) {
                    result = final_ids[keep - 1];
                }
            }

            *d_result = result;
        });
    });
}

} // namespace ops
