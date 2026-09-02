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
#include <thread>

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

namespace {

template <typename Fn>
void parallel_rows(int rows, Fn&& fn) {
    if (rows <= 0) return;
    unsigned hardware = std::thread::hardware_concurrency();
    if (hardware == 0) hardware = 1;
    unsigned threads = std::min<unsigned>(hardware, 16u);
    threads = std::min<unsigned>(threads, static_cast<unsigned>(rows));
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        const int start = static_cast<int>(static_cast<int64_t>(rows) * t / threads);
        const int end = static_cast<int>(static_cast<int64_t>(rows) * (t + 1) / threads);
        pool.emplace_back([&fn, start, end]() {
            for (int r = start; r < end; ++r) fn(r);
        });
    }
    for (auto& th : pool) th.join();
}

} // namespace

void diffusion_softmax_entropy(const bf16* logits, bf16* probs,
                               float* entropy, int rows, int cols) {
    if (cols <= 0) return;
    parallel_rows(rows, [&](int r) {
        const bf16* row = logits + static_cast<size_t>(r) * cols;
        bf16* prows = probs + static_cast<size_t>(r) * cols;
        float max_logit = -1.0e30f;
        for (int c = 0; c < cols; ++c) {
            max_logit = std::max(max_logit, static_cast<float>(row[c]));
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += std::exp(static_cast<float>(row[c]) - max_logit);
        }
        const float inv_sum = 1.0f / sum;
        float ent = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float p =
                std::exp(static_cast<float>(row[c]) - max_logit) * inv_sum;
            prows[c] = bf16(p);
            if (p > 0.0f) ent -= p * std::log(p);
        }
        entropy[r] = ent;
    });
}

void diffusion_temperature_scale(const float* logits, float* out, int rows,
                                 int cols, float temperature) {
    const int64_t n = static_cast<int64_t>(rows) * cols;
    const float inv = 1.0f / temperature;
    parallel_rows(static_cast<int>(rows), [&](int r) {
        const float* src = logits + static_cast<size_t>(r) * cols;
        float* dst = out + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) dst[c] = src[c] * inv;
    });
    (void)n;
}

void diffusion_accept_mask(const float* entropy, int rows, int cols,
                           float entropy_bound, uint8_t* accept) {
    parallel_rows(rows, [&](int r) {
        std::vector<std::pair<float, int>> order(static_cast<size_t>(cols));
        const float* erow = entropy + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            order[static_cast<size_t>(c)] = {erow[c], c};
        }
        std::sort(order.begin(), order.end(),
                  [](const std::pair<float, int>& a,
                     const std::pair<float, int>& b) { return a.first < b.first; });
        float cumulative = 0.0f;
        for (int k = 0; k < cols; ++k) {
            cumulative += order[static_cast<size_t>(k)].first;
            const bool selected =
                cumulative - order[static_cast<size_t>(k)].first <= entropy_bound;
            accept[static_cast<size_t>(r) * cols +
                   static_cast<size_t>(order[static_cast<size_t>(k)].second)] =
                selected ? 1 : 0;
        }
    });
}

void diffusion_argmax_device(Context& ctx, const float* logits_dev,
                             int* tokens_dev, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    constexpr int WG = 256;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_val(sycl::range<1>(WG), cgh);
        sycl::local_accessor<int, 1> local_idx(sycl::range<1>(WG), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const int row = static_cast<int>(item.get_group(0));
                const int lane = static_cast<int>(item.get_local_id(0));
                const float* rlogits =
                    logits_dev + static_cast<size_t>(row) * cols;

                float max_val = -1.0e30f;
                int min_idx = 0;
                for (int c = lane; c < cols; c += WG) {
                    const float v = rlogits[c];
                    if (v > max_val) {
                        max_val = v;
                        min_idx = c;
                    }
                }
                local_val[lane] = max_val;
                local_idx[lane] = min_idx;
                item.barrier(sycl::access::fence_space::local_space);

                for (int offset = WG / 2; offset > 0; offset >>= 1) {
                    if (lane < offset) {
                        const float other_val = local_val[lane + offset];
                        const int other_idx = local_idx[lane + offset];
                        if (other_val > local_val[lane] ||
                            (other_val == local_val[lane] &&
                             other_idx < local_idx[lane])) {
                            local_val[lane] = other_val;
                            local_idx[lane] = other_idx;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lane == 0) {
                    tokens_dev[row] = local_idx[0];
                }
            });
    });
}

void diffusion_softmax_entropy_device(Context& ctx, const float* logits_dev,
                                      float* entropy_dev, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    constexpr int WG = 256;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(WG), cgh);
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(WG), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const int row = static_cast<int>(item.get_group(0));
                const int lane = static_cast<int>(item.get_local_id(0));
                const float* rlogits =
                    logits_dev + static_cast<size_t>(row) * cols;

                float max_val = -1.0e30f;
                for (int c = lane; c < cols; c += WG) {
                    max_val = sycl::max(max_val, rlogits[c]);
                }
                local_max[lane] = max_val;
                item.barrier(sycl::access::fence_space::local_space);

                for (int offset = WG / 2; offset > 0; offset >>= 1) {
                    if (lane < offset) {
                        local_max[lane] = sycl::max(
                            local_max[lane], local_max[lane + offset]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float max_logit = local_max[0];
                float sum_exp = 0.0f;
                for (int c = lane; c < cols; c += WG) {
                    sum_exp += sycl::exp(rlogits[c] - max_logit);
                }
                local_sum[lane] = sum_exp;
                item.barrier(sycl::access::fence_space::local_space);

                for (int offset = WG / 2; offset > 0; offset >>= 1) {
                    if (lane < offset) {
                        local_sum[lane] += local_sum[lane + offset];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float inv_sum = 1.0f / local_sum[0];
                float ent = 0.0f;
                for (int c = lane; c < cols; c += WG) {
                    const float p = sycl::exp(rlogits[c] - max_logit) * inv_sum;
                    if (p > 0.0f) {
                        ent -= p * sycl::log(p);
                    }
                }
                local_sum[lane] = ent;
                item.barrier(sycl::access::fence_space::local_space);

                for (int offset = WG / 2; offset > 0; offset >>= 1) {
                    if (lane < offset) {
                        local_sum[lane] += local_sum[lane + offset];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lane == 0) {
                    entropy_dev[row] = local_sum[0];
                }
            });
    });
}

struct Philox4x32 {
    static constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53;
    static constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
    static constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9;
    static constexpr uint32_t PHILOX_W32_1   = 0xBB67AE85;

    static inline void round(uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3, uint32_t k0, uint32_t k1) {
        uint64_t prod0 = static_cast<uint64_t>(PHILOX_M4x32_0) * x0;
        uint64_t prod1 = static_cast<uint64_t>(PHILOX_M4x32_1) * x2;
        uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(prod0);
        uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(prod1);
        x0 = hi1 ^ x1 ^ k0;
        x1 = lo1;
        x2 = hi0 ^ x3 ^ k1;
        x3 = lo0;
    }

    static inline void generate4(uint32_t ctr0, uint32_t ctr1, uint32_t key0, uint32_t key1,
                                 float& u0, float& u1, float& u2, float& u3) {
        uint32_t x0 = ctr0, x1 = 0, x2 = ctr1, x3 = 0;
        uint32_t k0 = key0, k1 = key1;

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            round(x0, x1, x2, x3, k0, k1);
            k0 += PHILOX_W32_0;
            k1 += PHILOX_W32_1;
        }

        constexpr float scale = 1.0f / 4294967296.0f;
        u0 = sycl::fmax((x0 + 0.5f) * scale, 1.0e-30f);
        u1 = sycl::fmax((x1 + 0.5f) * scale, 1.0e-30f);
        u2 = sycl::fmax((x2 + 0.5f) * scale, 1.0e-30f);
        u3 = sycl::fmax((x3 + 0.5f) * scale, 1.0e-30f);
    }
};

void fused_post_logits_transform_stats_argmax(
    Context& ctx, const bf16* logits_bf16, float* logits_f32,
    int* argmax_dev, float* entropy_dev,
    float* row_max_dev, float* row_inv_sum_dev,
    float cap, float temperature, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    constexpr int WG_SIZE = 256;
    const float inv_cap = 1.0f / cap;
    const float inv_temp = 1.0f / temperature;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_max_val(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<int, 1> s_max_idx(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<float, 1> s_sum_exp(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<float, 1> s_sum_lexp(sycl::range<1>(WG_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(rows) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int row = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const size_t row_offset = static_cast<size_t>(row) * cols;
                const bf16* src = logits_bf16 + row_offset;
                float* dst = logits_f32 + row_offset;

                // Phase 1: 4-wide load, softcap, temperature scale, write out f32, local max & argmax
                float local_max = -1.0e30f;
                int local_argmax = 0;

                for (int c = lid * 4; c < cols; c += WG_SIZE * 4) {
                    if (c + WG_SIZE * 4 + 3 < cols) {
                        sycl::ext::oneapi::experimental::prefetch(src + c + WG_SIZE * 4);
                    }
                    if (c + 3 < cols) {
                        auto v_in = *reinterpret_cast<const sycl::vec<bf16, 4>*>(src + c);
                        sycl::vec<float, 4> v_out;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float raw = static_cast<float>(v_in[i]);
                            const float scaled = sycl::tanh(raw * inv_cap) * cap * inv_temp;
                            v_out[i] = scaled;
                            if (scaled > local_max) {
                                local_max = scaled;
                                local_argmax = c + i;
                            }
                        }
                        *reinterpret_cast<sycl::vec<float, 4>*>(dst + c) = v_out;
                    } else {
                        for (int i = 0; i < 4 && c + i < cols; ++i) {
                            const float raw = static_cast<float>(src[c + i]);
                            const float scaled = sycl::tanh(raw * inv_cap) * cap * inv_temp;
                            dst[c + i] = scaled;
                            if (scaled > local_max) {
                                local_max = scaled;
                                local_argmax = c + i;
                            }
                        }
                    }
                }

                s_max_val[lid] = local_max;
                s_max_idx[lid] = local_argmax;
                item.barrier(sycl::access::fence_space::local_space);

                #pragma unroll
                for (int offset = WG_SIZE / 2; offset > 0; offset >>= 1) {
                    if (lid < offset) {
                        if (s_max_val[lid + offset] > s_max_val[lid] ||
                            (s_max_val[lid + offset] == s_max_val[lid] && s_max_idx[lid + offset] < s_max_idx[lid])) {
                            s_max_val[lid] = s_max_val[lid + offset];
                            s_max_idx[lid] = s_max_idx[lid + offset];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float row_max = s_max_val[0];
                const int row_argmax = s_max_idx[0];
                if (lid == 0) {
                    if (argmax_dev) argmax_dev[row] = row_argmax;
                    if (row_max_dev) row_max_dev[row] = row_max;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Phase 2: 4-wide load from dst, compute sum(exp) and sum(l * exp)
                float local_sum_exp = 0.0f;
                float local_sum_lexp = 0.0f;

                for (int c = lid * 4; c < cols; c += WG_SIZE * 4) {
                    if (c + WG_SIZE * 4 + 3 < cols) {
                        sycl::ext::oneapi::experimental::prefetch(dst + c + WG_SIZE * 4);
                    }
                    if (c + 3 < cols) {
                        auto v = *reinterpret_cast<const sycl::vec<float, 4>*>(dst + c);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float val = v[i];
                            const float e = sycl::native::exp(val - row_max);
                            local_sum_exp += e;
                            local_sum_lexp += val * e;
                        }
                    } else {
                        for (int i = 0; i < 4 && c + i < cols; ++i) {
                            const float val = dst[c + i];
                            const float e = sycl::native::exp(val - row_max);
                            local_sum_exp += e;
                            local_sum_lexp += val * e;
                        }
                    }
                }

                s_sum_exp[lid] = local_sum_exp;
                s_sum_lexp[lid] = local_sum_lexp;
                item.barrier(sycl::access::fence_space::local_space);

                #pragma unroll
                for (int offset = WG_SIZE / 2; offset > 0; offset >>= 1) {
                    if (lid < offset) {
                        s_sum_exp[lid] += s_sum_exp[lid + offset];
                        s_sum_lexp[lid] += s_sum_lexp[lid + offset];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0) {
                    const float total_sum = s_sum_exp[0];
                    const float total_lexp = s_sum_lexp[0];
                    const float inv_sum = 1.0f / total_sum;
                    if (row_inv_sum_dev) row_inv_sum_dev[row] = inv_sum;
                    if (entropy_dev) {
                        entropy_dev[row] = row_max + sycl::native::log(total_sum) - total_lexp * inv_sum;
                    }
                }
            });
    });
}

void diffusion_gumbel_sample_device(Context& ctx, const float* logits_dev,
                                    const uint32_t* seeds_dev,
                                    uint32_t* /* mt_states_dev */,
                                    int* tokens_dev, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    constexpr int WG_SIZE = 256;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_max_val(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<int, 1> s_max_idx(sycl::range<1>(WG_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(rows) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int row = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const float* rlogits = logits_dev + static_cast<size_t>(row) * cols;

                float best_val = -1.0e30f;
                int best_idx = 0;

                const uint32_t row_seed = seeds_dev ? seeds_dev[row] : static_cast<uint32_t>(row * 1337 + 42);
                uint32_t rng_state = (row_seed ^ (static_cast<uint32_t>(row) * 0x9E3779B9u)) +
                                     static_cast<uint32_t>(lid * 1013904223u + 1);
                if (rng_state == 0) rng_state = 0xDEADBEEFu;

                auto next_gumbel = [&]() -> float {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 17;
                    rng_state ^= rng_state << 5;
                    constexpr float scale = 1.0f / 4294967296.0f;
                    const float u = sycl::fmax((rng_state + 0.5f) * scale, 1.0e-30f);
                    return -sycl::native::log(-sycl::native::log(u));
                };

                for (int c = lid * 8; c < cols; c += WG_SIZE * 8) {
                    if (c + WG_SIZE * 8 + 7 < cols) {
                        sycl::ext::oneapi::experimental::prefetch(rlogits + c + WG_SIZE * 8);
                    }
                    if (c + 7 < cols) {
                        auto v0 = *reinterpret_cast<const sycl::vec<float, 4>*>(rlogits + c);
                        auto v1 = *reinterpret_cast<const sycl::vec<float, 4>*>(rlogits + c + 4);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float g = next_gumbel();
                            const float val = v0[i] + g;
                            if (val > best_val) { best_val = val; best_idx = c + i; }
                        }
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float g = next_gumbel();
                            const float val = v1[i] + g;
                            if (val > best_val) { best_val = val; best_idx = c + 4 + i; }
                        }
                    } else {
                        for (int i = 0; i < 8 && c + i < cols; ++i) {
                            const float g = next_gumbel();
                            const float val = rlogits[c + i] + g;
                            if (val > best_val) { best_val = val; best_idx = c + i; }
                        }
                    }
                }

                s_max_val[lid] = best_val;
                s_max_idx[lid] = best_idx;
                item.barrier(sycl::access::fence_space::local_space);

                #pragma unroll
                for (int offset = WG_SIZE / 2; offset > 0; offset >>= 1) {
                    if (lid < offset) {
                        if (s_max_val[lid + offset] > s_max_val[lid] ||
                            (s_max_val[lid + offset] == s_max_val[lid] && s_max_idx[lid + offset] < s_max_idx[lid])) {
                            s_max_val[lid] = s_max_val[lid + offset];
                            s_max_idx[lid] = s_max_idx[lid + offset];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0) {
                    tokens_dev[row] = s_max_idx[0];
                }
            });
    });
}

void diffusion_accept_renoise_device(
    Context& ctx, const float* entropy_dev, const int* denoiser_dev,
    const int* /* prev_canvas_dev */, int* new_canvas_dev,
    const uint32_t* seeds_dev, float entropy_bound, int vocab_size,
    int num_canvases, int canvas_len) {
    if (num_canvases <= 0 || canvas_len <= 0) return;
    constexpr int WG_SIZE = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_ent(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<float, 1> s_sorted_ent(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<int, 1> s_sorted_orig_idx(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<uint8_t, 1> s_accept(sycl::range<1>(WG_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(num_canvases) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int b = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const size_t base_offset = static_cast<size_t>(b) * canvas_len;

                // Load entropy into SLM
                const float my_ent = (lid < canvas_len) ? entropy_dev[base_offset + lid] : 1.0e30f;
                s_ent[lid] = my_ent;
                item.barrier(sycl::access::fence_space::local_space);

                // Compute rank of lid by counting smaller elements
                if (lid < canvas_len) {
                    int rank = 0;
                    #pragma unroll 4
                    for (int j = 0; j < canvas_len; ++j) {
                        const float other_ent = s_ent[j];
                        if (other_ent < my_ent || (other_ent == my_ent && j < lid)) {
                            ++rank;
                        }
                    }
                    s_sorted_ent[rank] = my_ent;
                    s_sorted_orig_idx[rank] = lid;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Cumulative entropy and acceptance decision
                if (lid < canvas_len) {
                    float cum_sum_prev = 0.0f;
                    for (int k = 0; k < lid; ++k) {
                        cum_sum_prev += s_sorted_ent[k];
                    }
                    const bool is_acc = (cum_sum_prev <= entropy_bound);
                    const int orig_pos = s_sorted_orig_idx[lid];
                    s_accept[orig_pos] = is_acc ? 1 : 0;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Write out accepted denoiser token or generate fast pseudo-random token
                if (lid < canvas_len) {
                    const bool accepted = (s_accept[lid] != 0);
                    if (accepted) {
                        new_canvas_dev[base_offset + lid] = denoiser_dev[base_offset + lid];
                    } else {
                        const uint32_t seed_val = seeds_dev ? seeds_dev[b] : static_cast<uint32_t>(b * 1337 + 42);
                        uint32_t state = (seed_val ^ (static_cast<uint32_t>(b) * 0x9E3779B9u)) +
                                         static_cast<uint32_t>(lid * 1013904223u + 1);
                        if (state == 0) state = 0xDEADBEEFu;
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        const int rand_tok = static_cast<int>(state % static_cast<uint32_t>(vocab_size));
                        new_canvas_dev[base_offset + lid] = rand_tok;
                    }
                }
            });
    });
}

void diffusion_argmax(const float* logits, int* tokens, int rows, int cols) {
    parallel_rows(rows, [&](int r) {
        const float* row = logits + static_cast<size_t>(r) * cols;
        int best = 0;
        float best_value = row[0];
        for (int c = 1; c < cols; ++c) {
            if (row[c] > best_value) {
                best_value = row[c];
                best = c;
            }
        }
        tokens[r] = best;
    });
}

void diffusion_gumbel_sample(const float* logits, const float* uniforms,
                             int* tokens, int rows, int cols) {
    parallel_rows(rows, [&](int r) {
        const float* row = logits + static_cast<size_t>(r) * cols;
        const float* us = uniforms + static_cast<size_t>(r) * cols;
        int best = 0;
        float best_value = -1.0e30f;
        for (int c = 0; c < cols; ++c) {
            const float u = std::max(us[c], 1.0e-30f);
            const float gumbel = -std::log(-std::log(u));
            const float value = row[c] + gumbel;
            if (value > best_value) {
                best_value = value;
                best = c;
            }
        }
        tokens[r] = best;
    });
}

void diffusion_renoise(int* canvas, const uint8_t* accept, int rows, int cols,
                       int vocab_size, uint64_t seed, int* new_canvas) {
    if (rows <= 0 || cols <= 0) return;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, vocab_size - 1);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            const int random_token = dist(rng);
            new_canvas[idx] =
                accept[idx] ? canvas[idx] : random_token;
        }
    }
}

void gemma4_router_host(const float* hidden, const float* scale,
                        const float* proj, const float* per_expert_scale,
                        int tokens, int hidden_size, int num_experts,
                        int top_k, std::vector<int>& expert_ids,
                        std::vector<float>& expert_weights) {
    expert_ids.assign(static_cast<size_t>(tokens) * top_k, 0);
    expert_weights.assign(static_cast<size_t>(tokens) * top_k, 0.0f);
    const float root_size = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    const float eps = 1e-6f;

    parallel_rows(tokens, [&](int t) {
        const float* h = hidden + static_cast<size_t>(t) * hidden_size;
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum_sq += h[i] * h[i];
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

        // Match the observable dtype boundaries in DiffusionGemmaTextRouter:
        // RMSNorm returns bf16, both elementwise scales return bf16, and the
        // bf16 linear projection produces bf16 logits before fp32 softmax.
        std::vector<float> routed(static_cast<size_t>(hidden_size));
        for (int i = 0; i < hidden_size; ++i) {
            float value = static_cast<float>(bf16(h[i] * inv_rms));
            value = static_cast<float>(bf16(value * scale[i]));
            routed[static_cast<size_t>(i)] =
                static_cast<float>(bf16(value * root_size));
        }

        std::vector<float> scores(static_cast<size_t>(num_experts));
        float max_score = -1.0e30f;
        for (int e = 0; e < num_experts; ++e) {
            const float* prow = proj + static_cast<size_t>(e) * hidden_size;
            float acc = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                acc += routed[static_cast<size_t>(i)] * prow[i];
            }
            acc = static_cast<float>(bf16(acc));
            scores[static_cast<size_t>(e)] = acc;
            max_score = std::max(max_score, acc);
        }

        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            scores[static_cast<size_t>(e)] = std::exp(scores[static_cast<size_t>(e)] - max_score);
            sum += scores[static_cast<size_t>(e)];
        }
        const float inv_sum = 1.0f / sum;
        for (int e = 0; e < num_experts; ++e) {
            scores[static_cast<size_t>(e)] *= inv_sum;
        }

        std::vector<std::pair<float, int>> ranked(static_cast<size_t>(num_experts));
        for (int e = 0; e < num_experts; ++e) {
            ranked[static_cast<size_t>(e)] = {scores[static_cast<size_t>(e)], e};
        }
        std::partial_sort(
            ranked.begin(), ranked.begin() + top_k, ranked.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        float top_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            top_sum += ranked[static_cast<size_t>(k)].first;
        }
        for (int k = 0; k < top_k; ++k) {
            const int idx = t * top_k + k;
            expert_ids[static_cast<size_t>(idx)] = ranked[static_cast<size_t>(k)].second;
            expert_weights[static_cast<size_t>(idx)] =
                ranked[static_cast<size_t>(k)].first / top_sum *
                per_expert_scale[static_cast<size_t>(ranked[static_cast<size_t>(k)].second)];
        }
    });
}

} // namespace ops
