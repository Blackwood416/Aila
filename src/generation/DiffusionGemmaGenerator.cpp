#include "DiffusionGemmaGenerator.hpp"

#include "../ops/Ops.hpp"
#include "../utils/EnvUtils.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename F>
void parallel_rows(int rows, const F& fn) {
    if (rows <= 0) return;
    tbb::parallel_for(tbb::blocked_range<int>(0, rows),
                      [&](const tbb::blocked_range<int>& range) {
                          for (int r = range.begin(); r < range.end(); ++r) {
                              fn(r);
                          }
                      });
}

void softmax_entropy_host(const float* logits, float* entropy, int rows,
                          int cols) {
    parallel_rows(rows, [&](int r) {
        const float* row = logits + static_cast<size_t>(r) * cols;
        float max_logit = -1.0e30f;
        for (int c = 0; c < cols; ++c) {
            max_logit = std::max(max_logit, row[c]);
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += std::exp(row[c] - max_logit);
        }
        const float inv = 1.0f / sum;
        float ent = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float p = std::exp(row[c] - max_logit) * inv;
            if (p > 0.0f) ent -= p * std::log(p);
        }
        entropy[r] = ent;
    });
}

void gumbel_sample_host(const float* logits, std::mt19937& rng, int* tokens,
                        int rows, int cols) {
    // std::mt19937 is not thread-safe. Derive one deterministic stream per
    // row before entering the worker pool instead of racing a shared engine.
    std::vector<uint32_t> row_seeds(static_cast<size_t>(rows));
    for (uint32_t& seed : row_seeds) seed = rng();
    parallel_rows(rows, [&](int r) {
        std::mt19937 row_rng(row_seeds[static_cast<size_t>(r)]);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        const float* row = logits + static_cast<size_t>(r) * cols;
        int best = 0;
        float best_value = -1.0e30f;
        for (int c = 0; c < cols; ++c) {
            const float u = std::max(uniform(row_rng), 1.0e-30f);
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

} // namespace

bool DiffusionGemmaGenerator::generate(
    Context& ctx, DiffusionGemmaBnb4Backend& backend,
    const std::vector<int>& prompt_ids, const DiffusionGemmaGeneratorConfig& cfg,
    int max_new_tokens, std::vector<int>& output_ids,
    std::string* error_message, CanvasCallback callback) {
    if (max_new_tokens <= 0) {
        output_ids.clear();
        return true;
    }
    const int canvas_len = cfg.canvas_length;
    const int vocab = backend.vocab_size();
    const int hidden = backend.config().text.hidden_size;
    if (canvas_len <= 0 || vocab <= 0 || hidden <= 0) {
        if (error_message) *error_message = "invalid DiffusionGemma config";
        return false;
    }
    const auto generation_start = std::chrono::steady_clock::now();
    AILA_LOG_INFO(
        "[DiffusionGemma] Generating up to %d tokens (canvas=%d steps=%d)",
        max_new_tokens, canvas_len, cfg.max_denoising_steps);
    const bool profile = aila::env::read_flag("AILA_PROFILE_DG", false);
    double profile_encode_ms = 0.0;
    double profile_decode_ms = 0.0;
    double profile_host_ms = 0.0;
    double profile_soft_ms = 0.0;

    if (!prompt_ids.empty()) {
        const auto encode_start = std::chrono::steady_clock::now();
        AILA_LOG_DEBUG("[DiffusionGemma] Encoding prompt (%zu tokens)",
                       prompt_ids.size());
        int* prompt_device = static_cast<int*>(
            ctx.alloc_device(prompt_ids.size() * sizeof(int)));
        ctx.memcpy_h2d(prompt_device, prompt_ids.data(),
                       prompt_ids.size() * sizeof(int));
        std::string encode_error;
        const bool ok = backend.diffusion_encode(
            ctx, prompt_device, static_cast<int>(prompt_ids.size()),
            &encode_error);
        ctx.free_device(prompt_device);
        if (!ok) {
            if (error_message) *error_message = encode_error;
            return false;
        }
        ctx.synchronize();
        if (profile) {
            profile_encode_ms += std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() -
                                     encode_start)
                                     .count();
        }
        AILA_LOG_DEBUG("[DiffusionGemma] Prompt encoded");
    }

    output_ids.clear();
    output_ids.reserve(static_cast<size_t>(max_new_tokens));

    std::mt19937 rng(cfg.seed);
    std::uniform_int_distribution<int> token_dist(0, vocab - 1);

    std::vector<int> canvas(static_cast<size_t>(canvas_len));
    for (int i = 0; i < canvas_len; ++i) {
        canvas[static_cast<size_t>(i)] = token_dist(rng);
    }
    std::vector<bf16> sc_embeds(static_cast<size_t>(canvas_len) * hidden,
                                bf16(0.0f));

    const int max_canvases =
        (max_new_tokens + canvas_len - 1) / canvas_len;
    int* canvas_device = static_cast<int*>(
        ctx.alloc_device(static_cast<size_t>(canvas_len) * sizeof(int)));
    bf16* sc_device = static_cast<bf16*>(ctx.alloc_device(
        static_cast<size_t>(canvas_len) * hidden * sizeof(bf16)));

    std::vector<float> logits_host(static_cast<size_t>(canvas_len) * vocab);
    std::vector<float> processed_host(static_cast<size_t>(canvas_len) * vocab);
    std::vector<float> entropy(static_cast<size_t>(canvas_len));
    std::vector<int> denoiser(static_cast<size_t>(canvas_len));
    std::vector<int> new_argmax(static_cast<size_t>(canvas_len));
    std::vector<uint8_t> accept(static_cast<size_t>(canvas_len));
    std::vector<int> new_canvas(static_cast<size_t>(canvas_len));

    bool all_finished = false;
    int generated = 0;
    const bool debug_probes =
        aila::env::read_flag("AILA_DG_DEBUG_PROBES", false);
    const bool step_timing =
        aila::env::read_flag("AILA_DG_STEP_TIMING", false);
    const std::string dump_dir =
        aila::env::read_string("AILA_DG_DUMP_DIR", "");
    const std::string token_dump_path =
        aila::env::read_string("AILA_DG_TOKEN_DUMP", "");
    std::ofstream committed_token_dump;
    if (!token_dump_path.empty()) {
        // A token-only dump is intentionally independent of DUMP_DIR: it
        // records the committed stream without enabling layer tensor dumps,
        // debug synchronizations, or extra GPU work in a production-shaped
        // benchmark.
        committed_token_dump.open(token_dump_path,
                                  std::ios::binary | std::ios::trunc);
    } else if (!dump_dir.empty()) {
        committed_token_dump.open(dump_dir + "/committed_tokens.bin",
                                  std::ios::binary | std::ios::trunc);
    }
    const bool use_gpu_sampling =
        aila::env::read_flag("AILA_DG_GPU_SAMPLING", false);

    int* d_argmax = nullptr;
    float* d_entropy = nullptr;
    int* d_denoiser = nullptr;
    uint32_t* d_seeds = nullptr;
    uint32_t* d_mt_states = nullptr;

    if (use_gpu_sampling) {
        d_argmax = static_cast<int*>(
            ctx.alloc_device(static_cast<size_t>(canvas_len) * sizeof(int)));
        d_entropy = static_cast<float*>(
            ctx.alloc_device(static_cast<size_t>(canvas_len) * sizeof(float)));
        d_denoiser = static_cast<int*>(
            ctx.alloc_device(static_cast<size_t>(canvas_len) * sizeof(int)));
        d_seeds = static_cast<uint32_t*>(
            ctx.alloc_device(static_cast<size_t>(canvas_len) * sizeof(uint32_t)));
        d_mt_states = static_cast<uint32_t*>(
            ctx.alloc_device(static_cast<size_t>(canvas_len) * 624 * sizeof(uint32_t)));
    }

    for (int c = 0; c < max_canvases && !all_finished; ++c) {
        std::vector<int> prev_argmax(canvas);
        int equal_count = 0;
        bool finished_denoising = false;
        if (use_gpu_sampling) {
            ctx.queue().memset(
                sc_device, 0,
                static_cast<size_t>(canvas_len) * hidden * sizeof(bf16));
        }

        for (int cur_step = cfg.max_denoising_steps; cur_step >= 1;
             --cur_step) {
            const auto step_start = std::chrono::steady_clock::now();
            AILA_LOG_DEBUG("[DiffusionGemma] Canvas %d/%d step %d", c + 1,
                           max_canvases, cur_step);
            ctx.memcpy_h2d(canvas_device, canvas.data(),
                           static_cast<size_t>(canvas_len) * sizeof(int));
            if (!use_gpu_sampling) {
                ctx.memcpy_h2d(sc_device, sc_embeds.data(),
                               sc_embeds.size() * sizeof(bf16));
            }
            const auto h2d_done = std::chrono::steady_clock::now();
            const auto decode_start = h2d_done;
            Tensor& logits = backend.diffusion_decode_canvas(
                ctx, canvas_device, sc_device, canvas_len);
            const float temperature =
                cfg.t_min + (cfg.t_max - cfg.t_min) *
                                (static_cast<float>(cur_step) /
                                 static_cast<float>(cfg.max_denoising_steps));
            const auto decoder_submit_done = std::chrono::steady_clock::now();

            if (use_gpu_sampling && !debug_probes && !profile) {
                float* logits_ptr = static_cast<float*>(logits.data());
                ctx.queue().parallel_for(
                    sycl::range<1>(static_cast<size_t>(canvas_len) * vocab),
                    [=](sycl::id<1> idx) {
                        logits_ptr[idx[0]] = logits_ptr[idx[0]] / temperature;
                    });

                // Step 1: Argmax on GPU
                const auto argmax_start = std::chrono::steady_clock::now();
                ops::diffusion_argmax_device(
                    ctx, logits_ptr, d_argmax, canvas_len, vocab);
                ctx.memcpy_d2h_async(
                    new_argmax.data(), d_argmax,
                    static_cast<size_t>(canvas_len) * sizeof(int));

                if (cur_step == 1) {
                    ctx.synchronize();
                    const auto argmax_done = std::chrono::steady_clock::now();
                    if (step_timing) {
                        std::fprintf(stderr,
                            "[DG step] step=%d h2d=%.1f decode_logits=%.1f "
                            "decoder_submit=%.1f logits_d2h=0.0 argmax=%.1f "
                            "entropy=0.0 gumbel=0.0 "
                            "accept_renoise=0.0 host_total=0.0 "
                            "softmax_submit=0.0 soft_gemm_submit=0.0 "
                            "soft_embed_d2h=0.0 total=%.1f ms\n",
                            cur_step,
                            std::chrono::duration<double, std::milli>(
                                h2d_done - step_start).count(),
                            std::chrono::duration<double, std::milli>(
                                argmax_done - decode_start).count(),
                            std::chrono::duration<double, std::milli>(
                                decoder_submit_done - decode_start).count(),
                            std::chrono::duration<double, std::milli>(
                                argmax_done - argmax_start).count(),
                            std::chrono::duration<double, std::milli>(
                                argmax_done - step_start).count());
                    }
                    break;
                }

                // Step 2: Compute entropy on GPU
                const auto entropy_start = std::chrono::steady_clock::now();
                ops::diffusion_softmax_entropy_device(
                    ctx, logits_ptr, d_entropy, canvas_len, vocab);
                ctx.memcpy_d2h_async(
                    entropy.data(), d_entropy,
                    static_cast<size_t>(canvas_len) * sizeof(float));

                // Step 3: Fast GPU Gumbel sampling
                std::vector<uint32_t> row_seeds(static_cast<size_t>(canvas_len));
                for (uint32_t& s : row_seeds) s = rng();
                ctx.memcpy_h2d_async(
                    d_seeds, row_seeds.data(),
                    static_cast<size_t>(canvas_len) * sizeof(uint32_t));

                const auto gumbel_start = std::chrono::steady_clock::now();
                ops::diffusion_gumbel_sample_device(
                    ctx, logits_ptr, d_seeds, d_mt_states, d_denoiser,
                    canvas_len, vocab);
                ctx.memcpy_d2h_async(
                    denoiser.data(), d_denoiser,
                    static_cast<size_t>(canvas_len) * sizeof(int));
                ctx.synchronize();
                const auto gumbel_done = std::chrono::steady_clock::now();

                const auto host_start = gumbel_done;
                ops::diffusion_accept_mask(entropy.data(), 1, canvas_len,
                                           cfg.entropy_bound, accept.data());
                for (int i = 0; i < canvas_len; ++i) {
                    new_canvas[static_cast<size_t>(i)] =
                        accept[static_cast<size_t>(i)]
                            ? denoiser[static_cast<size_t>(i)]
                            : canvas[static_cast<size_t>(i)];
                }
                for (int i = 0; i < canvas_len; ++i) {
                    const int random_token = token_dist(rng);
                    canvas[static_cast<size_t>(i)] =
                        accept[static_cast<size_t>(i)]
                            ? new_canvas[static_cast<size_t>(i)]
                            : random_token;
                }

                if (new_argmax == prev_argmax) {
                    ++equal_count;
                } else {
                    equal_count = 0;
                }
                prev_argmax = new_argmax;

                float mean_entropy = 0.0f;
                for (int i = 0; i < canvas_len; ++i) {
                    mean_entropy += entropy[static_cast<size_t>(i)];
                }
                mean_entropy /= static_cast<float>(canvas_len);
                AILA_LOG_DEBUG("[DG] step=%d mean_entropy=%.4f", cur_step,
                               mean_entropy);
                const bool stable = equal_count >= cfg.stability_threshold;
                const bool confident =
                    mean_entropy < cfg.confidence_threshold;
                if (stable && confident) {
                    finished_denoising = true;
                }

                const auto host_done = std::chrono::steady_clock::now();
                const auto host_ms =
                    std::chrono::duration<double, std::milli>(
                        host_done - host_start).count();

                if (finished_denoising) break;

                const auto soft_start = std::chrono::steady_clock::now();
                Tensor& soft = backend.diffusion_compute_soft_embeds(
                    ctx, logits, canvas_len);
                ctx.queue().memcpy(sc_device, soft.data(),
                                   static_cast<size_t>(canvas_len) * hidden *
                                       sizeof(bf16));
                ctx.synchronize();
                const auto soft_submit_done = std::chrono::steady_clock::now();
                const auto soft_timing = backend.last_soft_embed_timing();
                const auto soft_done = soft_submit_done;

                if (step_timing) {
                    std::fprintf(stderr,
                        "[DG step] step=%d h2d=%.1f decode_logits=%.1f "
                        "decoder_submit=%.1f logits_d2h=0.0 argmax=%.1f "
                        "entropy_gpu=%.1f gumbel_gpu=%.1f "
                        "accept_renoise=%.2f host_total=%.2f "
                        "softmax_submit=%.1f soft_gemm_submit=%.1f "
                        "soft_embed_d2h=%.1f total=%.1f ms\n",
                        cur_step,
                        std::chrono::duration<double, std::milli>(
                            h2d_done - step_start).count(),
                        std::chrono::duration<double, std::milli>(
                            gumbel_done - decode_start).count(),
                        std::chrono::duration<double, std::milli>(
                            decoder_submit_done - decode_start).count(),
                        std::chrono::duration<double, std::milli>(
                            entropy_start - argmax_start).count(),
                        std::chrono::duration<double, std::milli>(
                            gumbel_start - entropy_start).count(),
                        std::chrono::duration<double, std::milli>(
                            gumbel_done - gumbel_start).count(),
                        host_ms,
                        host_ms,
                        soft_timing.softmax_submit_ms,
                        soft_timing.gemm_submit_ms,
                        std::chrono::duration<double, std::milli>(
                            soft_done - soft_submit_done).count(),
                        std::chrono::duration<double, std::milli>(
                            soft_done - step_start).count());
                }
            } else {
                // Legacy CPU fallback path
                const auto logits_d2h_start = decoder_submit_done;
                ctx.memcpy_d2h(logits_host.data(), logits.data(),
                               logits_host.size() * sizeof(float));
                const auto logits_d2h_done = std::chrono::steady_clock::now();
                ctx.synchronize();
                const auto logits_ready = std::chrono::steady_clock::now();
                if (profile) {
                    profile_decode_ms +=
                        std::chrono::duration<double, std::milli>(
                            logits_ready - decode_start)
                            .count();
                }

                for (size_t i = 0; i < logits_host.size(); ++i) {
                    processed_host[i] = logits_host[i] / temperature;
                }

                const auto argmax_start = std::chrono::steady_clock::now();
                ops::diffusion_argmax(processed_host.data(), new_argmax.data(),
                                      canvas_len, vocab);
                const auto argmax_done = std::chrono::steady_clock::now();
                if (cur_step == 1) {
                    if (step_timing) {
                        std::fprintf(stderr,
                            "[DG step] step=%d h2d=%.1f decode_logits=%.1f "
                            "decoder_submit=%.1f logits_d2h=%.1f argmax=%.1f "
                            "entropy=0.0 gumbel=0.0 "
                            "accept_renoise=0.0 host_total=0.0 "
                            "softmax_submit=0.0 soft_gemm_submit=0.0 "
                            "soft_embed_d2h=0.0 total=%.1f ms\n",
                            cur_step,
                            std::chrono::duration<double, std::milli>(
                                h2d_done - step_start).count(),
                            std::chrono::duration<double, std::milli>(
                                logits_ready - decode_start).count(),
                            std::chrono::duration<double, std::milli>(
                                decoder_submit_done - decode_start).count(),
                            std::chrono::duration<double, std::milli>(
                                logits_d2h_done - logits_d2h_start).count(),
                            std::chrono::duration<double, std::milli>(
                                argmax_done - argmax_start).count(),
                            std::chrono::duration<double, std::milli>(
                                argmax_done - step_start).count());
                    }
                    break;
                }

                const auto host_start = std::chrono::steady_clock::now();
                softmax_entropy_host(processed_host.data(), entropy.data(),
                                     canvas_len, vocab);
                const auto entropy_done = std::chrono::steady_clock::now();
                gumbel_sample_host(processed_host.data(), rng, denoiser.data(),
                                   canvas_len, vocab);
                const auto gumbel_done = std::chrono::steady_clock::now();
                if (debug_probes && cur_step == cfg.max_denoising_steps) {
                    float min_l = 1e30f, max_l = -1e30f;
                    double sum_l = 0.0;
                    for (size_t i = 0; i < processed_host.size(); ++i) {
                        min_l = std::min(min_l, processed_host[i]);
                        max_l = std::max(max_l, processed_host[i]);
                        sum_l += processed_host[i];
                    }
                    AILA_LOG_INFO(
                        "[DG-DEBUG] step=%d logits min=%.4f max=%.4f mean=%.4f "
                        "entropy0=%.4f argmax0=%d argmax1=%d argmax2=%d "
                        "argmax3=%d argmax4=%d",
                        cur_step, min_l, max_l,
                        static_cast<float>(sum_l / processed_host.size()),
                        entropy[0], new_argmax[0], new_argmax[1],
                        new_argmax[2], new_argmax[3], new_argmax[4]);
                }
                ops::diffusion_accept_mask(entropy.data(), 1, canvas_len,
                                           cfg.entropy_bound, accept.data());
                for (int i = 0; i < canvas_len; ++i) {
                    new_canvas[static_cast<size_t>(i)] =
                        accept[static_cast<size_t>(i)]
                            ? denoiser[static_cast<size_t>(i)]
                            : canvas[static_cast<size_t>(i)];
                }
                // Renoising: fresh random tokens for every position, keep accepted.
                for (int i = 0; i < canvas_len; ++i) {
                    const int random_token = token_dist(rng);
                    canvas[static_cast<size_t>(i)] =
                        accept[static_cast<size_t>(i)]
                            ? new_canvas[static_cast<size_t>(i)]
                            : random_token;
                }

                if (new_argmax == prev_argmax) {
                    ++equal_count;
                } else {
                    equal_count = 0;
                }
                prev_argmax = new_argmax;

                float mean_entropy = 0.0f;
                for (int i = 0; i < canvas_len; ++i) {
                    mean_entropy += entropy[static_cast<size_t>(i)];
                }
                mean_entropy /= static_cast<float>(canvas_len);
                AILA_LOG_DEBUG("[DG] step=%d mean_entropy=%.4f", cur_step,
                               mean_entropy);
                const bool stable = equal_count >= cfg.stability_threshold;
                const bool confident =
                    mean_entropy < cfg.confidence_threshold;
                if (stable && confident) {
                    finished_denoising = true;
                }

                if (profile) {
                    profile_host_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - host_start)
                            .count();
                }
                if (finished_denoising) break;

                const auto soft_start = std::chrono::steady_clock::now();
                ctx.memcpy_h2d(logits.data(), processed_host.data(),
                               processed_host.size() * sizeof(float));
                Tensor& soft = backend.diffusion_compute_soft_embeds(
                    ctx, logits, canvas_len);
                const auto soft_submit_done = std::chrono::steady_clock::now();
                const auto soft_timing = backend.last_soft_embed_timing();
                ctx.memcpy_d2h(sc_embeds.data(), soft.data(),
                               sc_embeds.size() * sizeof(bf16));
                ctx.synchronize();
                const auto soft_done = std::chrono::steady_clock::now();
                if (step_timing) {
                    std::fprintf(stderr,
                        "[DG step] step=%d h2d=%.1f decode_logits=%.1f "
                        "decoder_submit=%.1f logits_d2h=%.1f argmax=%.1f "
                        "entropy=%.1f gumbel=%.1f "
                        "accept_renoise=%.1f host_total=%.1f "
                        "softmax_submit=%.1f soft_gemm_submit=%.1f "
                        "soft_embed_d2h=%.1f total=%.1f ms\n",
                        cur_step,
                        std::chrono::duration<double, std::milli>(
                            h2d_done - step_start)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            logits_ready - decode_start)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            decoder_submit_done - decode_start)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            logits_d2h_done - logits_d2h_start)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            argmax_done - argmax_start)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            entropy_done - host_start)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            gumbel_done - entropy_done)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            soft_start - gumbel_done)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            soft_start - host_start)
                            .count(),
                        soft_timing.softmax_submit_ms,
                        soft_timing.gemm_submit_ms,
                        std::chrono::duration<double, std::milli>(
                            soft_done - soft_submit_done)
                            .count(),
                        std::chrono::duration<double, std::milli>(
                            soft_done - step_start)
                            .count());
                }
                if (profile) {
                    profile_soft_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - soft_start)
                            .count();
                }
            }

            // Yield 5ms between steps to allow Windows DWM/WDDM frame composition
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Commit the argmax canvas.
        const std::vector<int> commit = new_argmax;
        if (committed_token_dump) {
            committed_token_dump.write(
                reinterpret_cast<const char*>(commit.data()),
                static_cast<std::streamsize>(commit.size() * sizeof(int)));
        }
        if (debug_probes) {
            std::string tok_list;
            for (int i = 0; i < std::min(16, canvas_len); ++i) {
                tok_list += std::to_string(commit[static_cast<size_t>(i)]) + " ";
            }
            AILA_LOG_INFO("[DG-DEBUG] committed tokens: %s", tok_list.c_str());
        }
        output_ids.insert(output_ids.end(), commit.begin(), commit.end());
        generated += canvas_len;
        if (callback) callback(commit);

        // EOS handling: pad everything after the first EOS in this canvas.
        bool eos_found = false;
        std::vector<int> padded = commit;
        for (int i = 0; i < canvas_len; ++i) {
            const int token = padded[static_cast<size_t>(i)];
            const bool is_eos =
                std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(),
                          token) != cfg.eos_token_ids.end();
            if (is_eos) {
                eos_found = true;
                for (int j = i + 1; j < canvas_len; ++j) {
                    padded[static_cast<size_t>(j)] = cfg.pad_token_id;
                }
                break;
            }
        }
        if (eos_found) {
            all_finished = true;
        } else if (generated >= max_new_tokens) {
            all_finished = true;
        } else {
            // Encode the committed canvas as context for the next canvas.
            ctx.memcpy_h2d(canvas_device, padded.data(),
                           static_cast<size_t>(canvas_len) * sizeof(int));
            std::string encode_error;
            if (!backend.diffusion_encode(ctx, canvas_device, canvas_len,
                                          &encode_error)) {
                if (error_message) *error_message = encode_error;
                ctx.free_device(canvas_device);
                ctx.free_device(sc_device);
                return false;
            }
            // Next canvas starts fresh.
            for (int i = 0; i < canvas_len; ++i) {
                canvas[static_cast<size_t>(i)] = token_dist(rng);
            }
            std::fill(sc_embeds.begin(), sc_embeds.end(), bf16(0.0f));
        }
    }

    ctx.synchronize();
    ctx.free_device(canvas_device);
    ctx.free_device(sc_device);
    if (use_gpu_sampling) {
        ctx.free_device(d_argmax);
        ctx.free_device(d_entropy);
        ctx.free_device(d_denoiser);
        ctx.free_device(d_seeds);
        ctx.free_device(d_mt_states);
    }
    if (generated > max_new_tokens) {
        output_ids.resize(static_cast<size_t>(max_new_tokens));
    }
    const double generation_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() -
                                     generation_start)
                                     .count();
    AILA_LOG_INFO(
        "[DiffusionGemma] Generated %zu tokens in %d canvas(es), %.2f ms "
        "(memory current=%.2f MB peak=%.2f MB)",
        output_ids.size(), max_canvases, generation_ms,
        ctx.current_allocated_bytes() / (1024.0 * 1024.0),
        ctx.peak_allocated_bytes() / (1024.0 * 1024.0));
    if (profile) {
        AILA_LOG_INFO(
            "[DG Profile] encode=%.2f ms decode+logits=%.2f ms "
            "host_sampling=%.2f ms soft_embedding=%.2f ms",
            profile_encode_ms, profile_decode_ms, profile_host_ms,
            profile_soft_ms);
    }
    return true;
}
