#include "DiffusionGemmaGenerator.hpp"

#include "../ops/Ops.hpp"
#include "../utils/EnvUtils.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <thread>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

void parallel_rows(int rows, const std::function<void(int)>& fn) {
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
    std::ofstream committed_token_dump;
    if (!dump_dir.empty()) {
        committed_token_dump.open(dump_dir + "/committed_tokens.bin",
                                  std::ios::binary | std::ios::trunc);
    }
    for (int c = 0; c < max_canvases && !all_finished; ++c) {
        std::vector<int> prev_argmax(canvas);
        int equal_count = 0;
        bool finished_denoising = false;

        for (int cur_step = cfg.max_denoising_steps; cur_step >= 1;
             --cur_step) {
            const auto decode_start = std::chrono::steady_clock::now();
            AILA_LOG_DEBUG("[DiffusionGemma] Canvas %d/%d step %d", c + 1,
                           max_canvases, cur_step);
            ctx.memcpy_h2d(canvas_device, canvas.data(),
                           static_cast<size_t>(canvas_len) * sizeof(int));
            ctx.memcpy_h2d(sc_device, sc_embeds.data(),
                           sc_embeds.size() * sizeof(bf16));
            Tensor& logits = backend.diffusion_decode_canvas(
                ctx, canvas_device, sc_device, canvas_len);
            const float temperature =
                cfg.t_min + (cfg.t_max - cfg.t_min) *
                                (static_cast<float>(cur_step) /
                                 static_cast<float>(cfg.max_denoising_steps));
            // Divide the logits in place on the GPU with the same IEEE
            // division the host performed before, so the soft self-
            // conditioning path reads bitwise-identical processed logits
            // without a 256 MB host->device round trip per step.
            float* logits_ptr = static_cast<float*>(logits.data());
            ctx.queue().parallel_for(
                sycl::range<1>(static_cast<size_t>(canvas_len) * vocab),
                [=](sycl::id<1> idx) {
                    logits_ptr[idx[0]] = logits_ptr[idx[0]] / temperature;
                });
            ctx.memcpy_d2h(logits_host.data(), logits.data(),
                           logits_host.size() * sizeof(float));
            ctx.synchronize();
            const auto transfer_done = std::chrono::steady_clock::now();
            if (profile) {
                profile_decode_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - decode_start)
                        .count();
            }

            // The committed canvas is always the denoiser argmax. On the
            // final step no later iteration can consume entropy sampling,
            // renoising, or self-conditioning, so avoid one complete
            // full-vocabulary host pass and one probs@embedding GEMM.
            ops::diffusion_argmax(logits_host.data(), new_argmax.data(),
                                  canvas_len, vocab);
            if (cur_step == 1) break;

            const auto host_start = std::chrono::steady_clock::now();
            softmax_entropy_host(logits_host.data(), entropy.data(),
                                 canvas_len, vocab);
            gumbel_sample_host(logits_host.data(), rng, denoiser.data(),
                               canvas_len, vocab);
            if (debug_probes && cur_step == cfg.max_denoising_steps) {
                float min_l = 1e30f, max_l = -1e30f;
                double sum_l = 0.0;
                for (size_t i = 0; i < logits_host.size(); ++i) {
                    min_l = std::min(min_l, logits_host[i]);
                    max_l = std::max(max_l, logits_host[i]);
                    sum_l += logits_host[i];
                }
                AILA_LOG_INFO(
                    "[DG-DEBUG] step=%d logits min=%.4f max=%.4f mean=%.4f "
                    "entropy0=%.4f argmax0=%d argmax1=%d argmax2=%d "
                    "argmax3=%d argmax4=%d",
                    cur_step, min_l, max_l,
                    static_cast<float>(sum_l / logits_host.size()),
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
            Tensor& soft = backend.diffusion_compute_soft_embeds(
                ctx, logits, canvas_len);
            ctx.memcpy_d2h(sc_embeds.data(), soft.data(),
                           sc_embeds.size() * sizeof(bf16));
            ctx.synchronize();
            const auto soft_done = std::chrono::steady_clock::now();
            if (step_timing) {
                std::fprintf(stderr,
                    "[DG step] step=%d decode=%.1f ms transfer=%.1f ms "
                    "host=%.1f ms soft=%.1f ms\n",
                    cur_step,
                    std::chrono::duration<double, std::milli>(
                        transfer_done - decode_start)
                        .count(),
                    std::chrono::duration<double, std::milli>(
                        soft_start - transfer_done)
                        .count(),
                    std::chrono::duration<double, std::milli>(
                        soft_start - host_start)
                        .count(),
                    std::chrono::duration<double, std::milli>(
                        soft_done - soft_start)
                        .count());
            }
            if (profile) {
                profile_soft_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - soft_start)
                        .count();
            }

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

    ctx.free_device(canvas_device);
    ctx.free_device(sc_device);
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
