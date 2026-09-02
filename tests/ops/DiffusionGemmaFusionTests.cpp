#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Ops.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using bf16 = sycl::ext::oneapi::bfloat16;

static bool approx_equal(float a, float b, float tol = 0.05f) {
    return std::abs(a - b) <= tol * (std::abs(a) + std::abs(b) + 1.0e-3f);
}

void test_prefetch_compilation(Context& ctx) {
    Tensor t = Tensor::allocate(ctx, {1024}, dnnl::memory::data_type::f32);
    ctx.queue().submit([&](sycl::handler& cgh) {
        float* ptr = static_cast<float*>(t.data());
        cgh.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> item) {
            const int lid = static_cast<int>(item.get_local_id(0));
            // 1D raw prefetch
            sycl::ext::oneapi::experimental::prefetch(ptr + lid);
            // Simulated 2D Block Prefetch: [16, 64] float tile with pitch 64
            ops::Prefetch2D<float, 16, 64>::prefetch(ptr, 64, lid, 256);
            ptr[lid] = 1.0f;
        });
    });
    ctx.synchronize();
    std::cout << "[Test] 1D & Simulated 2D Block Prefetch verified!" << std::endl;
}

void test_fused_qkv_norm_rope(Context& ctx) {
    std::cout << "[Test] Fused Head RMSNorm + RoPE QKV..." << std::endl;
    const int seq_len = 256;
    const int start_pos = 0;
    const int num_heads = 16;
    const int num_kv_heads = 8;
    const int head_dim = 176;
    const int rotary_dim = 176;
    const float theta = 10000.0f;
    const float eps = 1e-6f;

    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> q_host(seq_len * q_dim);
    std::vector<bf16> k_host(seq_len * kv_dim);
    std::vector<bf16> v_host(seq_len * kv_dim);
    std::vector<bf16> q_norm_w(head_dim, bf16(1.0f));
    std::vector<bf16> k_norm_w(head_dim, bf16(1.0f));
    std::vector<bf16> ones_w(head_dim, bf16(1.0f));

    for (auto& x : q_host) x = bf16(dist(rng));
    for (auto& x : k_host) x = bf16(dist(rng));
    for (auto& x : v_host) x = bf16(dist(rng));

    Tensor q_dev = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(ctx, {seq_len, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(ctx, {seq_len, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor q_ref = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
    Tensor k_ref = Tensor::allocate(ctx, {seq_len, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor v_ref = Tensor::allocate(ctx, {seq_len, kv_dim}, dnnl::memory::data_type::bf16);

    Tensor q_norm_t = Tensor::allocate(ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_norm_t = Tensor::allocate(ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor ones_t = Tensor::allocate(ctx, {head_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(q_dev.data(), q_host.data(), q_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k_host.data(), k_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v_host.data(), v_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(q_ref.data(), q_host.data(), q_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_ref.data(), k_host.data(), k_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_ref.data(), v_host.data(), v_host.size() * sizeof(bf16));

    ctx.memcpy_h2d(q_norm_t.data(), q_norm_w.data(), q_norm_w.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_norm_t.data(), k_norm_w.data(), k_norm_w.size() * sizeof(bf16));
    ctx.memcpy_h2d(ones_t.data(), ones_w.data(), ones_w.size() * sizeof(bf16));

    // Baseline:
    ops::head_rms_norm(ctx, q_ref, q_norm_t, eps, seq_len, num_heads, head_dim);
    ops::head_rms_norm(ctx, k_ref, k_norm_t, eps, seq_len, num_kv_heads, head_dim);
    ops::head_rms_norm(ctx, v_ref, ones_t, eps, seq_len, num_kv_heads, head_dim);
    ops::apply_rope_partial(ctx, q_ref, k_ref, seq_len, start_pos, num_heads,
                            num_kv_heads, head_dim, rotary_dim, theta,
                            false, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, true);

    // Fused:
    ops::fused_head_rms_norm_rope_qkv(ctx, q_dev, k_dev, v_dev, q_norm_t, k_norm_t,
                                     ones_t, seq_len, start_pos, num_heads,
                                     num_kv_heads, head_dim, rotary_dim, theta,
                                     eps, true);

    ctx.synchronize();

    std::vector<bf16> q_res(q_host.size()), q_ref_h(q_host.size());
    std::vector<bf16> k_res(k_host.size()), k_ref_h(k_host.size());
    std::vector<bf16> v_res(v_host.size()), v_ref_h(v_host.size());

    ctx.memcpy_d2h(q_res.data(), q_dev.data(), q_res.size() * sizeof(bf16));
    ctx.memcpy_d2h(q_ref_h.data(), q_ref.data(), q_ref_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_res.data(), k_dev.data(), k_res.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_ref_h.data(), k_ref.data(), k_ref_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(v_res.data(), v_dev.data(), v_res.size() * sizeof(bf16));
    ctx.memcpy_d2h(v_ref_h.data(), v_ref.data(), v_ref_h.size() * sizeof(bf16));

    for (size_t i = 0; i < q_res.size(); ++i) {
        if (!approx_equal(static_cast<float>(q_res[i]), static_cast<float>(q_ref_h[i]))) {
            std::cerr << "Mismatch in Q at index " << i << ": fused="
                      << static_cast<float>(q_res[i]) << " ref="
                      << static_cast<float>(q_ref_h[i]) << std::endl;
            exit(1);
        }
    }
    for (size_t i = 0; i < k_res.size(); ++i) {
        if (!approx_equal(static_cast<float>(k_res[i]), static_cast<float>(k_ref_h[i]))) {
            std::cerr << "Mismatch in K at index " << i << ": fused="
                      << static_cast<float>(k_res[i]) << " ref="
                      << static_cast<float>(k_ref_h[i]) << std::endl;
            exit(1);
        }
    }
    for (size_t i = 0; i < v_res.size(); ++i) {
        if (!approx_equal(static_cast<float>(v_res[i]), static_cast<float>(v_ref_h[i]))) {
            std::cerr << "Mismatch in V at index " << i << ": fused="
                      << static_cast<float>(v_res[i]) << " ref="
                      << static_cast<float>(v_ref_h[i]) << std::endl;
            exit(1);
        }
    }
    std::cout << "[Test] Fused Head RMSNorm + RoPE QKV PASSED!" << std::endl;
}

void test_logits_row_max_inv_sum(Context& ctx) {
    std::cout << "[Test] Logits Row Max & InvSum Reduction..." << std::endl;
    const int canvas_len = 256;
    const int vocab = 262144;

    std::vector<float> logits(static_cast<size_t>(canvas_len) * vocab);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& x : logits) x = dist(rng);

    Tensor logits_dev = Tensor::allocate(ctx, {canvas_len, vocab}, dnnl::memory::data_type::f32);
    Tensor row_max_dev = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::f32);
    Tensor row_inv_sum_dev = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(logits_dev.data(), logits.data(), logits.size() * sizeof(float));

    ops::logits_row_max_inv_sum(ctx, static_cast<float*>(logits_dev.data()),
                                static_cast<float*>(row_max_dev.data()),
                                static_cast<float*>(row_inv_sum_dev.data()),
                                canvas_len, vocab);
    ctx.synchronize();

    std::vector<float> row_max_h(canvas_len);
    std::vector<float> row_inv_sum_h(canvas_len);
    ctx.memcpy_d2h(row_max_h.data(), row_max_dev.data(), canvas_len * sizeof(float));
    ctx.memcpy_d2h(row_inv_sum_h.data(), row_inv_sum_dev.data(), canvas_len * sizeof(float));

    // Verify against CPU reference for first few rows:
    for (int r = 0; r < 4; ++r) {
        float max_val = -1e30f;
        for (int c = 0; c < vocab; ++c) {
            float val = static_cast<float>(bf16(logits[static_cast<size_t>(r) * vocab + c]));
            max_val = std::max(max_val, val);
        }
        float sum = 0.0f;
        for (int c = 0; c < vocab; ++c) {
            float val = static_cast<float>(bf16(logits[static_cast<size_t>(r) * vocab + c]));
            sum += std::exp(val - max_val);
        }
        float inv_sum = 1.0f / sum;

        if (!approx_equal(row_max_h[r], max_val, 1e-4f)) {
            std::cerr << "Row max mismatch at row " << r << ": dev=" << row_max_h[r] << " ref=" << max_val << std::endl;
            exit(1);
        }
        if (!approx_equal(row_inv_sum_h[r], inv_sum, 1e-3f)) {
            std::cerr << "Row inv sum mismatch at row " << r << ": dev=" << row_inv_sum_h[r] << " ref=" << inv_sum << std::endl;
            exit(1);
        }
    }
    std::cout << "[Test] Logits Row Max & InvSum PASSED!" << std::endl;
}

static void test_fused_post_logits(Context& ctx) {
    constexpr int rows = 8;
    constexpr int vocab = 2048;
    constexpr float cap = 30.0f;
    constexpr float temperature = 0.8f;

    std::vector<bf16> logits_bf16(rows * vocab);
    std::vector<float> ref_logits_f32(rows * vocab);
    std::vector<int> ref_argmax(rows);
    std::vector<float> ref_entropy(rows);
    std::vector<float> ref_row_max(rows);
    std::vector<float> ref_row_inv_sum(rows);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < vocab; ++c) {
            float val = std::sin(static_cast<float>(r * vocab + c) * 0.05f) * 15.0f;
            logits_bf16[r * vocab + c] = bf16(val);
        }
        float max_v = -1e30f;
        int argm = 0;
        for (int c = 0; c < vocab; ++c) {
            float raw = static_cast<float>(logits_bf16[r * vocab + c]);
            float capped = std::tanh(raw / cap) * cap / temperature;
            ref_logits_f32[r * vocab + c] = capped;
            if (capped > max_v) {
                max_v = capped;
                argm = c;
            }
        }
        ref_row_max[r] = max_v;
        ref_argmax[r] = argm;

        float sum_e = 0.0f;
        float sum_le = 0.0f;
        for (int c = 0; c < vocab; ++c) {
            float e = std::exp(ref_logits_f32[r * vocab + c] - max_v);
            sum_e += e;
            sum_le += ref_logits_f32[r * vocab + c] * e;
        }
        ref_row_inv_sum[r] = 1.0f / sum_e;
        ref_entropy[r] = max_v + std::log(sum_e) - sum_le * ref_row_inv_sum[r];
    }

    Tensor d_bf16 = Tensor::allocate(ctx, {rows, vocab}, dnnl::memory::data_type::bf16);
    Tensor d_f32 = Tensor::allocate(ctx, {rows, vocab}, dnnl::memory::data_type::f32);
    Tensor d_argmax = Tensor::allocate(ctx, {rows}, dnnl::memory::data_type::s32);
    Tensor d_entropy = Tensor::allocate(ctx, {rows}, dnnl::memory::data_type::f32);
    Tensor d_row_max = Tensor::allocate(ctx, {rows}, dnnl::memory::data_type::f32);
    Tensor d_row_inv_sum = Tensor::allocate(ctx, {rows}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(d_bf16.data(), logits_bf16.data(), logits_bf16.size() * sizeof(bf16));

    ops::fused_post_logits_transform_stats_argmax(
        ctx, static_cast<const bf16*>(d_bf16.data()),
        static_cast<float*>(d_f32.data()),
        static_cast<int*>(d_argmax.data()),
        static_cast<float*>(d_entropy.data()),
        static_cast<float*>(d_row_max.data()),
        static_cast<float*>(d_row_inv_sum.data()),
        cap, temperature, rows, vocab);
    ctx.synchronize();

    std::vector<float> dev_f32(rows * vocab);
    std::vector<int> dev_argmax(rows);
    std::vector<float> dev_entropy(rows);
    std::vector<float> dev_row_max(rows);
    std::vector<float> dev_row_inv_sum(rows);

    ctx.memcpy_d2h(dev_f32.data(), d_f32.data(), dev_f32.size() * sizeof(float));
    ctx.memcpy_d2h(dev_argmax.data(), d_argmax.data(), dev_argmax.size() * sizeof(int));
    ctx.memcpy_d2h(dev_entropy.data(), d_entropy.data(), dev_entropy.size() * sizeof(float));
    ctx.memcpy_d2h(dev_row_max.data(), d_row_max.data(), dev_row_max.size() * sizeof(float));
    ctx.memcpy_d2h(dev_row_inv_sum.data(), d_row_inv_sum.data(), dev_row_inv_sum.size() * sizeof(float));

    for (int r = 0; r < rows; ++r) {
        if (dev_argmax[r] != ref_argmax[r]) {
            std::cerr << "Argmax mismatch at row " << r << ": dev=" << dev_argmax[r] << " ref=" << ref_argmax[r] << std::endl;
            exit(1);
        }
        if (!approx_equal(dev_row_max[r], ref_row_max[r], 1e-2f)) {
            std::cerr << "Row max mismatch at row " << r << ": dev=" << dev_row_max[r] << " ref=" << ref_row_max[r] << std::endl;
            exit(1);
        }
        if (!approx_equal(dev_entropy[r], ref_entropy[r], 2e-2f)) {
            std::cerr << "Entropy mismatch at row " << r << ": dev=" << dev_entropy[r] << " ref=" << ref_entropy[r] << std::endl;
            exit(1);
        }
    }
    std::cout << "[Test] Fused Post-Logits Transform Stats Argmax PASSED!" << std::endl;
}

void test_diffusion_accept_renoise(Context& ctx) {
    constexpr int canvas_len = 256;
    constexpr int vocab = 262144;
    const float entropy_bound = 0.1f;

    std::vector<float> entropy(canvas_len);
    std::vector<int> denoiser(canvas_len);
    std::vector<int> prev_canvas(canvas_len);
    std::vector<uint8_t> ref_accept(canvas_len);

    for (int i = 0; i < canvas_len; ++i) {
        entropy[i] = (i < 10) ? 0.001f * static_cast<float>(i + 1) : 0.5f + 0.01f * static_cast<float>(i);
        denoiser[i] = 1000 + i;
        prev_canvas[i] = 2000 + i;
    }

    ops::diffusion_accept_mask(entropy.data(), 1, canvas_len, entropy_bound, ref_accept.data());

    Tensor d_ent = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::f32);
    Tensor d_den = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::s32);
    Tensor d_can = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(d_ent.data(), entropy.data(), canvas_len * sizeof(float));
    ctx.memcpy_h2d(d_den.data(), denoiser.data(), canvas_len * sizeof(int));
    ctx.memcpy_h2d(d_can.data(), prev_canvas.data(), canvas_len * sizeof(int));

    ops::diffusion_accept_renoise_device(
        ctx, static_cast<const float*>(d_ent.data()),
        static_cast<const int*>(d_den.data()),
        static_cast<const int*>(d_can.data()),
        static_cast<int*>(d_can.data()),
        nullptr, entropy_bound, vocab, 1, canvas_len);
    ctx.synchronize();

    std::vector<int> dev_canvas(canvas_len);
    ctx.memcpy_d2h(dev_canvas.data(), d_can.data(), canvas_len * sizeof(int));

    int accepted_count = 0;
    for (int i = 0; i < canvas_len; ++i) {
        if (ref_accept[i]) {
            ++accepted_count;
            if (dev_canvas[i] != denoiser[i]) {
                std::cerr << "Accepted token mismatch at " << i << ": got " << dev_canvas[i] << " expected " << denoiser[i] << std::endl;
                exit(1);
            }
        } else {
            if (dev_canvas[i] < 0 || dev_canvas[i] >= vocab) {
                std::cerr << "Invalid renoised token at " << i << ": " << dev_canvas[i] << std::endl;
                exit(1);
            }
        }
    }

    if (accepted_count == 0) {
        std::cerr << "Expected some tokens to be accepted under entropy bound!" << std::endl;
        exit(1);
    }

    std::cout << "[Test] Diffusion Accept Renoise Device (" << accepted_count << " tokens accepted) PASSED!" << std::endl;
}

void run_prefetch_benchmark(Context& ctx) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "  Microbenchmark: 1D vs Simulated 2D Block Prefetch      " << std::endl;
    std::cout << "========================================================" << std::endl;

    const int K = 131072;
    const int N = 2816;
    const size_t total_elements = static_cast<size_t>(K) * N;
    const size_t total_bytes = total_elements * sizeof(bf16);
    double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Benchmarking 2D Matrix Streaming (" << (total_bytes / (1024 * 1024))
              << " MB, K=" << K << ", N=" << N << ")..." << std::endl;

    Tensor matrix = Tensor::allocate(ctx, {static_cast<int64_t>(total_elements)}, dnnl::memory::data_type::bf16);
    Tensor output = Tensor::allocate(ctx, {256, 2816}, dnnl::memory::data_type::bf16);
    ctx.queue().memset(matrix.data(), 0, total_bytes).wait();

    const bf16* mat_ptr = matrix.data_as<bf16>();
    bf16* out_ptr = output.data_as<bf16>();

    constexpr int BK = 16, BN = 64, WG_SIZE = 256;
    const int grid_n = (N + BN - 1) / BN;
    const int total_wgs = grid_n;

    auto run_kernel = [&](int mode) -> double {
        constexpr int iters = 20;
        constexpr int warmup = 5;

        for (int i = 0; i < warmup; ++i) {
            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE), [=](sycl::nd_item<1> item) {
                    const int wg_idx = static_cast<int>(item.get_group(0));
                    const int lid = static_cast<int>(item.get_local_id(0));
                    const int n0 = wg_idx * BN;
                    float acc = 0.0f;
                    for (int kb = 0; kb < K; kb += BK) {
                        if (mode == 1 && kb + BK < K) {
                            if (lid < (BN * BK) / 8) {
                                const int kh = lid / (BN / 8);
                                const int col8 = lid % (BN / 8);
                                const int src_n = n0 + col8 * 8;
                                if (src_n < N) {
                                    sycl::ext::oneapi::experimental::prefetch(
                                        mat_ptr + static_cast<size_t>(kb + BK + kh) * N + src_n);
                                }
                            }
                        } else if (mode == 2 && kb + BK < K) {
                            ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                mat_ptr + static_cast<size_t>(kb + BK) * N + n0, N, lid, WG_SIZE);
                        }
                        if (lid < (BN * BK) / 8) {
                            const int kh = lid / (BN / 8);
                            const int col8 = lid % (BN / 8);
                            const int src_n = n0 + col8 * 8;
                            if (src_n < N && kb + kh < K) {
                                auto v = *reinterpret_cast<const sycl::vec<bf16, 8>*>(
                                    mat_ptr + static_cast<size_t>(kb + kh) * N + src_n);
                                acc += static_cast<float>(v[0]);
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0 && wg_idx < 256) {
                        out_ptr[wg_idx] = bf16(acc);
                    }
                });
            });
        }
        ctx.synchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE), [=](sycl::nd_item<1> item) {
                    const int wg_idx = static_cast<int>(item.get_group(0));
                    const int lid = static_cast<int>(item.get_local_id(0));
                    const int n0 = wg_idx * BN;
                    float acc = 0.0f;
                    for (int kb = 0; kb < K; kb += BK) {
                        if (mode == 1 && kb + BK < K) {
                            if (lid < (BN * BK) / 8) {
                                const int kh = lid / (BN / 8);
                                const int col8 = lid % (BN / 8);
                                const int src_n = n0 + col8 * 8;
                                if (src_n < N) {
                                    sycl::ext::oneapi::experimental::prefetch(
                                        mat_ptr + static_cast<size_t>(kb + BK + kh) * N + src_n);
                                }
                            }
                        } else if (mode == 2 && kb + BK < K) {
                            ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                mat_ptr + static_cast<size_t>(kb + BK) * N + n0, N, lid, WG_SIZE);
                        }
                        if (lid < (BN * BK) / 8) {
                            const int kh = lid / (BN / 8);
                            const int col8 = lid % (BN / 8);
                            const int src_n = n0 + col8 * 8;
                            if (src_n < N && kb + kh < K) {
                                auto v = *reinterpret_cast<const sycl::vec<bf16, 8>*>(
                                    mat_ptr + static_cast<size_t>(kb + kh) * N + src_n);
                                acc += static_cast<float>(v[0]);
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0 && wg_idx < 256) {
                        out_ptr[wg_idx] = bf16(acc);
                    }
                });
            });
        }
        ctx.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        return ms;
    };

    double ms_none = run_kernel(0);
    double bw_none = (total_gb / (ms_none / 1000.0));
    std::cout << "  [Variant 0: No Prefetch]                 " << ms_none << " ms (" << bw_none << " GB/s)" << std::endl;

    double ms_1d = run_kernel(1);
    double bw_1d = (total_gb / (ms_1d / 1000.0));
    std::cout << "  [Variant 1: 1D Naive Prefetch]           " << ms_1d << " ms (" << bw_1d << " GB/s, "
              << (ms_none / ms_1d) << "x vs baseline)" << std::endl;

    double ms_2d = run_kernel(2);
    double bw_2d = (total_gb / (ms_2d / 1000.0));
    std::cout << "  [Variant 2: Simulated 2D Block Prefetch]   " << ms_2d << " ms (" << bw_2d << " GB/s, "
              << (ms_none / ms_2d) << "x vs baseline, " << (ms_1d / ms_2d) << "x vs 1D)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // Full Soft-Embedding XMX Benchmark with 1.47 GB weights
    const int vocab = 262144;
    const int hidden = 2816;
    const int canvas_len = 256;
    std::cout << "Benchmarking Soft-Embedding XMX Full Operator (1.47 GB Weight, Vocab=262144, Hidden=2816, Canvas=256)..." << std::endl;

    Tensor logits = Tensor::allocate(ctx, {canvas_len, vocab}, dnnl::memory::data_type::f32);
    Tensor embed_w = Tensor::allocate(ctx, {vocab, hidden}, dnnl::memory::data_type::bf16);
    Tensor row_max = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::f32);
    Tensor row_inv = Tensor::allocate(ctx, {canvas_len}, dnnl::memory::data_type::f32);
    Tensor embed_out = Tensor::allocate(ctx, {canvas_len, hidden}, dnnl::memory::data_type::bf16);

    ctx.queue().memset(logits.data(), 0, static_cast<size_t>(canvas_len) * vocab * sizeof(float)).wait();
    ctx.queue().memset(embed_w.data(), 0, static_cast<size_t>(vocab) * hidden * sizeof(bf16)).wait();
    ctx.queue().memset(row_max.data(), 0, static_cast<size_t>(canvas_len) * sizeof(float)).wait();
    ctx.queue().memset(row_inv.data(), 0, static_cast<size_t>(canvas_len) * sizeof(float)).wait();

    auto run_soft_embed = [&](int mode) -> double {
        using namespace sycl::ext::oneapi::experimental::matrix;
        constexpr int BM = 256, BN = 64, BK = 16, TM = 8, TN = 8, TK = 16, SG = 8;
        constexpr int TILES_PER_SG_M = 4;
        constexpr int SG_M = 8;
        constexpr int SG_N = 8;
        constexpr int NUM_SGS = SG_M * SG_N;
        constexpr int WG_SIZE = NUM_SGS * SG;

        const int grid_m = (canvas_len + BM - 1) / BM;
        const int grid_n = (hidden + BN - 1) / BN;
        const int total_wgs = grid_m * grid_n;

        const float* logits_ptr = logits.data_as<float>();
        const float* row_max_ptr = row_max.data_as<float>();
        const float* row_inv_sum_ptr = row_inv.data_as<float>();
        const bf16* B_ptr = embed_w.data_as<bf16>();
        bf16* C_ptr = embed_out.data_as<bf16>();
        const int M = canvas_len;
        const int N = hidden;
        const int K = vocab;

        constexpr int iters = 10;
        constexpr int warmup = 3;

        for (int i = 0; i < warmup; ++i) {
            ctx.queue().submit([&](sycl::handler& cgh) {
                sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
                sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(BK * BN), cgh);
                sycl::local_accessor<float, 1> s_row_max(sycl::range<1>(BM), cgh);
                sycl::local_accessor<float, 1> s_row_inv(sycl::range<1>(BM), cgh);

                cgh.parallel_for(
                    sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                        auto sg = item.get_sub_group();
                        const int wg_idx = static_cast<int>(item.get_group(0));
                        const int lid = static_cast<int>(item.get_local_id(0));
                        const int sg_id = lid / SG;
                        const int sg_row = sg_id / SG_N;
                        const int sg_col = sg_id % SG_N;
                        const int bm = wg_idx / grid_n;
                        const int bn = wg_idx % grid_n;
                        const int m0 = bm * BM;
                        const int n0 = bn * BN;

                        for (int k = lid; k < BM; k += WG_SIZE) {
                            s_row_max[k] = (m0 + k < M) ? row_max_ptr[m0 + k] : 0.0f;
                            s_row_inv[k] = (m0 + k < M) ? row_inv_sum_ptr[m0 + k] : 0.0f;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c[TILES_PER_SG_M];
                        #pragma unroll
                        for (int t = 0; t < TILES_PER_SG_M; ++t) {
                            joint_matrix_fill(sg, sub_c[t], 0.0f);
                        }

                        for (int kb = 0; kb < K; kb += BK) {
                            for (int pair_idx = lid; pair_idx < (BM * BK) / 2; pair_idx += WG_SIZE) {
                                const int r = pair_idx / (BK / 2);
                                const int c_pair = pair_idx % (BK / 2);
                                const int c0 = c_pair * 2;
                                const int c1 = c0 + 1;
                                const int gm_load = m0 + r;
                                if (gm_load < M) {
                                    const float r_max = s_row_max[r];
                                    const float r_inv = s_row_inv[r];
                                    bf16 val0 = bf16(0.0f);
                                    bf16 val1 = bf16(0.0f);
                                    if (kb + c0 < K) {
                                        const float l0 = logits_ptr[static_cast<size_t>(gm_load) * K + kb + c0];
                                        val0 = bf16(sycl::native::exp(l0 - r_max) * r_inv);
                                    }
                                    if (kb + c1 < K) {
                                        const float l1 = logits_ptr[static_cast<size_t>(gm_load) * K + kb + c1];
                                        val1 = bf16(sycl::native::exp(l1 - r_max) * r_inv);
                                    }
                                    const uint32_t packed_pair = (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&val1)) << 16) |
                                                                 (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&val0)));
                                    reinterpret_cast<uint32_t*>(As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = packed_pair;
                                } else {
                                    reinterpret_cast<uint32_t*>(As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = 0;
                                }
                            }

                            if (lid < (BN * BK) / 8) {
                                const int kh = lid / (BN / 8);
                                const int col8 = lid % (BN / 8);
                                const int src_n = n0 + col8 * 8;
                                if (src_n < N && kb + kh < K) {
                                    auto v = *reinterpret_cast<const sycl::vec<bf16, 8>*>(
                                        B_ptr + static_cast<size_t>(kb + kh) * N + src_n);
                                    #pragma unroll
                                    for (int j = 0; j < 8; ++j) {
                                        Bs[(col8 * 8 + j) * BK + kh] = v[j];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j = 0; j < 8; ++j) {
                                        Bs[(col8 * 8 + j) * BK + kh] = bf16(0.0f);
                                    }
                                }
                                if (mode == 1 && kb + BK + kh < K && src_n < N) {
                                    sycl::ext::oneapi::experimental::prefetch(
                                        B_ptr + static_cast<size_t>(kb + BK + kh) * N + src_n);
                                }
                            }

                            if (mode == 2 && kb + BK < K) {
                                ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                    B_ptr + static_cast<size_t>(kb + BK) * N + n0, N, lid, WG_SIZE);
                                ops::Prefetch2D<float, BM, BK>::prefetch(
                                    logits_ptr + static_cast<size_t>(m0) * K + kb + BK, K, lid, WG_SIZE);
                            } else if (mode == 3 && kb + 2 * BK < K) {
                                ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                    B_ptr + static_cast<size_t>(kb + 2 * BK) * N + n0, N, lid, WG_SIZE);
                                ops::Prefetch2D<float, BM, BK>::prefetch(
                                    logits_ptr + static_cast<size_t>(m0) * K + kb + 2 * BK, K, lid, WG_SIZE);
                            } else if (mode == 4 && kb + 4 * BK < K) {
                                ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                    B_ptr + static_cast<size_t>(kb + 4 * BK) * N + n0, N, lid, WG_SIZE);
                                ops::Prefetch2D<float, BM, BK>::prefetch(
                                    logits_ptr + static_cast<size_t>(m0) * K + kb + 4 * BK, K, lid, WG_SIZE);
                            }

                            item.barrier(sycl::access::fence_space::local_space);

                            joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;
                            auto pB = Bs.get_multi_ptr<sycl::access::decorated::no>() + (sg_col * TN) * BK;
                            joint_matrix_load(sg, sub_b, pB, BK);

                            #pragma unroll
                            for (int t = 0; t < TILES_PER_SG_M; ++t) {
                                joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
                                const int r_offset = (sg_row * TILES_PER_SG_M + t) * TM;
                                auto pA = As.get_multi_ptr<sycl::access::decorated::no>() + r_offset * BK;
                                joint_matrix_load(sg, sub_a, pA, BK);
                                joint_matrix_mad(sg, sub_c[t], sub_a, sub_b, sub_c[t]);
                            }

                            item.barrier(sycl::access::fence_space::local_space);
                        }
                    });
            });
        }
        ctx.synchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            ctx.queue().submit([&](sycl::handler& cgh) {
                sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
                sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(BK * BN), cgh);
                sycl::local_accessor<float, 1> s_row_max(sycl::range<1>(BM), cgh);
                sycl::local_accessor<float, 1> s_row_inv(sycl::range<1>(BM), cgh);

                cgh.parallel_for(
                    sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                        auto sg = item.get_sub_group();
                        const int wg_idx = static_cast<int>(item.get_group(0));
                        const int lid = static_cast<int>(item.get_local_id(0));
                        const int sg_id = lid / SG;
                        const int sg_row = sg_id / SG_N;
                        const int sg_col = sg_id % SG_N;
                        const int bm = wg_idx / grid_n;
                        const int bn = wg_idx % grid_n;
                        const int m0 = bm * BM;
                        const int n0 = bn * BN;

                        for (int k = lid; k < BM; k += WG_SIZE) {
                            s_row_max[k] = (m0 + k < M) ? row_max_ptr[m0 + k] : 0.0f;
                            s_row_inv[k] = (m0 + k < M) ? row_inv_sum_ptr[m0 + k] : 0.0f;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c[TILES_PER_SG_M];
                        #pragma unroll
                        for (int t = 0; t < TILES_PER_SG_M; ++t) {
                            joint_matrix_fill(sg, sub_c[t], 0.0f);
                        }

                        for (int kb = 0; kb < K; kb += BK) {
                            for (int pair_idx = lid; pair_idx < (BM * BK) / 2; pair_idx += WG_SIZE) {
                                const int r = pair_idx / (BK / 2);
                                const int c_pair = pair_idx % (BK / 2);
                                const int c0 = c_pair * 2;
                                const int c1 = c0 + 1;
                                const int gm_load = m0 + r;
                                if (gm_load < M) {
                                    const float r_max = s_row_max[r];
                                    const float r_inv = s_row_inv[r];
                                    bf16 val0 = bf16(0.0f);
                                    bf16 val1 = bf16(0.0f);
                                    if (kb + c0 < K) {
                                        const float l0 = logits_ptr[static_cast<size_t>(gm_load) * K + kb + c0];
                                        val0 = bf16(sycl::native::exp(l0 - r_max) * r_inv);
                                    }
                                    if (kb + c1 < K) {
                                        const float l1 = logits_ptr[static_cast<size_t>(gm_load) * K + kb + c1];
                                        val1 = bf16(sycl::native::exp(l1 - r_max) * r_inv);
                                    }
                                    const uint32_t packed_pair = (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&val1)) << 16) |
                                                                 (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&val0)));
                                    reinterpret_cast<uint32_t*>(As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = packed_pair;
                                } else {
                                    reinterpret_cast<uint32_t*>(As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = 0;
                                }
                            }

                            if (lid < (BN * BK) / 8) {
                                const int kh = lid / (BN / 8);
                                const int col8 = lid % (BN / 8);
                                const int src_n = n0 + col8 * 8;
                                if (src_n < N && kb + kh < K) {
                                    auto v = *reinterpret_cast<const sycl::vec<bf16, 8>*>(
                                        B_ptr + static_cast<size_t>(kb + kh) * N + src_n);
                                    #pragma unroll
                                    for (int j = 0; j < 8; ++j) {
                                        Bs[(col8 * 8 + j) * BK + kh] = v[j];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j = 0; j < 8; ++j) {
                                        Bs[(col8 * 8 + j) * BK + kh] = bf16(0.0f);
                                    }
                                }
                                if (mode == 1 && kb + BK + kh < K && src_n < N) {
                                    sycl::ext::oneapi::experimental::prefetch(
                                        B_ptr + static_cast<size_t>(kb + BK + kh) * N + src_n);
                                }
                            }

                            if (mode == 2 && kb + BK < K) {
                                ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                    B_ptr + static_cast<size_t>(kb + BK) * N + n0, N, lid, WG_SIZE);
                                ops::Prefetch2D<float, BM, BK>::prefetch(
                                    logits_ptr + static_cast<size_t>(m0) * K + kb + BK, K, lid, WG_SIZE);
                            } else if (mode == 3 && kb + 2 * BK < K) {
                                ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                    B_ptr + static_cast<size_t>(kb + 2 * BK) * N + n0, N, lid, WG_SIZE);
                                ops::Prefetch2D<float, BM, BK>::prefetch(
                                    logits_ptr + static_cast<size_t>(m0) * K + kb + 2 * BK, K, lid, WG_SIZE);
                            } else if (mode == 4 && kb + 4 * BK < K) {
                                ops::Prefetch2D<bf16, BK, BN>::prefetch(
                                    B_ptr + static_cast<size_t>(kb + 4 * BK) * N + n0, N, lid, WG_SIZE);
                                ops::Prefetch2D<float, BM, BK>::prefetch(
                                    logits_ptr + static_cast<size_t>(m0) * K + kb + 4 * BK, K, lid, WG_SIZE);
                            }

                            item.barrier(sycl::access::fence_space::local_space);

                            joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;
                            auto pB = Bs.get_multi_ptr<sycl::access::decorated::no>() + (sg_col * TN) * BK;
                            joint_matrix_load(sg, sub_b, pB, BK);

                            #pragma unroll
                            for (int t = 0; t < TILES_PER_SG_M; ++t) {
                                joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
                                const int r_offset = (sg_row * TILES_PER_SG_M + t) * TM;
                                auto pA = As.get_multi_ptr<sycl::access::decorated::no>() + r_offset * BK;
                                joint_matrix_load(sg, sub_a, pA, BK);
                                joint_matrix_mad(sg, sub_c[t], sub_a, sub_b, sub_c[t]);
                            }

                            item.barrier(sycl::access::fence_space::local_space);
                        }
                    });
            });
        }
        ctx.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    double xmx_none = run_soft_embed(0);
    double xmx_1d = run_soft_embed(1);
    double xmx_2d = run_soft_embed(2);
    double xmx_2d_dist2 = run_soft_embed(3);
    double xmx_2d_dist4 = run_soft_embed(4);

    double xmx_total_gb = (static_cast<double>(vocab) * hidden * sizeof(bf16) +
                           static_cast<double>(canvas_len) * vocab * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "  [Soft-Embed XMX 0: No Prefetch]               " << xmx_none << " ms ("
              << (xmx_total_gb / (xmx_none / 1000.0)) << " GB/s)" << std::endl;
    std::cout << "  [Soft-Embed XMX 1: 1D Naive Prefetch]         " << xmx_1d << " ms ("
              << (xmx_total_gb / (xmx_1d / 1000.0)) << " GB/s, " << (xmx_none / xmx_1d) << "x vs baseline)" << std::endl;
    std::cout << "  [Soft-Embed XMX 2: Simulated 2D (Lookahead 1)] " << xmx_2d << " ms ("
              << (xmx_total_gb / (xmx_2d / 1000.0)) << " GB/s, " << (xmx_none / xmx_2d) << "x vs baseline)" << std::endl;
    std::cout << "  [Soft-Embed XMX 3: Simulated 2D (Lookahead 2)] " << xmx_2d_dist2 << " ms ("
              << (xmx_total_gb / (xmx_2d_dist2 / 1000.0)) << " GB/s, " << (xmx_none / xmx_2d_dist2) << "x vs baseline)" << std::endl;
    std::cout << "  [Soft-Embed XMX 4: Simulated 2D (Lookahead 4)] " << xmx_2d_dist4 << " ms ("
              << (xmx_total_gb / (xmx_2d_dist4 / 1000.0)) << " GB/s, " << (xmx_none / xmx_2d_dist4) << "x vs baseline)" << std::endl;

    std::cout << "========================================================\n" << std::endl;
}

int main() {
    Context ctx;
    std::cout << "Running DiffusionGemma Fusion Unit Tests on: "
              << ctx.queue().get_device().get_info<sycl::info::device::name>() << std::endl;

    test_prefetch_compilation(ctx);
    test_fused_qkv_norm_rope(ctx);
    test_logits_row_max_inv_sum(ctx);
    test_fused_post_logits(ctx);
    test_diffusion_accept_renoise(ctx);

    run_prefetch_benchmark(ctx);

    std::cout << "All DiffusionGemma Fusion Unit Tests PASSED!" << std::endl;
    return 0;
}
