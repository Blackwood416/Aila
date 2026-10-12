// Included inside the kernel test executable's anonymous namespace.
template <int M, int SG, bool Slm>
void run_int8_dpas_probe(Context& ctx) {
    using namespace sycl::ext::oneapi::experimental::matrix;
    constexpr int K = 256, BK = 32, N = 8;
    std::vector<int8_t> a(M * K), b(N * K);
    for (int i = 0; i < M * K; ++i) a[i] = int8_t((i * 7 + 3) % 255 - 127);
    for (int i = 0; i < N * K; ++i) b[i] = int8_t((i * 11 + 5) % 255 - 127);
    auto da = Tensor::allocate(ctx, {M * K}, dnnl::memory::data_type::s8);
    auto db = Tensor::allocate(ctx, {N * K}, dnnl::memory::data_type::s8);
    auto dc = Tensor::allocate(ctx, {M * N}, dnnl::memory::data_type::s32);
    struct Drain { Context& c; ~Drain() { try { c.synchronize(); } catch (...) {} } } drain{ctx};
    ctx.memcpy_h2d(da.data(), a.data(), a.size());
    ctx.memcpy_h2d(db.data(), b.data(), b.size());
    auto* ap = da.data_as<int8_t>();
    auto* bp = db.data_as<int8_t>();
    auto* cp = dc.data_as<int32_t>();
    std::cout << "[INT8 DPAS PROBE] M=" << M << " SG=" << SG
              << " slm=" << Slm << " K=" << K << " N=" << N << '\n' << std::flush;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int8_t, 1> qa(Slm ? M * K : 1, cgh);
        sycl::local_accessor<int32_t, 1> acc(Slm ? M * N : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>(SG, SG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                if constexpr (Slm) {
                    for (int i = it.get_local_id(0); i < M * K; i += SG) qa[i] = ap[i];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                auto sg = it.get_sub_group();
                joint_matrix<sycl::sub_group, int8_t, use::a, M, BK, layout::row_major> am;
                joint_matrix<sycl::sub_group, int8_t, use::b, BK, N, layout::col_major> bm;
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, M, N> cm;
                joint_matrix_fill(sg, cm, int32_t(0));
                for (int k = 0; k < K; k += BK) {
                    if constexpr (Slm) {
                        joint_matrix_load(sg, am, qa.template get_multi_ptr<sycl::access::decorated::no>() + k, K);
                    } else {
                        joint_matrix_load(sg, am, sycl::address_space_cast<sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(ap + k), K);
                    }
                    joint_matrix_load(sg, bm, sycl::address_space_cast<sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(bp + k), K);
                    joint_matrix_mad(sg, cm, am, bm, cm);
                }
                if constexpr (Slm) {
                    joint_matrix_store(sg, cm, acc.template get_multi_ptr<sycl::access::decorated::no>(), N, layout::row_major);
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int i = it.get_local_id(0); i < M * N; i += SG) cp[i] = acc[i];
                } else {
                    joint_matrix_store(sg, cm, sycl::address_space_cast<sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(cp), N, layout::row_major);
                }
            });
    }).wait_and_throw();
    std::vector<int32_t> out(M * N);
    ctx.memcpy_d2h(out.data(), dc.data(), out.size() * sizeof(int32_t));
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        int32_t ref = 0;
        for (int k = 0; k < K; ++k) ref += int(a[m * K + k]) * int(b[n * K + k]);
        if (out[m * N + n] != ref) throw std::runtime_error("INT8 DPAS signed/col-major/SLM mismatch");
    }
    std::cout << "[INT8 DPAS PROBE PASS] errors=0/" << M * N << '\n';
}

void test_int8_dpas_probe(Context& ctx) {
    run_int8_dpas_probe<8, 8, false>(ctx);
    run_int8_dpas_probe<8, 8, true>(ctx);
    run_int8_dpas_probe<1, 8, true>(ctx);
}

void test_int8_dpas_attention(Context& ctx) {
    constexpr int H = 4, HK = 2, D = 256, tile = 128, stride = 272;
    auto run = [&](int len, bool sparse, bool gated) {
        const int count = sparse ? 1027 : len, tiles = (count + tile - 1) / tile;
        std::mt19937 rng(8127 + len);
        std::uniform_int_distribution<int> codes(-128, 127);
        std::uniform_real_distribution<float> queries(-6.0f, 6.0f);
        std::vector<int8_t> k(HK * len * D), v(k.size());
        std::vector<float> ks(HK * len), vs(ks.size());
        std::vector<bf16> q(H * D), gate(q.size()), got(q.size() + 8);
        for (auto& x : k) x = int8_t(codes(rng));
        for (auto& x : v) x = int8_t(codes(rng));
        for (int i = 0; i < HK * len; ++i) {
            ks[i] = 0.0005f * (1 + i % 11);
            vs[i] = 0.001f * (1 + i % 7);
        }
        for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d) {
            q[h * D + d] = bf16(h == 0 ? 0.0f : (h == 1 ? 0.75f : queries(rng)));
            gate[h * D + d] = bf16(float(d % 17 - 8) / 4.0f);
        }
        std::vector<int> indices(count);
        for (int t = 0; t < count; ++t) indices[t] = sparse ? (t * 37) % len : t;
        auto tq = Tensor::allocate(ctx, {1, H * D}, dnnl::memory::data_type::bf16);
        auto tg = Tensor::allocate(ctx, {1, H * D}, dnnl::memory::data_type::bf16);
        auto tk = Tensor::allocate(ctx, {HK, len, D}, dnnl::memory::data_type::s8);
        auto tv = Tensor::allocate(ctx, {HK, len, D}, dnnl::memory::data_type::s8);
        auto tks = Tensor::allocate(ctx, {HK, len}, dnnl::memory::data_type::f32);
        auto tvs = Tensor::allocate(ctx, {HK, len}, dnnl::memory::data_type::f32);
        auto ti = Tensor::allocate(ctx, {count}, dnnl::memory::data_type::s32);
        auto storage = Tensor::allocate(ctx, {H * D + 8}, dnnl::memory::data_type::bf16);
        auto output = Tensor::view(ctx, storage.data(), {1, H * D}, storage.dtype());
        auto scores = Tensor::allocate(ctx, {H, len}, dnnl::memory::data_type::f32);
        auto partials = Tensor::allocate(ctx, {H, tiles, stride}, dnnl::memory::data_type::f32);
        struct Drain { Context& c; ~Drain() { try { c.synchronize(); } catch (...) {} } } drain{ctx};
        ctx.memcpy_h2d(tq.data(), q.data(), q.size() * 2);
        ctx.memcpy_h2d(tg.data(), gate.data(), gate.size() * 2);
        ctx.memcpy_h2d(tk.data(), k.data(), k.size());
        ctx.memcpy_h2d(tv.data(), v.data(), v.size());
        ctx.memcpy_h2d(tks.data(), ks.data(), ks.size() * 4);
        ctx.memcpy_h2d(tvs.data(), vs.data(), vs.size() * 4);
        ctx.memcpy_h2d(ti.data(), indices.data(), indices.size() * 4);
        std::fill(got.begin(), got.end(), bf16(42.0f));
        ctx.memcpy_h2d(storage.data(), got.data(), got.size() * 2);
        const auto allocations = ctx.device_allocation_count(), pinned = ctx.pinned_host_allocation_count();
        ops::attention_decode(ctx, tq, tk, tv, output, scores, H, HK, D, len,
            &partials, gated ? &tg : nullptr, sparse ? ti.data_as<int>() : nullptr,
            sparse ? count : 0, nullptr, 0, &tks, &tvs);
        ctx.memcpy_d2h(got.data(), storage.data(), got.size() * 2);
        std::vector<float> parts(H * tiles * stride);
        ctx.memcpy_d2h(parts.data(), partials.data(), parts.size() * 4);
        double max_abs = 0.0, se = 0.0, norm = 0.0, max_logit_error = 0.0;
        for (int h = 0; h < H; ++h) {
            const int kh = h / (H / HK);
            std::vector<double> logits(count);
            // Independent f64 reference uses the original BF16 Q and logical
            // signed KV rows, with no matrix packing or Q quantization helper.
            for (int t = 0; t < count; ++t) {
                const int row = kh * len + indices[t];
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += double(float(q[h * D + d])) * int(k[row * D + d]);
                logits[t] = dot * double(ks[row]) / 16.0;
            }
            for (int t = 0; t < tiles; ++t) {
                const int end = std::min(count, (t + 1) * tile);
                const double m = *std::max_element(logits.begin() + t * tile, logits.begin() + end);
                const double actual = parts[(h * tiles + t) * stride];
                if (!std::isfinite(actual)) throw std::runtime_error("INT8 DPAS nonfinite tile max");
                max_logit_error = std::max(max_logit_error, std::abs(actual - m));
            }
            const double m = *std::max_element(logits.begin(), logits.end());
            double sum = 0.0;
            for (auto& x : logits) { x = std::exp(x - m); sum += x; }
            for (int d = 0; d < D; ++d) {
                double ref = 0.0;
                for (int t = 0; t < count; ++t) {
                    const int row = kh * len + indices[t];
                    ref += logits[t] * int(v[row * D + d]) * double(vs[row]);
                }
                ref = float(bf16(float(ref / sum)));
                if (gated) ref = float(bf16(float(ref / (1.0 + std::exp(-double(float(gate[h * D + d])))))));
                const double actual = float(got[h * D + d]), err = std::abs(actual - ref);
                if (!std::isfinite(actual) || err > 2e-5 + 0.008 * std::abs(ref))
                    throw std::runtime_error("INT8 DPAS attention vs independent f64 reference");
                max_abs = std::max(max_abs, err); se += err * err; norm += ref * ref;
            }
        }
        if (max_logit_error > 3e-4) throw std::runtime_error("INT8 DPAS Q residual/chunk/scales regression");
        for (size_t i = q.size(); i < got.size(); ++i)
            if (float(got[i]) != 42.0f) throw std::runtime_error("INT8 DPAS output canary changed");
        if (ctx.device_allocation_count() != allocations || ctx.pinned_host_allocation_count() != pinned)
            throw std::runtime_error("INT8 DPAS attention hot USM allocation");
        std::cout << "[INT8 DPAS ATTENTION PASS] len=" << len << " cap=len sparse=" << sparse
                  << " gated=" << gated << " f64_max_abs=" << max_abs
                  << " nrmse=" << std::sqrt(se / norm) << " tile_max_error=" << max_logit_error << '\n';
    };
    for (int len : {1024, 1025, 1031, 1032, 1152, 1834, 14336}) run(len, false, len % 2 == 1);
    run(1153, true, true);
}
