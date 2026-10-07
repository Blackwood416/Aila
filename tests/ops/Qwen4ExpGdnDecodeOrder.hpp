// Decode and short verifier windows must preserve the same recurrent trajectory.
void test_gdn_decode_order(Context& ctx) {
    constexpr int tokens = 128;
    auto g = ops::DeltaNetGeometry::create(16, 48, 128, 128, 4, 1e-6f, ops::GateActivation::Sigmoid);
    std::mt19937 rng(1954);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    auto bf = [&](int n) { std::vector<bf16> v(n); for (auto& x : v) x = bf16(dist(rng)); return v; };
    auto fp = [&](int n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
    auto input = bf(tokens * g.all_dim), conv = bf(g.qkv_dim * g.kernel), bias = bf(g.v_heads);
    auto initial = fp(g.v_heads * 128 * 128), history = fp(g.conv_rows * g.qkv_dim);
    auto norm = fp(128), alog = fp(g.v_heads);
    auto upload = [&](const auto& v, dnnl::memory::data_type type) {
        auto t = Tensor::allocate(ctx, {int64_t(v.size())}, type);
        ctx.memcpy_h2d(t.data(), v.data(), t.size_bytes()); return t;
    };
    auto x = upload(input, dnnl::memory::data_type::bf16);
    auto cw = upload(conv, dnnl::memory::data_type::bf16);
    auto db = upload(bias, dnnl::memory::data_type::bf16);
    auto nw = upload(norm, dnnl::memory::data_type::f32);
    auto al = upload(alog, dnnl::memory::data_type::f32);
    auto s = upload(initial, dnnl::memory::data_type::f32);
    auto h = upload(history, dnnl::memory::data_type::f32);
    auto out = Tensor::allocate(ctx, {tokens, g.kv_dim}, dnnl::memory::data_type::bf16);
    ops::DeltaNetWeights weights{cw.data_as<bf16>(), nw.data_as<float>(), al.data_as<float>(), db.data_as<bf16>()};
    const auto live = ctx.device_allocation_count(), pinned = ctx.pinned_host_allocation_count();
    std::vector<bf16> reference(tokens * g.kv_dim), actual(reference.size());
    std::vector<float> stateRef(initial.size()), stateActual(initial.size()), historyRef(history.size()), historyActual(history.size());
    for (auto act : {ops::GateActivation::Sigmoid, ops::GateActivation::Silu}) {
        g.gate_act = act;
        for (int variant = 0; variant < 2; ++variant) {
            ctx.memcpy_h2d(s.data(), initial.data(), s.size_bytes());
            ctx.memcpy_h2d(h.data(), history.data(), h.size_bytes());
            ops::DeltaNetState state{s.data_as<float>(), h.data_as<float>(), 2, false};
            int cycle = 0;
            for (int p = 0; p < tokens;) {
                const auto* row = x.data_as<bf16>() + size_t(p) * g.all_dim;
                auto* dst = out.data_as<bf16>() + size_t(p) * g.kv_dim;
                if (variant == 0) {
                    ops::run_linear_delta_decode_gpu(ctx, g, weights, state, row, row + g.qkv_dim,
                        row + g.qkv_dim + g.z_dim, row + g.qkv_dim + g.z_dim + g.v_heads, dst);
                    ++p;
                } else {
                    // Mixed widths exercise accepted windows and one-row rejection replay.
                    constexpr int widths[] = {1, 2, 4};
                    const int count = std::min(tokens - p, widths[cycle++ % 3]);
                    ops::run_linear_delta_prefill_gpu_batched(ctx, g, weights, state, row, dst, count, true);
                    p += count;
                }
            }
            ctx.memcpy_d2h(actual.data(), out.data(), out.size_bytes());
            ctx.memcpy_d2h(stateActual.data(), s.data(), s.size_bytes());
            ctx.memcpy_d2h(historyActual.data(), h.data(), h.size_bytes());
            if (state.conv_head != (2 + tokens) % g.conv_rows) throw std::runtime_error("GDN ring head mismatch");
            if (variant == 0) { reference = actual; stateRef = stateActual; historyRef = historyActual; }
            else if (std::memcmp(reference.data(), actual.data(), out.size_bytes()) ||
                     std::memcmp(stateRef.data(), stateActual.data(), s.size_bytes()) ||
                     std::memcmp(historyRef.data(), historyActual.data(), h.size_bytes()))
                throw std::runtime_error("GDN verifier differs from sequential decode");
        }
    }
    if (ctx.device_allocation_count() != live || ctx.pinned_host_allocation_count() != pinned)
        throw std::runtime_error("GDN verifier grew live USM");
    std::cout << "GDN 128-token mixed B1/B2/B4: output and FP32 state bitwise; both gates; live USM unchanged\n";
}
