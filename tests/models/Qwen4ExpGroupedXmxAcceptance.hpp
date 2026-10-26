// Included by Qwen4ExpToleranceTests.cpp, inside its test namespace.
struct X3Error {
    double square = 0, reference = 0, max_abs = 0;
    size_t count = 0;
    void Add(double ref, double got) {
        expect(std::isfinite(ref) && std::isfinite(got), "X3 nonfinite output");
        const double d = got - ref;
        square += d * d; reference += ref * ref; ++count;
        max_abs = std::max(max_abs, std::abs(d));
    }
    double Nrmse() const { return std::sqrt(square / std::max(reference, 1e-20)); }
};

void test_grouped_xmx_kernel() {
    std::cout << std::unitbuf;
    Context ctx(nullptr, true);
    constexpr int H = 64, I = 64, T = 40, R = 2, E = 3;
    const std::array<int, E> sizes{8, 9, 17};
    std::mt19937 rng(7381);
    std::vector<uint8_t> packed(E * 3 * H * I / 2), codes(E * 3 * H * I / 64);
    std::vector<float> am(codes.size()), nested_am((codes.size() + 31) / 32), nqmap(256);
    std::vector<bf16> x(T * H);
    std::array<float, 16> qmap{};
    for (int i = 0; i < 16; ++i) qmap[i] = kTestQuantMap[i] * 0.9f + 0.001f;
    for (int i = 0; i < 256; ++i) nqmap[i] = float(i + 1) / 256.f;
    for (auto& p : packed) p = uint8_t(rng());
    for (auto& p : codes) p = uint8_t(rng());
    for (size_t i = 0; i < nested_am.size(); ++i) nested_am[i] = .02f + .001f * float(i);
    for (size_t i = 0; i < am.size(); ++i) am[i] = nqmap[codes[i]] * nested_am[i / 32] + 0.01f;
    for (auto& v : x) v = bf16((int(rng() % 1000) - 500) / 500.f);
    auto bytes = [&](size_t n) { return Tensor::allocate(ctx, {int64_t(n)}, dnnl::memory::data_type::u8); };
    auto pd = bytes(packed.size()), cd = bytes(codes.size()), ad = bytes(am.size() * 4);
    auto nd = bytes(nested_am.size() * 4), nqd = bytes(256 * 4), qd = bytes(16 * 4), xd = bytes(x.size() * 2);
    ctx.memcpy_h2d(cd.data(), codes.data(), codes.size());
    ctx.memcpy_h2d(ad.data(), am.data(), am.size() * 4);
    ctx.memcpy_h2d(nd.data(), nested_am.data(), nested_am.size() * 4);
    ctx.memcpy_h2d(nqd.data(), nqmap.data(), 256 * 4);
    ctx.memcpy_h2d(qd.data(), qmap.data(), 16 * 4);
    ctx.memcpy_h2d(xd.data(), x.data(), x.size() * 2);
    std::vector<int> tokens, ranks, experts;
    std::vector<ops::MoeExpertSlotDev> slots;
    std::vector<ops::MoeExpertTile> tiles;
    for (int e = 0; e < E; ++e) {
        const int start = int(tokens.size());
        ops::MoeExpertSlotDev slot{};
        const size_t pbase = e * 3 * H * I / 2, abase = e * 3 * H * I / 64;
        slot.gate_up_packed = pd.data_as<uint8_t>() + pbase;
        slot.down_packed = slot.gate_up_packed + H * I;
        slot.gate_up_absmax = ad.data_as<float>() + abase;
        slot.down_absmax = slot.gate_up_absmax + 2 * H * I / 64;
        slot.gate_up_nested = {cd.data_as<uint8_t>() + abase, nd.data_as<float>() + abase / 32, nqd.data_as<float>(), .01f, 32};
        slot.down_nested = slot.gate_up_nested;
        slot.down_nested.codes += 2 * H * I / 64;
        slot.down_nested.nested_absmax += 2 * H * I / 64 / 32;
        for (int i = 0; i < sizes[e]; ++i) {
            const int unique = start + i;
            tokens.push_back((unique * 7) % T); ranks.push_back(unique % R);
            experts.push_back(e); slots.push_back(slot);
        }
        for (int i = 0; i < sizes[e]; i += 8) tiles.push_back({start + i, std::min(8, sizes[e] - i)});
    }
    auto td = bytes(tokens.size() * 4), rd = bytes(ranks.size() * 4), sd = bytes(slots.size() * sizeof(slots[0]));
    auto tiled = bytes(tiles.size() * sizeof(tiles[0]));
    auto hd = bytes((T * R * I + 16) * 2), dd = bytes((T * R * H + 16) * 4);
    ctx.memcpy_h2d(td.data(), tokens.data(), tokens.size() * 4);
    ctx.memcpy_h2d(rd.data(), ranks.data(), ranks.size() * 4);
    ctx.memcpy_h2d(sd.data(), slots.data(), slots.size() * sizeof(slots[0]));
    ctx.memcpy_h2d(tiled.data(), tiles.data(), tiles.size() * sizeof(tiles[0]));
    const auto weight = [&](int e, int projection, int n, int k) {
        const size_t idx = size_t(e) * 3 * H * I + projection * H * I + n * H + k;
        const int code = (packed[idx / 2] >> (idx % 2 ? 0 : 4)) & 15;
        return double(float(bf16(qmap[code] * am[idx / 64])));
    };
    std::vector<bool> touched(T * R, false);
    for (size_t j = 0; j < tokens.size(); ++j) {
        expect(!touched[tokens[j] * R + ranks[j]], "Duplicate synthetic rank");
        touched[tokens[j] * R + ranks[j]] = true;
    }
    for (bool nested : {false, true}) {
        std::vector<bf16> actual_h(T * R * I + 16, bf16(-9.f));
        std::vector<float> actual_d(T * R * H + 16, -9.f);
        ctx.memcpy_h2d(hd.data(), actual_h.data(), actual_h.size() * 2);
        ctx.memcpy_h2d(dd.data(), actual_d.data(), actual_d.size() * 4);
        // Independent DMA and overwrite queue tests both ready and final consumer.
        sycl::queue copy(ctx.queue().get_context(), ctx.queue().get_device());
        auto ready = copy.memcpy(pd.data(), packed.data(), packed.size());
        auto gu = ops::moe_grouped_xmx_gate_up(ctx, sd.data_as<ops::MoeExpertSlotDev>(),
            tiled.data_as<ops::MoeExpertTile>(), int(tiles.size()), td.data_as<int>(), rd.data_as<int>(),
            qd.data_as<float>(), xd.data_as<bf16>(), hd.data_as<bf16>(), R, H, I, nested, &ready);
        auto dn = ops::moe_grouped_xmx_down(ctx, sd.data_as<ops::MoeExpertSlotDev>(),
            tiled.data_as<ops::MoeExpertTile>(), int(tiles.size()), td.data_as<int>(), rd.data_as<int>(),
            qd.data_as<float>(), hd.data_as<bf16>(), dd.data_as<float>(), R, H, I, nested, &gu);
        auto reuse = copy.submit([&](sycl::handler& cgh) { cgh.depends_on(dn); cgh.memset(pd.data(), 0, packed.size()); });
        reuse.wait_and_throw();
        ctx.memcpy_d2h(actual_h.data(), hd.data(), actual_h.size() * 2);
        ctx.memcpy_d2h(actual_d.data(), dd.data(), actual_d.size() * 4);
        X3Error eh, ed;
        for (size_t j = 0; j < tokens.size(); ++j) {
            const int e = experts[j], t = tokens[j], r = ranks[j];
            std::vector<bf16> ref_h(I);
            for (int n = 0; n < I; ++n) {
                double g = 0, u = 0;
                for (int k = 0; k < H; ++k) {
                    g += float(x[t * H + k]) * weight(e, 0, n, k);
                    u += float(x[t * H + k]) * weight(e, 1, n, k);
                }
                ref_h[n] = bf16(float(g / (1 + std::exp(-g)) * u));
                eh.Add(float(ref_h[n]), float(actual_h[(t * R + r) * I + n]));
            }
            for (int n = 0; n < H; ++n) {
                double y = 0;
                for (int k = 0; k < I; ++k) y += float(ref_h[k]) * weight(e, 2, n, k);
                ed.Add(y, actual_d[(t * R + r) * H + n]);
            }
        }
        for (size_t i = 0; i < actual_h.size(); ++i)
            if (i >= T * R * I || !touched[i / I]) expect(float(actual_h[i]) == -9.f, "X3 h tail/empty rank overwrite");
        for (size_t i = 0; i < actual_d.size(); ++i)
            if (i >= T * R * H || !touched[i / H]) expect(actual_d[i] == -9.f, "X3 down tail/empty rank overwrite");
        std::cout << "[X3 KERNEL] nested=" << nested << " h_nrmse=" << eh.Nrmse()
                  << " down_nrmse=" << ed.Nrmse() << " max_abs=" << ed.max_abs
                  << " ready_consumer_reuse=pass guards=pass\n";
        expect(eh.Nrmse() < 0.001 && ed.Nrmse() < 0.001, "X3 CPU reference mismatch");
    }
}

double x3_quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1, size_t(q * (values.size() - 1)))];
}

struct X3ProbeScratch {
    Context& ctx;
    Tensor slots, tokens, ranks, tiles, h, down;
    explicit X3ProbeScratch(Context& c) : ctx(c) {
        const auto bytes = [&](size_t n) { return Tensor::allocate(ctx, {int64_t(n)}, dnnl::memory::data_type::u8); };
        slots = bytes(2560 * sizeof(ops::MoeExpertSlotDev));
        tokens = bytes(2560 * 4); ranks = bytes(2560 * 4);
        tiles = bytes(2560 * sizeof(ops::MoeExpertTile));
        h = bytes(2560 * 640 * 2); down = bytes(2560 * 2560 * 4);
    }
    ~X3ProbeScratch() { try { ctx.synchronize(); } catch (...) {} }
};
X3ProbeScratch* x3_probe_scratch = nullptr;

void x3_real_expert_probe(Qwen4ExpBnb4Backend& backend, int layer, int count,
    int first, int tileBegin, int numTiles, const sycl::event& ready) {
    static std::array<std::array<bool, 3>, 2> seen{};
    auto view = Access::Experts(backend);
    auto& ctx = view.ctx;
    auto readyCopy = ready;
    readyCopy.wait_and_throw();
    std::cout << "[X3 PROBE] layer=" << layer << " count=" << count << " first=" << first << " tiles=" << numTiles << '\n';
    for (int ti = tileBegin; ti < tileBegin + numTiles;) {
        const int begin = view.tiles[ti].first;
        expect(begin >= first && begin < count * 10, "X3 invalid probe tile offset");
        const auto slot = view.slots[begin];
        int n = 0;
        do { n += view.tiles[ti++].rows; }
        while (ti < tileBegin + numTiles && view.slots[view.tiles[ti].first].gate_up_packed == slot.gate_up_packed);
        const int bin = n <= 8 ? 0 : n <= 16 ? 1 : 2;
        if (seen[count > 64][bin]) continue;
        std::cout << "[X3 PROBE EXPERT] layer=" << layer << " first=" << begin << " rows=" << n << '\n';
        expect(n > 0 && begin + n <= count * 10, "X3 invalid probe expert span");
        seen[count > 64][bin] = true;
        constexpr int H = 2560, I = 640, R = 10;
        const int nt = (n + 7) / 8;
        std::vector<ops::MoeExpertTile> tiles(nt);
        for (int i = 0; i < nt; ++i) tiles[i] = {i * 8, std::min(8, n - i * 8)};
        expect(x3_probe_scratch && count <= 256, "X3 probe arena missing or too small");
        auto& sd = x3_probe_scratch->slots;
        auto& td = x3_probe_scratch->tokens;
        auto& rd = x3_probe_scratch->ranks;
        auto& tiled = x3_probe_scratch->tiles;
        auto& h = x3_probe_scratch->h;
        auto& d = x3_probe_scratch->down;
        ctx.memcpy_h2d(sd.data(), view.slots + begin, n * sizeof(slot));
        ctx.memcpy_h2d(td.data(), view.tokens + begin, n * 4);
        ctx.memcpy_h2d(rd.data(), view.ranks + begin, n * 4);
        ctx.memcpy_h2d(tiled.data(), tiles.data(), nt * sizeof(tiles[0]));
        auto run = [&](bool xmx) {
            sycl::event gu;
            if (xmx) {
                gu = ops::moe_grouped_xmx_gate_up(ctx, sd.data_as<ops::MoeExpertSlotDev>(),
                    tiled.data_as<ops::MoeExpertTile>(), nt, td.data_as<int>(), rd.data_as<int>(), nullptr,
                    view.x, h.data_as<bf16>(), R, H, I, view.nested);
                return ops::moe_grouped_xmx_down(ctx, sd.data_as<ops::MoeExpertSlotDev>(),
                    tiled.data_as<ops::MoeExpertTile>(), nt, td.data_as<int>(), rd.data_as<int>(), nullptr,
                    h.data_as<bf16>(), d.data_as<float>(), R, H, I, view.nested, &gu);
            }
            gu = ops::moe_expert_major_prefill_gate_up(ctx, sd.data_as<ops::MoeExpertSlotDev>(),
                td.data_as<int>(), rd.data_as<int>(), nullptr, view.x, h.data_as<bf16>(), n, R, H, I, 64,
                nullptr, view.nested, (n % 4) / 2, n / 4);
            return ops::moe_expert_major_prefill_down(ctx, sd.data_as<ops::MoeExpertSlotDev>(),
                td.data_as<int>(), rd.data_as<int>(), nullptr, h.data_as<bf16>(), d.data_as<float>(), n, R, I, H, 64,
                &gu, view.nested, (n % 4) / 2, n / 4);
        };
        // Read only the active token/rank rows; uninitialized holes are irrelevant.
        auto download = [&]() {
            std::pair<std::vector<float>, std::vector<float>> result;
            std::vector<bf16> row(I);
            result.first.resize(size_t(n) * I); result.second.resize(size_t(n) * H);
            for (int j = 0; j < n; ++j) {
                const int idx = view.tokens[begin + j] * R + view.ranks[begin + j];
                ctx.memcpy_d2h(row.data(), h.data_as<bf16>() + size_t(idx) * I, I * 2);
                for (int k = 0; k < I; ++k) result.first[size_t(j) * I + k] = float(row[k]);
                ctx.memcpy_d2h(result.second.data() + size_t(j) * H, d.data_as<float>() + size_t(idx) * H, H * 4);
            }
            return result;
        };
        run(false).wait_and_throw(); const auto ref = download();
        run(true).wait_and_throw(); const auto cand = download();
        X3Error eh, ed;
        std::vector<double> absolute;
        for (size_t i = 0; i < ref.first.size(); ++i) eh.Add(ref.first[i], cand.first[i]);
        for (size_t i = 0; i < ref.second.size(); ++i) {
            ed.Add(ref.second[i], cand.second[i]);
            absolute.push_back(std::abs(double(ref.second[i]) - cand.second[i]));
        }
        std::cout << "[X3 OPQUALITY] count=" << count << " layer=" << layer << " m=" << n
                  << " h_nrmse=" << eh.Nrmse() << " down_nrmse=" << ed.Nrmse()
                  << " down_rms=" << std::sqrt(ed.reference / ed.count) << " max_abs=" << ed.max_abs
                  << " p99_abs=" << x3_quantile(absolute, .99) << " warning=" << (ed.Nrmse() > .01) << '\n';
        for (int i = 0; i < 5; ++i) { run(false).wait_and_throw(); run(true).wait_and_throw(); }
        for (int series = 0; series < 3; ++series) {
            std::array<std::vector<double>, 2> times;
            for (int pair = 0; pair < 20; ++pair) {
                for (int order = 0; order < 2; ++order) {
                    const bool xmx = (pair + order) % 2 != 0;
                    const auto start = std::chrono::steady_clock::now();
                    run(xmx).wait_and_throw();
                    times[xmx].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                }
            }
            for (int mode = 0; mode < 2; ++mode) {
                const double med = x3_quantile(times[mode], .5);
                double sum = 0, sum2 = 0;
                for (double v : times[mode]) { sum += v; sum2 += v * v; }
                const double mean = sum / times[mode].size();
                const double cv = std::sqrt(std::max(0., sum2 / times[mode].size() - mean * mean)) / mean;
                std::cout << "[X3 OPTIME] count=" << count << " layer=" << layer << " m=" << n
                          << " series=" << series << " xmx=" << mode << " median_ms=" << med
                          << " p10=" << x3_quantile(times[mode], .1) << " p90=" << x3_quantile(times[mode], .9)
                          << " cv=" << cv << " noisy=" << (cv > .1) << '\n';
            }
        }
    }
}

void test_grouped_xmx_model(const std::string& directory) {
    std::cout << std::unitbuf;
    InferenceEngine engine;
    expect(engine.init(directory, 512), "X3 initialization failed");
    auto& ctx = Access::Ctx(engine);
    auto& backend = Access::Backend(engine);
    backend.set_prefill_xmx(true); backend.set_direct_prefill_io(true);
    Tokenizer tokenizer;
    expect(tokenizer.load(directory), "X3 tokenizer load failed");
    GenerationConfig config;
    config.do_sample = false; config.thinking_budget_tokens = 0; config.max_new_tokens = 48;
    aila::chat::ChatRequest request;
    std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config, request, &error), "X3 prompt parse failed");
    aila::chat::ChatFormatTextResult rendered;
    aila::chat::ChatFormatter formatter;
    expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error), "X3 format failed");
    auto story = tokenizer.encode(rendered.text);
    expect(story.size() == 30, "X3 story token count changed");
    std::string longText;
    while (tokenizer.encode(longText).size() < 256)
        longText += "The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
    auto longIds = tokenizer.encode(longText); longIds.resize(256);
    auto targets = tokenizer.encode("The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the horizon, the sea shines.");
    expect(targets.size() >= 48, "X3 continuation too short"); targets.resize(48);
    Tensor tokenBuffer = Tensor::allocate(ctx, {512}, dnnl::memory::data_type::s32);
    using Clock = std::chrono::steady_clock;
    struct Run { double prefill = 0, total = 0; uint64_t acts = 0; std::vector<std::vector<float>> logits; };
    const bool micro = aila::env::read_int("AILA_Q4E_DEMAND_MICROBATCH", 0) > 0;
    const bool diagnostic = aila::env::read_flag("AILA_TEST_X3_DIAGNOSTIC", false);
    const bool probeOnly = aila::env::read_flag("AILA_TEST_X3_PROBE_ONLY", false);
    // Preallocate test-only operator buffers before layer execution, as the
    // runtime does. Avoid allocation/free churn inside the live model callback.
    std::unique_ptr<X3ProbeScratch> probeScratch;
    if (!micro && (!diagnostic || probeOnly)) probeScratch = std::make_unique<X3ProbeScratch>(ctx);
    x3_probe_scratch = probeScratch.get();
    auto run = [&](const std::vector<int>& ids, bool xmx, bool capture, bool probe) {
        backend.reset(); Access::Grouped(backend, xmx ? 8 : 0); Access::Histogram(backend);
        Access::Probe(backend, probe ? x3_real_expert_probe : nullptr);
        ctx.memcpy_h2d(tokenBuffer.data(), ids.data(), ids.size() * 4);
        ctx.memcpy_h2d(tokenBuffer.data_as<int>() + ids.size(), targets.data(), targets.size() * 4);
        Run result;
        const uint64_t acts = Access::GroupedActs(backend);
        const auto start = Clock::now();
        std::cout << "[X3 RUN] count=" << ids.size() << " xmx=" << xmx << " capture=" << capture << " probe=" << probe << '\n';
        Tensor& initial = backend.forward(ctx, tokenBuffer.data_as<int>(), int(ids.size()), {SnapshotPolicy::ReusablePrefix, true});
        ctx.synchronize();
        std::cout << "[X3 PREFILL DONE] count=" << ids.size() << " xmx=" << xmx << '\n';
        result.prefill = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        result.acts = Access::GroupedActs(backend) - acts;
        if (capture) result.logits.push_back(extract_bf16_to_f32(ctx, initial));
        const int continuation = micro ? 3 : ids.size() == 30 ? 47 : 7;
        for (int i = 0; i < continuation; ++i) {
            Tensor& next = backend.forward(ctx, tokenBuffer.data_as<int>() + ids.size() + i, 1, {SnapshotPolicy::ReusablePrefix, true});
            if (capture) result.logits.push_back(extract_bf16_to_f32(ctx, next));
        }
        ctx.synchronize();
        result.total = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return result;
    };
    for (const auto& ids : {story, longIds}) {
        if (diagnostic && ids.size() == 30) continue;
        if (micro && ids.size() > 30) continue;
        run(ids, false, false, false); run(ids, true, false, !micro && (!diagnostic || probeOnly));
        if (probeOnly) continue;
        const auto ref = run(ids, false, true, false);
        const auto refHist = Access::Histogram(static_cast<const Qwen4ExpBnb4Backend&>(backend));
        const auto cand = run(ids, true, true, false);
        expect(cand.acts > 0, "X3 canary failed to dispatch grouped XMX");
        CheckpointMetrics metrics;
        for (size_t i = 0; i < ref.logits.size(); ++i) metrics.add(ref.logits[i], cand.logits[i], targets[i]);
        std::cout << "[X3 QUALITY] count=" << ids.size() << " rows=" << metrics.rows
                  << " mean_kl=" << metrics.kl / metrics.rows << " mean_nll_delta=" << metrics.nll_delta / metrics.rows
                  << " max_logits_nrmse=" << metrics.max_nrmse << " warning=" << (metrics.nll_delta / metrics.rows > .01) << '\n';
        for (int bin = 0; bin < 6; ++bin)
            std::cout << "[X3 HIST] count=" << ids.size() << " bin=" << bin << " requests=" << refHist[bin][0]
                      << " acts=" << refHist[bin][1] << " hits=" << refHist[bin][2] << " misses=" << refHist[bin][3] << '\n';
        for (int pair = 0; pair < (micro ? 1 : 3); ++pair) {
            for (int order = 0; order < 2; ++order) {
                const bool xmx = (pair + order) % 2 != 0;
                const auto sample = run(ids, xmx, false, false);
                std::cout << "[X3 TIME] count=" << ids.size() << " pair=" << pair << " xmx=" << xmx
                          << " prefill_ms=" << sample.prefill << " total_ms=" << sample.total << " xmx_acts=" << sample.acts << '\n';
            }
        }
        std::cout << "[X3 MEMORY] count=" << ids.size() << " device_bytes=" << ctx.current_allocated_bytes()
                  << " host_bytes=" << ctx.current_pinned_host_bytes() << " cache_slots=" << backend.adaptive_cache().NumGpuSlots() << '\n';
    }
    Access::Probe(backend, nullptr);
    x3_probe_scratch = nullptr;
}
