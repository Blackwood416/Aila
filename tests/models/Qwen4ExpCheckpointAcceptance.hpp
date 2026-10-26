// Included inside the tolerance test namespace; uses its logits conversion helpers.
// Optional real-model X2 canary, never part of the model-free CTest invocation.
template<class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template<class Snapshot>
void expect_same_checkpoint(const Snapshot& a, const Snapshot& b) {
    expect(a.length == b.length && a.layers.size() == b.layers.size(), "Snapshot identity changed");
    expect(a.ple_snapshot.prev_tokens == b.ple_snapshot.prev_tokens &&
           a.ple_snapshot.conv_head == b.ple_snapshot.conv_head &&
           same_bits(a.ple_snapshot.conv_state, b.ple_snapshot.conv_state), "Future tokens contaminated PLE snapshot");
    for (size_t i = 0; i < a.layers.size(); ++i) {
        expect(a.layers[i].linear_conv_head == b.layers[i].linear_conv_head &&
               a.layers[i].num_compressed_blocks == b.layers[i].num_compressed_blocks &&
               same_bits(a.layers[i].linear_state_host, b.layers[i].linear_state_host) &&
               same_bits(a.layers[i].linear_conv_state_host, b.layers[i].linear_conv_state_host),
               "Future tokens contaminated recurrent snapshot");
    }
}

struct CheckpointMetrics {
    double kl = 0, nll_delta = 0, max_nrmse = 0;
    size_t rows = 0;
    void add(const std::vector<float>& ref, const std::vector<float>& cand, int target) {
        expect(ref.size() == cand.size() && target >= 0 && target < static_cast<int>(ref.size()),
               "Invalid teacher-forcing logits/target");
        double sr = 0, sc = 0, sq = 0, err = 0;
        const double mr = *std::max_element(ref.begin(), ref.end());
        const double mc = *std::max_element(cand.begin(), cand.end());
        for (size_t i = 0; i < ref.size(); ++i) {
            expect(std::isfinite(ref[i]) && std::isfinite(cand[i]), "Nonfinite checkpoint logits");
            sr += std::exp(ref[i] - mr); sc += std::exp(cand[i] - mc);
            sq += double(ref[i]) * ref[i];
            const double d = double(cand[i]) - ref[i]; err += d * d;
        }
        const double zr = mr + std::log(sr), zc = mc + std::log(sc);
        double row_kl = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            row_kl += std::exp(ref[i] - zr) * ((ref[i] - zr) - (cand[i] - zc));
        kl += row_kl;
        nll_delta += (zc - cand[target]) - (zr - ref[target]);
        max_nrmse = std::max(max_nrmse, std::sqrt(err / std::max(sq, 1e-12)));
        ++rows;
    }
    void report(const char* label) const {
        expect(rows > 0, "Missing teacher-forcing observations");
        std::cout << "[X2 QUALITY] " << label << " rows=" << rows << " mean_kl=" << kl / rows
                  << " mean_nll_delta=" << nll_delta / rows << " max_logits_nrmse=" << max_nrmse
                  << " nll_warning=" << (nll_delta / rows > 0.01) << '\n';
    }
};

void test_checkpoint_acceptance(const std::string& directory) {
    std::cout << std::unitbuf;
    InferenceEngine engine;
    expect(engine.init(directory, 512), "X2 model initialization failed");
    auto& ctx = Access::Ctx(engine);
    auto& backend = Access::Backend(engine);
    auto& session = Access::Session(engine);
    backend.set_prefill_xmx(true);
    backend.set_direct_prefill_io(true);
    Tokenizer tokenizer;
    expect(tokenizer.load(directory), "Cannot load X2 tokenizer");
    GenerationConfig config;
    config.do_sample = false; config.thinking_budget_tokens = 0; config.max_new_tokens = 48;
    aila::chat::ChatRequest request;
    std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config, request, &error), "Cannot parse X2 prompt");
    aila::chat::ChatFormatTextResult rendered;
    aila::chat::ChatFormatter formatter;
    expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error), "Cannot format X2 prompt");
    const auto ids = tokenizer.encode(rendered.text);
    expect(ids.size() == 30, "X2 story must contain exactly 30 tokens");
    auto targets = tokenizer.encode("The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the horizon, the sea shines.");
    expect(targets.size() >= 48, "X2 continuation needs 48 target tokens");
    targets.resize(48);
    Tensor tokens = Tensor::allocate(ctx, {512}, dnnl::memory::data_type::s32);
    auto upload = [&](const std::vector<int>& values) {
        ctx.memcpy_h2d(tokens.data(), values.data(), values.size() * sizeof(int));
    };
    upload(ids);
    auto prefill = [&](bool interior, int anchor = 26) -> Tensor& {
        if (interior) return backend.forward(ctx, tokens.data_as<int>(), 30,
            {SnapshotPolicy::ReusablePrefix, true, false, anchor});
        backend.forward(ctx, tokens.data_as<int>(), anchor, {SnapshotPolicy::ReusablePrefix, false, true});
        return backend.forward(ctx, tokens.data_as<int>() + anchor, 30 - anchor,
            {SnapshotPolicy::ReusablePrefix, true});
    };

    // All routing is recomputed normally. Only the external continuation tokens are fixed.
    using Clock = std::chrono::steady_clock;
    struct Run { double prefill_ms = 0, total_ms = 0; std::vector<std::vector<float>> logits; };
    auto run = [&](bool interior, bool capture) {
        backend.reset(); upload(ids);
        ctx.memcpy_h2d(tokens.data_as<int>() + 30, targets.data(), targets.size() * sizeof(int));
        Run result;
        const auto start = Clock::now();
        Tensor& initial = prefill(interior);
        ctx.synchronize();
        result.prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (capture) result.logits.push_back(extract_bf16_to_f32(ctx, initial));
        for (int t = 0; t < 47; ++t) {
            Tensor& next = backend.forward(ctx, tokens.data_as<int>() + 30 + t, 1, {SnapshotPolicy::ReusablePrefix, true});
            if (capture) result.logits.push_back(extract_bf16_to_f32(ctx, next));
        }
        ctx.synchronize();
        result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return result;
    };
    for (const bool xmx : {false, true}) {
        backend.set_prefill_xmx(xmx);
        std::cout << "[X2] Warm both paths; xmx=" << xmx << "; GPU expert cache remains capacity-limited, host cache warmed\n";
        run(false, false); run(true, false);
        const auto ref = run(false, true);
        const auto cand = run(true, true);
        CheckpointMetrics metrics;
        for (size_t i = 0; i < targets.size(); ++i) metrics.add(ref.logits[i], cand.logits[i], targets[i]);
        metrics.report(xmx ? "story_26+4_vs_30_xmx" : "story_26+4_vs_30_onednn");

        // Separate timing runs do not download logits or perform numerical scoring.
        for (int pair = 0; pair < 3; ++pair) {
            for (int order = 0; order < 2; ++order) {
                const bool interior = ((pair + order) % 2) != 0;
                const auto stats = Bnb4BitLinear::get_prefill_dispatch_stats();
                const auto sample = run(interior, false);
                const auto after = Bnb4BitLinear::get_prefill_dispatch_stats();
                std::cout << "[X2 TIME] xmx=" << xmx << " pair=" << pair << " inline=" << interior
                          << " prefill_ms=" << sample.prefill_ms << " forward48_ms=" << sample.total_ms
                          << " fused=" << after.fused_gemm - stats.fused_gemm
                          << " dequant=" << after.on_demand_dequant - stats.on_demand_dequant << '\n';
            }
        }
        // Exercise the actual Engine helper and callback as a separate endpoint observation.
        for (int pair = 0; pair < 3; ++pair) {
            for (int order = 0; order < 2; ++order) {
                const bool interior = ((pair + order) % 2) != 0;
                _putenv_s("AILA_Q4E_INLINE_CHECKPOINT", interior ? "" : "0");
                engine.reset_context();
                expect(backend.supports_interior_checkpoint(30, 26) == interior, "Engine dispatch default/opt-out mismatch");
                double ttft = 0;
                const auto start = Clock::now();
                const auto text = engine.generate_chat_request_text(request, [&](const std::string& chunk) {
                    if (!chunk.empty() && ttft == 0)
                        ttft = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                });
                const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                expect(!text.empty() && ttft > 0 && engine.last_error_code() == EngineErrorCode::Ok, "Engine X2 generation failed");
                std::cout << "[X2 ENGINE] xmx=" << xmx << " pair=" << pair << " inline=" << interior
                          << " ttft_ms=" << ttft << " total_ms=" << elapsed
                          << " rendered_output_tokens=" << tokenizer.encode(text).size() << '\n';
                if (pair == 2) std::cout << "[X2 TEXT] inline=" << interior << " " << text << '\n';
            }
        }
    }

    // Same shape and same prefix with a radically different suffix: snapshots must be exact.
    // Split vs single-pass arithmetic may differ and is scored separately after restore.
    auto kv_prefix = [&](int length) {
        std::vector<bf16> result;
        for (const auto& cache : session.layer_caches_for_testing()) {
            for (const Tensor* tensor : {&cache.k, &cache.v}) {
                if (!tensor->valid()) continue;
                for (int h = 0; h < tensor->shape(0); ++h) {
                    const size_t offset = result.size();
                    result.resize(offset + static_cast<size_t>(length) * tensor->shape(2));
                    ctx.memcpy_d2h(result.data() + offset,
                        tensor->data_as<bf16>() + h * tensor->shape(1) * tensor->shape(2),
                        (result.size() - offset) * sizeof(bf16));
                }
            }
        }
        return result;
    };
    for (const int anchor : {26, 28, 16, 2}) {
        backend.reset(); upload(ids);
        backend.forward(ctx, tokens.data_as<int>(), anchor, {SnapshotPolicy::ReusablePrefix, false, true});
        const auto split_snapshot = *session.snapshot_for_testing(anchor);
        const auto split_logits = extract_bf16_to_f32(ctx, backend.forward(ctx, tokens.data_as<int>() + anchor, 30 - anchor));
        backend.reset(); upload(ids); prefill(true, anchor);
        const auto* saved = session.snapshot_for_testing(anchor);
        expect(saved && session.snapshot_for_testing(30), "Both checkpoint slots must be published");
        const auto snapshot = *saved;
        const auto prefix_kv = kv_prefix(anchor);
        expect(snapshot.ple_snapshot.prev_tokens == split_snapshot.ple_snapshot.prev_tokens &&
               snapshot.ple_snapshot.conv_head == split_snapshot.ple_snapshot.conv_head, "Prefix metadata differs");
        expect(backend.Truncate(session, anchor) == anchor, "Interior restore failed");
        // Capture restored state, including PLE, to check the complete host/device round trip.
        backend.capture_snapshot_for_testing(ctx, session, anchor, 1, true);
        expect_same_checkpoint(snapshot, *session.snapshot_for_testing(anchor));
        const auto replay = extract_bf16_to_f32(ctx, backend.forward(ctx, tokens.data_as<int>() + anchor, 30 - anchor));
        CheckpointMetrics restored_metrics;
        restored_metrics.add(split_logits, replay, targets[0]);
        restored_metrics.report(("restore_anchor_" + std::to_string(anchor)).c_str());
        // The identical restored implementation must be deterministic, regardless of tolerance.
        expect(backend.Truncate(session, anchor) == anchor, "Second restore failed");
        const auto replay_again = extract_bf16_to_f32(ctx, backend.forward(ctx, tokens.data_as<int>() + anchor, 30 - anchor));
        expect(same_bits(replay, replay_again), "Same snapshot continuation changed bits");
        auto changed = ids;
        for (int i = anchor; i < 30; ++i) changed[i] = targets[i - anchor];
        backend.reset(); upload(changed); prefill(true, anchor);
        expect_same_checkpoint(snapshot, *session.snapshot_for_testing(anchor));
        expect(same_bits(prefix_kv, kv_prefix(anchor)), "Future suffix contaminated visible KV prefix");
        std::cout << "[X2 STATE PASS] anchor=" << anchor << " suffix_independent=1 restore_exact=1 replay_exact=1\n";
    }

    // Nonzero initial GDN/PLE/KV state; checkpoint_after is relative to this forward.
    auto seed = [&] {
        backend.reset();
        upload(std::vector<int>(targets.begin(), targets.begin() + 4));
        backend.forward(ctx, tokens.data_as<int>(), 4, {SnapshotPolicy::None, false});
        upload(ids);
    };
    seed(); prefill(true, 16);
    expect(session.CurrentLength() == 34 && session.snapshot_for_testing(20), "Nonzero checkpoint used a relative position as absolute");
    const auto nonzero_snapshot = *session.snapshot_for_testing(20);
    expect(backend.Truncate(session, 20) == 20, "Nonzero checkpoint restore failed");
    backend.capture_snapshot_for_testing(ctx, session, 20, 1, true);
    expect_same_checkpoint(nonzero_snapshot, *session.snapshot_for_testing(20));
    const auto nonzero_replay = extract_bf16_to_f32(ctx, backend.forward(ctx, tokens.data_as<int>() + 16, 14));
    seed();
    backend.forward(ctx, tokens.data_as<int>(), 16, {SnapshotPolicy::ReusablePrefix, false});
    const auto nonzero_split = extract_bf16_to_f32(ctx, backend.forward(ctx, tokens.data_as<int>() + 16, 14));
    CheckpointMetrics nonzero;
    nonzero.add(nonzero_split, nonzero_replay, targets[0]); nonzero.report("nonzero_start4_anchor20");

    // Long canary exercises Engine's conservative fallback, not a new long-context algorithm.
    std::string long_text;
    while (tokenizer.encode(long_text).size() < 256)
        long_text += "The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
    auto long_ids = tokenizer.encode(long_text); long_ids.resize(256);
    std::vector<float> long_ref;
    for (bool enabled : {false, true}) {
        engine.reset_context();
        _putenv_s("AILA_Q4E_INLINE_CHECKPOINT", enabled ? "1" : "0");
        expect(!backend.supports_interior_checkpoint(256, 252), "Long prefill must keep split fallback");
        const auto start = Clock::now();
        auto* output = engine.forward_prompt_tokens(long_ids, 0, 256, 252);
        expect(output && session.CurrentLength() == 256 && session.snapshot_for_testing(252), "Long fallback lost prefix checkpoint");
        const auto logits = extract_bf16_to_f32(ctx, *output);
        if (!enabled) long_ref = logits;
        else expect(same_bits(long_ref, logits), "Opt-in changed unsupported long-prefill result");
        std::cout << "[X2 LONG] tokens=256 inline_optin=" << enabled << " fallback=252+4 ms="
                  << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << '\n';
    }
    _putenv_s("AILA_Q4E_INLINE_CHECKPOINT", "");
    std::cout << "[X2 PASS] State semantics verified; numerical warnings above are canary observations, not general quality certification\n";
}

void test_prefill_memory(const std::string& directory) {
    std::cout << std::unitbuf;
    const int count = aila::env::read_int("AILA_Q4E_MEMORY_CANARY_TOKENS", 256);
    const int tailCount = aila::env::read_int_raw("AILA_Q4E_MEMORY_CANARY_TAIL", 0);
    expect(count == 256 || count == 512 || count == 1024 || count == 2052, "Unsupported memory canary token count");
    expect(tailCount == 0 || tailCount == 8, "Unsupported memory canary tail");
    InferenceEngine engine;
    expect(engine.init(directory, std::max(512, count + tailCount)), "Memory canary initialization failed");
    auto& ctx = Access::Ctx(engine);
    auto& backend = Access::Backend(engine);
    Tokenizer tokenizer;
    expect(tokenizer.load(directory), "Cannot load memory canary tokenizer");
    auto report = [&](const char* stage) {
        std::cout << "[X2 MEMORY] stage=" << stage << " device_bytes=" << ctx.current_allocated_bytes()
                  << " peak_device_bytes=" << ctx.peak_allocated_bytes()
                  << " pinned_host_bytes=" << ctx.current_pinned_host_bytes() << '\n';
    };
    report("warmup");
    std::string text;
    while (tokenizer.encode(text).size() < size_t(count))
        text += "The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
    const auto all = tokenizer.encode(text);
    Tensor tailIds;
    if(tailCount) {
        tailIds=Tensor::allocate(ctx,{tailCount},dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(tailIds.data(),all.data(),tailCount*sizeof(int));
    }
    // Extra identical passes let the profiling harness retain a complete warm
    // interval even when the collector loses trailing events at process exit.
    const int repeats = std::clamp(aila::env::read_int("AILA_Q4E_MEMORY_CANARY_REPEATS", 2), 2, 8);
    uint64_t expected_hash = 0;
    std::vector<uint64_t> expectedTail;
    for (int pass = 0; pass < repeats; ++pass) {
        engine.reset_context();
        const std::vector<int> ids(all.begin(), all.begin() + count);
        const auto start = std::chrono::steady_clock::now();
        auto* output = engine.forward_prompt_tokens(ids, 0, count, 0, true, {SnapshotPolicy::None, true});
        expect(output && backend.get_current_context_len() == count, "Memory canary forward failed");
        const auto logits = extract_bf16_to_f32(ctx, *output);
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 0; i < logits.size(); ++i) {
            expect(std::isfinite(logits[i]), "Nonfinite memory canary logits");
            const auto* bytes = reinterpret_cast<const uint8_t*>(&logits[i]);
            for (size_t b = 0; b < sizeof(float); ++b) { hash ^= bytes[b]; hash *= 1099511628211ULL; }
        }
        if (pass == 0) expected_hash = hash;
        else expect(hash == expected_hash, "Identical memory canary replay changed logits bits");
        std::cout << "[X2 MEMORY FORWARD] tokens=" << count << " ms="
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                  << " logits_fnv64=" << hash << '\n';
        report(std::to_string(count).c_str());
        if(tailCount) {
            const auto live=ctx.device_allocation_count(),pinned=ctx.pinned_host_allocation_count();
            std::vector<uint64_t> hashes;std::vector<double> ms;
            for(int t=0;t<tailCount;++t) {
                auto begin=std::chrono::steady_clock::now();
                auto& next=Access::VerifyRows(engine,tailIds.data_as<int>()+t,1);ctx.synchronize();
                ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
                const auto values=extract_bf16_to_f32(ctx,next);uint64_t h=14695981039346656037ULL;
                for(float v:values) {
                    expect(std::isfinite(v),"Nonfinite memory canary tail");
                    const auto* bytes=reinterpret_cast<const uint8_t*>(&v);
                    for(size_t b=0;b<sizeof(float);++b){h^=bytes[b];h*=1099511628211ULL;}
                }
                hashes.push_back(h);
            }
            if(pass==0)expectedTail=hashes;else expect(hashes==expectedTail,"Memory canary tail changed bits");
            expect(ctx.device_allocation_count()==live && ctx.pinned_host_allocation_count()==pinned,"Memory canary decode grew live USM");
            auto print=[](const auto& v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';};
            std::cout<<"[S4 TAIL] {\"pass\":"<<pass<<",\"hashes\":";print(hashes);
            std::cout<<",\"ms\":";print(ms);std::cout<<",\"live_usm_delta\":0}\n";
            report("tail");
        }
    }
}
