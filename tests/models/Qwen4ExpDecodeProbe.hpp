// Optional real-model diagnostic, included inside the tolerance test namespace.
// Fixed teacher forcing measures target work, not generated/sampled throughput.
void test_decode_probe(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    constexpr int steps = 128;
    const int prompt_count = aila::env::read_int("AILA_Q4E_DECODE_PROBE_PROMPT", 30);
    const int rows = aila::env::read_int("AILA_Q4E_DECODE_PROBE_ROWS", 1);
    const int repeats = aila::env::read_int("AILA_Q4E_DECODE_PROBE_REPEATS", 4);
    expect((prompt_count == 30 || prompt_count == 512) && (rows == 1 || rows == 2 || rows == 4) &&
           repeats >= 2 && repeats <= 8, "Invalid decode probe configuration");
    InferenceEngine engine;
    expect(engine.init(directory, std::max(512, prompt_count + steps)), "Decode probe initialization failed");
    auto& ctx = Access::Ctx(engine);
    auto& backend = Access::Backend(engine);
    auto& cache = Access::Cache(engine);
    Tokenizer tokenizer;
    expect(tokenizer.load(directory), "Cannot load decode probe tokenizer");
    std::vector<int> prompt;
    if (prompt_count == 30) {
        GenerationConfig config;
        config.do_sample = false; config.thinking_budget_tokens = 0;
        aila::chat::ChatRequest request;
        std::string error;
        expect(aila::chat::parse_chat_request_json(
            R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
            config, request, &error), "Cannot parse decode probe prompt");
        aila::chat::ChatFormatTextResult rendered;
        aila::chat::ChatFormatter formatter;
        expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error),
               "Cannot render decode probe prompt");
        prompt = tokenizer.encode(rendered.text);
        expect(prompt.size() == 30, "Decode probe story prompt changed");
    } else {
        std::string text;
        while (tokenizer.encode(text).size() < size_t(prompt_count))
            text += "The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
        prompt = tokenizer.encode(text); prompt.resize(prompt_count);
    }
    auto continuation = tokenizer.encode(
        "The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. "
        "Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. "
        "As the sun breaches the horizon, the sea shines. A fishing boat moves slowly toward the harbor, leaving a narrow trail "
        "across the calm water. On the shore, a keeper closes the lighthouse door and watches the gulls circle overhead. "
        "The air smells of salt and wet stone. Behind the dunes, the village begins to wake, and the first voices mingle with "
        "the steady sound of the tide. He writes the time in his logbook, checks the weather, and sets a kettle on the stove.");
    expect(continuation.size() >= steps, "Decode probe continuation too short");
    continuation.resize(steps);
    auto hash_bytes = [](const void* data, size_t size) {
        uint64_t hash = 14695981039346656037ULL;
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
        return hash;
    };
    Tensor ids = Tensor::allocate(ctx, {steps}, dnnl::memory::data_type::s32);
    struct Drain {
        Context& context;
        ~Drain() { try { context.synchronize(); } catch (...) {} }
    } drain{ctx}; // Drain submitted token readers before ids is destroyed on errors.
    ctx.memcpy_h2d(ids.data(), continuation.data(), steps * sizeof(int));
    Access::ReserveVerifyLogits(engine, rows);
    std::vector<bf16> reference, block;
    std::vector<uint64_t> expected_hashes(steps);
    std::cout << "[DECODE INPUT] prompt=" << prompt_count << " steps=" << steps << " rows=" << rows
              << " prompt_fnv64=" << hash_bytes(prompt.data(), prompt.size() * sizeof(int))
              << " continuation_fnv64=" << hash_bytes(continuation.data(), continuation.size() * sizeof(int)) << '\n';
    for (int pass = 0; pass < repeats; ++pass) {
        engine.reset_context();
        const auto prefill_start = std::chrono::steady_clock::now();
        auto* initial = engine.forward_prompt_tokens(prompt, 0, prompt_count, 0, true, {SnapshotPolicy::None, true});
        expect(initial && backend.get_current_context_len() == prompt_count, "Decode probe prefill failed");
        ctx.synchronize();
        const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefill_start).count();
        const int vocabulary = static_cast<int>(initial->shape(1));
        block.resize(size_t(rows) * vocabulary);
        if (pass == 0) reference.resize(size_t(steps) * vocabulary);
        const auto before = cache.stats();
        const auto device_count = ctx.device_allocation_count(), host_count = ctx.pinned_host_allocation_count();
        std::vector<double> times; times.reserve(steps / rows);
        std::vector<uint64_t> hashes; hashes.reserve(steps);
        std::vector<int> top1; top1.reserve(steps);
        double core_ms = 0;
        const auto loop_start = std::chrono::steady_clock::now();
        for (int t = 0; t < steps; t += rows) {
            const auto start = std::chrono::steady_clock::now();
            auto& logits = Access::VerifyRows(engine, ids.data_as<int>() + t, rows);
            ctx.synchronize();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            core_ms += ms; times.push_back(ms);
            expect(logits.shape(0) == rows && logits.shape(1) == vocabulary, "Missing verification logits rows");
            ctx.memcpy_d2h(block.data(), logits.data(), block.size() * sizeof(bf16));
            for (int r = 0; r < rows; ++r) {
                const auto* row = block.data() + size_t(r) * vocabulary;
                int best = 0;
                for (int v = 0; v < vocabulary; ++v) {
                    expect(std::isfinite(float(row[v])), "Nonfinite decode logits");
                    if (float(row[v]) > float(row[best])) best = v;
                }
                const auto hash = hash_bytes(row, size_t(vocabulary) * sizeof(bf16));
                if (pass == 0) {
                    expected_hashes[t+r] = hash;
                    std::copy(row, row + vocabulary, reference.data() + size_t(t+r) * vocabulary);
                } else {
                    expect(hash == expected_hashes[t+r], "Identical continuation replay changed logits bits");
                }
                hashes.push_back(hash); top1.push_back(best);
            }
        }
        const double loop_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
        expect(backend.get_current_context_len() == prompt_count + steps, "Wrong number of consumed continuation tokens");
        expect(ctx.device_allocation_count() == device_count && ctx.pinned_host_allocation_count() == host_count,
               "Decode/verification changed the live USM allocation count");
        const auto after = cache.stats();
        auto sorted = times; std::sort(sorted.begin(), sorted.end());
        std::cout << "[DECODE PASS] {\"pass\":" << pass << ",\"prompt_tokens\":" << prompt_count
                  << ",\"steps\":" << steps << ",\"rows\":" << rows << ",\"forwards\":" << times.size()
                  << ",\"prefill_ms\":" << prefill_ms << ",\"core_ms\":" << core_ms << ",\"loop_ms\":" << loop_ms
                  << ",\"core_ms_per_input\":" << core_ms/steps << ",\"loop_ms_per_input\":" << loop_ms/steps
                  << ",\"forward_p10_ms\":" << sorted[sorted.size()/10]
                  << ",\"forward_median_ms\":" << sorted[sorted.size()/2]
                  << ",\"forward_p90_ms\":" << sorted[sorted.size()*9/10]
                  << ",\"hits\":" << after.hit_count-before.hit_count << ",\"misses\":" << after.miss_count-before.miss_count
                  << ",\"demand_bytes\":" << after.total_transferred_bytes-before.total_transferred_bytes
                  << ",\"host_hits\":" << after.host_pool_hits-before.host_pool_hits
                  << ",\"host_misses\":" << after.host_pool_misses-before.host_pool_misses
                  << ",\"fill_ms\":" << (after.fill_ns-before.fill_ns)*1e-6
                  << ",\"prefetch_wait_ms\":" << (after.prefetch_wait_ns-before.prefetch_wait_ns)*1e-6
                  << ",\"prefetch_issued_bytes\":" << after.prefetch_issued_bytes-before.prefetch_issued_bytes
                  << ",\"prefetch_useful_bytes\":" << after.prefetch_useful_bytes-before.prefetch_useful_bytes
                  << ",\"device_bytes\":" << ctx.current_allocated_bytes()
                  << ",\"pinned_host_bytes\":" << ctx.current_pinned_host_bytes()
                  << ",\"live_usm_count_delta\":0,\"step_hashes\":[";
        for (size_t i=0;i<hashes.size();++i) std::cout << (i ? "," : "") << '"' << hashes[i] << '"';
        std::cout << "],\"top1\":[";
        for (size_t i=0;i<top1.size();++i) std::cout << (i ? "," : "") << top1[i];
        std::cout << "],\"forward_ms\":[";
        for (size_t i=0;i<times.size();++i) std::cout << (i ? "," : "") << times[i];
        std::cout << "]}\n";
    }
    const auto path = aila::env::read_string("AILA_Q4E_DECODE_PROBE_LOGITS", "");
    if (!path.empty()) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(reference.data()), reference.size() * sizeof(bf16));
        file.close();
        expect(bool(file), "Cannot write probe logits artifact");
    }
    std::cout << "[DECODE PASS] deterministic_replay=1 finite_all_rows=1 target_only=1 sampling=0 rollback=0\n";
}
