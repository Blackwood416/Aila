void test_qwen38_window_quality(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    const auto path = aila::env::read_string("AILA_Q4E_WINDOW_PROMPT_FILE", "");
    std::ifstream file(path); expect(bool(file), "R1 prompt file");
    const std::string body((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    GenerationConfig config; config.do_sample=false;config.thinking_budget_tokens=0;
    aila::chat::ChatRequest request; std::string error;
    expect(aila::chat::parse_chat_request_json(body,config,request,&error),"R1 prompt JSON");
    expect(request.messages.size()==1 && request.messages[0].role==aila::chat::ChatRole::User,"R1 single user prompt");
    // Match the probe's line-oriented interactive CLI input normalization exactly.
    for(auto& part:request.messages[0].content) {
        std::string flat;bool newline=false;
        for(char c:part.text) {if(c=='\r'||c=='\n'){if(!newline)flat+=' ';newline=true;}else{flat+=c;newline=false;}}
        part.text=std::move(flat);
    }
    aila::chat::ChatMessage system;system.role=aila::chat::ChatRole::System;
    system.content.push_back(aila::chat::ChatContentPart::text_part("You are a helpful assistant."));
    request.messages.insert(request.messages.begin(),std::move(system));
    InferenceEngine engine;expect(engine.init(directory,2560),"R1 model init");
    auto& ctx=Access::Ctx(engine);auto& backend=Access::Backend(engine);
    aila::chat::ChatFormatter formatter;aila::chat::ChatFormatTextResult rendered;
    expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),"R1 formatter");
    Tokenizer tokenizer;expect(tokenizer.load(directory),"R1 tokenizer");
    const auto prompt=tokenizer.encode(rendered.text);expect(prompt.size()==1833,"R1 CLI token identity changed");
    const auto target=tokenizer.encode("Begin by agreeing on stable identifiers and a single request register. Mara should define the permission decisions and choose a small set of pilot requests. Meanwhile, Jon can prepare the receiving room and arrange the wall inspection. The technician can check the cable route and rehearse restoring a sample backup before connecting the capture workflow to shared storage.");
    expect(target.size()>=48,"R1 teacher sequence");
    auto id=Tensor::allocate(ctx,{1},dnnl::memory::data_type::s32);
    struct Drain {Context& c;~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    std::vector<std::vector<float>> reference;CheckpointMetrics metrics;
    const int tile = aila::env::read_int("AILA_Q4E_PREFILL_ATTN_TILE",0);
    const int compact = aila::env::read_int("AILA_Q4E_MOE_COMPACT_ACTS",0);
    for(int mode=0;mode<2;++mode) {
        const int chunk=(tile||compact)?2048:(mode?2048:512);
        if(compact) Access::CompactMoe(backend,mode?compact:0);
        else if(tile) Access::AttentionTile(backend,mode?tile:0);
        _putenv_s("AILA_Q35_PREFILL_CHUNK",std::to_string(chunk).c_str());backend.reset();
        expect(engine.forward_prompt_tokens(prompt,0,int(prompt.size()),int(prompt.size())-4)!=nullptr,"R1 prefill");
        for(int step=0;step<48;++step) {
            const auto values=extract_bf16_to_f32(ctx,Access::Logits(backend));
            if(mode==0)reference.push_back(values);else metrics.add(reference[step],values,target[step]);
            ctx.memcpy_h2d(id.data(),&target[step],sizeof(int));backend.forward(ctx,id.data_as<int>(),1);
        }
    }
    metrics.report(compact?"R1b_natural1833_compact_moe":tile?"R1b_natural1833_untiled_vs_tiled":"R1_natural1833_chunk512_vs2048");
    std::cout<<((tile||compact)?"[R1b QUALITY]":"[R1 QUALITY]")<<" prompt=1833 rows=48 tile="<<tile<<" compact="<<compact<<" finite=1 nll_warning="<<(metrics.nll_delta/metrics.rows>.01)<<"\n";
}

// Focused R0 state/quality canary for the actual CLI's 41-token prompt.
void test_qwen38_preset_checkpoint(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    InferenceEngine engine;
    expect(engine.init(directory, 512), "R0 initialization failed");
    auto& ctx = Access::Ctx(engine);
    auto& backend = Access::Backend(engine);
    auto& session = Access::Session(engine);
    Tokenizer tokenizer; expect(tokenizer.load(directory), "R0 tokenizer");
    GenerationConfig config; config.do_sample = false; config.thinking_budget_tokens = 0;
    aila::chat::ChatRequest request; std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config, request, &error), "R0 request");
    aila::chat::ChatFormatter formatter; aila::chat::ChatFormatTextResult rendered;
    expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error), "R0 render");
    const auto prompt = tokenizer.encode(rendered.text);
    expect(prompt.size() == 41, "R0 CLI prompt must contain 41 tokens");
    const auto target = tokenizer.encode("The morning sun had barely crested the horizon, painting the sea in shades of gold and rose. A quiet breeze carried the scent of salt across the shore. Small waves folded over the sand while distant fishing boats crossed the brightening water. The lighthouse keeper opened a window and listened to gulls calling above the harbor.");
    expect(target.size() >= 48, "R0 teacher sequence");
    Tensor ids = Tensor::allocate(ctx, {64}, dnnl::memory::data_type::s32);
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    auto prefill = [&](const std::vector<int>& input, bool enabled) {
        backend.reset();
        _putenv_s("AILA_Q4E_INLINE_CHECKPOINT", enabled ? "1" : "0");
        expect(backend.supports_interior_checkpoint(41, 37) == enabled, "R0 dispatch");
        const auto begin = std::chrono::steady_clock::now();
        auto* logits = engine.forward_prompt_tokens(input, 0, 41, 37);
        expect(logits && session.CurrentLength() == 41 && session.snapshot_for_testing(37), "R0 checkpoint publication");
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    };
    std::vector<std::vector<float>> reference;
    CheckpointMetrics metrics;
    for (bool enabled : {false, true}) {
        prefill(prompt, enabled);
        for (int step = 0; step < 48; ++step) {
            auto logits = extract_bf16_to_f32(ctx, Access::Logits(backend));
            if (!enabled) reference.push_back(logits);
            else metrics.add(reference[step], logits, target[step]);
            if (enabled && step == 15) metrics.report("R0_cli41_first16");
            ctx.memcpy_h2d(ids.data(), &target[step], sizeof(int));
            backend.forward(ctx, ids.data_as<int>(), 1);
        }
    }
    metrics.report("R0_cli41_inline37");
    const bool quality_gate = metrics.kl / metrics.rows <= 0.01 && metrics.nll_delta / metrics.rows <= 0.01;
    prefill(prompt, true);
    const auto saved = *session.snapshot_for_testing(37);
    auto prefix_kv = [&] {
        std::vector<bf16> result;
        for (const auto& cache : session.layer_caches_for_testing()) for (const Tensor* tensor : {&cache.k, &cache.v}) {
            if (!tensor->valid()) continue;
            for (int h = 0; h < tensor->shape(0); ++h) {
                const auto offset = result.size();
                result.resize(offset + 37 * tensor->shape(2));
                ctx.memcpy_d2h(result.data() + offset, tensor->data_as<bf16>() + h * tensor->shape(1) * tensor->shape(2),
                    (result.size() - offset) * sizeof(bf16));
            }
        }
        return result;
    };
    const auto kv = prefix_kv();
    auto changed = prompt; for (int i = 37; i < 41; ++i) changed[i] = target[i - 37];
    prefill(changed, true);
    expect_same_checkpoint(saved, *session.snapshot_for_testing(37));
    expect(same_bits(kv, prefix_kv()), "R0 future suffix changed visible KV");
    std::vector<float> replay;
    for (int repeat = 0; repeat < 2; ++repeat) {
        expect(backend.Truncate(session, 37) == 37, "R0 restore");
        backend.capture_snapshot_for_testing(ctx, session, 37, 1, true);
        expect_same_checkpoint(saved, *session.snapshot_for_testing(37));
        auto* logits = engine.forward_prompt_tokens(prompt, 37, 4, 0);
        auto values = extract_bf16_to_f32(ctx, *logits);
        if (repeat == 0) replay = values; else expect(same_bits(replay, values), "R0 nondeterministic replay");
    }
    for (int pair = 0; pair < 3; ++pair) for (int order = 0; order < 2; ++order) {
        const bool enabled = (pair + order) % 2;
        const double ms = prefill(prompt, enabled);
        std::cout << "[R0 TIME] pair=" << pair << " inline=" << enabled << " prefill_ms=" << ms << "\n";
    }
    _putenv_s("AILA_Q4E_INLINE_CHECKPOINT", "1");
    std::cout << "[R0 STATE PASS] prompt=41 anchor=37 teacher_rows=48 future_independent=1 restore_exact=1 replay_exact=1\n";
    expect(quality_gate, "R0 48-step KL/NLL canary gate 0.01 not met");
    std::cout << "[R0 PASS] state_and_quality=1\n";
}
