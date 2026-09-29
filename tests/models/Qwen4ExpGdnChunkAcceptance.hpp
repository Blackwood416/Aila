// Optional real-model X4 canary, included after the common tolerance helpers.
void test_gdn_chunk_model(const std::string& directory) {
    std::cout<<std::unitbuf;
    InferenceEngine engine;
    expect(engine.init(directory,512),"X4 model initialization failed");
    auto& ctx=Access::Ctx(engine);auto& backend=Access::Backend(engine);auto& session=Access::Session(engine);
    backend.set_prefill_xmx(true);backend.set_direct_prefill_io(true);Access::Grouped(backend,0);
    Tokenizer tokenizer;expect(tokenizer.load(directory),"X4 tokenizer load failed");
    GenerationConfig config;config.do_sample=false;config.thinking_budget_tokens=0;config.max_new_tokens=48;
    aila::chat::ChatRequest request;std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config,request,&error),"X4 prompt parse failed");
    aila::chat::ChatFormatTextResult rendered;aila::chat::ChatFormatter formatter;
    expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),"X4 format failed");
    const auto story=tokenizer.encode(rendered.text);expect(story.size()==30,"X4 story count changed");
    std::string text;
    while(tokenizer.encode(text).size()<256) text+="The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
    auto long_ids=tokenizer.encode(text);long_ids.resize(256);
    auto targets=tokenizer.encode("The horizon ignites with a soft, amber glow, piercing the lingering indigo of the pre-dawn sky. Gentle waves ripple beneath the emerging light, their crests catching the first golden rays like scattered diamonds. As the sun breaches the horizon, the sea shines.");
    expect(targets.size()>=48,"X4 continuation too short");targets.resize(48);
    Tensor tokens=Tensor::allocate(ctx,{512},dnnl::memory::data_type::s32);
    struct Run { double prefill=0,total=0;uint64_t calls=0;std::vector<std::vector<float>> logits; };
    using Clock=std::chrono::steady_clock;
    auto run=[&](const std::vector<int>& ids,int mode,bool capture) {
        backend.reset();Access::GdnMode(backend,mode);
        ctx.memcpy_h2d(tokens.data(),ids.data(),ids.size()*4);
        ctx.memcpy_h2d(tokens.data_as<int>()+ids.size(),targets.data(),targets.size()*4);
        const uint64_t calls=Access::GdnCalls(backend);
        Run result;const auto begin=Clock::now();
        Tensor& first=backend.forward(ctx,tokens.data_as<int>(),int(ids.size()),{SnapshotPolicy::ReusablePrefix,true});
        ctx.synchronize();result.prefill=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        if(capture) result.logits.push_back(extract_bf16_to_f32(ctx,first));
        for(int t=0;t<(ids.size()==30?47:7);++t) {
            Tensor& next=backend.forward(ctx,tokens.data_as<int>()+ids.size()+t,1,{SnapshotPolicy::ReusablePrefix,true});
            if(capture) result.logits.push_back(extract_bf16_to_f32(ctx,next));
        }
        ctx.synchronize();result.total=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        result.calls=Access::GdnCalls(backend)-calls;
        expect(result.calls==(mode?36:0),"X4 unexpected prefill/decode dispatch count");
        return result;
    };
    for(const auto& ids:{story,long_ids}) {
        for(int mode=0;mode<3;++mode) {run(ids,mode,false);std::cout<<"[X4 WARM] count="<<ids.size()<<" mode="<<mode<<'\n';}
        const auto ref=run(ids,0,true);
        for(int mode=1;mode<3;++mode) {
            const auto cand=run(ids,mode,true);CheckpointMetrics metrics;
            for(size_t i=0;i<ref.logits.size();++i) metrics.add(ref.logits[i],cand.logits[i],targets[i]);
            std::cout<<"[X4 QUALITY] count="<<ids.size()<<" mode="<<mode<<" rows="<<metrics.rows
                <<" mean_kl="<<metrics.kl/metrics.rows<<" mean_nll_delta="<<metrics.nll_delta/metrics.rows
                <<" max_logits_nrmse="<<metrics.max_nrmse<<" warning="<<(metrics.nll_delta/metrics.rows>.01)<<'\n';
        }
        for(int pair=0;pair<3;++pair) for(int order=0;order<3;++order) {
            const int mode=(pair+order)%3;const auto sample=run(ids,mode,false);
            std::cout<<"[X4 TIME] count="<<ids.size()<<" mode="<<mode<<" pair="<<pair<<" prefill_ms="<<sample.prefill
                <<" total_ms="<<sample.total<<" chunk_calls="<<sample.calls<<'\n';
        }
        std::cout<<"[X4 MEMORY] count="<<ids.size()<<" device_bytes="<<ctx.current_allocated_bytes()
            <<" host_bytes="<<ctx.current_pinned_host_bytes()<<" cache_slots="<<Access::Cache(engine).num_slots()<<'\n';
    }
    // Nonzero context and an interior anchor inside a T16 tail. Prefix state
    // cannot depend on future input; exact restore/replay is a separate contract.
    _putenv_s("AILA_Q4E_INLINE_CHECKPOINT","1");
    for(int mode:{1,2}) {
        Access::GdnMode(backend,mode);
        auto seed=[&](const std::vector<int>& ids) {
            backend.reset();ctx.memcpy_h2d(tokens.data(),targets.data(),4*sizeof(int));
            backend.forward(ctx,tokens.data_as<int>(),4,{SnapshotPolicy::None,false});
            ctx.memcpy_h2d(tokens.data(),ids.data(),ids.size()*4);
        };
        auto capture=[&]() {
            backend.forward(ctx,tokens.data_as<int>(),30,{SnapshotPolicy::ReusablePrefix,true,false,17});
            ctx.synchronize();expect(session.CurrentLength()==34 && session.snapshot_for_testing(21),"X4 absolute checkpoint length mismatch");
        };
        seed(story);capture();const auto saved=*session.snapshot_for_testing(21);
        expect(backend.Truncate(session,21)==21,"X4 restore failed");
        backend.capture_snapshot_for_testing(ctx,session,21,1,true);
        expect_same_checkpoint(saved,*session.snapshot_for_testing(21));
        const auto replay=extract_bf16_to_f32(ctx,backend.forward(ctx,tokens.data_as<int>()+17,13));
        expect(backend.Truncate(session,21)==21,"X4 second restore failed");
        const auto again=extract_bf16_to_f32(ctx,backend.forward(ctx,tokens.data_as<int>()+17,13));
        expect(same_bits(replay,again),"X4 restored replay not deterministic");
        auto changed=story;for(int i=17;i<30;++i) changed[i]=targets[i-17];
        seed(changed);capture();expect_same_checkpoint(saved,*session.snapshot_for_testing(21));
        std::cout<<"[X4 STATE] mode="<<mode<<" start=4 anchor=21 restore=bitwise replay=bitwise suffix_independent=1\n";
    }
    _putenv_s("AILA_Q4E_INLINE_CHECKPOINT","0");
    Access::GdnMode(backend,0);
}
