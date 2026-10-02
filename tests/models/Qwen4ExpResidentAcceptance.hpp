// Real-model H1/H2 acceptance; timings include diagnostics and are not CLI TPS.
void test_resident_model(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    aila::ApplyQwen38A770Preset();
    _putenv_s("AILA_Q4E_RESIDENT_DECODE", "2"); // reserve both candidates before any decode
    InferenceEngine engine; expect(engine.init(directory,2560),"Resident model init");
    auto& ctx=Access::Ctx(engine);auto& backend=Access::Backend(engine);auto& cache=Access::Cache(engine);
    auto& session=Access::Session(engine);
    std::ifstream file(aila::env::read_string("AILA_Q4E_RESIDENT_PROMPT_FILE",""));expect(bool(file),"Resident prompt file");
    const std::string body((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    GenerationConfig config;config.do_sample=false;config.thinking_budget_tokens=0;
    aila::chat::ChatRequest request;std::string error;
    expect(aila::chat::parse_chat_request_json(body,config,request,&error),"Resident request");
    expect(request.messages.size()==1 && request.messages[0].role==aila::chat::ChatRole::User,"Resident single user prompt");
    for(auto& part:request.messages[0].content){std::string flat;bool newline=false;for(char c:part.text){if(c=='\r'||c=='\n'){if(!newline)flat+=' ';newline=true;}else{flat+=c;newline=false;}}part.text=std::move(flat);}
    aila::chat::ChatMessage system;system.role=aila::chat::ChatRole::System;system.content.push_back(aila::chat::ChatContentPart::text_part("You are a helpful assistant."));request.messages.insert(request.messages.begin(),std::move(system));
    aila::chat::ChatFormatter formatter;aila::chat::ChatFormatTextResult rendered;
    expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),"Resident chat render");
    Tokenizer tokenizer;expect(tokenizer.load(directory),"Resident tokenizer");const auto prompt=tokenizer.encode(rendered.text);
    expect(prompt.size()>4 && prompt.size()<2400,"Resident prompt bounds");
    const auto targets=tokenizer.encode(prompt.size()>256 ?
        "Begin by agreeing on stable identifiers and a single request register. Mara should define the permission decisions and choose a small set of pilot requests. Meanwhile, Jon can prepare the receiving room and arrange the wall inspection. The technician can check the cable route and rehearse restoring a sample backup before connecting the capture workflow to shared storage." :
        "The morning sun had barely crested the horizon, painting the sea in shades of gold and rose. A quiet breeze carried the scent of salt across the shore. Small waves folded over the sand while distant fishing boats crossed the brightening water. The lighthouse keeper opened a window and listened to gulls calling above the harbor.");
    expect(targets.size()>=48,"Resident teacher targets");
    auto ids=Tensor::allocate(ctx,{48},dnnl::memory::data_type::s32);ctx.memcpy_h2d(ids.data(),targets.data(),48*sizeof(int));
    struct Drain{Context& ctx;~Drain(){try{ctx.synchronize();}catch(...){}}}drain{ctx};
    std::vector<std::vector<float>> reference;
    for(int mode=0;mode<=2;++mode){
        Access::ResidentMode(backend,mode);backend.reset();
        const auto start=std::chrono::steady_clock::now();
        expect(engine.forward_prompt_tokens(prompt,0,int(prompt.size()),int(prompt.size())-4)!=nullptr,"Resident prefill");
        ctx.synchronize();const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto before=cache.stats();const auto device_count=ctx.device_allocation_count(),host_count=ctx.pinned_host_allocation_count();
        CheckpointMetrics metrics;int exact=0,top1=0;const int anchor=int(prompt.size())+24;
        for(int step=0;step<48;++step){
            auto values=extract_bf16_to_f32(ctx,Access::Logits(backend));
            if(mode==0)reference.push_back(values);
            else{metrics.add(reference[step],values,targets[step]);exact+=same_bits(reference[step],values);
                top1+=(std::max_element(reference[step].begin(),reference[step].end())-reference[step].begin())==(std::max_element(values.begin(),values.end())-values.begin());}
            backend.forward(ctx,ids.data_as<int>()+step,1,{step==23?SnapshotPolicy::ReusablePrefix:SnapshotPolicy::None,true,step==23});
        }
        const auto after=cache.stats();
        expect(ctx.device_allocation_count()==device_count && ctx.pinned_host_allocation_count()==host_count,"Resident decode allocated USM");
        if(mode){metrics.report(mode==1?"H1_resident_teacher48":"H2_resident_teacher48");
            expect(metrics.kl/metrics.rows<=.01 && metrics.nll_delta/metrics.rows<=.01 && metrics.max_nrmse<=.15,"Resident quality gate");
            expect(after.direct_entries-before.direct_entries + after.cold_request_entries-before.cold_request_entries==48ULL*48*10,"Resident route contributions missing");}
        expect(session.snapshot_for_testing(anchor),"Resident checkpoint missing");auto snapshot=*session.snapshot_for_testing(anchor);
        const auto original=extract_bf16_to_f32(ctx,Access::Logits(backend));
        for(int pass=0;pass<2;++pass){
            expect(backend.Truncate(session,anchor)==anchor,"Resident truncate");
            for(int step=24;step<48;++step)backend.forward(ctx,ids.data_as<int>()+step,1,{SnapshotPolicy::None,true});
            expect(same_bits(original,extract_bf16_to_f32(ctx,Access::Logits(backend))),"Resident replay changed logits across cache bindings");
            expect_same_checkpoint(snapshot,*session.snapshot_for_testing(anchor));
        }
        std::cout<<"[HOTSET MODEL] mode="<<mode<<" prompt="<<prompt.size()<<" teacher=48 exact_rows="<<exact<<" top1_match="<<top1
            <<" prefill_ms="<<prefill_ms<<" direct="<<after.direct_entries-before.direct_entries<<" cold="<<after.cold_request_entries-before.cold_request_entries
            <<" acquire_calls="<<after.acquire_calls-before.acquire_calls<<" demand_bytes="<<after.total_transferred_bytes-before.total_transferred_bytes
            <<" metadata_bytes="<<after.resident_metadata_bytes-before.resident_metadata_bytes<<" replay_exact=1 snapshot_exact=1 live_usm_delta=0 performance_comparable=0\n";
    }
    std::cout<<"[HOTSET MODEL PASS] quality=1 continuation=1 checkpoint_replay=1 modes=0,1,2\n";
}
