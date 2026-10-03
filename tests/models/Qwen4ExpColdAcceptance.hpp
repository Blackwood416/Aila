// C1/H3: one model and one frozen logical hotset shared by both compute arms.
// Performance calibration observes prompt routes only. Quality runs additionally
// retain the original reference's cache history; their timings are diagnostic.
void test_cold_model(const std::string& directory,bool benchmark) {
    std::cout<<std::unitbuf<<std::setprecision(10);
    aila::ApplyQwen38A770Preset();
    _putenv_s("AILA_Q4E_COLD_MODE","1");
    InferenceEngine engine;expect(engine.init(directory,2560),"Cold model init");
    auto& ctx=Access::Ctx(engine);auto& backend=Access::Backend(engine);auto& cache=Access::Cache(engine);
    auto& session=Access::Session(engine);
    const bool adaptive=cache.adaptive_hotset();
    if(adaptive)cache.RestartHotsetForTesting(false);
    Access::ColdMode(backend,0,false);
    std::ifstream file(aila::env::read_string("AILA_Q4E_RESIDENT_PROMPT_FILE",""));expect(bool(file),"Cold prompt file");
    const std::string body((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    GenerationConfig config;config.do_sample=false;config.thinking_budget_tokens=0;config.max_new_tokens=48;
    config.decode_chunk_size=1;config.stream_chunk_size=1;
    aila::chat::ChatRequest request;std::string error;
    expect(aila::chat::parse_chat_request_json(body,config,request,&error),"Cold request");
    expect(request.messages.size()==1 && request.messages[0].role==aila::chat::ChatRole::User,"Cold user prompt");
    for(auto& part:request.messages[0].content){std::string flat;bool newline=false;for(char c:part.text){if(c=='\r'||c=='\n'){if(!newline)flat+=' ';newline=true;}else{flat+=c;newline=false;}}part.text=std::move(flat);}
    aila::chat::ChatMessage system;system.role=aila::chat::ChatRole::System;system.content.push_back(aila::chat::ChatContentPart::text_part("You are a helpful assistant."));request.messages.insert(request.messages.begin(),std::move(system));
    aila::chat::ChatFormatter formatter;aila::chat::ChatFormatTextResult rendered;
    expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),"Cold render");
    Tokenizer tokenizer;expect(tokenizer.load(directory),"Cold tokenizer");const auto prompt=tokenizer.encode(rendered.text);
    expect(prompt.size()>4 && prompt.size()<2400,"Cold prompt bounds");
    const auto targets=tokenizer.encode(prompt.size()>256 ?
        "Begin by agreeing on stable identifiers and a single request register. Mara should define the permission decisions and choose a small set of pilot requests. Meanwhile, Jon can prepare the receiving room and arrange the wall inspection. The technician can check the cable route and rehearse restoring a sample backup before connecting the capture workflow to shared storage." :
        "The morning sun had barely crested the horizon, painting the sea in shades of gold and rose. A quiet breeze carried the scent of salt across the shore. Small waves folded over the sand while distant fishing boats crossed the brightening water. The lighthouse keeper opened a window and listened to gulls calling above the harbor.");
    expect(targets.size()>=48,"Cold teacher targets");
    auto ids=Tensor::allocate(ctx,{48},dnnl::memory::data_type::s32);ctx.memcpy_h2d(ids.data(),targets.data(),48*sizeof(int));
    struct Drain{Context& ctx;~Drain(){try{ctx.synchronize();}catch(...){}}}drain{ctx};
    auto prefill=[&]{backend.reset();expect(engine.forward_prompt_tokens(prompt,0,int(prompt.size()),int(prompt.size())-4)!=nullptr,"Cold prefill");ctx.synchronize();};
    std::vector<std::vector<float>> reference;
    if(!benchmark) {
        prefill();
        for(int step=0;step<48;++step) {reference.push_back(extract_bf16_to_f32(ctx,Access::Logits(backend)));backend.forward(ctx,ids.data_as<int>()+step,1,{SnapshotPolicy::None,true});}
    }
    // Rebuild from prompt demand before freezing. Calibration time is reported.
    const auto calibration_begin=std::chrono::steady_clock::now();
    if(adaptive)cache.RestartHotsetForTesting(true);
    else {prefill();cache.FreezeReady(512);}
    const auto identity=cache.FrozenIdentity();
    const auto calibration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-calibration_begin).count();
    std::cout<<"[COLD CALIBRATION] prompt="<<prompt.size()<<" slots="<<cache.frozen_slots()<<" identity="<<identity<<" ms="<<calibration_ms<<"\n";
    if(aila::env::read_flag("AILA_Q4E_COLD_PIPELINE_DIAGNOSTIC",false)) {
        struct SpotDone {};
        struct Diagnostic {Context* context;int mode=1,index=0;std::vector<std::vector<bf16>> x,y;std::vector<std::vector<float>> down;bool different=false;};
        Diagnostic diagnostic{&ctx};
        for(int mode=1;mode<=2;++mode) {
            Access::ColdMode(backend,mode,false);prefill();diagnostic.mode=mode;diagnostic.index=0;
            Access::ResidentProbe(backend,&diagnostic,[](void* p,Qwen4ExpBnb4Backend& b,int layer) {
                auto& d=*static_cast<Diagnostic*>(p);auto& context=*d.context;
                std::vector<bf16> x(2560),y(2560);std::vector<float> down(25600);
                context.memcpy_d2h(x.data(),Access::MoeInput(b).data(),x.size()*sizeof(bf16));
                context.memcpy_d2h(y.data(),Access::MoeOutput(b).data(),y.size()*sizeof(bf16));
                context.memcpy_d2h(down.data(),Access::ResidentDown(b).data(),down.size()*sizeof(float));
                if(d.mode==1) {d.x.push_back(x);d.y.push_back(y);d.down.push_back(down);}
                else if(std::memcmp(y.data(),d.y[d.index].data(),y.size()*sizeof(bf16))) {
                    std::cout<<"[COLD FIRST DIVERGENCE] token="<<d.index/48<<" layer="<<layer<<"\n";
                    q4e_cpu_diagnostic::compare("pipeline_x",d.x[d.index].data(),x.data(),x.size());
                    q4e_cpu_diagnostic::compare("pipeline_y",d.y[d.index].data(),y.data(),y.size());
                    for(int r=0;r<10;++r){std::cout<<"[COLD RANK] rank="<<r<<"\n";q4e_cpu_diagnostic::compare("pipeline_down",d.down[d.index].data()+r*2560,down.data()+r*2560,2560);}
                    d.different=true;throw SpotDone{};
                }
                ++d.index;
            });
            try{for(int i=0;i<8;++i)backend.forward(ctx,ids.data_as<int>()+i,1,{SnapshotPolicy::None,true});}
            catch(const SpotDone&){}
            Access::ResidentProbe(backend,nullptr,nullptr);
        }
        std::cout<<"[COLD PIPELINE DIAGNOSTIC DONE] different="<<diagnostic.different<<"\n";return;
    }
    if(aila::env::read_flag("AILA_Q4E_COLD_DIAGNOSTIC",false)) {
        struct Diagnostic {Qwen4ExpBnb4Backend* backend;q4e_cpu_diagnostic::State state;};
        Diagnostic diagnostic{&backend,{0,2}};
        Access::ColdMode(backend,2,false);prefill();
        backend.expert_executor().set_cpu_observer_for_testing([](void* p,Context& context,int layer,const aila::RouteRankEntry& entry,const ops::MoeExpertSlotHost& host,const bf16* x,const bf16* h,const float* out) {
            auto& diagnostic=*static_cast<Diagnostic*>(p);std::vector<bf16> original(2560);
            context.memcpy_d2h(original.data(),Access::MoeInput(*diagnostic.backend).data(),2560*sizeof(bf16));
            q4e_cpu_diagnostic::compare("cold_input_copy",original.data(),x,2560);
            q4e_cpu_diagnostic::observe(&diagnostic.state,context,layer,entry,host,x,h,out);
        },&diagnostic);
        try{for(int i=0;i<48;++i)backend.forward(ctx,ids.data_as<int>()+i,1,{SnapshotPolicy::None,true});}
        catch(const q4e_cpu_diagnostic::Complete&){}
        backend.expert_executor().set_cpu_observer_for_testing(nullptr,nullptr);
        std::cout<<"[COLD DIAGNOSTIC DONE] samples="<<diagnostic.state.samples<<"\n";return;
    }
    if(benchmark) {
        const bool h3=aila::env::read_flag("AILA_Q4E_COLD_BENCH_H3",false);
        const bool reverse=aila::env::read_flag("AILA_Q4E_COLD_BENCH_REVERSE",false);
        const int order[4]={1,2,2,1};
        for(int i=0;i<4;++i) {
            const int mode=reverse?3-order[i]:order[i];Access::ColdMode(backend,mode,h3);engine.reset_context();
            if(adaptive)cache.RestartHotsetForTesting(true);
            const auto before=cache.stats();const auto cpu0=backend.hetero_stats();
            const auto begin=std::chrono::steady_clock::now();double first_ms=0,last_ms=0;int callbacks=0;
            const auto output=engine.generate_chat_request_text(request,[&](const std::string&){last_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(!callbacks++)first_ms=last_ms;});
            ctx.synchronize();const auto request_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            const auto after=cache.stats();const auto cpu=backend.hetero_stats();
            if(!adaptive)expect(cache.FrozenIdentity()==identity,"Benchmark changed frozen residency");expect(!output.empty(),"Empty cold output");
            std::cout<<"[COLD BENCH] index="<<i<<" mode="<<mode<<" h3="<<h3<<" adaptive="<<adaptive<<" prompt="<<prompt.size()<<" identity="<<identity
                <<" ttft_ms="<<first_ms<<" request_ms="<<request_ms<<" last_callback_ms="<<last_ms<<" callbacks="<<callbacks<<" demand_bytes="<<after.total_transferred_bytes-before.total_transferred_bytes
                <<" direct="<<after.direct_entries-before.direct_entries<<" acquire_calls="<<after.acquire_calls-before.acquire_calls
                <<" cpu_tasks="<<cpu.cpu_tasks_completed-cpu0.cpu_tasks_completed<<" compute_ns="<<cpu.cpu_compute_ns-cpu0.cpu_compute_ns
                <<" join_ns="<<cpu.cpu_join_wait_ns-cpu0.cpu_join_wait_ns<<" activation_ns="<<cpu.cpu_activation_wait_ns-cpu0.cpu_activation_wait_ns
                <<" result_bytes="<<cpu.cpu_result_bytes-cpu0.cpu_result_bytes<<"\n[COLD TEXT] "<<output<<"\n[/COLD TEXT]\n";
        }
        std::cout<<"[COLD BENCH PASS] frozen_identity="<<int(!adaptive)<<" adaptive="<<adaptive<<" generated_requests=4\n";return;
    }
    uint64_t cpu_total=0,h3_direct=0;bool nll_warning=false;
    const int h3_filter=aila::env::read_int_raw("AILA_Q4E_COLD_TEST_H3",-1);
    for(int variant=0;variant<4;++variant) {
        const int mode=variant%2+1;const bool h3=variant>=2;Access::ColdMode(backend,mode,h3);
        if(h3_filter>=0 && int(h3)!=h3_filter)continue;
        if(adaptive)cache.RestartHotsetForTesting(true);
        const auto pre_stats=cache.stats();prefill();h3_direct+=h3?cache.stats().direct_entries-pre_stats.direct_entries:0;
        const auto before=cache.stats();const auto cpu0=backend.hetero_stats();
        const auto device_count=ctx.device_allocation_count(),host_count=ctx.pinned_host_allocation_count();
        CheckpointMetrics metrics;int exact=0;const int anchor=int(prompt.size())+24;
        for(int step=0;step<48;++step) {
            const auto values=extract_bf16_to_f32(ctx,Access::Logits(backend));metrics.add(reference[step],values,targets[step]);exact+=same_bits(reference[step],values);
            backend.forward(ctx,ids.data_as<int>()+step,1,{step==23?SnapshotPolicy::ReusablePrefix:SnapshotPolicy::None,true,step==23});
        }
        const auto after=cache.stats();const auto cpu=backend.hetero_stats();cpu_total+=cpu.cpu_tasks_completed-cpu0.cpu_tasks_completed;
        metrics.report("C1_H3_teacher48");
        // The original tolerance plan defines +.01 nat/token as an alert,
        // not a calibrated universal pass threshold. Keep the alert visible.
        nll_warning=nll_warning || metrics.nll_delta/metrics.rows>.01;
        expect(metrics.kl/metrics.rows<=.01 && metrics.max_nrmse<=.15,"Cold KL/NRMSE gate");
        expect(ctx.device_allocation_count()==device_count && ctx.pinned_host_allocation_count()==host_count,"Cold decode allocated USM");
        expect(after.direct_entries-before.direct_entries+after.cold_request_entries-before.cold_request_entries==48ULL*48*10,"Cold route coverage");
        if(!adaptive)expect(cache.FrozenIdentity()==identity,"Frozen set drift");
        else if(aila::env::read_int_raw("AILA_Q4E_HOTSET_SWAPS",96)>0)
            expect(after.epoch_swaps>before.epoch_swaps,"H4 did not adapt during decode");
        else expect(after.epoch_swaps==before.epoch_swaps,"Disabled online admission changed the hotset");
        expect(session.snapshot_for_testing(anchor),"Cold checkpoint missing");auto snapshot=*session.snapshot_for_testing(anchor);
        const auto original=extract_bf16_to_f32(ctx,Access::Logits(backend));
        for(int pass=0;pass<2;++pass) {
            expect(backend.Truncate(session,anchor)==anchor,"Cold truncate");
            for(int step=24;step<48;++step)backend.forward(ctx,ids.data_as<int>()+step,1,{SnapshotPolicy::None,true});
            const auto replay=extract_bf16_to_f32(ctx,Access::Logits(backend));CheckpointMetrics replay_metrics;replay_metrics.add(original,replay,targets[47]);
            expect(replay_metrics.max_nrmse<=.15 && replay_metrics.kl<=.01,"Cold replay quality");
            if(mode==1)expect(same_bits(original,replay),"GPU frozen replay not exact");
            expect_same_checkpoint(snapshot,*session.snapshot_for_testing(anchor));
        }
        std::cout<<"[COLD MODEL] mode="<<mode<<" h3="<<h3<<" exact_rows="<<exact<<" cpu_tasks="<<cpu.cpu_tasks_completed-cpu0.cpu_tasks_completed
            <<" identity="<<identity<<" replay=1 checkpoint=1 live_usm_delta=0\n";
    }
    expect(cpu_total>0,"C1 ran no CPU tasks");if(h3_filter!=0)expect(h3_direct>0,"H3 ran no direct batches");
    // Cancel at the actual CPU result boundary, then recover with reset.
    struct CancelState{bool observed=false;};CancelState cancelled;
    backend.expert_executor().set_cpu_observer_for_testing([](void* p,Context&,int,const aila::RouteRankEntry&,const ops::MoeExpertSlotHost&,const bf16*,const bf16*,const float*) {
        static_cast<CancelState*>(p)->observed=true;throw std::runtime_error("cold test cancellation");
    },&cancelled);
    Access::ColdMode(backend,2,true);prefill();
    try {for(int i=0;i<48;++i)backend.forward(ctx,ids.data_as<int>()+i,1,{SnapshotPolicy::None,true});}
    catch(const std::runtime_error& e){expect(std::string(e.what())=="cold test cancellation","Unexpected cold cancellation error");}
    backend.expert_executor().set_cpu_observer_for_testing(nullptr,nullptr);
    expect(cancelled.observed && !cache.is_faulted(),"Cold cancellation not exercised/poisoned cache");
    Access::ColdMode(backend,1,true);prefill();backend.forward(ctx,ids.data_as<int>(),1,{SnapshotPolicy::None,true});ctx.synchronize();
    if(!adaptive)expect(cache.FrozenIdentity()==identity,"Cancellation lost frozen identity");
    std::cout<<"[COLD MODEL PASS] cpu_tasks="<<cpu_total<<" prefill_direct="<<h3_direct<<" frozen_identity="<<int(!adaptive)<<" adaptive="<<adaptive<<" kl_nrmse=1 nll_warning="<<nll_warning<<" checkpoint=1 cancellation=1\n";
}
