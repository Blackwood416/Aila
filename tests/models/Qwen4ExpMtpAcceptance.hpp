// Included in the tolerance executable's anonymous namespace.
void test_mtp_probe(const std::string& directory) {
    std::cout<<std::unitbuf<<std::setprecision(10);
    const int promptCount=aila::env::read_int("AILA_Q4E_MTP_PROMPT",30);
    const int anchorsRequested=aila::env::read_int("AILA_Q4E_MTP_ANCHORS",60);
    const int repeats=aila::env::read_int("AILA_Q4E_MTP_REPEATS",3);
    const int verifyPasses=aila::env::read_int_raw("AILA_Q4E_MTP_VERIFY_PASSES",3);
    const std::string prefix=aila::env::read_string("AILA_Q4E_MTP_OUTPUT","");
    expect((promptCount==30 || promptCount==512) && anchorsRequested>=1 && anchorsRequested<=60 &&
           repeats>=1 && repeats<=4 && verifyPasses>=0 && verifyPasses<=4,"Invalid MTP probe options");
    InferenceEngine engine;
    expect(engine.init(directory,640),"MTP target init failed");
    auto& ctx=Access::Ctx(engine); auto& target=Access::Backend(engine);
    Tokenizer tokenizer; expect(tokenizer.load(directory),"MTP tokenizer failed");
    std::vector<int> tokens;
    if (promptCount==30) {
        GenerationConfig config; config.do_sample=false; config.thinking_budget_tokens=0;
        aila::chat::ChatRequest request; std::string error;
        expect(aila::chat::parse_chat_request_json(
            R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
            config,request,&error),"MTP prompt parse failed");
        aila::chat::ChatFormatTextResult rendered; aila::chat::ChatFormatter formatter;
        expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),error.c_str());
        tokens=tokenizer.encode(rendered.text); expect(tokens.size()==30,"MTP prompt identity changed");
    } else {
        std::string text;
        while (tokenizer.encode(text).size()<size_t(promptCount))
            text+="The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
        tokens=tokenizer.encode(text); tokens.resize(promptCount);
    }
    const int maxNew=anchorsRequested+4;
    Tensor ids=Tensor::allocate(ctx,{640},dnnl::memory::data_type::s32);
    Tensor targetHidden=Tensor::allocate(ctx,{promptCount+maxNew,10240},dnnl::memory::data_type::bf16);
    Tensor draftHidden=Tensor::allocate(ctx,{1,10240},dnnl::memory::data_type::bf16);
    Access::ReserveVerifyLogits(engine,4);
    Qwen4ExpMtpProbe draft(engine,640);
    struct Drain { Context& c; ~Drain(){try{c.synchronize();}catch(...){}} } drain{ctx};
    auto dump=[&](const std::string& suffix,const void* data,size_t bytes) {
        if(prefix.empty())return;
        std::vector<uint8_t> host(bytes); ctx.memcpy_d2h(host.data(),data,bytes);
        std::ofstream stream(prefix+suffix,std::ios::binary);stream.write(reinterpret_cast<const char*>(host.data()),host.size());
        expect(bool(stream),"MTP artifact write failed");
    };
    std::vector<bf16> hostLogits(target.vocab_size());
    uint64_t lastLogitsHash=0;
    auto argmax=[&](const bf16* logits) {
        ctx.memcpy_d2h(hostLogits.data(),logits,hostLogits.size()*sizeof(bf16));
        int best=0;
        for(int i=0;i<target.vocab_size();++i) {
            expect(std::isfinite(float(hostLogits[i])),"MTP nonfinite logits");
            if(float(hostLogits[i])>float(hostLogits[best]))best=i;
        }
        lastLogitsHash=14695981039346656037ULL;
        const auto* bytes=reinterpret_cast<const uint8_t*>(hostLogits.data());
        for(size_t i=0;i<hostLogits.size()*sizeof(bf16);++i)lastLogitsHash=(lastLogitsHash^bytes[i])*1099511628211ULL;
        return best;
    };
    auto now=[] {return std::chrono::steady_clock::now();};
    auto elapsed=[&](auto start){return std::chrono::duration<double,std::milli>(now()-start).count();};
    engine.reset_context(); ctx.memcpy_h2d(ids.data(),tokens.data(),tokens.size()*sizeof(int));
    auto prefillStart=now();
    auto& promptLogits=Access::MtpTargetPrompt(engine,ids.data_as<int>(),promptCount);
    ctx.synchronize(); double prefillMs=elapsed(prefillStart);
    ctx.queue().memcpy(targetHidden.data(),Access::FrontHidden(target).data(),size_t(promptCount)*10240*sizeof(bf16));
    int next=argmax(promptLogits.data_as<bf16>());
    std::vector<double> mainTimes;
    for(int t=0;t<maxNew;++t) {
        tokens.push_back(next);
        if(next==248044 || next==248046)break; // Never manufacture acceptance after EOS.
        ctx.memcpy_h2d(ids.data(),&next,sizeof(int)); auto start=now();
        auto& logits=Access::VerifyRows(engine,ids.data_as<int>(),1);ctx.synchronize();
        mainTimes.push_back(elapsed(start));
        ctx.queue().memcpy(targetHidden.data_as<bf16>()+size_t(promptCount+t)*10240,
            Access::FrontHidden(target).data(),10240*sizeof(bf16));
        next=argmax(logits.data_as<bf16>());
    }
    const int anchors=std::min(anchorsRequested,int(tokens.size())-promptCount-3);
    expect(anchors>0,"Target stopped before any complete three-draft anchor");
    auto printInts=[](const auto& values) { std::cout<<'[';bool first=true;for(auto v:values){if(!first)std::cout<<',';first=false;std::cout<<v;}std::cout<<']'; };
    std::cout<<"[MTP TARGET] {\"prompt_tokens\":"<<promptCount<<",\"anchors\":"<<anchors<<",\"prefill_ms\":"<<prefillMs
             <<",\"target_single_ms\":";printInts(mainTimes);std::cout<<",\"tokens\":";printInts(tokens);std::cout<<"}\n";
    std::vector<int> referenceDrafts;
    std::vector<uint64_t> referenceHashes;
    std::vector<int> referenceAccepted;
    for(int pass=0;pass<repeats;++pass) {
        draft.Reset();
        for(int p=0;p<promptCount-1;++p) {
            const bool referenceCase=pass==0 && p<4 && !prefix.empty();
            if(referenceCase)dump(".ref"+std::to_string(p)+".input.bf16",targetHidden.data_as<bf16>()+size_t(p)*10240,10240*sizeof(bf16));
            draft.Step(targetHidden.data_as<bf16>()+size_t(p)*10240,tokens[p+1],p,referenceCase,
                referenceCase ? prefix+".ref"+std::to_string(p) : "");
            if(referenceCase) {
                dump(".ref"+std::to_string(p)+".hidden.bf16",Access::FrontHidden(target).data(),10240*sizeof(bf16));
                dump(".ref"+std::to_string(p)+".mixed.bf16",Access::FrontX(target).data(),2560*sizeof(bf16));
                dump(".ref"+std::to_string(p)+".logits.bf16",Access::Logits(target).data(),size_t(target.vocab_size())*sizeof(bf16));
                std::cout<<"[MTP REF] {\"position\":"<<p<<",\"token\":"<<tokens[p+1]<<",\"experts\":";
                std::vector<int> route(Access::FrontIds(target),Access::FrontIds(target)+10);printInts(route);std::cout<<"}\n";
            }
        }
        ctx.synchronize();
        const auto allocations=ctx.device_allocation_count(), pinned=ctx.pinned_host_allocation_count();
        const auto misses=draft.experts.stats().miss_count;
        const auto draftBytes=draft.experts.stats().total_transferred_bytes;
        std::vector<int> accepted, predictions; std::vector<double> core,loop;
        std::vector<uint64_t> hashes;
        for(int a=0;a<anchors;++a) {
            const int p=promptCount-1+a;
            const bf16* hidden=targetHidden.data_as<bf16>()+size_t(p)*10240;
            int input=tokens[p+1], matched=0; bool prefixMatches=true;
            const auto loopStart=now(); double coreMs=0;
            for(int step=0;step<3;++step) {
                const auto start=now(); draft.Step(hidden,input,p+step);ctx.synchronize();coreMs+=elapsed(start);
                input=argmax(Access::Logits(target).data_as<bf16>());predictions.push_back(input);
                hashes.push_back(lastLogitsHash);
                if(prefixMatches && input==tokens[p+2+step])++matched;else prefixMatches=false;
                ctx.queue().memcpy(draftHidden.data(),Access::FrontHidden(target).data(),10240*sizeof(bf16));
                hidden=draftHidden.data_as<bf16>();
            }
            ctx.synchronize(); loop.push_back(elapsed(loopStart)); core.push_back(coreMs); accepted.push_back(matched);
        }
        if(pass==0){referenceDrafts=predictions;referenceHashes=hashes;referenceAccepted=accepted;}
        else expect(predictions==referenceDrafts && hashes==referenceHashes,"MTP repeated trajectory changed logits");
        expect(ctx.device_allocation_count()==allocations && ctx.pinned_host_allocation_count()==pinned,"MTP grew live USM allocations");
        const auto newMisses=draft.experts.stats().miss_count-misses;
        if(draft.experts.num_slots()==512)expect(newMisses==0,"Resident MTP experts unexpectedly missed");
        std::cout<<"[MTP PASS] {\"pass\":"<<pass<<",\"anchors\":"<<anchors<<",\"core_ms\":";printInts(core);
        std::cout<<",\"loop_ms\":";printInts(loop);std::cout<<",\"accepted_prefix\":";printInts(accepted);
        std::cout<<",\"draft_tokens\":";printInts(predictions);
        std::cout<<",\"live_usm_delta\":0,\"draft_misses\":"<<newMisses<<",\"draft_demand_bytes\":"<<draft.experts.stats().total_transferred_bytes-draftBytes<<"}\n";
    }
    // Counterfactual greedy cycles on this fixed target trajectory. Rebuild
    // the accepted target-aligned MTP KV rows before each new anchor. Skipping
    // these catch-up rows would underestimate draft cost after multi-accept.
    draft.Reset();
    for(int p=0;p<promptCount-1;++p)draft.Step(targetHidden.data_as<bf16>()+size_t(p)*10240,tokens[p+1],p,false);
    ctx.synchronize();
    std::vector<int> cycleAnchors,cycleAccepted,catchupRows;
    std::vector<double> cycleCore,cycleLoop;
    int lastCorrect=promptCount-2;
    for(int a=0;a<anchors;) {
        const int p=promptCount-1+a;const auto loopStart=now(),coreStart=now();
        int catchup=0;
        for(int r=lastCorrect+1;r<p;++r){draft.Step(targetHidden.data_as<bf16>()+size_t(r)*10240,tokens[r+1],r,false);++catchup;}
        ctx.synchronize();double coreMs=elapsed(coreStart);
        const bf16* hidden=targetHidden.data_as<bf16>()+size_t(p)*10240;
        int input=tokens[p+1],matched=0;bool good=true;
        for(int step=0;step<3;++step){
            const auto start=now();draft.Step(hidden,input,p+step);ctx.synchronize();coreMs+=elapsed(start);
            input=argmax(Access::Logits(target).data_as<bf16>());
            if(good && input==tokens[p+2+step])++matched;else good=false;
            ctx.queue().memcpy(draftHidden.data(),Access::FrontHidden(target).data(),10240*sizeof(bf16));hidden=draftHidden.data_as<bf16>();
        }
        ctx.synchronize();expect(matched==referenceAccepted[a],"MTP target-aligned KV catch-up changed acceptance");
        cycleAnchors.push_back(a);cycleAccepted.push_back(matched);catchupRows.push_back(catchup);
        cycleCore.push_back(coreMs);cycleLoop.push_back(elapsed(loopStart));
        lastCorrect=p;a+=1+matched;
    }
    std::cout<<"[MTP CYCLES] {\"anchor\":";printInts(cycleAnchors);std::cout<<",\"accepted_prefix\":";printInts(cycleAccepted);
    std::cout<<",\"catchup_rows\":";printInts(catchupRows);std::cout<<",\"core_ms\":";printInts(cycleCore);std::cout<<",\"loop_ms\":";printInts(cycleLoop);std::cout<<"}\n";
    // Measure target verification with the draft weights/pool actually resident
    // and the admission reserve charged. Fixed greedy trajectory, no rollback.
    const int verifyTokens=(int(tokens.size())-promptCount)/4*4;
    ctx.memcpy_h2d(ids.data(),tokens.data()+promptCount,size_t(verifyTokens)*sizeof(int));
    std::vector<int> prompt(tokens.begin(),tokens.begin()+promptCount);
    std::vector<uint64_t> expectedVerifyHashes;
    for(int pass=0;pass<verifyPasses;++pass) {
        engine.reset_context();auto start=now();
        engine.forward_prompt_tokens(prompt,0,promptCount,0,true,{SnapshotPolicy::None,true});ctx.synchronize();
        double pf=elapsed(start);std::vector<double> times;
        const auto allocations=ctx.device_allocation_count(), pinned=ctx.pinned_host_allocation_count();
        const auto before=Access::Cache(engine).stats();std::vector<uint64_t> hashes;
        for(int t=0;t<verifyTokens;t+=4) {
            start=now();Access::VerifyRows(engine,ids.data_as<int>()+t,4);ctx.synchronize();times.push_back(elapsed(start));
            for(int r=0;r<4;++r){argmax(Access::Logits(target).data_as<bf16>()+size_t(r)*target.vocab_size());hashes.push_back(lastLogitsHash);}
        }
        if(pass==0)expectedVerifyHashes=hashes;else expect(hashes==expectedVerifyHashes,"MTP resident verification changed logits");
        expect(ctx.device_allocation_count()==allocations && ctx.pinned_host_allocation_count()==pinned,"MTP verify live allocation growth");
        const auto after=Access::Cache(engine).stats();
        std::cout<<"[MTP VERIFY] {\"pass\":"<<pass<<",\"prefill_ms\":"<<pf<<",\"tokens\":"<<verifyTokens<<",\"block_ms\":";printInts(times);
        std::cout<<",\"misses\":"<<after.miss_count-before.miss_count<<",\"demand_bytes\":"<<after.total_transferred_bytes-before.total_transferred_bytes<<"}\n";
    }
    if(verifyPasses>0) {
        std::vector<int> continuation(tokens.begin()+promptCount,tokens.end());
        if(continuation.size()>=6)draft.CheckTargetRollback(engine,prompt,continuation);
    }
    std::cout<<"[MTP END] conditional_greedy_probe=1 production_speculative_decode=0 accepted_prefix_commit=0\n";
}
