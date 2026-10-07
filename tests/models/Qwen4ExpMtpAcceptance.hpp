// Included in the tolerance executable's anonymous namespace.
void test_mtp_probe(const std::string& directory) {
    std::cout<<std::unitbuf<<std::setprecision(10);
    int promptCount=aila::env::read_int("AILA_Q4E_MTP_PROMPT",30);
    const int anchorsRequested=aila::env::read_int("AILA_Q4E_MTP_ANCHORS",60);
    const int repeats=aila::env::read_int("AILA_Q4E_MTP_REPEATS",3);
    const int verifyPasses=aila::env::read_int_raw("AILA_Q4E_MTP_VERIFY_PASSES",3);
    const int capacity=aila::env::read_int("AILA_Q4E_MTP_MAX_CONTEXT",640);
    const bool compareKv=aila::env::read_flag("AILA_Q4E_MTP_COMPARE_KV",false);
    const bool windowSweep=aila::env::read_flag("AILA_Q4E_MTP_WINDOW_SWEEP",false);
    const bool compareResident=aila::env::read_flag("AILA_Q4E_MTP_COMPARE_RESIDENT",false);
    const std::string promptFile=aila::env::read_string("AILA_Q4E_MTP_PROMPT_FILE","");
    const std::string prefix=aila::env::read_string("AILA_Q4E_MTP_OUTPUT","");
    expect((promptCount==30 || promptCount==512) && anchorsRequested>=1 && anchorsRequested<=124 &&
           repeats>=1 && repeats<=4 && verifyPasses>=0 && verifyPasses<=4,"Invalid MTP probe options");
    InferenceEngine engine;
    expect(capacity==640 || capacity==2048,"MTP context must be 640 or 2048");
    expect(engine.init(directory,capacity),"MTP target init failed");
    auto& ctx=Access::Ctx(engine); auto& target=Access::Backend(engine);
    Tokenizer tokenizer; expect(tokenizer.load(directory),"MTP tokenizer failed");
    std::vector<int> tokens;
    if (promptCount==30 || !promptFile.empty()) {
        GenerationConfig config; config.do_sample=false; config.thinking_budget_tokens=0;
        aila::chat::ChatRequest request; std::string error;
        std::string messages=R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])";
        if(!promptFile.empty()) {
            std::ifstream in(promptFile,std::ios::binary);expect(bool(in),"MTP prompt file missing");
            messages.assign(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
        }
        expect(aila::chat::parse_chat_request_json(messages,config,request,&error),"MTP prompt parse failed");
        aila::chat::ChatFormatTextResult rendered; aila::chat::ChatFormatter formatter;
        expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),error.c_str());
        tokens=tokenizer.encode(rendered.text);
        if(promptFile.empty())expect(tokens.size()==30,"MTP prompt identity changed");
        else {expect(tokens.size()>0 && tokens.size()+anchorsRequested+4<=size_t(capacity),"MTP corpus prompt exceeds context budget");promptCount=int(tokens.size());}
    } else {
        std::string text;
        while (tokenizer.encode(text).size()<size_t(promptCount))
            text+="The lighthouse overlooks the ocean. At sunrise the keeper watches the waves and writes in the logbook. ";
        tokens=tokenizer.encode(text); tokens.resize(promptCount);
    }
    const int maxNew=anchorsRequested+4;
    Tensor ids=Tensor::allocate(ctx,{capacity},dnnl::memory::data_type::s32);
    Tensor targetHidden=Tensor::allocate(ctx,{promptCount+maxNew,10240},dnnl::memory::data_type::bf16);
    Tensor draftHidden=Tensor::allocate(ctx,{1,10240},dnnl::memory::data_type::bf16);
    Access::ReserveVerifyLogits(engine,4);
    Qwen4ExpMtpProbe draft(engine,capacity);
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
            if(windowSweep && !referenceCase)draft.AppendTargetKv(targetHidden.data_as<bf16>()+size_t(p)*10240,tokens[p+1],p);
            else draft.Step(targetHidden.data_as<bf16>()+size_t(p)*10240,tokens[p+1],p,referenceCase,
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
    if(compareKv)draft.CheckTargetKv(targetHidden.data_as<bf16>(),tokens,promptCount+anchors-1);
    for(int cyclePass=0;cyclePass<(windowSweep ? 4*repeats : compareKv ? 2*repeats : 1);++cyclePass) {
    // AB / BA / AB ordering, each path starts from an independently rebuilt KV.
    const int width=windowSweep ? ((cyclePass/4)%2 ? 4-cyclePass%4 : 1+cyclePass%4) : 4;
    const bool kvOnly=windowSweep || (compareKv && ((cyclePass/2)%2 ? cyclePass%2==0 : cyclePass%2==1));
    auto catchUp=[&](int p) {
        const auto* h=targetHidden.data_as<bf16>()+size_t(p)*10240;
        if(kvOnly)draft.AppendTargetKv(h,tokens[p+1],p);else draft.Step(h,tokens[p+1],p,false);
    };
    draft.Reset();
    if(width>1)for(int p=0;p<promptCount-1;++p)catchUp(p);
    ctx.synchronize();
    const auto cycleStats=draft.experts.stats();
    const auto cycleLive=ctx.device_allocation_count(),cyclePinned=ctx.pinned_host_allocation_count();
    std::vector<int> cycleAnchors,cycleAccepted,catchupRows;
    std::vector<double> cycleCore,cycleLoop,catchupTimes;
    int lastCorrect=promptCount-2;
    for(int a=0;a<anchors;) {
        const int p=promptCount-1+a;const auto loopStart=now(),coreStart=now();
        int catchup=0;
        if(width>1)for(int r=lastCorrect+1;r<p;++r){catchUp(r);++catchup;}
        ctx.synchronize();double coreMs=elapsed(coreStart);
        catchupTimes.push_back(coreMs);
        const bf16* hidden=targetHidden.data_as<bf16>()+size_t(p)*10240;
        int input=tokens[p+1],matched=0;bool good=true;
        for(int step=0;step<width-1;++step){
            const auto start=now();draft.Step(hidden,input,p+step);ctx.synchronize();coreMs+=elapsed(start);
            input=argmax(Access::Logits(target).data_as<bf16>());
            expect(input==referenceDrafts[a*3+step] && lastLogitsHash==referenceHashes[a*3+step],
                "MTP catch-up changed subsequent draft logits");
            if(good && input==tokens[p+2+step])++matched;else good=false;
            ctx.queue().memcpy(draftHidden.data(),Access::FrontHidden(target).data(),10240*sizeof(bf16));hidden=draftHidden.data_as<bf16>();
        }
        ctx.synchronize();expect(matched==std::min(width-1,referenceAccepted[a]),"MTP target-aligned KV catch-up changed acceptance");
        cycleAnchors.push_back(a);cycleAccepted.push_back(matched);catchupRows.push_back(catchup);
        cycleCore.push_back(coreMs);cycleLoop.push_back(elapsed(loopStart));
        lastCorrect=p;a+=1+matched;
    }
    expect(ctx.device_allocation_count()==cycleLive && ctx.pinned_host_allocation_count()==cyclePinned,"MTP cycles grew live USM");
    std::cout<<"[MTP CYCLES] {\"pass\":"<<cyclePass<<",\"width\":"<<width<<",\"kv_only\":"<<(kvOnly?"true":"false")<<",\"anchor\":";printInts(cycleAnchors);std::cout<<",\"accepted_prefix\":";printInts(cycleAccepted);
    std::cout<<",\"catchup_rows\":";printInts(catchupRows);std::cout<<",\"catchup_ms\":";printInts(catchupTimes);
    std::cout<<",\"core_ms\":";printInts(cycleCore);std::cout<<",\"loop_ms\":";printInts(cycleLoop);
    std::cout<<",\"draft_misses\":"<<draft.experts.stats().miss_count-cycleStats.miss_count
             <<",\"draft_demand_bytes\":"<<draft.experts.stats().total_transferred_bytes-cycleStats.total_transferred_bytes
             <<",\"continuation_bitwise\":true,\"live_usm_delta\":0}\n";
    }
    // Measure target verification with the draft weights/pool actually resident
    // and the admission reserve charged. Fixed greedy trajectory, no rollback.
    const int continuationTokens=int(tokens.size())-promptCount;
    ctx.memcpy_h2d(ids.data(),tokens.data()+promptCount,size_t(continuationTokens)*sizeof(int));
    std::vector<int> prompt(tokens.begin(),tokens.begin()+promptCount);
    std::array<std::vector<uint64_t>,4> expectedVerifyHashes;
    for(int pass=0;pass<verifyPasses;++pass) {
      for(int w=0;w<(windowSweep?4:1);++w) {
       for(int variant=0;variant<(compareResident?2:1);++variant) {
        const bool residentSplit=compareResident && (pass%2 ? variant==0 : variant==1);
        if(compareResident)Access::ResidentSplit(target,residentSplit);
        const int width=windowSweep ? (pass%2 ? 4-w : w+1) : 4;
        const int verifyTokens=continuationTokens/width*width;
        engine.reset_context();auto start=now();
        engine.forward_prompt_tokens(prompt,0,promptCount,0,true,{SnapshotPolicy::None,true});ctx.synchronize();
        double pf=elapsed(start);std::vector<double> times;
        const auto allocations=ctx.device_allocation_count(), pinned=ctx.pinned_host_allocation_count();
        const auto before=Access::Cache(engine).stats();std::vector<uint64_t> hashes;
        std::vector<int> unionCounts,readyCounts,verifyTop1;
        for(int t=0;t<verifyTokens;t+=width) {
            if(windowSweep)Access::Histogram(target);
            start=now();Access::VerifyRows(engine,ids.data_as<int>()+t,width);ctx.synchronize();times.push_back(elapsed(start));
            if(windowSweep) {
                const auto& histogram=Access::Histogram(static_cast<const Qwen4ExpBnb4Backend&>(target));
                int unique=0,ready=0;for(const auto& b:histogram){unique+=int(b[0]);ready+=int(b[2]);}
                unionCounts.push_back(unique);readyCounts.push_back(ready);
            }
            for(int r=0;r<width;++r){verifyTop1.push_back(argmax(Access::Logits(target).data_as<bf16>()+size_t(r)*target.vocab_size()));hashes.push_back(lastLogitsHash);}
        }
        if(expectedVerifyHashes[width-1].empty())expectedVerifyHashes[width-1]=hashes;else expect(hashes==expectedVerifyHashes[width-1],"MTP resident verification changed logits");
        expect(ctx.device_allocation_count()==allocations && ctx.pinned_host_allocation_count()==pinned,"MTP verify live allocation growth");
        const auto after=Access::Cache(engine).stats();
        std::cout<<"[MTP VERIFY] {\"pass\":"<<pass<<",\"width\":"<<width<<",\"resident_split\":"<<(residentSplit?"true":"false")<<",\"prefill_ms\":"<<pf<<",\"tokens\":"<<verifyTokens<<",\"block_ms\":";printInts(times);
        std::cout<<",\"union_counts\":";printInts(unionCounts);std::cout<<",\"ready_counts\":";printInts(readyCounts);
        std::cout<<",\"top1\":";printInts(verifyTop1);
        std::cout<<",\"hashes\":";printInts(hashes);
        std::cout<<",\"misses\":"<<after.miss_count-before.miss_count<<",\"demand_bytes\":"<<after.total_transferred_bytes-before.total_transferred_bytes<<"}\n";
       }
      }
    }
    if(verifyPasses>0) {
        std::vector<int> continuation(tokens.begin()+promptCount,tokens.end());
        if(continuation.size()>=6)draft.CheckTargetRollback(engine,prompt,continuation);
    }
    const int liveWidth = aila::env::read_int_raw("AILA_Q4E_MTP_LIVE_WIDTH", 0);
    const bool liveCompare = aila::env::read_flag("AILA_Q4E_MTP_LIVE_COMPARE_WIDTHS", false);
    if (liveCompare) {
        for (int width : {1, 2, 4}) draft.RunLive(engine, prompt, tokens, maxNew, width, ids, targetHidden, draftHidden);
    } else if (liveWidth) draft.RunLive(engine, prompt, tokens, maxNew, liveWidth, ids, targetHidden, draftHidden);
    std::cout<<"[MTP END] conditional_greedy_probe=1 production_speculative_decode=0 accepted_prefix_commit="<<int(liveWidth>0 || liveCompare)<<"\n";
}
