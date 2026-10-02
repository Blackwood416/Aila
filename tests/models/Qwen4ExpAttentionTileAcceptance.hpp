// Bounded R1b state/quality canary. Numerical tolerances do not relax snapshots.
void test_attention_tile_acceptance(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    aila::ApplyQwen38A770Preset();
    InferenceEngine engine; expect(engine.init(directory, 2560), "R1b init");
    auto& ctx = Access::Ctx(engine); auto& b = Access::Backend(engine);
    auto& session = Access::Session(engine);
    struct Drain { Context& c; ~Drain(){try{c.synchronize();}catch(...){}} } drain{ctx};
    // Reserve the untiled shape once so both arithmetic paths use the same pool.
    Access::AttentionTile(b, 0, 257);
    Tokenizer tokenizer; expect(tokenizer.load(directory), "R1b tokenizer");
    std::ifstream file(aila::env::read_string("AILA_Q4E_WINDOW_PROMPT_FILE", ""));
    expect(bool(file), "R1b prompt");
    std::string body((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto prompt = tokenizer.encode(body); expect(prompt.size() > 257, "R1b source length"); prompt.resize(257);
    auto target = tokenizer.encode("First establish a clear inventory and agree on stable identifiers. Then record the condition of each item and the permissions needed for access.");
    expect(target.size() >= 8, "R1b targets");
    auto ids = Tensor::allocate(ctx, {257}, dnnl::memory::data_type::s32);
    auto upload = [&](const std::vector<int>& v){ctx.memcpy_h2d(ids.data(),v.data(),v.size()*sizeof(int));};
    auto prefill = [&](const std::vector<int>& v, int tile, bool snapshot = false) {
        b.reset(); Access::AttentionTile(b,tile); upload(v);
        if (snapshot) {
            // Interior checkpoints remain limited to <=64 rows in production.
            // Exercise a real reusable prefix at the tiled call boundary instead.
            b.forward(ctx,ids.data_as<int>(),129,{SnapshotPolicy::ReusablePrefix,false,true});
            b.forward(ctx,ids.data_as<int>()+129,128,{SnapshotPolicy::ReusablePrefix,true});
            expect(session.snapshot_for_testing(129), "R1b checkpoint publish");
        } else b.forward(ctx,ids.data_as<int>(),257,{SnapshotPolicy::None,true});
        expect(session.CurrentLength()==257, "R1b window publication");
    };
    std::vector<std::vector<float>> reference;
    CheckpointMetrics metrics;
    for(int tile:{0,128}) {
        prefill(prompt,tile);
        for(int step=0;step<8;++step) {
            auto logits=extract_bf16_to_f32(ctx,Access::Logits(b));
            if(!tile)reference.push_back(logits);else metrics.add(reference[step],logits,target[step]);
            ctx.memcpy_h2d(ids.data(),&target[step],sizeof(int));b.forward(ctx,ids.data_as<int>(),1);
        }
    }
    metrics.report("R1b_257_tile128_teacher8");
    // Early numerical signal only; the normal CLI natural-input run is separate.
    expect(metrics.kl/metrics.rows<=.02 && metrics.max_nrmse<=.15,"R1b bounded quality gate");
    auto kv_prefix = [&] {
        std::vector<bf16> values;
        for(const auto& c:session.layer_caches_for_testing())for(const Tensor* t:{&c.k,&c.v}) {
            if(!t->valid())continue;
            for(int h=0;h<t->shape(0);++h) {
                size_t offset=values.size(), n=129*t->shape(2);values.resize(offset+n);
                ctx.memcpy_d2h(values.data()+offset,t->data_as<bf16>()+h*t->shape(1)*t->shape(2),n*sizeof(bf16));
            }
        }
        return values;
    };
    prefill(prompt,128,true); auto saved=*session.snapshot_for_testing(129);auto kv=kv_prefix();
    auto changed=prompt;for(int i=129;i<257;++i)changed[i]=target[i%target.size()];
    prefill(changed,128,true);
    expect_same_checkpoint(saved,*session.snapshot_for_testing(129));
    expect(same_bits(kv,kv_prefix()),"R1b future suffix changed KV");
    std::vector<float> replay;
    for(int pass=0;pass<2;++pass) {
        expect(b.Truncate(session,129)==129,"R1b truncate");upload(prompt);
        b.forward(ctx,ids.data_as<int>()+129,128,{SnapshotPolicy::None,true});
        auto got=extract_bf16_to_f32(ctx,Access::Logits(b));
        if(!pass)replay=got;else expect(same_bits(replay,got),"R1b replay changed");
    }
    std::cout<<"[R1b STATE] rows=257 tile=128 anchor=129 future_independent=1 replay_exact=1 tail_one_avoided=1\n";

    if (aila::env::read_int("AILA_Q4E_MOE_COMPACT_ACTS",0)>0) {
        // A changed suffix can change expert frequency, grouping and cache order.
        // Compare the visible output of the entire window, not only split calls.
        prefill(prompt,128);
        Tensor prefix=Tensor::view(ctx,Access::FrontHidden(b).data(),{129,10240});
        auto original=extract_bf16_to_f32(ctx,prefix);auto original_kv=kv_prefix();
        prefill(changed,128);
        auto future_changed=extract_bf16_to_f32(ctx,prefix);
        expect(same_bits(original,future_changed),"Compact full-window prefix depends on future routing");
        expect(same_bits(original_kv,kv_prefix()),"Compact full-window KV depends on future routing");
        std::cout<<"[COMPACT CAUSAL] window=257 visible=129 hidden_exact=1 kv_exact=1\n";
    }

    // Isolate one real full-attention layer across the 2052 sparse boundary.
    // Prefix KV/raw indexer K are zeros, incoming x is deterministic BF16.
    const int layer=Access::FullLayer(b), H=2560, count=257, start=1796;
    std::vector<bf16> x(count*H);for(size_t i=0;i<x.size();++i)x[i]=bf16(float(int(i%113)-56)/64.f);
    auto reset_layer = [&] {
        b.reset();auto& c=session.layer_caches_for_testing()[layer];
        for(Tensor* t:{&c.k,&c.v,&c.indexer_k,&c.indexer_compressed_k})if(t->valid())ctx.queue().memset(t->data(),0,t->size_bytes());
        c.num_compressed_blocks=0;ctx.memcpy_h2d(Access::FrontX(b).data(),x.data(),x.size()*sizeof(bf16));
    };
    std::vector<float> full,tiled;
    for(int tile:{0,128}) {
        reset_layer();Access::AttentionTile(b,tile);
        auto& out=Access::AttentionOnly(b,layer,count,start);
        Tensor view=Tensor::view(ctx,out.data(),{count,H});
        auto values=extract_bf16_to_f32(ctx,view);
        if(!tile)full=values;else tiled=values;
    }
    double sq=0,err=0,ma=0;
    for(size_t i=0;i<full.size();++i){expect(std::isfinite(tiled[i])&&std::isfinite(full[i]),"R1b attention finite");double d=tiled[i]-full[i];sq+=double(full[i])*full[i];err+=d*d;ma=std::max(ma,std::abs(d));}
    double nrmse=std::sqrt(err/std::max(sq,1e-30));
    expect(nrmse<=.05,"R1b sparse boundary error");
    for(int row=128;row<count;++row)for(int j=0;j<H;++j)x[row*H+j]=bf16(.1f);
    reset_layer();Access::AttentionTile(b,128);auto& out=Access::AttentionOnly(b,layer,count,start);
    Tensor visible=Tensor::view(ctx,out.data(),{128,H});auto got=extract_bf16_to_f32(ctx,visible);
    expect(std::equal(got.begin(),got.end(),tiled.begin()),"R1b attention sees future tile");
    std::cout<<"[R1b ATTENTION] layer="<<layer<<" start="<<start<<" count="<<count<<" nrmse="<<nrmse<<" max_abs="<<ma<<" bound_nrmse=0.05 future_independent=1\n";
    std::cout<<"[R1b PASS] attention_tiling_state_and_bounded_quality=1\n";
}
