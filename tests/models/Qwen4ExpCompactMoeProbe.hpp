// Two real experts; unchanged NF4 arithmetic, compact addressing and Q36 merge.
void test_compact_moe(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    constexpr int E=2, M=32, H=2560, I=640, R=10, A=E*M;
    _putenv_s("AILA_Q4E_HOT_RAM_MB", "0"); // This fixture needs only two resident slots.
    Context ctx;
    auto weights = LoadModelWeightsLazy(directory, ctx);
    Qwen4ExpExpertCache cache;
    std::string error;
    if (!cache.init(ctx, weights, 1, size_t(E)*2534400, 512, H, I, 64, &error))
        throw std::runtime_error("R2a cache: " + error);
    expect(cache.uses_nested_codes(), "R2a requires compact nested NF4");
    const int experts[E] = {7,113}; int slot_ids[E];
    cache.acquire(0, experts, E, slot_ids).wait_and_throw();
    Bnb4BitHostWeightRef source;
    expect(LoadBnb4BitHostRef(weights, "model.language_model.layers.0.mlp.experts.gate_up_proj", source, &error, H), "R2a source");
    auto alloc = [&](std::vector<int64_t> shape, dnnl::memory::data_type dtype) { return Tensor::allocate(ctx, shape, dtype); };
    const auto bf = dnnl::memory::data_type::bf16, fp = dnnl::memory::data_type::f32, i32 = dnnl::memory::data_type::s32;
    auto x=alloc({M,H},bf), h=alloc({M,R,I},bf), down=alloc({M,R,H},fp), output=alloc({M,H},bf);
    auto sd=alloc({int64_t(A*sizeof(ops::MoeExpertSlotDev))},dnnl::memory::data_type::u8);
    auto td=alloc({A},i32), rd=alloc({A},i32), tile=alloc({int64_t(E*M/8*sizeof(ops::MoeExpertTile))},dnnl::memory::data_type::u8);
    auto route_weights=alloc({M,R},fp), qmap=alloc({16},fp);
    std::vector<ops::MoeExpertSlotDev> slots(A);
    std::vector<int> token(A), rank(A);
    std::vector<ops::MoeExpertTile> tiles(E*M/8);
    std::vector<float> rw(M*R,0.f);
    for (int e=0;e<E;++e) {
        for (int m=0;m<M;++m) { slots[e*M+m]=cache.get_slot_dev(slot_ids[e]); token[e*M+m]=(m*7)%M; rank[e*M+m]=e; rw[token[e*M+m]*R+e]=.5f; }
        for (int m=0;m<M;m+=8) tiles[e*M/8+m/8]={e*M+m,8};
    }
    std::mt19937 rng(61003); std::normal_distribution<float> normal(0.f,.25f);
    std::vector<bf16> x_host(M*H); for (auto& v:x_host) v=bf16(normal(rng));
    ctx.memcpy_h2d(x.data(),x_host.data(),x_host.size()*2);
    ctx.memcpy_h2d(sd.data(),slots.data(),slots.size()*sizeof(slots[0]));
    ctx.memcpy_h2d(td.data(),token.data(),A*4); ctx.memcpy_h2d(rd.data(),rank.data(),A*4);
    ctx.memcpy_h2d(tile.data(),tiles.data(),tiles.size()*sizeof(tiles[0]));
    ctx.memcpy_h2d(route_weights.data(),rw.data(),rw.size()*4);
    ctx.memcpy_h2d(qmap.data(),source.quant_map.data(),16*4);
    ctx.queue().memset(down.data(),0,down.size_bytes()).wait_and_throw();
    auto compact=alloc({A,H},fp), inverse=alloc({M,R},i32);
    auto sum=alloc({int64_t(M)*H*8},dnnl::memory::data_type::u8), invalid=alloc({1},i32);
    struct Drain {Context& c;~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    auto* sp=sd.data_as<ops::MoeExpertSlotDev>();auto* tp=td.data_as<int>();auto* rp=rd.data_as<int>();
    ctx.queue().memset(inverse.data(),255,inverse.size_bytes());
    ops::moe_build_inverse_map(ctx,tp,rp,inverse.data_as<int>(),A,R);
    auto gate=[&]{return ops::moe_expert_major_prefill_gate_up(ctx,sp,tp,rp,qmap.data_as<float>(),
        x.data_as<bf16>(),h.data_as<bf16>(),A,R,H,I,64,nullptr,true,0,A/4);};
    auto run=[&](bool bounded,bool reverse=false) {
        auto ev=gate();
        if(!bounded){
            ev=ops::moe_expert_major_prefill_down(ctx,sp,tp,rp,qmap.data_as<float>(),h.data_as<bf16>(),
                down.data_as<float>(),A,R,I,H,64,&ev,true,0,A/4);
            ops::moe_ordered_accumulate_down_rows(ctx,down.data_as<float>(),route_weights.data_as<float>(),
                output.data_as<bf16>(),M,R,H,nullptr,nullptr,&ev);
        }else{
            ctx.queue().memset(sum.data(),0,sum.size_bytes());ctx.queue().memset(invalid.data(),0,sizeof(int));
            // Four disjoint 16-act tiles, also tested in reverse submission order.
            for(int i=0;i<4;++i){int first=(reverse?3-i:i)*16;
                ev=ops::moe_expert_major_prefill_down(ctx,sp+first,tp+first,rp+first,qmap.data_as<float>(),
                    h.data_as<bf16>(),compact.data_as<float>(),16,R,I,H,64,&ev,true,0,4,true);
                ev=ops::moe_merge_compact_down(ctx,compact.data_as<float>(),tp,inverse.data_as<int>(),
                    route_weights.data_as<float>(),sum.data_as<int64_t>(),invalid.data_as<int>(),first,16,R,H,&ev);
            }
            ops::moe_finish_compact_down(ctx,sum.data_as<int64_t>(),output.data_as<bf16>(),M,H);
        }
        ctx.synchronize();
    };
    run(false);
    std::vector<float> raw(M*R*H);ctx.memcpy_d2h(raw.data(),down.data(),raw.size()*4);
    std::vector<bf16> baseline(M*H),got(M*H),reversed(M*H);
    ctx.memcpy_d2h(baseline.data(),output.data(),baseline.size()*2);
    // Mixed quad/pair/single dispatch, nonzero map offset and untouched guards.
    for(int first:{0,5})for(int n:{1,3,11,27}){
        ctx.queue().memset(compact.data(),0x5a,compact.size_bytes());
        const int nq=n/4,np=(n%4)/2;
        ops::moe_expert_major_prefill_down(ctx,sp+first,tp+first,rp+first,qmap.data_as<float>(),
            h.data_as<bf16>(),compact.data_as<float>(),n,R,I,H,64,nullptr,true,np,nq,true);
        std::vector<float> v(A*H);ctx.memcpy_d2h(v.data(),compact.data(),v.size()*4);
        for(int i=0;i<n;++i)for(int j=0;j<H;++j)
            expect(v[i*H+j]==raw[(token[first+i]*R+rank[first+i])*H+j],"Compact down address/arithmetic mismatch");
        const auto* bytes=reinterpret_cast<const unsigned char*>(v.data());
        for(size_t i=size_t(n)*H*4;i<v.size()*4;++i)expect(bytes[i]==0x5a,"Compact down overrun");
    }
    run(true);ctx.memcpy_d2h(got.data(),output.data(),got.size()*2);
    std::vector<int64_t> qsum(M*H);ctx.memcpy_d2h(qsum.data(),sum.data(),qsum.size()*8);
    size_t errors=0;double err=0,sq=0,ma=0;
    for(int t=0;t<M;++t)for(int j=0;j<H;++j){
        int64_t exact=0;for(int k=0;k<R;++k)exact+=int64_t(std::nearbyint(double(rw[t*R+k]*raw[(t*R+k)*H+j])*68719476736.));
        expect(qsum[t*H+j]==exact,"Q36 CPU integer reference mismatch");
        double ref=float(baseline[t*H+j]),d=float(got[t*H+j])-ref;
        expect(std::isfinite(float(got[t*H+j])),"Compact finite");
        errors+=std::abs(d)>2e-4+.02*std::abs(ref);err+=d*d;sq+=ref*ref;ma=std::max(ma,std::abs(d));
    }
    run(true,true);ctx.memcpy_d2h(reversed.data(),output.data(),reversed.size()*2);
    expect(same_bits(got,reversed),"Compact depends on tile submission order");
    int bad=0;ctx.memcpy_d2h(&bad,invalid.data(),sizeof(bad));expect(!bad,"Unexpected Q36 range failure");
    expect(errors==0 && std::sqrt(err/std::max(sq,1e-30))<=.02,"Compact merge tolerance");
    std::cout<<"[COMPACT CHECK] raw_errors=0 cpu_integer_errors=0 order_independent=1 errors="<<errors
        <<" nrmse="<<std::sqrt(err/std::max(sq,1e-30))<<" max_abs="<<ma<<" rel_tol=0.02 abs_tol=0.0002\n";
    // Guard a deliberately out-of-range contribution before floating-to-int conversion.
    float one=1.f;ctx.memcpy_h2d(route_weights.data(),&one,4);
    for(uint32_t bits:{0x4a000000U,0x7fc00000U,0x7f800000U}) {
        ctx.memcpy_h2d(compact.data(),&bits,4);ctx.queue().memset(invalid.data(),0,4);
        ops::moe_merge_compact_down(ctx,compact.data_as<float>(),tp,inverse.data_as<int>(),route_weights.data_as<float>(),
            sum.data_as<int64_t>(),invalid.data_as<int>(),0,1,R,H);
        ctx.memcpy_d2h(&bad,invalid.data(),4);expect(bad==1,"Q36 range/nonfinite guard missing");
    }
    std::cout<<"[COMPACT GUARD] cases=3 errors=0 finite_range_nan_inf=1\n";
    ctx.memcpy_h2d(route_weights.data(),rw.data(),rw.size()*4);
    if(aila::env::read_flag("AILA_Q4E_MATURE_PROFILE",false)){
        std::cout<<"[COMPACT PASS] profile_only=1\n";return;
    }
    const auto live=ctx.device_allocation_count();
    for(int i=0;i<5;++i){run(false);run(true);}
    for(int round=0;round<3;++round){std::array<std::vector<double>,2> times;
        for(int i=0;i<20;++i)for(int order=0;order<2;++order){int mode=(i+round+order)%2;
            auto start=std::chrono::steady_clock::now();run(mode!=0);
            times[mode].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());}
        for(int mode=0;mode<2;++mode){std::sort(times[mode].begin(),times[mode].end());std::cout<<"[COMPACT TIME] round="<<round<<" mode="<<mode<<" median_us="<<(times[mode][9]+times[mode][10])/2<<" p10_us="<<times[mode][2]<<" p90_us="<<times[mode][18]<<"\n";}
    }
    expect(live==ctx.device_allocation_count(),"Compact hot allocation");
    std::cout<<"[COMPACT PASS] no_hot_usm_growth=1\n";
}
