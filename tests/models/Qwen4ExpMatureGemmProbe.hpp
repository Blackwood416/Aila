// R2a: two real NF4 experts, one M=32 bucket, bounded BF16 weight staging.
// A standalone component experiment; no production dispatch or full-model claim.
void test_mature_moe_gemm(const std::string& directory) {
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
    auto gu_w=alloc({E,2*I,H},bf), dn_w=alloc({E,H,I},bf);
    auto gathered=alloc({E,M,H},bf), gu=alloc({E,M,2*I},fp), middle=alloc({E,M,I},bf), result=alloc({E,M,H},fp);
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
    using md = dnnl::memory::desc; using dims = dnnl::memory::dims;
    const md a_md({E,M,H},bf,dnnl::memory::format_tag::abc);
    const md guw_md({E,H,2*I},bf,dims{2*I*H,1,H});
    const md gu_md({E,M,2*I},fp,dnnl::memory::format_tag::abc);
    const md mid_md({E,M,I},bf,dnnl::memory::format_tag::abc);
    const md dnw_md({E,I,H},bf,dims{H*I,1,I});
    const md out_md({E,M,H},fp,dnnl::memory::format_tag::abc);
    dnnl::primitive_attr attr; attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
    const dnnl::matmul::primitive_desc gu_pd(ctx.engine(),a_md,guw_md,gu_md,attr);
    const dnnl::matmul::primitive_desc dn_pd(ctx.engine(),mid_md,dnw_md,out_md,attr);
    const dnnl::matmul gu_mm(gu_pd), dn_mm(dn_pd);
    const size_t scratch_bytes=std::max(gu_pd.scratchpad_desc().get_size(),dn_pd.scratchpad_desc().get_size());
    auto scratch=alloc({int64_t(std::max<size_t>(1,scratch_bytes))},dnnl::memory::data_type::u8);
    auto mem=[&](const md& desc,void* ptr) { return dnnl::sycl_interop::make_memory(desc,ctx.engine(),dnnl::sycl_interop::memory_kind::usm,ptr); };
    std::unordered_map<int,dnnl::memory> gu_args{{DNNL_ARG_SRC,mem(a_md,gathered.data())},{DNNL_ARG_WEIGHTS,mem(guw_md,gu_w.data())},
        {DNNL_ARG_DST,mem(gu_md,gu.data())}};
    std::unordered_map<int,dnnl::memory> dn_args{{DNNL_ARG_SRC,mem(mid_md,middle.data())},{DNNL_ARG_WEIGHTS,mem(dnw_md,dn_w.data())},
        {DNNL_ARG_DST,mem(out_md,result.data())}};
    if(gu_pd.scratchpad_desc().get_size())gu_args.emplace(DNNL_ARG_SCRATCHPAD,mem(gu_pd.scratchpad_desc(),scratch.data()));
    if(dn_pd.scratchpad_desc().get_size())dn_args.emplace(DNNL_ARG_SCRATCHPAD,mem(dn_pd.scratchpad_desc(),scratch.data()));
    struct Drain {Context& c; ~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    auto* sp=sd.data_as<ops::MoeExpertSlotDev>(); auto* tp=td.data_as<int>(); auto* rp=rd.data_as<int>();
    auto* xp=x.data_as<bf16>(); auto* hp=h.data_as<bf16>(); auto* dp=down.data_as<float>();
    auto* gp=gu_w.data_as<bf16>(); auto* wp=dn_w.data_as<bf16>(); auto* qp=qmap.data_as<float>();
    auto* ap=gathered.data_as<bf16>(); auto* up=gu.data_as<float>(); auto* mp=middle.data_as<bf16>(); auto* op=result.data_as<float>();
    auto dequant=[&] {
        ctx.queue().parallel_for(sycl::range<1>(size_t(E)*3*H*I),[=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(16)]] {
            const size_t e=idx[0]/(3*H*I), w=idx[0]%(3*H*I); const bool dn=w>=2*H*I;
            const size_t j=dn?w-2*H*I:w; const auto s=sp[e*M];
            const auto ns=dn?s.down_nested:s.gate_up_nested; const auto* packed=dn?s.down_packed:s.gate_up_packed;
            const float scale=sycl::fma(ns.nested_quant_map[ns.codes[j/64]],ns.nested_absmax[j/64/ns.blocksize],0.f)+ns.offset;
            const int code=(packed[j/2]>>(j%2?0:4))&15;
            (dn?wp+e*H*I:gp+e*2*H*I)[j]=bf16(qp[code]*scale);
        });
    };
    auto gather=[&] {ctx.queue().parallel_for(sycl::range<1>(A*H),[=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(16)]] {const int a=idx[0]/H,k=idx[0]%H;ap[idx]=xp[tp[a]*H+k];});};
    auto activation=[&] {ctx.queue().parallel_for(sycl::range<1>(A*I),[=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(16)]] {const int a=idx[0]/I,k=idx[0]%I;const float g=up[a*2*I+k];mp[idx]=bf16(g/(1.f+sycl::exp(-g))*up[a*2*I+I+k]);});};
    auto scatter=[&] {ctx.queue().parallel_for(sycl::range<1>(A*H),[=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(16)]] {const int a=idx[0]/H,k=idx[0]%H;dp[(tp[a]*R+rp[a])*H+k]=op[idx];});};
    auto run=[&](int mode) {
        if (mode==2) { dequant();gather();gu_mm.execute(ctx.stream(),gu_args);activation();dn_mm.execute(ctx.stream(),dn_args);scatter(); }
        else if (mode==1) {
            auto ev=ops::moe_grouped_xmx_gate_up(ctx,sp,tile.data_as<ops::MoeExpertTile>(),E*M/8,tp,rp,qp,xp,hp,R,H,I,true);
            ops::moe_grouped_xmx_down(ctx,sp,tile.data_as<ops::MoeExpertTile>(),E*M/8,tp,rp,qp,hp,dp,R,H,I,true,&ev);
        } else {
            auto ev=ops::moe_expert_major_prefill_gate_up_4token(ctx,sp,tp,rp,qp,xp,hp,A/4,R,H,I,64,nullptr,true);
            ops::moe_expert_major_prefill_down_4token(ctx,sp,tp,rp,qp,hp,dp,A/4,R,I,H,64,&ev,true);
        }
        ops::moe_ordered_accumulate_down_rows(ctx,dp,route_weights.data_as<float>(),output.data_as<bf16>(),M,R,H);
    };
    std::cout<<"[R2 CONFIG] experts=2 ids=7,113 layer=0 rows_per_expert=32 H=2560 I=640 top_k=10 bf16_weight_bytes="<<gu_w.size_bytes()+dn_w.size_bytes()
        <<" scratch_bytes="<<scratch_bytes<<" device_bytes="<<ctx.current_allocated_bytes()<<" input=synthetic_bf16 weights=real_nf4 h2d_excluded=1\n";
    std::cout<<"[R2 LIBRARY] gate_up="<<gu_pd.impl_info_str()<<" down="<<dn_pd.impl_info_str()<<" src=bf16 weight=bf16 dst=f32 weights_strided=1 fpmath=strict\n";
    std::vector<bf16> baseline(M*H), got(M*H);
    for (int mode:{0,1,2}) {
        run(mode);ctx.synchronize();ctx.memcpy_d2h(got.data(),output.data(),got.size()*2);
        if (mode==0) baseline=got;
        double error_sq=0,ref_sq=0,max_abs=0,max_rel=0;size_t failures=0;
        for (size_t i=0;i<got.size();++i) {
            expect(std::isfinite(float(got[i])),"R2 nonfinite");
            const double ref=float(baseline[i]), d=float(got[i])-ref;
            error_sq+=d*d;ref_sq+=ref*ref;max_abs=std::max(max_abs,std::abs(d));
            max_rel=std::max(max_rel,std::abs(d)/std::max(std::abs(ref),1e-12));
            failures+=std::abs(d)>.02*std::abs(ref)+2e-4;
        }
        const double nrmse=std::sqrt(error_sq/std::max(ref_sq,1e-30));
        std::cout<<"[R2 CHECK] mode="<<mode<<" nrmse_vs_nf4="<<nrmse<<" max_abs="<<max_abs<<" max_rel_err="<<max_rel
            <<" rel_tol=0.02 abs_tol=0.0002 errors="<<failures<<" bound_nrmse=0.02 reference=original_nf4_alu semantics_id=two_expert_gate_swiglu_down_merge accuracy_mode=bf16_staging\n";
        expect(nrmse<=.02 && failures==0,"R2 candidate exceeded operator error budget");
    }
    // Independent f64 reference of the staged BF16 matrices, two gathered rows per expert.
    std::vector<bf16> gu_host(E*2*I*H),dn_host(E*H*I);
    std::vector<float> matmul_host(A*H);
    ctx.memcpy_d2h(gu_host.data(),gu_w.data(),gu_host.size()*2);ctx.memcpy_d2h(dn_host.data(),dn_w.data(),dn_host.size()*2);
    ctx.memcpy_d2h(matmul_host.data(),result.data(),matmul_host.size()*4);
    double cpu_err=0,cpu_ref=0,cpu_abs=0,cpu_rel=0;size_t cpu_failures=0;
    for(int e=0;e<E;++e)for(int m:{0,M-1}) {
        std::vector<bf16> hh(I);
        for(int n=0;n<I;++n) {double g=0,u=0;for(int k=0;k<H;++k){const double x=float(x_host[token[e*M+m]*H+k]);g+=x*float(gu_host[(e*2*I+n)*H+k]);u+=x*float(gu_host[(e*2*I+I+n)*H+k]);}hh[n]=bf16(float(g/(1.+std::exp(-g))*u));}
        for(int n=0;n<H;++n) {
            double y=0;for(int k=0;k<I;++k)y+=double(float(hh[k]))*float(dn_host[(e*H+n)*I+k]);
            const double d=matmul_host[(e*M+m)*H+n]-y;cpu_err+=d*d;cpu_ref+=y*y;cpu_abs=std::max(cpu_abs,std::abs(d));
            cpu_rel=std::max(cpu_rel,std::abs(d)/std::max(std::abs(y),1e-12));
            cpu_failures+=std::abs(d)>1e-4*std::abs(y)+1e-6;
        }
    }
    const double cpu_nrmse=std::sqrt(cpu_err/std::max(cpu_ref,1e-30));
    std::cout<<"[R2 CPU] reference=f64_from_bf16_staging rows=4 nrmse="<<cpu_nrmse<<" max_abs="<<cpu_abs<<" max_rel_err="<<cpu_rel
        <<" rel_tol=0.0001 abs_tol=0.000001 errors="<<cpu_failures<<" bound_nrmse=0.01\n";
    expect(cpu_nrmse<=.01 && cpu_failures==0,"R2 oneDNN CPU reference mismatch");
    std::cout<<"[R2 CONTRACT] accuracy_class=relaxed_matched comparable_for_speedup=true semantics_id=two_expert_gate_swiglu_down_merge accuracy_mode=bf16_staging reference=original_nf4_and_cpu_f64\n";
    if(aila::env::read_flag("AILA_Q4E_MATURE_PROFILE",false)) {
        std::cout<<"[R2 PROFILE PASS] correctness_checked=1 timing_skipped=1\n";
        return;
    }
    const auto live=ctx.device_allocation_count();
    for(int mode:{0,1,2})for(int i=0;i<5;++i){run(mode);ctx.synchronize();}
    for(int round=0;round<3;++round) {
        std::array<std::vector<double>,3> times;
        for(int sample=0;sample<20;++sample)for(int order=0;order<3;++order) {
            const int mode=(round+sample+order)%3;const auto begin=std::chrono::steady_clock::now();run(mode);ctx.synchronize();
            times[mode].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count());
        }
        for(int mode=0;mode<3;++mode){std::sort(times[mode].begin(),times[mode].end());std::cout<<"[R2 TIME] round="<<round<<" mode="<<mode<<" median_us="<<(times[mode][9]+times[mode][10])/2<<" p10_us="<<times[mode][2]<<" p90_us="<<times[mode][18]<<"\n";}
    }
    expect(ctx.device_allocation_count()==live,"R2 hot-path USM grew");
    std::cout<<"[R2 PASS] dequant_gather_gemm_swiglu_scatter_merge_included=1 no_hot_usm_growth=1\n";
}
