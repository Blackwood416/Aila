void test_q4e_attention_pair(Context& ctx) {
    constexpr int cap=2048,heads=24,dim=256,qd=heads*dim,needed=2*heads*(cap/128)*272;
    auto q=Tensor::allocate(ctx,{2,qd},dnnl::memory::data_type::bf16);
    auto g=Tensor::allocate(ctx,{2,qd},dnnl::memory::data_type::bf16);
    auto k=Tensor::allocate(ctx,{2,cap,dim},dnnl::memory::data_type::bf16);
    auto v=Tensor::allocate(ctx,{2,cap,dim},dnnl::memory::data_type::bf16);
    auto out=Tensor::allocate(ctx,{2*qd+16},dnnl::memory::data_type::bf16);
    auto ref=Tensor::allocate(ctx,{2*qd+16},dnnl::memory::data_type::bf16);
    auto scratch=Tensor::allocate(ctx,{needed+32},dnnl::memory::data_type::f32);
    auto partial=Tensor::allocate(ctx,{heads,cap/128,272},dnnl::memory::data_type::f32);
    auto scores=Tensor::allocate(ctx,{heads,cap},dnnl::memory::data_type::f32);
    struct Drain{Context& c;~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    std::mt19937 rng(61006);std::uniform_real_distribution<float> dist(-2.0f,2.0f);
    for(auto* tensor:{&q,&g,&k,&v}) {
        std::vector<bf16> host(tensor->numel());for(auto& x:host)x=bf16(dist(rng));
        ctx.memcpy_h2d(tensor->data(),host.data(),host.size()*2);
    }
    ctx.queue().memset(out.data(),0x4a,out.size_bytes());ctx.queue().memset(ref.data(),0x4a,ref.size_bytes());
    ctx.queue().memset(scratch.data(),0x4a,scratch.size_bytes());
    const auto live=ctx.device_allocation_count(),pinned=ctx.pinned_host_allocation_count();
    auto run=[&](bool pair,int first){
        if(pair){ops::q4e_attention_decode_pair(ctx,q,k,v,out,scratch,first,g);return;}
        for(int row=0;row<2;++row){
            auto qr=Tensor::view(ctx,q.data_as<bf16>()+row*qd,{1,qd});
            auto gr=Tensor::view(ctx,g.data_as<bf16>()+row*qd,{1,qd});
            auto o=Tensor::view(ctx,ref.data_as<bf16>()+row*qd,{1,qd});
            ops::attention_decode(ctx,qr,k,v,o,scores,heads,2,dim,first+row,&partial,&gr);
        }
    };
    for(int first:{1024,1025,1792,1833,2047}) {
        run(false,first);run(true,first);
        std::vector<bf16> a(out.numel()),b(ref.numel());
        ctx.memcpy_d2h(a.data(),out.data(),out.size_bytes());ctx.memcpy_d2h(b.data(),ref.data(),ref.size_bytes());
        if(std::memcmp(a.data(),b.data(),out.size_bytes())) {
            int n=0;float diff=0;for(int i=0;i<2*qd;++i){n+=std::memcmp(&a[i],&b[i],2)!=0;diff=std::max(diff,std::abs(float(a[i])-float(b[i])));}
            std::cout<<"[PAIR FAIL] first_len="<<first<<" different="<<n<<" max_abs="<<diff<<'\n';
            throw std::runtime_error("Attention pair must match two B1 calls bitwise");
        }
        for(int i=2*qd;i<2*qd+16;++i)if(sycl::bit_cast<uint16_t>(a[i])!=0x4a4a)throw std::runtime_error("Pair output canary");
    }
    uint32_t guard[32];ctx.memcpy_d2h(guard,scratch.data_as<float>()+needed,sizeof(guard));
    for(auto x:guard)if(x!=0x4a4a4a4a)throw std::runtime_error("Pair partial canary");
    std::vector<double> times[2];
    for(int sample=0;sample<23;++sample)for(int j=0;j<2;++j){
        int mode=(j+sample)%2;auto start=std::chrono::steady_clock::now();run(mode==1,1833);ctx.synchronize();
        if(sample>=3)times[mode].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    for(int mode=0;mode<2;++mode){std::sort(times[mode].begin(),times[mode].end());
        std::cout<<"[PAIR WALL] pair="<<mode<<" median_ms="<<(times[mode][9]+times[mode][10])*0.5<<" p10="<<times[mode][2]<<" p90="<<times[mode][17]<<'\n';}
    if(ctx.device_allocation_count()!=live || ctx.pinned_host_allocation_count()!=pinned)throw std::runtime_error("Pair hot USM");
    std::cout<<"[PAIR PASS] 5 causal lengths, fused gate bitwise vs B1, output/partial canaries, live_usm_delta=0\n";
}
