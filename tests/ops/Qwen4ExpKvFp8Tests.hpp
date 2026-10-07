// Included inside the kernel test executable's anonymous namespace.
void test_q4e_kv_fp8(Context& ctx) {
    // Independent standard E4M3FN table; no production codec in the reference.
    std::vector<float> values(127);
    for (int b=0;b<127;++b) values[b]=(b<8) ? std::ldexp(float(b),-9)
        : std::ldexp(1.0f+float(b&7)/8.0f,(b>>3)-7);
    auto encode=[&](float x) -> uint8_t {
        const uint32_t bits=sycl::bit_cast<uint32_t>(x), mag=bits&0x7fffffff;
        if(mag>0x7f800000) return 0x7f;
        const uint8_t sign=uint8_t((bits>>24)&0x80);
        const float a=sycl::bit_cast<float>(mag);
        auto hi=std::lower_bound(values.begin(),values.end(),a);
        if(hi==values.end()) return sign|126;
        int b=int(hi-values.begin());
        if(b && (a-values[b-1]<values[b]-a ||
            (a-values[b-1]==values[b]-a && ((b-1)&1)==0))) --b;
        return uint8_t(sign|b);
    };
    for(int b=0;b<256;++b) {
        const float actual=fp8_e4m3fn_to_float(uint8_t(b));
        if((b&127)==127) {
            if((sycl::bit_cast<uint32_t>(actual)&0x7fffffff)<=0x7f800000) throw std::runtime_error("FP8 NaN decode");
        } else {
            const float expected=(b&128) ? -values[b&127] : values[b];
            if(sycl::bit_cast<uint32_t>(actual)!=sycl::bit_cast<uint32_t>(expected) ||
               float_to_fp8_e4m3fn(expected)!=b) throw std::runtime_error("FP8 standard table/roundtrip");
        }
    }
    for(int b=0;b<126;++b) {
        const float middle=(values[b]+values[b+1])*0.5f;
        for(float x:{std::nextafter(middle,0.0f),middle,std::nextafter(middle,1000.0f)})
            for(float sign:{1.0f,-1.0f}) if(float_to_fp8_e4m3fn(sign*x)!=encode(sign*x))
                throw std::runtime_error("FP8 midpoint rounding");
    }
    std::vector<float> inputs(65536),decoded(65536);
    std::vector<uint8_t> encoded(65536);
    for(uint32_t b=0;b<65536;++b) inputs[b]=sycl::bit_cast<float>(b<<16);
    auto input=Tensor::allocate(ctx,{65536},dnnl::memory::data_type::f32);
    auto codes=Tensor::allocate(ctx,{65536},dnnl::memory::data_type::u8);
    auto roundtrip=Tensor::allocate(ctx,{65536},dnnl::memory::data_type::f32);
    constexpr int cap=2048, len=1834, hq=24, hkv=2, dim=256, qdim=hq*dim;
    auto src=Tensor::allocate(ctx,{2,len,hkv*dim},dnnl::memory::data_type::bf16);
    auto k8=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::f8_e4m3);
    auto v8=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::f8_e4m3);
    auto kr=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
    auto vr=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
    auto q=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
    auto g=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
    auto out=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
    auto ref=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
    auto scores=Tensor::allocate(ctx,{hq,cap},dnnl::memory::data_type::f32);
    auto partials=Tensor::allocate(ctx,{hq,cap/128,272},dnnl::memory::data_type::f32);
    struct Drain {Context& c;~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    ctx.memcpy_h2d(input.data(),inputs.data(),inputs.size()*4);
    auto* ip=input.data_as<float>();auto* cp=codes.data_as<uint8_t>();auto* rp=roundtrip.data_as<float>();
    ctx.queue().parallel_for(sycl::nd_range<1>(65536,256),[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int i=item.get_global_id(0); cp[i]=float_to_fp8_e4m3fn(ip[i]);rp[i]=fp8_e4m3fn_to_float(cp[i]);
    });
    ctx.memcpy_d2h(encoded.data(),codes.data(),encoded.size());
    ctx.memcpy_d2h(decoded.data(),roundtrip.data(),decoded.size()*4);
    for(size_t i=0;i<inputs.size();++i) {
        const auto expected=encode(inputs[i]);
        if(encoded[i]!=expected || float_to_fp8_e4m3fn(inputs[i])!=expected) throw std::runtime_error("FP8 BF16-domain encode at "+std::to_string(i));
        if((expected&127)!=127) {
            const float value=(expected&128) ? -values[expected&127] : values[expected];
            if(sycl::bit_cast<uint32_t>(decoded[i])!=sycl::bit_cast<uint32_t>(value)) throw std::runtime_error("FP8 GPU decode");
        }
    }
    std::cout<<"[FP8 CODEC PASS] standard E4M3FN, 256 codes, midpoint neighbors, 65536 BF16 bit patterns CPU/GPU\n";
    std::mt19937 rng(100601);std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    std::vector<bf16> host(2*len*hkv*dim),qh(2*qdim),gh(2*qdim);
    for(auto& x:host)x=bf16(dist(rng));for(auto& x:qh)x=bf16(dist(rng));for(auto& x:gh)x=bf16(dist(rng));
    ctx.memcpy_h2d(src.data(),host.data(),host.size()*2);ctx.memcpy_h2d(q.data(),qh.data(),qh.size()*2);ctx.memcpy_h2d(g.data(),gh.data(),gh.size()*2);
    ctx.queue().memset(k8.data(),0xa5,k8.size_bytes());ctx.queue().memset(v8.data(),0xa5,v8.size_bytes());
    const auto live=ctx.device_allocation_count(), pinned=ctx.pinned_host_allocation_count();
    for(int side=0;side<2;++side) {
        auto& cache=side ? v8:k8; auto& expanded=side ? vr:kr;
        auto prefix=Tensor::view(ctx,src.data_as<bf16>()+side*len*hkv*dim,{len-2,hkv*dim});
        auto append=Tensor::view(ctx,src.data_as<bf16>()+(side*len+len-2)*hkv*dim,{2,hkv*dim});
        ops::copy_to_cache(ctx,prefix,cache,len-2,0,hkv,dim,cap);
        ops::copy_to_cache(ctx,append,cache,2,len-2,hkv,dim,cap);
        std::vector<uint8_t> packed(cache.size_bytes());ctx.memcpy_d2h(packed.data(),cache.data(),packed.size());
        std::vector<bf16> deq(hkv*cap*dim,bf16(0));
        for(int h=0;h<hkv;++h)for(int t=0;t<cap;++t)for(int d=0;d<dim;++d) {
            const int offset=(h*cap+t)*dim+d;
            const uint8_t expected=t<len ? encode(float(host[((side*len+t)*hkv+h)*dim+d])) : 0xa5;
            if(packed[offset]!=expected)throw std::runtime_error("FP8 append/layout/canary");
            if(t<len)deq[offset]=bf16((expected&128) ? -values[expected&127] : values[expected]);
        }
        ctx.memcpy_h2d(expanded.data(),deq.data(),deq.size()*2);
    }
    auto run=[&](bool quantized) {
        for(int row=0;row<2;++row) {
            auto qr=Tensor::view(ctx,q.data_as<bf16>()+row*qdim,{1,qdim});
            auto gr=Tensor::view(ctx,g.data_as<bf16>()+row*qdim,{1,qdim});
            auto o=Tensor::view(ctx,(quantized?out:ref).data_as<bf16>()+row*qdim,{1,qdim});
            ops::attention_decode(ctx,qr,quantized?k8:kr,quantized?v8:vr,o,scores,hq,hkv,dim,len-1+row,&partials,&gr);
        }
    };
    run(false);run(true);
    std::vector<bf16> actual(2*qdim),expected(2*qdim);
    ctx.memcpy_d2h(actual.data(),out.data(),actual.size()*2);ctx.memcpy_d2h(expected.data(),ref.data(),expected.size()*2);
    float maxdiff=0;int different=0;
    for(size_t i=0;i<actual.size();++i) {
        float diff=std::abs(float(actual[i])-float(expected[i]));maxdiff=std::max(maxdiff,diff);different+=diff!=0;
        if(!(diff<=1e-3f))throw std::runtime_error("FP8 attention vs expanded-cache reference");
    }
    std::vector<double> times[2];
    for(int sample=0;sample<23;++sample)for(int j=0;j<2;++j) {
        int mode=(j+sample)%2;auto start=std::chrono::steady_clock::now();
        run(mode==1);ctx.synchronize();
        if(sample>=3)times[mode].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    for(int mode=0;mode<2;++mode){std::sort(times[mode].begin(),times[mode].end());
        std::cout<<"[KV ATTENTION WALL] format="<<(mode?"fp8":"bf16-expanded")<<" B=2 Hq=24 Hkv=2 D=256 lengths=1833,1834 samples=20 median_ms="
            <<(times[mode][9]+times[mode][10])*0.5<<" p10="<<times[mode][2]<<" p90="<<times[mode][17]<<'\n';}
    if(ctx.device_allocation_count()!=live || ctx.pinned_host_allocation_count()!=pinned)throw std::runtime_error("FP8 KV hot USM allocation");
    std::cout<<"[KV FP8 PASS] append bytes/canaries exact, expanded-reference max_abs="<<maxdiff<<" different="<<different<<" live_usm_delta=0\n";
}
