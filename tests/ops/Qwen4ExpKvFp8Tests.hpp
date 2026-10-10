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

    // ================================================================
    // S0 regression: the head_dim==128 fast path of the decode-prepare
    // kernels used to encode only the first vec4 half of K and V. The bf16
    // cache holds the exact pre-quantization values the fp8 path sees, so the
    // fp8 bytes must equal the standard-table encode of the bf16 cache for all
    // 128 dimensions. start_pos=0 makes RoPE the identity.
    // ================================================================
    {
        constexpr int s_hq=4,s_hkv=2,s_dim=128,s_cap=16;
        auto qn=Tensor::allocate(ctx,{s_dim},dnnl::memory::data_type::bf16);
        auto kn=Tensor::allocate(ctx,{s_dim},dnnl::memory::data_type::bf16);
        auto rf=Tensor::allocate(ctx,{s_dim},dnnl::memory::data_type::f32);
        std::vector<bf16> ones(s_dim,bf16(1.0f));std::vector<float> freq(s_dim,1.0f);
        ctx.memcpy_h2d(qn.data(),ones.data(),s_dim*2);
        ctx.memcpy_h2d(kn.data(),ones.data(),s_dim*2);
        ctx.memcpy_h2d(rf.data(),freq.data(),s_dim*4);
        std::mt19937 rng0(4242);std::uniform_real_distribution<float> d0(-1.0f,1.0f);
        std::vector<bf16> s_q(s_hq*s_dim),s_k(s_hkv*s_dim),s_v(s_hkv*s_dim);
        for(auto& x:s_q)x=bf16(d0(rng0));for(auto& x:s_k)x=bf16(d0(rng0));for(auto& x:s_v)x=bf16(d0(rng0));
        auto check=[&](bool partial) {
            auto q1=Tensor::allocate(ctx,{1,s_hq*s_dim},dnnl::memory::data_type::bf16);
            auto k1=Tensor::allocate(ctx,{1,s_hkv*s_dim},dnnl::memory::data_type::bf16);
            auto v1=Tensor::allocate(ctx,{1,s_hkv*s_dim},dnnl::memory::data_type::bf16);
            auto q2=Tensor::allocate(ctx,{1,s_hq*s_dim},dnnl::memory::data_type::bf16);
            auto k2=Tensor::allocate(ctx,{1,s_hkv*s_dim},dnnl::memory::data_type::bf16);
            auto v2=Tensor::allocate(ctx,{1,s_hkv*s_dim},dnnl::memory::data_type::bf16);
            for(Tensor* t:{&q1,&q2})ctx.memcpy_h2d(t->data(),s_q.data(),s_q.size()*2);
            for(Tensor* t:{&k1,&k2})ctx.memcpy_h2d(t->data(),s_k.data(),s_k.size()*2);
            for(Tensor* t:{&v1,&v2})ctx.memcpy_h2d(t->data(),s_v.data(),s_v.size()*2);
            auto kb=Tensor::allocate(ctx,{s_hkv,s_cap,s_dim},dnnl::memory::data_type::bf16);
            auto vb=Tensor::allocate(ctx,{s_hkv,s_cap,s_dim},dnnl::memory::data_type::bf16);
            auto kf=Tensor::allocate(ctx,{s_hkv,s_cap,s_dim},dnnl::memory::data_type::f8_e4m3);
            auto vf=Tensor::allocate(ctx,{s_hkv,s_cap,s_dim},dnnl::memory::data_type::f8_e4m3);
            // Dirty the fp8 cache first: the allocator hands back recycled
            // memory, so without this an unwritten second half could still hold
            // a previous run's correct codes and hide the S0 defect.
            ctx.queue().memset(kf.data(),0xa5,kf.size_bytes());
            ctx.queue().memset(vf.data(),0xa5,vf.size_bytes());
            if(partial) {
                ops::decode_prepare_qkv_partial(ctx,q1,k1,v1,qn,kn,kb,vb,0,s_hq,s_hkv,s_dim,1e-6f,s_dim,10000.0f,false,nullptr,nullptr,nullptr,0,0,0,0,0);
                ops::decode_prepare_qkv_partial(ctx,q2,k2,v2,qn,kn,kf,vf,0,s_hq,s_hkv,s_dim,1e-6f,s_dim,10000.0f,false,nullptr,nullptr,nullptr,0,0,0,0,0);
            } else {
                ops::decode_prepare_qkv(ctx,q1,k1,v1,rf,qn,kn,kb,vb,0,s_hq,s_hkv,s_dim,1e-6f,10000.0f);
                ops::decode_prepare_qkv(ctx,q2,k2,v2,rf,qn,kn,kf,vf,0,s_hq,s_hkv,s_dim,1e-6f,10000.0f);
            }
            const size_t n=(size_t)s_hkv*s_cap*s_dim;
            std::vector<bf16> hkb(n),hvb(n);
            ctx.memcpy_d2h(hkb.data(),kb.data(),n*2);
            ctx.memcpy_d2h(hvb.data(),vb.data(),n*2);
            std::vector<uint8_t> ckf(n),cvf(n);
            ctx.memcpy_d2h(ckf.data(),kf.data(),n);
            ctx.memcpy_d2h(cvf.data(),vf.data(),n);
            int mism=0;
            for(int h=0;h<s_hkv;++h)for(int d=0;d<s_dim;++d) {
                const int off=(h*s_cap+0)*s_dim+d;
                const uint8_t ek=encode(float(hkb[off])),ev2=encode(float(hvb[off]));
                if(ckf[off]!=ek||cvf[off]!=ev2)++mism;
            }
            if(mism)throw std::runtime_error(std::string("S0 head_dim=128 fast path (")+
                (partial?"qkv_partial":"qkv")+"): "+std::to_string(mism)+" K/V codes differ from the bf16-domain encode");
        };
        check(false);check(true);
        std::cout<<"[S0 PASS] decode_prepare head_dim=128 fast path encodes all 128 dims (qkv + qkv_partial, canary-filled fp8 cache, bf16-vs-fp8 code equality)\n";
    }

    // ================================================================
    // Three-arm KV microtest (bf16 true / fp8 / int8), production geometry.
    // ================================================================
    std::cout<<"\n[KV THREE-ARM] Hq=24 Hkv=2 D=256; arms=bf16-true,fp8,int8\n";
    auto run_arm=[&](int cap,int len) {
        constexpr int hq=24,hkv=2,dim=256,qdim=hq*dim;
        const int ntiles=(cap+127)/128;
        auto src=Tensor::allocate(ctx,{2,len,hkv*dim},dnnl::memory::data_type::bf16);
        auto q=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto g=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto scores=Tensor::allocate(ctx,{hq,cap},dnnl::memory::data_type::f32);
        auto partials=Tensor::allocate(ctx,{hq,ntiles,272},dnnl::memory::data_type::f32);
        auto k_bf=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
        auto v_bf=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
        auto k_f8=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::f8_e4m3);
        auto v_f8=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::f8_e4m3);
        auto k_i8=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::s8);
        auto v_i8=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::s8);
        auto ks=Tensor::allocate(ctx,{hkv,cap},dnnl::memory::data_type::f32);
        auto vs=Tensor::allocate(ctx,{hkv,cap},dnnl::memory::data_type::f32);
        auto k_f8e=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
        auto v_f8e=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
        auto k_i8e=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
        auto v_i8e=Tensor::allocate(ctx,{hkv,cap,dim},dnnl::memory::data_type::bf16);
        auto o_true=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto o_f8e=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto o_f8=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto o_i8e=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto o_i8=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto o_scratch=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto q2=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto pc_scores=Tensor::allocate(ctx,{hq,2,len},dnnl::memory::data_type::f32);
        auto p_true=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto p_f8e=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto p_f8=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto p_i8e=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);
        auto p_i8=Tensor::allocate(ctx,{2,qdim},dnnl::memory::data_type::bf16);

        std::mt19937 rng(100601+(uint32_t)len);std::uniform_real_distribution<float> dist(-1.0f,1.0f);
        std::vector<bf16> host(2*len*hkv*dim),qh(2*qdim),gh(2*qdim);
        for(auto& x:host)x=bf16(dist(rng));for(auto& x:qh)x=bf16(dist(rng));for(auto& x:gh)x=bf16(dist(rng));
        ctx.memcpy_h2d(src.data(),host.data(),host.size()*2);
        ctx.memcpy_h2d(q.data(),qh.data(),qh.size()*2);
        ctx.memcpy_h2d(q2.data(),qh.data(),qh.size()*2);
        ctx.memcpy_h2d(g.data(),gh.data(),gh.size()*2);
        for(Tensor* t:{&k_f8,&v_f8,&k_i8,&v_i8})
            ctx.queue().memset(t->data(),0xa5,t->size_bytes());

        const auto live=ctx.device_allocation_count(),pinned=ctx.pinned_host_allocation_count();

        // Write all four caches through the production write ops (two chunks).
        auto write=[&](Tensor& kc,Tensor& vc,Tensor* ksp,Tensor* vsp) {
            for(int side=0;side<2;++side) {
                auto& cache=side?vc:kc;
                Tensor* sp=side?vsp:ksp;
                auto prefix=Tensor::view(ctx,src.data_as<bf16>()+side*len*hkv*dim,{len-2,hkv*dim});
                auto append=Tensor::view(ctx,src.data_as<bf16>()+(side*len+len-2)*hkv*dim,{2,hkv*dim});
                ops::copy_to_cache(ctx,prefix,cache,len-2,0,hkv,dim,cap,sp);
                ops::copy_to_cache(ctx,append,cache,2,len-2,hkv,dim,cap,sp);
            }
        };
        write(k_bf,v_bf,nullptr,nullptr);
        write(k_f8,v_f8,nullptr,nullptr);
        write(k_i8,v_i8,&ks,&vs);

        // Canary + int8 code/scale replication against the host reference.
        std::vector<uint8_t> pf8(k_f8.size_bytes()),vf8(v_f8.size_bytes());
        std::vector<uint8_t> pi8(k_i8.size_bytes()),vi8(v_i8.size_bytes());
        ctx.memcpy_d2h(pf8.data(),k_f8.data(),pf8.size());ctx.memcpy_d2h(vf8.data(),v_f8.data(),vf8.size());
        ctx.memcpy_d2h(pi8.data(),k_i8.data(),pi8.size());ctx.memcpy_d2h(vi8.data(),v_i8.data(),vi8.size());
        std::vector<float> hks(hkv*cap,0.0f),hvs(hkv*cap,0.0f);
        ctx.memcpy_d2h(hks.data(),ks.data(),hks.size()*4);
        ctx.memcpy_d2h(hvs.data(),vs.data(),hvs.size()*4);
        int bad_code=0,exact_code_mismatch=0;float worst_scale_rel=0.0f;
        for(int h=0;h<hkv;++h)for(int t=0;t<cap;++t) {
            for(int d=0;d<dim;++d) {
                const int off=(h*cap+t)*dim+d;
                if(t>=len) {
                    if(pf8[off]!=0xa5||vf8[off]!=0xa5||pi8[off]!=(uint8_t)0xa5||vi8[off]!=(uint8_t)0xa5)
                        throw std::runtime_error("KV append/canary: unwritten cache byte modified");
                    continue;
                }
                const float kv=float(host[((0*len+t)*hkv+h)*dim+d]);
                const float vv=float(host[((1*len+t)*hkv+h)*dim+d]);
                if(pf8[off]!=encode(kv)||vf8[off]!=encode(vv))
                    throw std::runtime_error("FP8 append/layout/canary");
            }
        }
        // Dedicated per-row amax replication for the int8 codes.
        for(int h=0;h<hkv;++h)for(int t=0;t<len;++t) {
            float kamax=0.0f,vamax=0.0f;
            for(int d=0;d<dim;++d) {
                kamax=std::max(kamax,std::abs(float(host[((0*len+t)*hkv+h)*dim+d])));
                vamax=std::max(vamax,std::abs(float(host[((1*len+t)*hkv+h)*dim+d])));
            }
            const float ksc=kamax/127.0f,vsc=vamax/127.0f;
            const float rs=std::max(std::abs(ksc-hks[h*cap+t])/(std::abs(ksc)+1e-30f),
                                    std::abs(vsc-hvs[h*cap+t])/(std::abs(vsc)+1e-30f));
            worst_scale_rel=std::max(worst_scale_rel,rs);
            for(int d=0;d<dim;++d) {
                const int off=(h*cap+t)*dim+d;
                const auto qk=[&](float v,float s){ return s>0.0f?int(std::rint(std::min(std::max(v/s,-127.0f),127.0f))):0; };
                const int ek=qk(float(host[((0*len+t)*hkv+h)*dim+d]),ksc);
                const int ev=qk(float(host[((1*len+t)*hkv+h)*dim+d]),vsc);
                const int dk=std::abs(int((int8_t)pi8[off])-ek);
                const int dv=std::abs(int((int8_t)vi8[off])-ev);
                if(dk>1||dv>1)++bad_code;
                else if(dk||dv)++exact_code_mismatch;
            }
        }
        if(bad_code)throw std::runtime_error("INT8 code replication: "+std::to_string(bad_code)+" codes differ by >1 LSB");
        if(worst_scale_rel>1e-6f)throw std::runtime_error("INT8 scale replication mismatch");
        std::cout<<"[KV INT8 WRITE] len="<<len<<" canaries exact, code_1lsb_diffs="<<exact_code_mismatch
                 <<" code_gt_1lsb="<<bad_code<<" scale_rel_max="<<worst_scale_rel<<"\n";

        // Expanded bf16 caches: fp8 from the standard table, int8 from device scale.
        std::vector<bf16> deq(hkv*cap*dim,bf16(0));
        for(int h=0;h<hkv;++h)for(int t=0;t<len;++t)for(int d=0;d<dim;++d) {
            const int off=(h*cap+t)*dim+d;
            deq[off]=bf16((pf8[off]&128)?-values[pf8[off]&127]:values[pf8[off]]);
        }
        ctx.memcpy_h2d(k_f8e.data(),deq.data(),deq.size()*2);
        for(int h=0;h<hkv;++h)for(int t=0;t<len;++t)for(int d=0;d<dim;++d) {
            const int off=(h*cap+t)*dim+d;
            deq[off]=bf16((vf8[off]&128)?-values[vf8[off]&127]:values[vf8[off]]);
        }
        ctx.memcpy_h2d(v_f8e.data(),deq.data(),deq.size()*2);
        for(int h=0;h<hkv;++h)for(int t=0;t<len;++t)for(int d=0;d<dim;++d) {
            const int off=(h*cap+t)*dim+d;
            deq[off]=bf16(float((int8_t)pi8[off])*hks[h*cap+t]);
        }
        ctx.memcpy_h2d(k_i8e.data(),deq.data(),deq.size()*2);
        for(int h=0;h<hkv;++h)for(int t=0;t<len;++t)for(int d=0;d<dim;++d) {
            const int off=(h*cap+t)*dim+d;
            deq[off]=bf16(float((int8_t)vi8[off])*hvs[h*cap+t]);
        }
        ctx.memcpy_h2d(v_i8e.data(),deq.data(),deq.size()*2);

        auto attn=[&](Tensor& kc,Tensor& vc,Tensor& out,const Tensor* ksp,const Tensor* vsp) {
            for(int row=0;row<2;++row) {
                auto qr=Tensor::view(ctx,q.data_as<bf16>()+row*qdim,{1,qdim});
                auto gr=Tensor::view(ctx,g.data_as<bf16>()+row*qdim,{1,qdim});
                auto o=Tensor::view(ctx,out.data_as<bf16>()+row*qdim,{1,qdim});
                ops::attention_decode(ctx,qr,kc,vc,o,scores,hq,hkv,dim,len-1+row,&partials,&gr,
                                      nullptr,0,nullptr,0,ksp,vsp);
            }
        };
        attn(k_bf,v_bf,o_true,nullptr,nullptr);
        attn(k_f8e,v_f8e,o_f8e,nullptr,nullptr);
        attn(k_f8,v_f8,o_f8,nullptr,nullptr);
        attn(k_i8e,v_i8e,o_i8e,nullptr,nullptr);
        attn(k_i8,v_i8,o_i8,&ks,&vs);

        std::vector<bf16> ht(2*qdim),hf8e(2*qdim),hf8(2*qdim),hi8e(2*qdim),hi8(2*qdim);
        ctx.memcpy_d2h(ht.data(),o_true.data(),ht.size()*2);
        ctx.memcpy_d2h(hf8e.data(),o_f8e.data(),hf8e.size()*2);
        ctx.memcpy_d2h(hf8.data(),o_f8.data(),hf8.size()*2);
        ctx.memcpy_d2h(hi8e.data(),o_i8e.data(),hi8e.size()*2);
        ctx.memcpy_d2h(hi8.data(),o_i8.data(),hi8.size()*2);
        auto diff=[&](const std::vector<bf16>& a,const std::vector<bf16>& b) {
            float m=0.0f;for(size_t i=0;i<a.size();++i)m=std::max(m,std::abs(float(a[i])-float(b[i])));return m;
        };
        if(!(diff(hf8,hf8e)<=1e-3f))throw std::runtime_error("FP8 attention vs expanded-cache reference");
        if(!(diff(hi8,hi8e)<=1e-3f))throw std::runtime_error("INT8 attention vs expanded-cache reference");
        auto report=[&](const char* name,const std::vector<bf16>& a) {
            float maxabs=0.0f;double se=0.0,sb=0.0,dot=0.0,na=0.0,nb=0.0;
            for(size_t i=0;i<a.size();++i) {
                const double x=double(float(a[i])),y=double(float(ht[i]));
                maxabs=std::max(maxabs,(float)std::abs(x-y));
                se+=(x-y)*(x-y);sb+=y*y;dot+=x*y;na+=x*x;nb+=y*y;
            }
            const double nrmse=sb>0.0?std::sqrt(se/sb):0.0;
            const double cos=(na>0.0&&nb>0.0)?dot/std::sqrt(na*nb):1.0;
            std::cout<<"[KV ACC] len="<<len<<" arm="<<name<<" vs_bf16_true max_abs="<<maxabs
                     <<" nrmse="<<nrmse<<" cos="<<cos<<" (bitwise_eq="<<(diff(a,ht)==0.0f)<<")\n";
        };
        report("bf16-expanded-fp8",hf8e);
        report("fp8",hf8);
        report("int8-expanded",hi8e);
        report("int8",hi8);

        // Chunked incremental prefill through attention_prefill_cached (int8 read path).
        auto prefill=[&](Tensor& kc,Tensor& vc,Tensor& out,const Tensor* ksp,const Tensor* vsp) {
            ops::attention_prefill_cached(ctx,q2,kc,vc,out,pc_scores,2,len-2,hq,hkv,dim,cap,nullptr,ksp,vsp);
        };
        prefill(k_bf,v_bf,p_true,nullptr,nullptr);
        prefill(k_f8e,v_f8e,p_f8e,nullptr,nullptr);
        prefill(k_f8,v_f8,p_f8,nullptr,nullptr);
        prefill(k_i8e,v_i8e,p_i8e,nullptr,nullptr);
        prefill(k_i8,v_i8,p_i8,&ks,&vs);
        std::vector<bf16> pt(2*qdim),pe8(2*qdim),p8v(2*qdim),pie8(2*qdim),pi8v(2*qdim);
        ctx.memcpy_d2h(pt.data(),p_true.data(),pt.size()*2);
        ctx.memcpy_d2h(pe8.data(),p_f8e.data(),pe8.size()*2);
        ctx.memcpy_d2h(p8v.data(),p_f8.data(),p8v.size()*2);
        ctx.memcpy_d2h(pie8.data(),p_i8e.data(),pie8.size()*2);
        ctx.memcpy_d2h(pi8v.data(),p_i8.data(),pi8v.size()*2);
        if(!(diff(p8v,pe8)<=1e-3f))throw std::runtime_error("FP8 prefill_cached vs expanded reference");
        if(!(diff(pi8v,pie8)<=1e-3f))throw std::runtime_error("INT8 prefill_cached vs expanded reference");
        std::cout<<"[KV PREFILL_CACHED] len="<<len<<" start_pos="<<(len-2)
                 <<" fp8_maxabs="<<diff(p8v,pe8)<<" int8_maxabs="<<diff(pi8v,pie8)
                 <<" int8_vs_true_maxabs="<<diff(pi8v,pt)<<"\n";

        // Interleaved timing of the three attention arms.
        std::vector<double> times[3];
        for(int sample=0;sample<23;++sample)for(int mode=0;mode<3;++mode) {
            auto start=std::chrono::steady_clock::now();
            if(mode==0)attn(k_bf,v_bf,o_scratch,nullptr,nullptr);
            else if(mode==1)attn(k_f8,v_f8,o_scratch,nullptr,nullptr);
            else attn(k_i8,v_i8,o_scratch,&ks,&vs);
            ctx.synchronize();
            if(sample>=3)times[mode].push_back(std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count());
        }
        const char* names[3]={"bf16","fp8","int8"};
        for(int mode=0;mode<3;++mode) {
            std::sort(times[mode].begin(),times[mode].end());
            std::cout<<"[KV WALL] len="<<len<<" B=2 Hq=24 Hkv=2 D=256 arm="<<names[mode]
                     <<" samples=20 median_ms="<<(times[mode][9]+times[mode][10])*0.5
                     <<" p10="<<times[mode][2]<<" p90="<<times[mode][17]<<'\n';
        }
        if(ctx.device_allocation_count()!=live||ctx.pinned_host_allocation_count()!=pinned)
            throw std::runtime_error("KV hot USM allocation");
    };
    run_arm(512,333);
    run_arm(2048,1834);
    run_arm(16384,14336);
    std::cout<<"[KV THREE-ARM PASS] bf16/fp8/int8 equivalence, canaries, host replication and timing complete\n";
}
