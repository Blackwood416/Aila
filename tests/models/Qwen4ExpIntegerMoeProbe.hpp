// R2b test-only prototype. Packed BNB NF4 is unchanged; Q8 activations use
// symmetric 64-element blocks. One row/SG16, uint32 weight loads, integer LUT
// in SLM. dot_acc is a compiler idiom: native DP4A must be verified separately.
static constexpr int8_t kIntegerNf4[16] = {
    -127,-88,-67,-50,-36,-23,-12,0,10,20,31,43,56,71,92,127
};

// Independent exact-rational RTE oracle. Host fast-math may rewrite even a
// double division to reciprocal multiplication and perturb an exact half tie.
int integer_q8_reference(bf16 value,bf16 maximum) {
    uint16_t v,m;std::memcpy(&v,&value,2);std::memcpy(&m,&maximum,2);
    const int ve=(v>>7)&255,me=(m>>7)&255;
    const int shift=std::max(1,me)-std::max(1,ve);
    if((m&0x7fff)==0 || shift>=16)return 0;
    const uint32_t n=((v&127)+(ve?128:0))*127;
    const uint32_t d=((m&127)+(me?128:0))<<shift;
    uint32_t q=n/d,r=n%d;
    if(2*r>d || (2*r==d && (q&1)))++q;
    return (v&0x8000)?-int(q):int(q);
}

template<bool CorrectedReciprocal=false>
sycl::event integer_quantize(Context& ctx, const bf16* src, int8_t* dst,
                            float* scales, int rows, int width, int top_k,
                            const int* tokens=nullptr, const int* ranks=nullptr) {
    expect(width%64==0,"R2b Q8 block alignment");
    const int blocks=width/64;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>((size_t(rows)*blocks+7)/8*128,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int block=it.get_global_linear_id()/16, lane=it.get_local_linear_id()%16;
                if(block>=rows*blocks)return; // Entire subgroup, before collectives.
                const int row=block/blocks,k=(block%blocks)*64+lane*4;
                const int srcrow=tokens?tokens[row]*top_k+ranks[row]:row;
                sycl::vec<float,4> v;
                float a=0.f;
                for(int j=0;j<4;++j){
                    // Check raw BF16 bits BEFORE float conversion: fast-math can
                    // assume the conversion result is finite and erase a later test.
                    uint16_t bits=reinterpret_cast<const uint16_t*>(src)[srcrow*width+k+j];
                    if((bits&0x7f80U)==0x7f80U)bits=0;
                    const float f=float(sycl::bit_cast<bf16>(bits));
                    v[j]=f;a=sycl::fmax(a,sycl::fabs(f));
                }
                a=sycl::reduce_over_group(it.get_sub_group(),a,sycl::maximum<float>());
                const float scale=a/127.f;
                if(lane==0)scales[block]=scale;
                sycl::vec<int8_t,4> q;
                // BF16 * 127 is exact in f32. Explicit correctly-rounded division
                // prevents fast reciprocal approximations changing half-way ties.
                for(int j=0;j<4;++j){
                    if constexpr(CorrectedReciprocal){
                        // BF16 * small integer is exact in f32. Repair approximate
                        // reciprocal rounding by comparing exact half thresholds.
                        // /256 keeps products finite even near max BF16. Tiny
                        // blocks use the correctly-rounded division fallback.
                        if(a>0x1p-110f){
                            int mag=int(sycl::rint(sycl::clamp(sycl::fabs(v[j])*(127.f/a),0.f,127.f)));
                            const float value=sycl::fma(sycl::fabs(v[j]),254.f/256.f,0.f);
                            const float lower=sycl::fma(a,float(2*mag-1)/256.f,0.f);
                            const float upper=sycl::fma(a,float(2*mag+1)/256.f,0.f);
                            if(value>upper || (value==upper && (mag&1)))++mag;
                            else if(value<lower || (value==lower && (mag&1)))--mag;
                            q[j]=int8_t(v[j]<0.f?-mag:mag);
                        }else q[j]=int8_t(sycl::rint(a>0.f?sycl::ext::intel::math::fdiv_rn(v[j]*127.f,a):0.f));
                    }else q[j]=int8_t(sycl::rint(sycl::clamp(a>0.f?sycl::ext::intel::math::fdiv_rn(v[j]*127.f,a):0.f,-127.f,127.f)));
                }
                *reinterpret_cast<sycl::vec<int8_t,4>*>(dst+row*width+k)=q;
            });
    });
}

inline float integer_nested_scale(const ops::MoeNestedScaleDev& ns,size_t block) {
    const float product=sycl::fma(ns.nested_quant_map[ns.codes[block]],ns.nested_absmax[block/ns.blocksize],0.f);
    return sycl::fma(product,1.f,ns.offset);
}

template<class Lookup>
inline sycl::vec<int32_t,2> integer_unpack8(uint32_t packed,const Lookup& lut) {
    sycl::vec<int8_t,4> a,b;
    // Each byte is high nibble then low nibble (BNB), little-endian uint32.
    for(int j=0;j<4;++j) {
        a[j]=lut[(packed>>((j/2)*8+(j%2?0:4)))&15];
        b[j]=lut[(packed>>((j/2+2)*8+(j%2?0:4)))&15];
    }
    return {a.template as<int32_t>(),b.template as<int32_t>()};
}

template<bool Gate,int Rows=1>
sycl::event integer_projection(Context& ctx,const ops::MoeExpertSlotDev* slots,
    const int* tokens,const int* ranks,const int8_t* x,const float* xs,
    bf16* h,float* down,int acts,int top_k,int width,int outputs,
    float* diagnostic=nullptr) {
    expect(width%128==0 && outputs%8==0 && acts%Rows==0,"R2b projection alignment");
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int8_t,1> lut(16,cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(acts/Rows)*outputs*16,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid=it.get_local_linear_id(),lane=lid%16;
                if(lid<16)lut[lid]=kIntegerNf4[lid];
                it.barrier(sycl::access::fence_space::local_space);
                const int index=it.get_global_linear_id()/16,a=index/outputs*Rows,row=index%outputs;
                int xr[Rows],yr[Rows];float g[Rows]={},u[Rows]={};
#pragma unroll
                for(int r=0;r<Rows;++r){xr[r]=Gate?tokens[a+r]:a+r;yr[r]=tokens[a+r]*top_k+ranks[a+r];}
                // Fixture guarantees each quad belongs to the same resident expert.
                const auto slot=slots[a];
                const auto ns=Gate?slot.gate_up_nested:slot.down_nested;
                const uint8_t* w=Gate?slot.gate_up_packed:slot.down_packed;
                const int blocks=width/64;
                for(int word=lane;word<width/8;word+=16) {
                    const size_t bi=size_t(row)*blocks+word/8;
                    const float s=(lane==0||lane==8)?integer_nested_scale(ns,bi):0.f;
                    const float s0=sycl::group_broadcast(it.get_sub_group(),s,0);
                    const float s8=sycl::group_broadcast(it.get_sub_group(),s,8);
                    const uint32_t gw=*reinterpret_cast<const uint32_t*>(w+size_t(row)*width/2+word*4);
                    const auto gp=integer_unpack8(gw,lut);
                    float us0=0.f,us8=0.f;sycl::vec<int32_t,2> up;
                    if constexpr(Gate) {
                        const float us=(lane==0||lane==8)?integer_nested_scale(ns,bi+size_t(outputs)*blocks):0.f;
                        us0=sycl::group_broadcast(it.get_sub_group(),us,0);
                        us8=sycl::group_broadcast(it.get_sub_group(),us,8);
                        const uint32_t uw=*reinterpret_cast<const uint32_t*>(w+size_t(row+outputs)*width/2+word*4);
                        up=integer_unpack8(uw,lut);
                    }
#pragma unroll
                    for(int r=0;r<Rows;++r){
                        const float ax=xs[xr[r]*blocks+word/8]/127.f;
                        const auto qx=*reinterpret_cast<const sycl::vec<int32_t,2>*>(x+xr[r]*width+word*8);
                        const int gd=sycl::ext::oneapi::dot_acc(gp[1],qx[1],sycl::ext::oneapi::dot_acc(gp[0],qx[0],int32_t(0)));
                        g[r]=sycl::fma(float(gd),ax*(lane<8?s0:s8),g[r]);
                        if constexpr(Gate){
                            const int ud=sycl::ext::oneapi::dot_acc(up[1],qx[1],sycl::ext::oneapi::dot_acc(up[0],qx[0],int32_t(0)));
                            u[r]=sycl::fma(float(ud),ax*(lane<8?us0:us8),u[r]);
                        }
                    }
                }
#pragma unroll
                for(int r=0;r<Rows;++r){
                    g[r]=sycl::reduce_over_group(it.get_sub_group(),g[r],sycl::plus<float>());
                    if constexpr(Gate)u[r]=sycl::reduce_over_group(it.get_sub_group(),u[r],sycl::plus<float>());
                    if(lane==0) {
                        if constexpr(Gate) {
                            h[yr[r]*outputs+row]=bf16(g[r]/(1.f+sycl::native::exp(-g[r]))*u[r]);
                            if(diagnostic){diagnostic[(a+r)*2*outputs+row]=g[r];diagnostic[(a+r)*2*outputs+outputs+row]=u[r];}
                        } else down[yr[r]*outputs+row]=g[r];
                    }
                }
            });
    });
}

#include "Qwen4ExpDpasMoeProbe.hpp"

void test_integer_moe(const std::string& directory,bool dpas=false) {
    std::cout<<std::unitbuf<<std::setprecision(10);
    constexpr int E=2,H=2560,I=640,R=10;
    _putenv_s("AILA_Q4E_HOT_RAM_MB","0");
    const bool device_profile=aila::env::read_flag("AILA_Q4E_INTEGER_DEVICE_PROFILE",false);
    Context ctx(nullptr,device_profile);auto weights=LoadModelWeightsLazy(directory,ctx);
    Qwen4ExpExpertCache cache;std::string error;
    if(!cache.init(ctx,weights,1,size_t(E)*2534400,512,H,I,64,&error))throw std::runtime_error("R2b cache: "+error);
    expect(cache.uses_nested_codes(),"R2b needs nested NF4");
    const int experts[E]={7,113};int ids[E];cache.acquire(0,experts,E,ids).wait_and_throw();
    Bnb4BitHostWeightRef source;
    expect(LoadBnb4BitHostRef(weights,"model.language_model.layers.0.mlp.experts.gate_up_proj",source,&error,H),"R2b LUT");
    for(int c=0;c<16;++c)expect(int(std::nearbyint(reinterpret_cast<const float*>(source.quant_map.data())[c]*127.f))==kIntegerNf4[c],"R2b wrong NF4 LUT");
    auto alloc=[&](std::vector<int64_t> s,dnnl::memory::data_type t){return Tensor::allocate(ctx,s,t);};
    const auto bf=dnnl::memory::data_type::bf16,fp=dnnl::memory::data_type::f32,
        si=dnnl::memory::data_type::s32,byte=dnnl::memory::data_type::u8;
    // Host reference copies only these two experts. No full-model INT8 cache.
    struct HostExpert {std::vector<uint8_t> gu,dn;std::vector<float> gs,ds;};
    std::array<HostExpert,E> host;
    auto copy_weight=[&](const uint8_t* packed,const ops::MoeNestedScaleDev& ns,int count,
                         std::vector<uint8_t>& w,std::vector<float>& scales) {
        w.resize(count/2);scales.resize(count/64);std::vector<uint8_t> code(count/64);
        std::vector<float> abs((code.size()+ns.blocksize-1)/ns.blocksize),map(256);
        ctx.memcpy_d2h(w.data(),packed,w.size());ctx.memcpy_d2h(code.data(),ns.codes,code.size());
        ctx.memcpy_d2h(abs.data(),ns.nested_absmax,abs.size()*4);ctx.memcpy_d2h(map.data(),ns.nested_quant_map,map.size()*4);
        for(size_t b=0;b<code.size();++b){volatile float product=map[code[b]]*abs[b/ns.blocksize];scales[b]=product+ns.offset;}
    };
    for(int e=0;e<E;++e){auto s=cache.get_slot_dev(ids[e]);copy_weight(s.gate_up_packed,s.gate_up_nested,2*I*H,host[e].gu,host[e].gs);copy_weight(s.down_packed,s.down_nested,H*I,host[e].dn,host[e].ds);}
    for(int M:(dpas?(aila::env::read_flag("AILA_Q4E_MATURE_PROFILE",false)?std::vector<int>{32}:std::vector<int>{8,16,24,32,48,64}):std::vector<int>{1,32})) {
        const int A=E*M;
        auto x=alloc({M,H},bf),h=alloc({M,R,I},bf),down=alloc({M,R,H},fp),out=alloc({M,H},bf);
        auto sd=alloc({int64_t(A*sizeof(ops::MoeExpertSlotDev))},byte),td=alloc({A},si),rd=alloc({A},si);
        auto rwdev=alloc({M,R},fp),qm=alloc({16},fp),qx=alloc({M,H},byte),qh=alloc({A,I},byte);
        auto sx=alloc({M,H/64},fp),sh=alloc({A,I/64},fp),diagnostic=alloc({A,2*I},fp);
        auto staged=alloc({dpas?int64_t(E)*2*I*H:1},byte),staged_scale=alloc({dpas?int64_t(E)*2*I*H/64:1},fp);
        auto vnni=alloc({dpas?int64_t(E)*((M+31)/32)*32*H:1},byte),gathered=alloc({dpas?int64_t(A)*H:1},byte),gathered_scale=alloc({dpas?int64_t(A)*H/64:1},fp);
        using ExecutableGraph=sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
        std::array<std::unique_ptr<ExecutableGraph>,4> replay;
        std::unique_ptr<IntegerDpasLibrary> library;
        if(dpas)try{library=std::make_unique<IntegerDpasLibrary>(ctx,E,M,H,I,staged.data_as<int8_t>(),gathered.data_as<int8_t>(),qh.data_as<int8_t>());}
        catch(const dnnl::error& err){std::cout<<"[R2B DPAS LIBRARY] unavailable=1 reason="<<err.what()<<"\n";}
        struct Drain{Context& c;~Drain(){try{c.synchronize();}catch(...){}}}drain{ctx};
        std::vector<ops::MoeExpertSlotDev> slots(A);std::vector<int> tokens(A),ranks(A);std::vector<float> rw(M*R,0.f);
        for(int e=0;e<E;++e)for(int m=0;m<M;++m){const int a=e*M+m;slots[a]=cache.get_slot_dev(ids[e]);tokens[a]=(m*7)%M;ranks[a]=e;rw[tokens[a]*R+e]=.5f;}
        std::mt19937 rng(61003);std::normal_distribution<float> normal(0.f,.25f);
        std::vector<bf16> input(M*H);for(auto& v:input)v=bf16(normal(rng));
        ctx.memcpy_h2d(sd.data(),slots.data(),sd.size_bytes());ctx.memcpy_h2d(td.data(),tokens.data(),A*4);ctx.memcpy_h2d(rd.data(),ranks.data(),A*4);
        ctx.memcpy_h2d(qm.data(),source.quant_map.data(),64);ctx.memcpy_h2d(rwdev.data(),rw.data(),rw.size()*4);
        ctx.queue().memset(down.data(),0,down.size_bytes());
        auto* sp=sd.data_as<ops::MoeExpertSlotDev>();auto* tp=td.data_as<int>();auto* rp=rd.data_as<int>();
        auto* xp=x.data_as<bf16>();auto* hp=h.data_as<bf16>();auto* dp=down.data_as<float>();auto* qp=qm.data_as<float>();
        auto* x8=qx.data_as<int8_t>();auto* h8=qh.data_as<int8_t>();auto* xs=sx.data_as<float>();auto* hs=sh.data_as<float>();
        // Isolated quantizer edge cases, including exact ties (RTE), zero and nonfinite.
        std::vector<bf16> edge(M*H,bf16(0.f));
        edge[64]=bf16(127.f);edge[65]=bf16(.5f);edge[66]=bf16(1.5f);edge[67]=bf16(-2.5f);edge[68]=bf16(-127.f);
        const uint16_t infinity_bits=0x7f80, nan_bits=0x7fc0;
        std::memcpy(&edge[128],&infinity_bits,2);std::memcpy(&edge[129],&nan_bits,2);
        ctx.memcpy_h2d(x.data(),edge.data(),edge.size()*2);
        for(bool fast:{false,true}){
        if(fast)integer_quantize<true>(ctx,xp,x8,xs,M,H,1);else integer_quantize(ctx,xp,x8,xs,M,H,1);
        std::vector<int8_t> edgeq(M*H);std::vector<float> edges(M*H/64);
        ctx.memcpy_d2h(edgeq.data(),qx.data(),edgeq.size());ctx.memcpy_d2h(edges.data(),sx.data(),edges.size()*4);
        std::cout<<"[R2B EDGE] scales="<<edges[0]<<","<<edges[1]<<","<<edges[2]<<" codes="<<int(edgeq[64])<<","<<int(edgeq[65])<<","<<int(edgeq[66])<<","<<int(edgeq[67])<<","<<int(edgeq[68])<<"\n";
        expect(edges[0]==0.f && edges[1]==1.f && edges[2]==0.f,"R2b quantizer scale edge");
        for(int k=0;k<M*H;++k){int want=k==64?127:k==66?2:k==67?-2:k==68?-127:0;expect(int(edgeq[k])==want,"R2b quantizer RTE/zero/nonfinite");}
        }
        ctx.memcpy_h2d(x.data(),input.data(),input.size()*2);
        std::array<std::array<std::vector<double>,12>,12> stage_times;
        auto run=[&](int mode,bool diagnostics=false,bool capture=false){
            std::array<sycl::event,12> stages;
            if(mode>=7 && mode!=10)stages[11]=ctx.queue().ext_oneapi_graph(*replay[mode==11?3:mode-7]);
            else if(mode>=4){
                auto* wp=staged.data_as<int8_t>();auto* sw=staged_scale.data_as<float>();auto* vp=vnni.data_as<int8_t>();auto* gx=gathered.data_as<int8_t>();auto* gs=gathered_scale.data_as<float>();
                stages[0]=integer_quantize<true>(ctx,xp,x8,xs,M,H,1);
                if(mode==10)stages[5]=integer_dpas_pack<true,true>(ctx,x8,xs,vp,gx,gs,tp,E,M,H);
                else stages[5]=integer_dpas_pack<true>(ctx,x8,xs,mode!=5?vp:nullptr,gx,gs,tp,E,M,H);
                stages[6]=integer_dpas_stage<true>(ctx,sp,wp,sw,E,M,H,I);
                if(mode==10)stages[1]=integer_dpas_projection<true,8,true>(ctx,wp,sw,vp,gs,tp,rp,hp,nullptr,E,M,H,I,R,diagnostics?diagnostic.data_as<float>():nullptr);
                else if(mode==6)stages[1]=integer_dpas_projection<true,32>(ctx,wp,sw,vp,gs,tp,rp,hp,nullptr,E,M,H,I,R,diagnostics?diagnostic.data_as<float>():nullptr);
                else if(mode==4)stages[1]=integer_dpas_projection<true>(ctx,wp,sw,vp,gs,tp,rp,hp,nullptr,E,M,H,I,R,diagnostics?diagnostic.data_as<float>():nullptr);
                else{
                    stages[9]=library->Run(ctx,0);
                    stages[1]=integer_dpas_library_reduce<true>(ctx,library->partial.data_as<int32_t>(),sw,gs,tp,rp,hp,nullptr,E,M,H,I,R,diagnostics?diagnostic.data_as<float>():nullptr);
                }
                stages[2]=integer_quantize<true>(ctx,hp,h8,hs,A,I,R,tp,rp);
                if(mode==10)stages[7]=integer_dpas_pack<false,true>(ctx,h8,hs,vp,nullptr,nullptr,nullptr,E,M,I);
                else if(mode!=5)stages[7]=integer_dpas_pack<false>(ctx,h8,hs,vp,nullptr,nullptr,nullptr,E,M,I);
                stages[8]=integer_dpas_stage<false>(ctx,sp,wp,sw,E,M,I,H);
                if(mode==10)stages[3]=integer_dpas_projection<false,8,true>(ctx,wp,sw,vp,hs,tp,rp,nullptr,dp,E,M,I,H,R);
                else if(mode==6)stages[3]=integer_dpas_projection<false,32>(ctx,wp,sw,vp,hs,tp,rp,nullptr,dp,E,M,I,H,R);
                else if(mode==4)stages[3]=integer_dpas_projection<false>(ctx,wp,sw,vp,hs,tp,rp,nullptr,dp,E,M,I,H,R);
                else{
                    stages[10]=library->Run(ctx,1);
                    stages[3]=integer_dpas_library_reduce<false>(ctx,library->partial.data_as<int32_t>(),sw,hs,tp,rp,nullptr,dp,E,M,I,H,R);
                }
            }else if(mode) {
                stages[0]=mode==3?integer_quantize<true>(ctx,xp,x8,xs,M,H,1):integer_quantize(ctx,xp,x8,xs,M,H,1);
                if((mode==2 || mode==3) && M>=8)stages[1]=integer_projection<true,4>(ctx,sp,tp,rp,x8,xs,hp,nullptr,A,R,H,I,diagnostics?diagnostic.data_as<float>():nullptr);
                else stages[1]=integer_projection<true>(ctx,sp,tp,rp,x8,xs,hp,nullptr,A,R,H,I,diagnostics?diagnostic.data_as<float>():nullptr);
                stages[2]=mode==3?integer_quantize<true>(ctx,hp,h8,hs,A,I,R,tp,rp):integer_quantize(ctx,hp,h8,hs,A,I,R,tp,rp);
                if((mode==2 || mode==3) && M>=8)stages[3]=integer_projection<false,4>(ctx,sp,tp,rp,h8,hs,nullptr,dp,A,R,I,H);
                else stages[3]=integer_projection<false>(ctx,sp,tp,rp,h8,hs,nullptr,dp,A,R,I,H);
            }else if(M==1){
                auto ev=ops::moe_decode_gate_up_swiglu_grouped(ctx,sp,qp,xp,hp,A,H,I,64,nullptr,tp,true);
                stages[1]=ev;stages[3]=ops::moe_decode_down_to_rank_f32(ctx,sp,rp,hp,dp,A,R,I,H,64,tp,qp,&ev,true);
            }else{
                auto ev=ops::moe_expert_major_prefill_gate_up_4token(ctx,sp,tp,rp,qp,xp,hp,A/4,R,H,I,64,nullptr,true);
                stages[1]=ev;stages[3]=ops::moe_expert_major_prefill_down_4token(ctx,sp,tp,rp,qp,hp,dp,A/4,R,I,H,64,&ev,true);
            }
            if(mode<7 || mode==10)stages[4]=ops::moe_ordered_accumulate_down_rows(ctx,dp,rwdev.data_as<float>(),out.data_as<bf16>(),M,R,H);
            if(capture)return;
            ctx.synchronize();
            if(device_profile)for(int j=0;j<12;++j)if((mode>=7 && mode!=10)?j==11:(mode>=4?(mode==5?(j<11 && j!=7):j<9):(j<5 && (mode || (j!=0 && j!=2))))){
                const auto begin=stages[j].get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto end=stages[j].get_profiling_info<sycl::info::event_profiling::command_end>();
                stage_times[mode][j].push_back(double(end-begin)/1000.);
            }
        };
        run(false);std::vector<bf16> baseline(M*H),got(M*H);ctx.memcpy_d2h(baseline.data(),out.data(),baseline.size()*2);
        std::vector<int> modes=dpas?(library?std::vector<int>{0,3,4,5,6,10}:std::vector<int>{0,3,4,6,10}):(M==32?std::vector<int>{0,1,2,3}:std::vector<int>{0,1,3});
        for(int checked_mode:modes){if(checked_mode==0)continue;
        run(checked_mode,true);ctx.memcpy_d2h(got.data(),out.data(),got.size()*2);
        std::vector<int8_t> xq(M*H),hq(A*I);std::vector<float> xscale(M*H/64),hscale(A*I/64),gu(A*2*I),raw(M*R*H);
        std::vector<bf16> mid(M*R*I);
        ctx.memcpy_d2h(xq.data(),qx.data(),xq.size());ctx.memcpy_d2h(hq.data(),qh.data(),hq.size());
        ctx.memcpy_d2h(xscale.data(),sx.data(),xscale.size()*4);ctx.memcpy_d2h(hscale.data(),sh.data(),hscale.size()*4);
        ctx.memcpy_d2h(gu.data(),diagnostic.data(),gu.size()*4);ctx.memcpy_d2h(raw.data(),down.data(),raw.size()*4);ctx.memcpy_d2h(mid.data(),h.data(),mid.size()*2);
        // CPU reconstructs scale per64, signed products, BNB ordering and gathers.
        auto check_q=[&](const std::vector<bf16>& in,const std::vector<int8_t>& q,const std::vector<float>& scale,int rows,int width,bool gather){
            for(int a=0;a<rows;++a)for(int b=0;b<width/64;++b){const int srcrow=gather?tokens[a]*R+ranks[a]:a;float mx=0.f;
                for(int j=0;j<64;++j)mx=std::max(mx,std::abs(float(in[srcrow*width+b*64+j])));
                expect(std::abs(scale[a*width/64+b]-mx/127.f)<=1e-7f*mx+1e-12f,"R2b CPU Q8 scale");
                for(int j=0;j<64;++j){float f=float(in[srcrow*width+b*64+j]);int want=integer_q8_reference(in[srcrow*width+b*64+j],bf16(mx));
                    if(int(q[a*width+b*64+j])!=want)std::cout<<"[R2B QFAIL] gather="<<gather<<" a="<<a<<" b="<<b<<" j="<<j<<" f="<<f<<" mx="<<mx<<" ratio="<<double(f)*127./double(mx)<<" expected="<<want<<" got="<<int(q[a*width+b*64+j])<<"\n";
                    expect(int(q[a*width+b*64+j])==want,"R2b CPU Q8 payload");}
            }
        };
        check_q(input,xq,xscale,M,H,false);check_q(mid,hq,hscale,A,I,true);
        double ce=0,cr=0,ca=0;size_t errors=0,checked=0;
        auto dot=[&](const std::vector<uint8_t>& w,const std::vector<float>& s,int row,int width,const int8_t* q,const float* as){double z=0.;
            for(int b=0;b<width/64;++b){int sum=0;for(int j=0;j<64;++j){const int k=row*width+b*64+j;int c=(w[k/2]>>(k%2?0:4))&15;sum+=int(kIntegerNf4[c])*int(q[b*64+j]);}
                z+=double(sum)*s[row*width/64+b]*as[b]/127.;}return z;};
        auto check=[&](double ref,double v){double d=v-ref;ce+=d*d;cr+=ref*ref;ca=std::max(ca,std::abs(d));errors+=std::abs(d)>1e-6+1e-4*std::abs(ref);++checked;};
        // All outputs for first/last row per expert, with permuted token mapping.
        for(int e=0;e<E;++e)for(int m=0;m<M;m+=(M==1?1:M-1)){int a=e*M+m,t=tokens[a];
            for(int n=0;n<2*I;++n)check(dot(host[e].gu,host[e].gs,n,H,xq.data()+t*H,xscale.data()+t*H/64),gu[a*2*I+n]);
            for(int n=0;n<H;++n)check(dot(host[e].dn,host[e].ds,n,I,hq.data()+a*I,hscale.data()+a*I/64),raw[(t*R+e)*H+n]);
        }
        expect(errors==0,"R2b integer CPU semantics mismatch");
        // Validate SwiGLU rank addressing and the complete ordered merge as well.
        for(int a=0;a<A;++a)for(int n=0;n<I;++n){
            const double g=gu[a*2*I+n],u=gu[a*2*I+I+n],expected=g/(1.+std::exp(-g))*u;
            expect(std::abs(float(mid[(tokens[a]*R+ranks[a])*I+n])-expected)<=1e-7+.008*std::abs(expected),"R2b SwiGLU reference/address");
        }
        for(int t=0;t<M;++t)for(int n=0;n<H;++n){
            float acc=0.f;for(int r=0;r<R;++r)acc+=rw[t*R+r]*raw[(t*R+r)*H+n];
            const bf16 expected(acc);expect(std::memcmp(&expected,&got[t*H+n],2)==0,"R2b ordered merge reference");
        }
        double err=0,ref=0,ma=0;size_t relaxed_errors=0;
        for(size_t j=0;j<got.size();++j){const double v=float(got[j]),r=float(baseline[j]),d=v-r;expect(std::isfinite(v),"R2b nonfinite output");err+=d*d;ref+=r*r;ma=std::max(ma,std::abs(d));relaxed_errors+=std::abs(d)>2e-4+.02*std::abs(r);}
        std::cout<<"[R2B CONFIG] M="<<M<<" E=2 H=2560 I=640 top_k=10 layer=0 experts=7,113 input=synthetic_bf16 host_pool=0 q8_block=64 integer_weight_cache_bytes=0 q8_scratch_bytes="<<qx.size_bytes()+qh.size_bytes()+sx.size_bytes()+sh.size_bytes()<<" baseline="<<(M==1?"decode_grouped":"prefill_g3_four_token")<<"\n";
        std::cout<<"[R2B CPU] M="<<M<<" mode="<<checked_mode<<" checked="<<checked<<" errors="<<errors<<" quantizer_errors=0 edge_errors=0 activation_errors=0 merge_errors=0 nrmse="<<std::sqrt(ce/cr)<<" max_abs="<<ca<<" abs_tol=1e-6 rel_tol=1e-4\n";
        std::cout<<"[R2B APPROX] M="<<M<<" mode="<<checked_mode<<" nrmse="<<std::sqrt(err/ref)<<" max_abs="<<ma<<" pointwise_errors="<<relaxed_errors<<" elements="<<got.size()<<" abs_tol=0.0002 rel_tol=0.02 model_quality_validated=0\n";
        }
        if(dpas && !device_profile && ctx.queue().get_device().has(sycl::aspect::ext_oneapi_limited_graph)){
            namespace ex=sycl::ext::oneapi::experimental;
            const int ordinary[4]={4,0,3,10};
            auto bytes=[&](Tensor& tensor){std::vector<uint8_t> v(tensor.size_bytes());ctx.memcpy_d2h(v.data(),tensor.data(),v.size());return v;};
            for(int g=0;g<4;++g){
                const int graph_mode=g==3?11:7+g;
                const auto build_start=std::chrono::steady_clock::now();
                ex::command_graph<ex::graph_state::modifiable> recording(ctx.queue().get_context(),ctx.queue().get_device());
                recording.begin_recording(ctx.queue());
                try{run(ordinary[g],false,true);}catch(...){recording.end_recording(ctx.queue());throw;}
                recording.end_recording(ctx.queue());replay[g]=std::make_unique<ExecutableGraph>(recording.finalize());
                const double build_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-build_start).count();
                double first_us=0;
                for(int mutation=0;mutation<2;++mutation){
                    if(mutation){
                        auto changed_input=input;auto changed_tokens=tokens;auto changed_slots=slots;
                        for(size_t j=0;j<changed_input.size();++j)changed_input[j]=bf16(float(input[j])*(j%2?-.5f:.5f));
                        for(int a=0;a<A;++a){changed_tokens[a]=M-1-tokens[a];changed_slots[a]=slots[(1-a/M)*M+a%M];}
                        ctx.memcpy_h2d(x.data(),changed_input.data(),changed_input.size()*2);
                        ctx.memcpy_h2d(td.data(),changed_tokens.data(),A*4);
                        ctx.memcpy_h2d(sd.data(),changed_slots.data(),sd.size_bytes());
                    }
                    // Validate the exact replay path used for timing, including h
                    // and FP32 rank output, before and after device-buffer updates.
                    run(ordinary[g]);const auto expected_h=bytes(h),expected_down=bytes(down),expected_out=bytes(out);
                    const auto start=std::chrono::steady_clock::now();run(graph_mode);
                    if(!mutation)first_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
                    expect(bytes(h)==expected_h && bytes(down)==expected_down && bytes(out)==expected_out,"R2b graph differs from ordinary path");
                }
                ctx.memcpy_h2d(x.data(),input.data(),input.size()*2);ctx.memcpy_h2d(td.data(),tokens.data(),A*4);ctx.memcpy_h2d(sd.data(),slots.data(),sd.size_bytes());
                modes.push_back(graph_mode);
                std::cout<<"[R2B DPAS GRAPH] M="<<M<<" mode="<<graph_mode<<" ordinary="<<ordinary[g]<<" exact_h_down_output=1 updated_buffers_cases=2 build_ms="<<build_ms<<" first_replay_warm_kernels_us="<<first_us<<"\n";
            }
        }
        if(dpas)std::cout<<"[R2B DPAS CONFIG] M="<<M<<" tile=8x8x32 scale_block=64 weight_stage_bytes="<<staged.size_bytes()<<" scale_stage_bytes="<<staged_scale.size_bytes()<<" stage_reused_gate_down=1 full_path_conversion_included=1 q8_pack_bytes="<<vnni.size_bytes()+gathered.size_bytes()+gathered_scale.size_bytes()<<"\n";
        const auto live=ctx.device_allocation_count();
        if(!aila::env::read_flag("AILA_Q4E_MATURE_PROFILE",false)){
            for(int i=0;i<8;++i)for(int mode:modes)run(mode);
            if(device_profile){
                for(auto& mode:stage_times)for(auto& stage:mode)stage.clear();
                for(int i=0;i<20;++i)for(int mode:modes)run(mode);
                for(int mode:modes)for(int j=0;j<12;++j){auto& v=stage_times[mode][j];if(v.empty())continue;
                    std::sort(v.begin(),v.end());std::cout<<"[R2B DEVICE] M="<<M<<" mode="<<mode<<" stage="<<j<<" median_us="<<(v[9]+v[10])/2<<" p10_us="<<v[2]<<" p90_us="<<v[18]<<"\n";}
            } else for(int round=0;round<3;++round){std::array<std::vector<double>,12> times;
                for(int i=0;i<20;++i)for(int order=0;order<int(modes.size());++order){int mode=modes[(i+round+order)%modes.size()];auto start=std::chrono::steady_clock::now();run(mode);times[mode].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());}
                for(int mode:modes){auto& v=times[mode];std::sort(v.begin(),v.end());std::cout<<"[R2B TIME] M="<<M<<" round="<<round<<" mode="<<mode<<" median_us="<<(v[9]+v[10])/2<<" p10_us="<<v[2]<<" p90_us="<<v[18]<<"\n";}
            }
        }
        expect(live==ctx.device_allocation_count(),"R2b hot USM allocation");
    }
    std::cout<<"[R2B PASS] integer_semantics=1 no_hot_usm_growth=1 performance_includes_two_quantizations=1 production_enabled=0\n";
}
