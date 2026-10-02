// Diagnostic-only R2b real-activation replay. The model first runs NF4; an
// optional callback replaces every prefill expert contribution with one fixed
// integer-LUT/Q8 arithmetic policy. Bucket heat never selects numeric format.
template<bool Gate,bool Quantized>
sycl::event integer_ablation_projection(Context& ctx,const ops::MoeExpertSlotDev* slots,
    const float* qmap,const bf16* input,const int8_t* qinput,const float* input_scale,
    bf16* middle,float* output,int rows,int width,int outputs){
    return ctx.queue().submit([&](sycl::handler& cgh){
        sycl::local_accessor<float,1> lut(16,cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(rows)*outputs*16,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid=it.get_local_linear_id(),lane=lid%16;
                if(lid<16)lut[lid]=qmap[lid];it.barrier(sycl::access::fence_space::local_space);
                const int a=it.get_global_linear_id()/16/outputs,n=it.get_global_linear_id()/16%outputs;
                const auto slot=slots[0];const auto ns=Gate?slot.gate_up_nested:slot.down_nested;
                const auto* p=Gate?slot.gate_up_packed:slot.down_packed;
                const int B=width/64;float g=0.f,u=0.f;
                for(int word=lane;word<width/8;word+=16){
                    const size_t b=size_t(n)*B+word/8;
                    const float ls=(lane==0||lane==8)?integer_nested_scale(ns,b):0.f;
                    const float s0=sycl::group_broadcast(it.get_sub_group(),ls,0),s8=sycl::group_broadcast(it.get_sub_group(),ls,8);
                    const float gs=lane<8?s0:s8;
                    float us=0.f;uint32_t uw=0;
                    const uint32_t gw=*reinterpret_cast<const uint32_t*>(p+size_t(n)*width/2+word*4);
                    if constexpr(Gate){
                        const float ul=(lane==0||lane==8)?integer_nested_scale(ns,b+size_t(outputs)*B):0.f;
                        const float u0=sycl::group_broadcast(it.get_sub_group(),ul,0),u8=sycl::group_broadcast(it.get_sub_group(),ul,8);
                        us=lane<8?u0:u8;uw=*reinterpret_cast<const uint32_t*>(p+size_t(n+outputs)*width/2+word*4);
                    }
#pragma unroll
                    for(int j=0;j<8;++j){
                        float x;
                        if constexpr(Quantized)x=float(qinput[a*width+word*8+j])*input_scale[a*B+word/8];
                        else x=float(input[a*width+word*8+j]);
                        const int shift=(j/2)*8+(j%2?0:4);
                        g=sycl::fma(x,lut[(gw>>shift)&15]*gs,g);
                        if constexpr(Gate)u=sycl::fma(x,lut[(uw>>shift)&15]*us,u);
                    }
                }
                g=sycl::reduce_over_group(it.get_sub_group(),g,sycl::plus<float>());
                if constexpr(Gate)u=sycl::reduce_over_group(it.get_sub_group(),u,sycl::plus<float>());
                if(lane==0){if constexpr(Gate)middle[a*outputs+n]=bf16(g/(1.f+sycl::native::exp(-g))*u);else output[a*outputs+n]=g;}
            });
    });
}

struct IntegerQualityReplay {
    static constexpr int M=32,H=2560,I=640,R=10;
    using Graph=sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    Context& ctx;Qwen4ExpBnb4Backend& backend;
    Tensor x,reference_h,reference_down,verify_down,h,down,qx,qh,xs,hs,weights,scales,vnni,slots,tokens,ranks,source_tokens,source_ranks,gu,qmap,lutmap;
    struct TileParams {ops::MoeExpertSlotDev slot;const bf16* input;float* output;int token[M],rank[M],count;};
    static constexpr int MaxTiles=2048;
    Tensor params,cursor;
    std::vector<TileParams> host_params;
    std::unique_ptr<Graph> graph,stream_graph;
    int pending_tiles=0;

    std::array<std::vector<int>,512> buckets;
    std::array<int,48> samples{};
    std::array<uint64_t,48> layer_acts{},layer_tiles{};
    uint64_t total_acts=0,replaced=0,tiles=0,hot_acts=0,experts=0;
    int phase=0; // 0 NF4 observation / 1 integer replacement; only count>1 callbacks.
    std::string prefix;
    explicit IntegerQualityReplay(Context& c,Qwen4ExpBnb4Backend& b,const std::string& path):ctx(c),backend(b),prefix(path){
        auto alloc=[&](int64_t n,dnnl::memory::data_type type){return Tensor::allocate(ctx,{n},type);};
        const auto bf=dnnl::memory::data_type::bf16,fp=dnnl::memory::data_type::f32,byte=dnnl::memory::data_type::u8,si=dnnl::memory::data_type::s32;
        x=alloc(M*H,bf);reference_h=alloc(M*I,bf);reference_down=alloc(M*H,fp);verify_down=alloc(M*H,fp);h=alloc(M*I,bf);down=alloc(M*H,fp);
        qx=alloc(M*H,byte);qh=alloc(M*I,byte);xs=alloc(M*H/64,fp);hs=alloc(M*I/64,fp);
        weights=alloc(2*I*H,byte);scales=alloc(2*I*H/64,fp);vnni=alloc(M*H,byte);
        slots=alloc(M*sizeof(ops::MoeExpertSlotDev),byte);tokens=alloc(M,si);ranks=alloc(M,si);
        source_tokens=alloc(M,si);source_ranks=alloc(M,si);gu=alloc(M*2*I,fp);qmap=alloc(16,fp);lutmap=alloc(16,fp);
        std::array<int,M> identity{};for(int i=0;i<M;++i)identity[i]=i;
        ctx.memcpy_h2d(tokens.data(),identity.data(),M*4);ctx.queue().memset(ranks.data(),0,M*4);
        ctx.memcpy_h2d(qmap.data(),kTestQuantMap,64);float integer_map[16];for(int i=0;i<16;++i)integer_map[i]=float(kIntegerNf4[i])/127.f;
        ctx.memcpy_h2d(lutmap.data(),integer_map,64);
        params=alloc(int64_t(MaxTiles)*sizeof(TileParams),byte);cursor=alloc(1,si);host_params.resize(MaxTiles);
        for(auto& v:buckets)v.reserve(2048);
        namespace ex=sycl::ext::oneapi::experimental;
        expect(ctx.queue().get_device().has(sycl::aspect::ext_oneapi_limited_graph),"Integer quality needs limited graph");
        ex::command_graph<ex::graph_state::modifiable> recording(ctx.queue().get_context(),ctx.queue().get_device());
        recording.begin_recording(ctx.queue());
        try{Enqueue();}catch(...){recording.end_recording(ctx.queue());throw;}
        recording.end_recording(ctx.queue());graph=std::make_unique<Graph>(recording.finalize());
        // Fixed device parameter table + GPU cursor: one table upload per
        // microbatch, no three blocking host copies per expert tile.
        ex::command_graph<ex::graph_state::modifiable> stream(ctx.queue().get_context(),ctx.queue().get_device());
        stream.begin_recording(ctx.queue());
        const auto* table=params.data_as<TileParams>();auto* index=cursor.data_as<int>();
        auto* xp=x.data_as<bf16>();auto* sp=slots.data_as<ops::MoeExpertSlotDev>();auto* result=down.data_as<float>();
        ctx.queue().parallel_for(sycl::range<1>(M*H),[=](sycl::id<1> id) [[sycl::reqd_sub_group_size(16)]] {
            const auto& p=table[*index];const int row=id[0]/H,col=id[0]%H;
            xp[id]=row<p.count?p.input[p.token[row]*H+col]:bf16(0.f);
            if(id[0]==0)sp[0]=p.slot;
        });
        Enqueue();
        ctx.queue().parallel_for(sycl::range<1>(M*H),[=](sycl::id<1> id) [[sycl::reqd_sub_group_size(16)]] {
            const auto& p=table[*index];const int row=id[0]/H,col=id[0]%H;
            if(row<p.count)p.output[(p.token[row]*R+p.rank[row])*H+col]=result[id];
        });
        // Separate ordered node: advancing in scatter's thread zero could race
        // other workgroups still reading the old cursor.
        ctx.queue().single_task([=]{++*index;});
        stream.end_recording(ctx.queue());stream_graph=std::make_unique<Graph>(stream.finalize());
    }
    ~IntegerQualityReplay(){try{ctx.synchronize();}catch(...){}Access::ExpertQualityProbe(backend,nullptr,nullptr);}
    void Enqueue(){
        auto* sp=slots.data_as<ops::MoeExpertSlotDev>();auto* w=weights.data_as<int8_t>();auto* s=scales.data_as<float>();auto* v=vnni.data_as<int8_t>();
        auto* t=tokens.data_as<int>();auto* r=ranks.data_as<int>();
        integer_quantize<true>(ctx,x.data_as<bf16>(),qx.data_as<int8_t>(),xs.data_as<float>(),M,H,1);
        integer_dpas_pack<false>(ctx,qx.data_as<int8_t>(),nullptr,v,nullptr,nullptr,nullptr,1,M,H);
        integer_dpas_stage<true>(ctx,sp,w,s,1,M,H,I);
        integer_dpas_projection<true>(ctx,w,s,v,xs.data_as<float>(),t,r,h.data_as<bf16>(),nullptr,1,M,H,I,1,gu.data_as<float>());
        integer_quantize<true>(ctx,h.data_as<bf16>(),qh.data_as<int8_t>(),hs.data_as<float>(),M,I,1);
        integer_dpas_pack<false>(ctx,qh.data_as<int8_t>(),nullptr,v,nullptr,nullptr,nullptr,1,M,I);
        integer_dpas_stage<false>(ctx,sp,w,s,1,M,I,H);
        integer_dpas_projection<false>(ctx,w,s,v,hs.data_as<float>(),t,r,nullptr,down.data_as<float>(),1,M,I,H,1);
    }
    sycl::event Gather(const std::vector<int>& positions,int first,int count,const Access::ExpertView& view){
        std::array<int,M> ts{},rs{};std::array<ops::MoeExpertSlotDev,M> desc;
        for(int j=0;j<count;++j){ts[j]=view.tokens[positions[first+j]];rs[j]=view.ranks[positions[first+j]];}
        desc.fill(view.slots[positions[first]]);
        // Blocking copies keep these host arrays alive, and drain the previous
        // tile before reusing this execution slot's fixed descriptor buffers.
        ctx.memcpy_h2d(source_tokens.data(),ts.data(),M*4);ctx.memcpy_h2d(source_ranks.data(),rs.data(),M*4);
        ctx.memcpy_h2d(slots.data(),desc.data(),sizeof(desc));
        const auto* st=source_tokens.data_as<int>();const auto* sr=source_ranks.data_as<int>();
        const auto* input=view.x;const auto* mh=Access::ExpertIntermediate(backend).data_as<bf16>();const auto* md=Access::ExpertDown(backend).data_as<float>();
        auto* xp=x.data_as<bf16>();auto* rh=reference_h.data_as<bf16>();auto* rd=reference_down.data_as<float>();
        return ctx.queue().parallel_for(sycl::range<1>(M*H),[=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
            const int row=index[0]/H,col=index[0]%H;const bool valid=row<count;
            xp[index]=valid?input[st[row]*H+col]:bf16(0.f);
            rd[index]=valid?md[(st[row]*R+sr[row])*H+col]:0.f;
            if(col<I)rh[row*I+col]=valid?mh[(st[row]*R+sr[row])*I+col]:bf16(0.f);
        });
    }
    sycl::event Scatter(int count){
        const auto* src=down.data_as<float>();const auto* t=source_tokens.data_as<int>();const auto* r=source_ranks.data_as<int>();auto* dst=Access::ExpertDown(backend).data_as<float>();
        return ctx.queue().parallel_for(sycl::range<1>(count*H),[=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
            const int row=index[0]/H,col=index[0]%H;dst[(t[row]*R+r[row])*H+col]=src[index];
        });
    }
    struct Score {double sq=0,err=0,max_abs=0,max_ref=0;uint64_t violations=0;
        double nrmse()const{return std::sqrt(err/std::max(sq,1e-30));}
        void add(float ref,float value){uint32_t bits;std::memcpy(&bits,&value,4);expect((bits&0x7f800000U)!=0x7f800000U,"Nonfinite real expert output");
            double d=double(value)-ref;sq+=double(ref)*ref;err+=d*d;max_abs=std::max(max_abs,std::abs(d));max_ref=std::max(max_ref,std::abs(double(ref)));violations+=std::abs(d)>2e-4+.02*std::abs(ref);}
    };
    void Sample(int layer,int expert,const ops::MoeExpertSlotDev& slot){
        // Capture is bounded to two experts at each of layers 0/23/47.
        std::vector<bf16> xhost(M*H),href(M*I),hcand(M*I);std::vector<float> ref(M*H),cand(M*H),guh(M*2*I);
        ctx.memcpy_d2h(xhost.data(),x.data(),xhost.size()*2);ctx.memcpy_d2h(href.data(),reference_h.data(),href.size()*2);ctx.memcpy_d2h(ref.data(),reference_down.data(),ref.size()*4);
        ctx.queue().ext_oneapi_graph(*graph).wait_and_throw();
        ctx.memcpy_d2h(cand.data(),down.data(),cand.size()*4);ctx.memcpy_d2h(hcand.data(),h.data(),hcand.size()*2);ctx.memcpy_d2h(guh.data(),gu.data(),guh.size()*4);
        std::vector<int8_t> xq(M*H),hq(M*I);std::vector<float> sx(M*H/64),sh(M*I/64);
        ctx.memcpy_d2h(xq.data(),qx.data(),xq.size());ctx.memcpy_d2h(hq.data(),qh.data(),hq.size());ctx.memcpy_d2h(sx.data(),xs.data(),sx.size()*4);ctx.memcpy_d2h(sh.data(),hs.data(),sh.size()*4);
        auto qcheck=[&](const std::vector<bf16>& raw,const std::vector<int8_t>& quant,const std::vector<float>& scale,int width){
            for(int a=0;a<M;++a)for(int b=0;b<width/64;++b){float mx=0.f;
                for(int j=0;j<64;++j){uint16_t bits;std::memcpy(&bits,&raw[a*width+b*64+j],2);expect((bits&0x7f80U)!=0x7f80U,"Nonfinite real hidden input");mx=std::max(mx,std::abs(float(raw[a*width+b*64+j])));}
                expect(std::abs(scale[a*width/64+b]-mx/127.f)<1e-7f*mx+1e-12f,"Real Q8 scale mismatch");
                for(int j=0;j<64;++j)expect(int(quant[a*width+b*64+j])==integer_q8_reference(raw[a*width+b*64+j],bf16(mx)),"Real Q8 integer oracle mismatch");
            }
        };qcheck(xhost,xq,sx,H);qcheck(hcand,hq,sh,I);
        // Independent CPU check of real packed payload/scales and integer dots.
        std::vector<uint8_t> gw(2*I*H/2),dw(H*I/2);std::vector<float> gs(2*I*H/64),ds(H*I/64);
        auto copy=[&](const uint8_t* packed,const ops::MoeNestedScaleDev& ns,std::vector<uint8_t>& w,std::vector<float>& scales){
            ctx.memcpy_d2h(w.data(),packed,w.size());std::vector<uint8_t> codes(scales.size());std::vector<float> abs((codes.size()+ns.blocksize-1)/ns.blocksize),map(256);
            ctx.memcpy_d2h(codes.data(),ns.codes,codes.size());ctx.memcpy_d2h(abs.data(),ns.nested_absmax,abs.size()*4);ctx.memcpy_d2h(map.data(),ns.nested_quant_map,1024);
            for(size_t b=0;b<codes.size();++b){volatile float product=map[codes[b]]*abs[b/ns.blocksize];scales[b]=product+ns.offset;}
        };copy(slot.gate_up_packed,slot.gate_up_nested,gw,gs);copy(slot.down_packed,slot.down_nested,dw,ds);
        auto dot=[&](const std::vector<uint8_t>& w,const std::vector<float>& scale,int n,int K,const int8_t* q,const float* as){double v=0;
            for(int b=0;b<K/64;++b){int sum=0;for(int j=0;j<64;++j){const int k=n*K+b*64+j,c=(w[k/2]>>(k%2?0:4))&15;sum+=int(kIntegerNf4[c])*int(q[b*64+j]);}v+=double(sum)*scale[n*(K/64)+b]*as[b]/127.;}return v;};
        double max_cpu=0;for(int a:{0,M-1}){
            auto check=[&](double v,double got){max_cpu=std::max(max_cpu,std::abs(v-got));expect(std::abs(v-got)<=1e-6+1e-4*std::abs(v),"Real DPAS CPU integer reference");};
            for(int n=0;n<2*I;++n)check(dot(gw,gs,n,H,xq.data()+a*H,sx.data()+a*H/64),guh[a*2*I+n]);
            for(int n=0;n<H;++n)check(dot(dw,ds,n,I,hq.data()+a*I,sh.data()+a*I/64),cand[a*H+n]);
        }
        Score full;for(size_t j=0;j<ref.size();++j)full.add(ref[j],cand[j]);
        double x2=0,h2=0,xmax=0,hmax=0;for(auto v:xhost){double f=float(v);x2+=f*f;xmax=std::max(xmax,std::abs(f));}for(auto v:href){double f=float(v);h2+=f*f;hmax=std::max(hmax,std::abs(f));}
        std::cout<<"[R2B REAL] layer="<<layer<<" expert="<<expert<<" rows=32 x_rms="<<std::sqrt(x2/xhost.size())<<" x_max="<<xmax<<" h_rms="<<std::sqrt(h2/href.size())<<" h_max="<<hmax<<" dpas_nrmse="<<full.nrmse()<<" max_abs="<<full.max_abs<<" violations="<<full.violations<<" elements="<<ref.size()<<" integer_errors=0 max_cpu_abs="<<max_cpu<<"\n";
        // Self-contained replay corpus: little-endian header then real X, h,
        // down reference, packed weights and decoded per64 scales. No pointers.
        const auto path=prefix+".L"+std::to_string(layer)+".E"+std::to_string(expert)+".bin";
        std::ofstream file(path,std::ios::binary);expect(bool(file),"Real input artifact");
        const uint32_t header[8]={0x52324251U,1,uint32_t(layer),uint32_t(expert),M,H,I,64};file.write(reinterpret_cast<const char*>(header),sizeof(header));
        auto write=[&](const auto& v){file.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));};write(xhost);write(href);write(ref);write(gw);write(gs);write(dw);write(ds);expect(bool(file),"Real input artifact write");
        // Same full chain, changing one source of approximation at a time.
        // mode0: original LUT/raw X/raw h; 1 LUT only; 2 X-Q8 only;
        // 3 intermediate-Q8 only; 4 all approximations in float ALU.
        for(int mode=0;mode<5;++mode){
            const bool xquant=mode==2||mode==4,hquant=mode==3||mode==4;
            const float* lut=(mode==1||mode==4)?lutmap.data_as<float>():qmap.data_as<float>();
            if(xquant)integer_ablation_projection<true,true>(ctx,slots.data_as<ops::MoeExpertSlotDev>(),lut,x.data_as<bf16>(),qx.data_as<int8_t>(),xs.data_as<float>(),h.data_as<bf16>(),nullptr,M,H,I);
            else integer_ablation_projection<true,false>(ctx,slots.data_as<ops::MoeExpertSlotDev>(),lut,x.data_as<bf16>(),nullptr,nullptr,h.data_as<bf16>(),nullptr,M,H,I);
            if(hquant){integer_quantize<true>(ctx,h.data_as<bf16>(),qh.data_as<int8_t>(),hs.data_as<float>(),M,I,1);integer_ablation_projection<false,true>(ctx,slots.data_as<ops::MoeExpertSlotDev>(),lut,h.data_as<bf16>(),qh.data_as<int8_t>(),hs.data_as<float>(),nullptr,down.data_as<float>(),M,I,H);}
            else integer_ablation_projection<false,false>(ctx,slots.data_as<ops::MoeExpertSlotDev>(),lut,h.data_as<bf16>(),nullptr,nullptr,nullptr,down.data_as<float>(),M,I,H);
            std::vector<float> got(M*H);ctx.memcpy_d2h(got.data(),down.data(),got.size()*4);Score score;for(size_t j=0;j<got.size();++j)score.add(ref[j],got[j]);
            std::cout<<"[R2B ABLATION] layer="<<layer<<" expert="<<expert<<" mode="<<mode<<" nrmse="<<score.nrmse()<<" max_abs="<<score.max_abs<<" violations="<<score.violations<<"\n";
            if(mode==0)expect(score.nrmse()<1e-4 && score.max_abs<1e-6+1e-4*score.max_ref,"Ablation NF4 baseline mismatch");
        }
        // Validate streamed parameter tables/cursor and partial-tile scatter on
        // real model buffers, then restore NF4 before the model consumes them.
        auto& p=host_params[0];p.slot=slot;p.input=Access::FrontX(backend).data_as<bf16>();p.output=Access::ExpertDown(backend).data_as<float>();
        ctx.memcpy_d2h(p.token,source_tokens.data(),M*4);ctx.memcpy_d2h(p.rank,source_ranks.data(),M*4);
        for(int trial=0;trial<3;++trial){
            const int take=trial==1?3:M,entries=trial==2?2:1;
            p.count=trial==2?3:take;host_params[1]=p;host_params[1].count=M;
            ctx.memcpy_h2d(params.data(),host_params.data(),entries*sizeof(p));ctx.queue().memset(cursor.data(),0,4);
            for(int j=0;j<entries;++j)ctx.queue().ext_oneapi_graph(*stream_graph);pending_tiles=entries;CheckCompletion();
            const auto* pp=params.data_as<TileParams>();auto* check=verify_down.data_as<float>();auto* saved=reference_down.data_as<float>();
            ctx.queue().parallel_for(sycl::range<1>(M*H),[=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]]{
                const int row=index[0]/H,col=index[0]%H;const int dest=(pp[0].token[row]*R+pp[0].rank[row])*H+col;
                check[index]=pp[0].output[dest];pp[0].output[dest]=saved[index];
            });
            std::vector<float> streamed(M*H);ctx.memcpy_d2h(streamed.data(),verify_down.data(),streamed.size()*4);
            for(int row=0;row<M;++row)for(int col=0;col<H;++col){const int j=row*H+col;const float expected=row<take?cand[j]:ref[j];expect(std::memcmp(&streamed[j],&expected,4)==0,"Stream graph/scatter changed real expert values");}
        }
        std::cout<<"[R2B STREAM CHECK] layer="<<layer<<" expert="<<expert<<" full_and_partial=1 table_entries=2 cursor_exact=1 nf4_restored=1\n";

    }
    static sycl::event Hook(void* data,Qwen4ExpBnb4Backend& b,int layer,int count,int first,int acts,const sycl::event&){
        auto& self=*static_cast<IntegerQualityReplay*>(data);auto view=Access::Experts(b);
        for(auto& rows:self.buckets)rows.clear();std::vector<int> used;used.reserve(32);
        const auto* ids=Access::FrontIds(b);
        for(int a=first;a<first+acts;++a){const int expert=ids[view.tokens[a]*R+view.ranks[a]];expect(expert>=0 && expert<512,"Quality expert id");auto& v=self.buckets[expert];if(v.empty())used.push_back(expert);v.push_back(a);}
        self.total_acts+=acts;self.layer_acts[layer]+=acts;self.experts+=used.size();
        int batch_tiles=0;
        if(self.phase==1)self.CheckCompletion();
        for(int expert:used){auto& rows=self.buckets[expert];const int n=int(rows.size());if(n>=32)self.hot_acts+=n;
            if(self.phase==0 && count>32 && n>=32 && (layer==0||layer==23||layer==47) && self.samples[layer]<2){
                self.Gather(rows,0,M,view);self.Sample(layer,expert,view.slots[rows[0]]);++self.samples[layer];
            }
            if(self.phase==1)for(int begin=0;begin<n;begin+=M){
                expect(batch_tiles<MaxTiles,"Integer parameter table capacity");auto& p=self.host_params[batch_tiles++];p.slot=view.slots[rows[begin]];p.input=view.x;p.output=Access::ExpertDown(b).data_as<float>();
                p.count=std::min(M,n-begin);for(int j=0;j<p.count;++j){p.token[j]=view.tokens[rows[begin+j]];p.rank[j]=view.ranks[rows[begin+j]];}
                self.replaced+=p.count;++self.tiles;++self.layer_tiles[layer];
            }
        }
        if(self.phase==1 && batch_tiles){
            // CheckCompletion drained the previous readers before host_params
            // was reused. This table stays immutable until the next callback.
            self.ctx.queue().memcpy(self.params.data(),self.host_params.data(),batch_tiles*sizeof(TileParams));
            self.ctx.queue().memset(self.cursor.data(),0,4);
            for(int j=0;j<batch_tiles;++j)self.ctx.queue().ext_oneapi_graph(*self.stream_graph);
            self.pending_tiles=batch_tiles;
            if(self.layer_acts[layer]==uint64_t(count)*R)std::cout<<"[R2B INJECT QUEUED] layer="<<layer<<" rows="<<count<<" acts="<<self.layer_acts[layer]<<" tiles="<<self.layer_tiles[layer]<<"\n";
        }
        // A real fence also covers all-quad ALU facade calls returning {}.
        return self.ctx.queue().ext_oneapi_submit_barrier();
    }
    void CheckCompletion(){
        if(pending_tiles){int done=0;ctx.memcpy_d2h(&done,cursor.data(),4);expect(done==pending_tiles,"Integer graph cursor coverage");pending_tiles=0;}
    }
    void ResetCounts(){CheckCompletion();total_acts=replaced=tiles=hot_acts=experts=0;layer_acts.fill(0);layer_tiles.fill(0);}
    void Report(const char* label){std::cout<<"[R2B COVERAGE] phase="<<label<<" acts="<<total_acts<<" replaced="<<replaced<<" padded_tiles="<<tiles<<" hot32_acts="<<hot_acts<<" experts="<<experts<<" padded_work_acts="<<tiles*M<<"\n";}
};

void test_integer_quality(const std::string& directory,bool state_only=false){
    std::cout<<std::unitbuf<<std::setprecision(10);aila::ApplyQwen38A770Preset();
    _putenv_s("AILA_Q35_PREFILL_CHUNK","2048");
    InferenceEngine engine;expect(engine.init(directory,2560),"R2b quality model init");
    auto& ctx=Access::Ctx(engine);auto& b=Access::Backend(engine);
    IntegerQualityReplay replay(ctx,b,aila::env::read_string("AILA_Q4E_INTEGER_QUALITY_PREFIX","tmp/r2b-quality"));
    std::ifstream input(aila::env::read_string("AILA_Q4E_WINDOW_PROMPT_FILE",""));expect(bool(input),"R2b quality prompt");
    const std::string body((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
    GenerationConfig config;config.do_sample=false;config.thinking_budget_tokens=0;
    aila::chat::ChatRequest request;std::string error;
    expect(aila::chat::parse_chat_request_json(body,config,request,&error),"R2b quality request");
    expect(request.messages.size()==1 && request.messages[0].role==aila::chat::ChatRole::User,"R2b single user prompt");
    for(auto& part:request.messages[0].content){std::string flat;bool newline=false;for(char c:part.text){if(c=='\r'||c=='\n'){if(!newline)flat+=' ';newline=true;}else{flat+=c;newline=false;}}part.text=std::move(flat);}
    aila::chat::ChatMessage system;system.role=aila::chat::ChatRole::System;system.content.push_back(aila::chat::ChatContentPart::text_part("You are a helpful assistant."));request.messages.insert(request.messages.begin(),std::move(system));
    aila::chat::ChatFormatter formatter;aila::chat::ChatFormatTextResult rendered;
    expect(formatter.render_text(engine.make_chat_format_input(request),true,rendered,&error),"R2b format");
    Tokenizer tokenizer;expect(tokenizer.load(directory),"R2b tokenizer");auto prompt=tokenizer.encode(rendered.text);expect(prompt.size()>64 && prompt.size()<2400,"R2b prompt length");
    const auto target=tokenizer.encode("Begin by agreeing on stable identifiers and a single request register. Mara should define the permission decisions and choose a small set of pilot requests. Meanwhile, Jon can prepare the receiving room and arrange the wall inspection. The technician can check the cable route and rehearse restoring a sample backup before connecting the capture workflow to shared storage.");
    expect(target.size()>=48,"R2b teacher targets");auto id=Tensor::allocate(ctx,{state_only?257:1},dnnl::memory::data_type::s32);
    struct Drain{Context& c;~Drain(){try{c.synchronize();}catch(...){}}}drain{ctx};
    if(state_only){
        // Isolate the longer causal check from checkpoint/replay so each job
        // stays inside the runner's 240-second watchdog. Both retain count>1.
        static const auto part=aila::env::read_string(
            "AILA_Q4E_INTEGER_STATE_PART","causal");
        expect(part=="causal" || part=="checkpoint","Integer state part");
        const bool causal=part=="causal";
        const int count=causal?257:65,anchor=causal?129:33,tail=count-anchor;
        expect(prompt.size()>=size_t(count),"Integer state prompt too short");
        prompt.resize(count);replay.phase=1;Access::Grouped(b,std::numeric_limits<int>::max());Access::ExpertQualityProbe(b,&replay,&IntegerQualityReplay::Hook);
        auto& session=Access::Session(engine);
        auto upload=[&](const std::vector<int>& v){ctx.memcpy_h2d(id.data(),v.data(),v.size()*4);};
        auto prefill=[&](const std::vector<int>& v,bool split){
            replay.CheckCompletion();b.reset();replay.ResetCounts();upload(v);
            if(split){b.forward(ctx,id.data_as<int>(),anchor,{SnapshotPolicy::ReusablePrefix,false,true});b.forward(ctx,id.data_as<int>()+anchor,tail,{SnapshotPolicy::None,true});}
            else b.forward(ctx,id.data_as<int>(),count,{SnapshotPolicy::None,true});
            replay.CheckCompletion();expect(session.CurrentLength()==count,"Integer state length");
            expect(replay.replaced==uint64_t(count)*48*10 && replay.total_acts==replay.replaced,"Integer state coverage");
            replay.Report(split?"state_split":"state_whole");
        };
        auto hidden_prefix=[&]{std::vector<bf16> values(anchor*10240);ctx.memcpy_d2h(values.data(),Access::FrontHidden(b).data(),values.size()*2);return values;};
        auto kv_prefix=[&]{std::vector<bf16> values;
            for(const auto& c:session.layer_caches_for_testing())for(const Tensor* tensor:{&c.k,&c.v})if(tensor->valid())for(int head=0;head<tensor->shape(0);++head){
                const size_t old=values.size(),n=anchor*tensor->shape(2);values.resize(old+n);ctx.memcpy_d2h(values.data()+old,tensor->data_as<bf16>()+head*tensor->shape(1)*tensor->shape(2),n*2);
            }return values;};
        auto changed=prompt;for(int i=anchor;i<count;++i)changed[i]=target[i%target.size()];
        std::cout<<"[R2B STATE START] part="<<part<<" tokens="<<count<<" prefix="<<anchor<<"\n";
        if(causal){
            prefill(prompt,false);auto prefix_hidden=hidden_prefix(),prefix_kv=kv_prefix();
            prefill(changed,false);expect(same_bits(prefix_hidden,hidden_prefix()),"Integer whole-window prefix depends on future routing");expect(same_bits(prefix_kv,kv_prefix()),"Integer KV sees future routing");
            std::cout<<"[R2B STATE PASS] part=causal whole_window=257 prefix=129 hidden_exact=1 kv_exact=1 cursor_checked=1 format_policy=all_prefill_experts\n";
        }else{
            prefill(prompt,true);expect(session.snapshot_for_testing(anchor),"Integer checkpoint missing");
            auto saved=*session.snapshot_for_testing(anchor);auto saved_kv=kv_prefix();
            const auto original_logits=extract_bf16_to_f32(ctx,Access::Logits(b));
            prefill(changed,true);expect(session.snapshot_for_testing(anchor),"Integer changed checkpoint missing");
            expect_same_checkpoint(saved,*session.snapshot_for_testing(anchor));expect(same_bits(saved_kv,kv_prefix()),"Integer split KV changed");
            for(int pass=0;pass<2;++pass){
                replay.CheckCompletion();expect(b.Truncate(session,anchor)==anchor,"Integer truncate");
                replay.ResetCounts();upload(prompt);b.forward(ctx,id.data_as<int>()+anchor,tail,{SnapshotPolicy::None,true});replay.CheckCompletion();
                expect(session.CurrentLength()==count,"Integer replay length");
                expect(replay.replaced==uint64_t(tail)*48*10 && replay.total_acts==replay.replaced,"Integer replay coverage");
                expect(same_bits(original_logits,extract_bf16_to_f32(ctx,Access::Logits(b))),"Integer restored replay differs from original continuation");
                expect(same_bits(saved_kv,kv_prefix()),"Integer replay changed prefix KV");
                replay.Report("state_replay");
            }
            std::cout<<"[R2B STATE PASS] part=checkpoint tokens=65 prefix=33 snapshot_exact=1 kv_exact=1 replay_exact=1 original_continuation_exact=1 cursor_checked=1 format_policy=all_prefill_experts\n";
        }
        Access::ExpertQualityProbe(b,nullptr,nullptr);Access::Grouped(b,0);return;
    }
    std::vector<std::vector<float>> reference;CheckpointMetrics control,candidate;
    for(int mode=0;mode<3;++mode){
        b.reset();replay.ResetCounts();replay.phase=mode==2?1:0;
        Access::Grouped(b,mode?std::numeric_limits<int>::max():0);
        Access::ExpertQualityProbe(b,mode?&replay:nullptr,mode?&IntegerQualityReplay::Hook:nullptr);
        std::cout<<"[R2B MODEL START] mode="<<mode<<" prompt="<<prompt.size()<<"\n";
        const auto start=std::chrono::steady_clock::now();
        expect(engine.forward_prompt_tokens(prompt,0,int(prompt.size()),int(prompt.size())-4)!=nullptr,"R2b prefill");
        ctx.synchronize();replay.CheckCompletion();const auto prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        for(int step=0;step<48;++step){auto logits=extract_bf16_to_f32(ctx,Access::Logits(b));
            if(mode==0)reference.push_back(logits);
            else if(mode==1){control.add(reference[step],logits,target[step]);expect(same_bits(reference[step],logits),"R2b NF4 probe changed model logits");}
            else {
                const double kl=candidate.kl,nll=candidate.nll_delta;candidate.add(reference[step],logits,target[step]);
                const int ref_top=int(std::max_element(reference[step].begin(),reference[step].end())-reference[step].begin());
                const int cand_top=int(std::max_element(logits.begin(),logits.end())-logits.begin());
                std::cout<<"[R2B TEACHER] step="<<step<<" target="<<target[step]<<" kl="<<candidate.kl-kl<<" nll_delta="<<candidate.nll_delta-nll<<" top1_match="<<(ref_top==cand_top)<<"\n";
            }
            ctx.memcpy_h2d(id.data(),&target[step],sizeof(int));b.forward(ctx,id.data_as<int>(),1);
        }
        replay.Report(mode==0?"default_nf4":mode==1?"observed_nf4":"integer_prefill");
        std::cout<<"[R2B MODEL RUN] mode="<<mode<<" prompt="<<prompt.size()<<" teacher=48 prefill_ms="<<prefill_ms<<" performance_comparable=0\n";
        if(mode==1)expect(replay.samples[0]==2 && replay.samples[23]==2 && replay.samples[47]==2,"R2b insufficient real activation samples");
        if(mode==2)expect(replay.replaced==replay.total_acts && replay.total_acts==uint64_t(prompt.size())*48*10,"R2b missing prefill expert coverage");
    }
    control.report("R2b_probe_nf4_exact");candidate.report("R2b_all_prefill_integer_teacher48");
    const bool pass=candidate.kl/candidate.rows<=.01 && candidate.nll_delta/candidate.rows<=.01 && candidate.max_nrmse<=.15;
    std::cout<<"[R2B MODEL QUALITY] gate_pass="<<pass<<" kl_limit=0.01 nll_delta_limit=0.01 logits_nrmse_limit=0.15 prefill_all_experts=1 decode_nf4=1 production_enabled=0\n";
    Access::ExpertQualityProbe(b,nullptr,nullptr);Access::Grouped(b,0);
    std::cout<<"[R2B QUALITY DONE] arithmetic_checked=1 model_canary_pass="<<pass<<" hot_dispatch_not_integrated=1\n";
}
