// Included inside the tolerance-test namespace. No runtime graph dispatch is enabled.
void test_front_graph(const std::string& directory) {
    namespace ex = sycl::ext::oneapi::experimental;
    std::cout << std::unitbuf;
    InferenceEngine engine;
    expect(engine.init(directory,512),"Front graph model init failed");
    auto& ctx=Access::Ctx(engine); auto& b=Access::Backend(engine); auto& q=ctx.queue();
    const bool limited=q.get_device().has(sycl::aspect::ext_oneapi_limited_graph);
    const bool full=q.get_device().has(sycl::aspect::ext_oneapi_graph);
    std::cout<<"[FRONT GRAPH CAPS] limited="<<limited<<" full="<<full<<'\n';
    if(!limited&&!full) {std::cout<<"[FRONT GRAPH SKIP] static graph unsupported\n";return;}
    const auto& g=Access::FrontGeometry(b);
    expect(g.conv_rows==3,"Graph fixture expects three conv heads");
    auto state=Tensor::allocate(ctx,{int64_t(g.v_heads)*g.head_k_dim*g.head_v_dim},dnnl::memory::data_type::f32);
    auto history=Tensor::allocate(ctx,{int64_t(g.conv_rows)*g.qkv_dim},dnnl::memory::data_type::f32);
    std::vector<float> initial(state.numel()), initial_history(history.numel());
    std::vector<bf16> input(4*2560);
    using Executable=ex::command_graph<ex::graph_state::executable>;
    std::array<std::unique_ptr<Executable>,3> graphs;
    // All host input arrays precede the drain guard, and every graph precedes tensor destruction.
    struct Drain {Context& c;~Drain(){try{c.synchronize();}catch(...){}}} drain{ctx};
    auto reset=[&](int seed) {
        for(size_t j=0;j<initial.size();++j) initial[j]=float(int((j*17+seed*31)%101)-50)*.0002f;
        for(size_t j=0;j<initial_history.size();++j) initial_history[j]=float(int((j*7+seed*13)%101)-50)*.002f;
        for(size_t j=0;j<input.size();++j) input[j]=bf16(float(int((j*11+seed*3)%103)-51)*.015f);
        ctx.memcpy_h2d(state.data(),initial.data(),state.size_bytes());
        ctx.memcpy_h2d(history.data(),initial_history.data(),history.size_bytes());
        ctx.memcpy_h2d(Access::FrontHidden(b).data(),input.data(),input.size()*2);
        return ops::DeltaNetState{state.data_as<float>(),history.data_as<float>(),seed%3,false};
    };
    auto ds=reset(0);Access::Front(b,ds);ctx.synchronize(); // all ordinary lazy preparation first
    for(int head=0;head<3;++head) {
        ds=reset(head);
        ex::command_graph<ex::graph_state::modifiable> recording(q.get_context(),q.get_device());
        recording.begin_recording(q);
        try {Access::Front(b,ds);recording.end_recording(q);}
        catch(...) {recording.end_recording(q);throw;}
        graphs[head]=std::make_unique<Executable>(recording.finalize());
    }
    auto execute=[&](ops::DeltaNetState& s,int steps,bool graph) {
        for(int i=0;i<steps;++i) {
            if(graph) {q.ext_oneapi_graph(*graphs[s.conv_head]);s.conv_head=(s.conv_head+1)%3;s.device_state_dirty=true;}
            else Access::Front(b,s);
        }
        ctx.synchronize();
    };
    auto snapshot=[&]() {
        std::vector<uint8_t> bytes;
        auto device=[&](const void* p,size_t n) {const auto offset=bytes.size();bytes.resize(offset+n);ctx.memcpy_d2h(bytes.data()+offset,p,n);};
        auto host=[&](const void* p,size_t n) {const auto* v=static_cast<const uint8_t*>(p);bytes.insert(bytes.end(),v,v+n);};
        device(state.data(),state.size_bytes());device(history.data(),history.size_bytes());
        device(Access::FrontHidden(b).data(),4*2560*2);device(Access::FrontX(b).data(),2560*2);
        device(Access::FrontScores(b).data(),512*2);
        device(Access::FrontInject(b).data(),4*sizeof(float));
        host(Access::FrontIds(b),10*sizeof(int));host(Access::FrontWeights(b),10*sizeof(float));
        return bytes;
    };
    for(int seed=0;seed<3;++seed) {
        const int steps=seed==0?1:seed==1?2:7;
        ds=reset(seed);execute(ds,steps,false);auto reference=snapshot();const int head=ds.conv_head;
        ds=reset(seed);execute(ds,steps,true);auto actual=snapshot();
        expect(head==ds.conv_head&&same_bits(reference,actual),"Graph front output/state/router parity failed");
        std::cout<<"[FRONT GRAPH CHECK] seed="<<seed<<" steps="<<steps<<" head="<<head<<" state_router_output=bitwise\n";
    }
    for(int repeat=0;repeat<3;++repeat) {
        std::array<std::vector<double>,2> times;
        for(int warm=0;warm<4;++warm) {ds=reset(warm);execute(ds,36,warm%2);}
        for(int pair=0;pair<20;++pair) for(int order=0;order<2;++order) {
            const int mode=(pair+order+repeat)%2;
            ds=reset(pair%3);
            const auto t0=std::chrono::steady_clock::now();execute(ds,36,mode);
            times[mode].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/36);
        }
        for(int mode=0;mode<2;++mode) std::cout<<"[FRONT GRAPH TIME] repeat="<<repeat<<" graph="<<mode
            <<" samples=20 steps=36 ms_per_front="<<x3_quantile(times[mode],.5)
            <<" p10="<<x3_quantile(times[mode],.1)<<" p90="<<x3_quantile(times[mode],.9)<<'\n';
    }
    std::cout<<"[FRONT GRAPH MEMORY] device="<<ctx.current_allocated_bytes()<<" host="<<ctx.current_pinned_host_bytes()<<'\n';
}
