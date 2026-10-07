void test_mtp_greedy(Context& ctx) {
    constexpr int vocab=248320;
    auto logits=Tensor::allocate(ctx,{4,vocab},dnnl::memory::data_type::bf16);
    auto scratch=Tensor::allocate(ctx,{4*mtp_probe::GreedyGroups*int64_t(sizeof(mtp_probe::GreedyResult))},dnnl::memory::data_type::u8);
    auto result=Tensor::allocate(ctx,{int64_t(5*sizeof(mtp_probe::GreedyResult))},dnnl::memory::data_type::u8);
    std::vector<bf16> host(vocab);
    std::mt19937 rng(8724); std::uniform_real_distribution<float> dist(-4,4);
    const auto live=ctx.device_allocation_count(), pinned=ctx.pinned_host_allocation_count();
    const mtp_probe::GreedyResult canary{123.0f,98765,0x12345678,0x87654321};
    for (int test=0;test<8;++test) {
        for(auto& v:host)v=bf16(dist(rng));
        if(test==1) {std::fill(host.begin(),host.end(),bf16(-3e38f));host.back()=bf16(-2e38f);}
        if(test==2) host.back()=bf16(99);
        if(test==3) for(int i:{3,256,65535,vocab-1})host[i]=bf16(99);
        if(test==4) {std::fill(host.begin(),host.end(),bf16(-1));host[7]=bf16(0.0f);host[2]=bf16(-0.0f);}
        if(test>=5) {uint16_t bits=test==5?0x7fc1:test==6?0x7f80:0xff80;std::memcpy(&host[test==5?vocab-1:0],&bits,sizeof(bits));}
        int expected=0;uint32_t bad=0;
        for(int i=0;i<vocab;++i) {
            uint16_t bits;std::memcpy(&bits,&host[i],sizeof(bits));bad|=uint32_t((bits&0x7f80)==0x7f80);
            if(float(host[i])>float(host[expected]))expected=i;
        }
        ctx.memcpy_h2d(logits.data(),host.data(),host.size()*sizeof(bf16));
        ctx.memcpy_h2d(static_cast<mtp_probe::GreedyResult*>(result.data())+1,&canary,sizeof(canary));
        mtp_probe::Greedy(ctx,logits.data_as<bf16>(),vocab,static_cast<mtp_probe::GreedyResult*>(scratch.data()),static_cast<mtp_probe::GreedyResult*>(result.data()));
        mtp_probe::GreedyResult out[2];ctx.memcpy_d2h(out,result.data(),sizeof(out));
        if(out[0].nonfinite!=bad || (!bad && (out[0].index!=expected || out[0].value!=float(host[expected]))) ||
            std::memcmp(&out[1],&canary,sizeof(canary)))throw std::runtime_error("MTP greedy mismatch/canary failure");
    }
    for(int rows:{2,4}) for(int invalid:{0,1}) {
        std::vector<bf16> batch(size_t(rows)*vocab);
        for(auto& v:batch)v=bf16(dist(rng));
        for(int r=0;r<rows;++r) for(int i:{3+r,256,65535,vocab-1})batch[size_t(r)*vocab+i]=bf16(20+r);
        if(invalid) {uint16_t bits=0x7fc1;std::memcpy(&batch.back(),&bits,sizeof(bits));}
        ctx.memcpy_h2d(logits.data(),batch.data(),batch.size()*sizeof(bf16));
        ctx.memcpy_h2d(static_cast<mtp_probe::GreedyResult*>(result.data())+rows,&canary,sizeof(canary));
        mtp_probe::Greedy(ctx,logits.data_as<bf16>(),vocab,static_cast<mtp_probe::GreedyResult*>(scratch.data()),static_cast<mtp_probe::GreedyResult*>(result.data()),rows);
        mtp_probe::GreedyResult out[5];ctx.memcpy_d2h(out,result.data(),size_t(rows+1)*sizeof(out[0]));
        for(int r=0;r<rows;++r) if(out[r].nonfinite!=uint32_t(invalid && r==rows-1) ||
            (!out[r].nonfinite && (out[r].index!=3+r || out[r].value!=float(20+r))))
            throw std::runtime_error("MTP greedy rows mixed results or finite flags");
        if(std::memcmp(&out[rows],&canary,sizeof(canary)))throw std::runtime_error("MTP greedy rows overrun");
    }
    if(ctx.device_allocation_count()!=live || ctx.pinned_host_allocation_count()!=pinned)throw std::runtime_error("MTP greedy grew USM");
    std::cout << "MTP greedy: vocab=248320 B1/B2/B4; random, negative extremes, tail, cross-WG ties, signed zeros, per-row NaN/Inf, canary PASS; live USM unchanged\n";
}
