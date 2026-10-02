#include "ops/QsaIndexerOps.hpp"

// Diagnostic only: checkpoint-backed Q4E MTP, dense-QSA range (positions < 640).
// Semantics source: vLLM qwen4_exp/nvidia/mtp.py, residual_linear_shared,
// scheme A. Target/draft share embedding and LM head, but never KV state.
struct Qwen4ExpMtpProbe {
    Context& ctx;
    Qwen4ExpBnb4Backend& target;
    ModelWeights& weights;
    Qwen4ExpLayer layer;
    Qwen4ExpHyperConnectionLayer mixer;
    Bnb4BitLinear fcEmbedding, fcHidden;
    Tensor* normEmbedding=nullptr;
    Tensor* normHidden=nullptr;
    Tensor embed, embedNorm, embedFc, hiddenNorm;
    Qwen4ExpLayerCache state;
    Qwen4ExpExpertCache experts;
    std::array<ops::MoeExpertSlotDev,10> slots{};
    std::array<int,10> slotIds{};
    std::vector<bf16> hostEmbed;
    HostTensorView embeddingTable;
    int capacity;
    size_t deviceBytes=0, hostBytes=0;

    static void Require(bool ok, const std::string& message) {
        if (!ok) throw std::runtime_error(message);
    }
    Tensor Allocate(int rows, int cols, dnnl::memory::data_type type=dnnl::memory::data_type::bf16) {
        return Tensor::allocate(ctx,{rows,cols},type);
    }
    Tensor* Norm(const std::string& name, bool addOneBf16=false) {
        auto v=weights.host_view(name);
        Require(v.valid() && v.dtype==DT_BF16,"MTP norm missing: "+name);
        std::vector<bf16> src(v.bytes/2);
        std::memcpy(src.data(),v.ptr,v.bytes);
        if (addOneBf16) {
            for (auto& x:src) x=bf16(float(x)+1.0f);
            auto t=Allocate(1,int(src.size())); ctx.memcpy_h2d(t.data(),src.data(),v.bytes);
            weights.put(name,std::move(t));
        } else {
            std::vector<float> f(src.size());
            std::transform(src.begin(),src.end(),f.begin(),[](bf16 x){return float(x);});
            auto t=Allocate(1,int(f.size()),dnnl::memory::data_type::f32);
            ctx.memcpy_h2d(t.data(),f.data(),f.size()*sizeof(float)); weights.put(name,std::move(t));
        }
        return &weights.get(name);
    }
    void Quant(const std::string& name, Bnb4BitLinear& p) {
        for (const auto* suffix:{"",".absmax",".quant_map",".nested_absmax",".nested_quant_map",".quant_state.bitsandbytes__nf4"})
            if (weights.has_host(name+suffix)) weights.upload(name+suffix,ctx);
        Bnb4BitWeightRef ref; std::string error;
        Require(LoadBnb4BitWeightRef(ctx,weights,name,ref,&error),error);
        Require(p.init(ctx,ref,&error),error);
    }
    void EraseFusedSource(const std::string& name) {
        // Fused projections own their packed rows/scales. Their quant_map can
        // still refer to the first source, matching the main backend loader.
        for(const auto* suffix:{"",".absmax",".nested_absmax",".nested_quant_map",".quant_state.bitsandbytes__nf4"})
            if(weights.has(name+suffix))weights.erase(name+suffix);
    }
    void Hc(const std::string& prefix,Qwen4ExpHyperConnectionLayer& hc,bool inject) {
        hc.hc_norm_weight=Norm(prefix+"hc_norm.weight");
        hc.mix_down_w=&weights.upload(prefix+"input_mix_weight_down.weight",ctx);
        hc.mix_up_w=&weights.upload(prefix+"input_mix_weight_up.weight",ctx);
        if (inject) hc.inject_w=&weights.upload(prefix+"block_inject_weight.weight",ctx);
        // Decode uses raw BF16 GEMV; no unused oneDNN primitives for these HCs.
    }
    explicit Qwen4ExpMtpProbe(InferenceEngine& engine,int cap)
        :ctx(Access::Ctx(engine)),target(Access::Backend(engine)),weights(Access::Weights(engine)),capacity(cap) {
        Require(cap>0 && cap<=640,"MTP probe only supports dense-QSA text positions <=640");
        const auto before=ctx.current_allocated_bytes(), hostBefore=ctx.current_pinned_host_bytes();
        normEmbedding=Norm("mtp.pre_fc_norm_embedding.weight");
        normHidden=Norm("mtp.pre_fc_norm_hidden.weight");
        Quant("mtp.fc_embedding.weight",fcEmbedding); Quant("mtp.fc_hidden.weight",fcHidden);
        Require(fcEmbedding.in_features()==2560 && fcEmbedding.out_features()==2560 &&
                fcHidden.in_features()==2560 && fcHidden.out_features()==2560,"Unexpected MTP FC geometry");
        Hc("mtp.layers.0.attn_hyper_connection.",layer.attn_hc,true);
        Hc("mtp.layers.0.mlp_hyper_connection.",layer.mlp_hc,true);
        Hc("mtp.hyper_connection_mixer.",mixer,false);
        Bnb4BitLinear q,k,v,g,u;
        Quant("mtp.layers.0.self_attn.q_proj.weight",q);
        Quant("mtp.layers.0.self_attn.k_proj.weight",k);
        Quant("mtp.layers.0.self_attn.v_proj.weight",v);
        std::string error;
        Require(layer.qkv_proj.init_fused_rows(ctx,q,k,v,&error),error);
        for(const auto* name:{"q_proj","k_proj","v_proj"})EraseFusedSource(std::string("mtp.layers.0.self_attn.")+name+".weight");
        Quant("mtp.layers.0.self_attn.o_proj.weight",layer.o_proj);
        layer.q_norm_weight=Norm("mtp.layers.0.self_attn.q_norm.weight",true);
        layer.k_norm_weight=Norm("mtp.layers.0.self_attn.k_norm.weight",true);
        layer.index_qk_weight=&weights.upload("mtp.layers.0.self_attn.indexer.index_qk_proj.weight",ctx);
        layer.index_k_weight=Tensor::view(ctx,layer.index_qk_weight->data_as<bf16>()+512*2560,{128,2560});
        layer.index_k_norm_weight=Norm("mtp.layers.0.self_attn.indexer.k_layernorm.weight",true);
        layer.router_w=&weights.upload("mtp.layers.0.mlp.gate.weight",ctx);
        layer.shared_expert_gate_w=&weights.upload("mtp.layers.0.mlp.shared_expert_gate.weight",ctx);
        Quant("mtp.layers.0.mlp.shared_expert.gate_proj.weight",g);
        Quant("mtp.layers.0.mlp.shared_expert.up_proj.weight",u);
        Require(layer.shared_gate_up_proj.init_fused_rows(ctx,g,u,&error),error);
        EraseFusedSource("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
        EraseFusedSource("mtp.layers.0.mlp.shared_expert.up_proj.weight");
        Quant("mtp.layers.0.mlp.shared_expert.down_proj.weight",layer.shared_down_proj);
        embed=Allocate(1,2560); embedNorm=Allocate(1,2560); embedFc=Allocate(1,2560);
        hiddenNorm=Allocate(4,2560); hostEmbed.resize(2560);
        embeddingTable=weights.host_view("model.language_model.embed_tokens.weight");
        Require(embeddingTable.valid() && embeddingTable.dtype==DT_BF16,"MTP shared embedding missing");
        state.k=Tensor::allocate(ctx,{2,cap,256},dnnl::memory::data_type::bf16);
        state.v=Tensor::allocate(ctx,{2,cap,256},dnnl::memory::data_type::bf16);
        state.indexer_k=Allocate(cap,128); state.indexer_compressed_k=Allocate((cap+3)/4,128);
        // Test either all-resident or a small GPU LRU. ReadFile and the cache's
        // existing bounded staging/fences handle misses; no CPU expert math.
        constexpr size_t slotBytes=2534400;
        const int mtpSlots=aila::env::read_int("AILA_Q4E_MTP_SLOTS",512);
        Require(mtpSlots==64 || mtpSlots==512,"Unsupported MTP cache probe size");
        // A separate trace prevents the draft cache from truncating the target
        // cache's file when both caches live in the same diagnostic process.
        Require(experts.init(ctx,weights,1,mtpSlots*slotBytes,512,2560,640,64,&error,true,".mtp"),error);
        Bnb4BitHostWeightRef gu,dn;
        Require(LoadBnb4BitHostRef(weights,"mtp.layers.0.mlp.experts.gate_up_proj",gu,&error,2560),error);
        Require(LoadBnb4BitHostRef(weights,"mtp.layers.0.mlp.experts.down_proj",dn,&error,640),error);
        experts.set_layer_expert_ref(0,gu,dn);
        Require(experts.num_slots()==mtpSlots,"MTP expert slot budget mismatch");
        // Touch all source experts before measurement in both modes. A small
        // LRU only retains its capacity; the following prompt replay warms it.
        for (int start=0;start<512;start+=16) {
            int ids[16], output[16]; for(int i=0;i<16;++i)ids[i]=start+i;
            experts.acquire(0,ids,16,output); experts.wait_copy();
        }
        ctx.synchronize();
        deviceBytes=ctx.current_allocated_bytes()-before;
        hostBytes=ctx.current_pinned_host_bytes()-hostBefore;
        std::cout<<"[MTP LOAD] device_bytes="<<deviceBytes<<" pinned_host_bytes="<<hostBytes
                 <<" gpu_expert_slots="<<experts.num_slots()<<" host_pool_bytes="<<experts.host_pool_bytes()
                 <<" target_slots="<<target.expert_cache_.num_slots()<<'\n';
    }
    ~Qwen4ExpMtpProbe() { try { experts.wait_prefetch(); experts.wait_copy(); ctx.synchronize(); } catch (...) {} }

    void Reset() {
        ctx.queue().memset(state.k.data(),0,state.k.size_bytes());
        ctx.queue().memset(state.v.data(),0,state.v.size_bytes());
        ctx.queue().memset(state.indexer_k.data(),0,state.indexer_k.size_bytes());
        ctx.queue().memset(state.indexer_compressed_k.data(),0,state.indexer_compressed_k.size_bytes());
        state.num_compressed_blocks=0;
    }
    // Input hidden is [4,H] from target/prior draft BEFORE its final mixer.
    // Position is the hidden's position; the input token ID is shifted by one.
    // Sequential overwriting of an attention-only suffix implements draft
    // rollback here: the visible length is always position+1, never capacity.
    void Dump(const std::string& prefix,const char* name,const Tensor& tensor,int count) {
        if(prefix.empty())return;
        std::vector<bf16> host(count);ctx.memcpy_d2h(host.data(),tensor.data(),host.size()*sizeof(bf16));
        std::ofstream out(prefix+name,std::ios::binary);out.write(reinterpret_cast<const char*>(host.data()),host.size()*sizeof(bf16));
        Require(bool(out),"MTP stage dump failed");
    }
    void AppendTargetKv(const bf16* hidden,int token,int position) {
        Step(hidden,token,position,false,"",true);
    }
    void Step(const bf16* hidden,int token,int position,bool head=true,const std::string& debug="",bool kvOnly=false) {
        Require(position>=0 && position<capacity,"MTP position outside probe capacity");
        auto& b=target.buf_; auto& scratch=target.linear_scratch_;
        Require(token>=0 && token<target.cfg_.vocab_size,"MTP token out of range");
        std::memcpy(hostEmbed.data(),embeddingTable.ptr+size_t(token)*2560*sizeof(bf16),2560*sizeof(bf16));
        // This blocking copy also protects the reusable host row on exceptions.
        ctx.memcpy_h2d(embed.data(),hostEmbed.data(),2560*sizeof(bf16));
        ops::grouped_rms_norm(ctx,embed.data_as<bf16>(),normEmbedding->data_as<float>(),1e-6f,
            embedNorm.data_as<bf16>(),1,1,2560,true);
        fcEmbedding.ForwardDecodeRows(ctx,scratch,embedNorm,embedFc,1);
        // Global 10240-wide norm, then the same FC on each 2560-wide stream.
        ops::grouped_rms_norm(ctx,hidden,normHidden->data_as<float>(),1e-6f,
            hiddenNorm.data_as<bf16>(),1,1,10240,true);
        fcHidden.ForwardDecodeRows(ctx,scratch,hiddenNorm,b.hidden4,4);
        auto* h=b.hidden4.data_as<bf16>(); auto* e=embedFc.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(10240),[=](sycl::id<1> i){h[i]=bf16(float(h[i])+float(e[i%2560]));});
        Dump(debug,".fc.bf16",b.hidden4,10240);
        target.RunHyperConnection(ctx,layer.attn_hc,1,true);
        Dump(debug,".attn_x.bf16",b.x,2560);
        layer.qkv_proj.ForwardDecodeRows(ctx,scratch,b.x,b.full_qkv,1);
        auto* p=b.full_qkv.data_as<bf16>();
        auto q=Tensor::view(ctx,p,{1,12288}),k=Tensor::view(ctx,p+12288,{1,512}),v=Tensor::view(ctx,p+12800,{1,512});
        ops::decode_prepare_qgkv_packed_partial(ctx,q,k,v,b.q,b.z,*layer.q_norm_weight,*layer.k_norm_weight,
            state.k,state.v,position,24,2,256,1e-6f,64,10000000.0f,true,nullptr,nullptr,nullptr,0,0,11,11,10);
        Dump(debug,".q.bf16",b.q,6144);Dump(debug,".k.bf16",k,512);
        Dump(debug,".v.bf16",v,512);Dump(debug,".gate.bf16",b.z,6144);
        ops::bf16_gemv_bf16(ctx,layer.index_k_weight,b.x.data_as<bf16>(),
            state.indexer_k.data_as<bf16>()+size_t(position)*128,128,2560);
        if ((position+1)%4==0) {
            ops::qsa_compress_key_blocks(ctx,state.indexer_k,*layer.index_k_norm_weight,state.indexer_compressed_k,
                position/4,1,10000000.0f,true,nullptr,nullptr,nullptr,0,0,11,11,10);
        }
        // Keep the original fused QKV and indexer arithmetic. Catch-up only
        // needs these persistent rows; neither attention nor MoE feeds them.
        // Scratch hidden4/x are intentionally not a completed draft output.
        if (kvOnly) return;
        ops::attention_decode(ctx,b.q,state.k,state.v,b.attn_out,b.decode_scores,24,2,256,position+1,&b.decode_attn_partials,&b.z);
        Dump(debug,".attn_out.bf16",b.attn_out,6144);
        layer.o_proj.ForwardDecodeRows(ctx,scratch,b.attn_out,b.y,1);
        target.RunHyperConnectionCombined(ctx,layer.mlp_hc,1,true,b.y.data_as<bf16>(),b.hc_inject_f32.data_as<float>());
        Dump(debug,".moe_x.bf16",b.x,2560);
        ops::bf16_gemv_bf16(ctx,*layer.router_w,b.x.data_as<bf16>(),b.router_scores.data_as<bf16>(),512,2560);
        Dump(debug,".router.bf16",b.router_scores,512);
        auto route=ops::router_softmax_topk_device(ctx,b.router_scores.data_as<bf16>(),nullptr,
            target.router_host_ids_,target.router_host_weights_,1,512,10,true);
        ops::bf16_gemv_bf16(ctx,*layer.shared_expert_gate_w,b.x.data_as<bf16>(),b.shared_gate_buf.data_as<bf16>(),1,2560);
        if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(ctx,layer.shared_gate_up_proj,b.x,b.shared_inter_buf,640)) {
            layer.shared_gate_up_proj.ForwardDecodeRows(ctx,scratch,b.x,b.shared_gate_up_buf,1);
            ops::moe_silu_gate_up_mul_flat(ctx,b.shared_gate_up_buf.data_as<bf16>(),b.shared_inter_buf.data_as<bf16>(),1,640);
        }
        layer.shared_down_proj.ForwardDecodeRows(ctx,scratch,b.shared_inter_buf,b.shared_out_buf,1);
        route.wait_and_throw();
        auto ready=experts.acquire(0,target.router_host_ids_,10,slotIds.data());
        experts.get_slots_dev(slotIds.data(),10,slots.data());
        ops::moe_decode_gate_up_swiglu_grouped(ctx,slots.data(),nullptr,b.x.data_as<bf16>(),b.moe_h.data_as<bf16>(),10,2560,640,64,ready);
        auto done=ops::moe_decode_down_weighted_accumulate(ctx,slots.data(),nullptr,target.router_host_weights_,b.moe_h.data_as<bf16>(),
            nullptr,b.y.data_as<bf16>(),10,640,2560,64,b.shared_out_buf.data_as<bf16>(),b.shared_gate_buf.data_as<bf16>());
        experts.record_consumer_reader_fences(slotIds.data(),10,done);
        ops::hc_inject_add(ctx,b.hidden4.data_as<bf16>(),b.y.data_as<bf16>(),b.hc_inject_f32.data_as<float>(),1,2560,4);
        if (head) {
            target.RunHyperConnection(ctx,mixer,1,false);
            target.lm_head_.forward(ctx,b.x,b.logits,1);
        }
    }

    void CheckTargetKv(const bf16* hidden,const std::vector<int>& tokens,int rows) {
        Require(rows>0 && rows+1<capacity && int(tokens.size())>rows+1,"Invalid KV equivalence range");
        auto copy=[&](const Tensor& t) {
            std::vector<uint8_t> out(t.size_bytes());ctx.memcpy_d2h(out.data(),t.data(),out.size());return out;
        };
        const std::array<Tensor*,4> caches{&state.k,&state.v,&state.indexer_k,&state.indexer_compressed_k};
        std::array<std::vector<uint8_t>,4> reference;
        const auto live=ctx.device_allocation_count(),pinned=ctx.pinned_host_allocation_count();
        Reset();
        for(int p=0;p<rows;++p)Step(hidden+size_t(p)*10240,tokens[p+1],p,false);
        for(size_t i=0;i<caches.size();++i)reference[i]=copy(*caches[i]);
        Step(hidden+size_t(rows)*10240,tokens[rows+1],rows);ctx.synchronize();
        // Only one output row is written; do not copy unused scratch tails.
        std::vector<bf16> refHidden(10240),actualHidden(10240);
        ctx.memcpy_d2h(refHidden.data(),target.buf_.hidden4.data(),refHidden.size()*sizeof(bf16));
        std::vector<bf16> refLogits(target.vocab_size()),actual(refLogits.size());
        ctx.memcpy_d2h(refLogits.data(),target.buf_.logits.data(),refLogits.size()*sizeof(bf16));
        Reset();const auto stats=experts.stats();
        for(int p=0;p<rows;++p)AppendTargetKv(hidden+size_t(p)*10240,tokens[p+1],p);
        for(size_t i=0;i<caches.size();++i)Require(copy(*caches[i])==reference[i],"KV-only changed persistent cache");
        Require(experts.stats().miss_count==stats.miss_count && experts.stats().hit_count==stats.hit_count &&
            experts.stats().total_transferred_bytes==stats.total_transferred_bytes,"KV-only accessed experts");
        Step(hidden+size_t(rows)*10240,tokens[rows+1],rows);ctx.synchronize();
        ctx.memcpy_d2h(actual.data(),target.buf_.logits.data(),actual.size()*sizeof(bf16));
        ctx.memcpy_d2h(actualHidden.data(),target.buf_.hidden4.data(),actualHidden.size()*sizeof(bf16));
        Require(std::memcmp(refHidden.data(),actualHidden.data(),10240*sizeof(bf16))==0 &&
            std::memcmp(refLogits.data(),actual.data(),actual.size()*sizeof(bf16))==0,"KV-only changed subsequent full draft");
        Require(ctx.device_allocation_count()==live && ctx.pinned_host_allocation_count()==pinned,"KV-only grew live USM");
        std::cout<<"[MTP KV EQUIVALENCE] {\"rows\":"<<rows<<",\"cache_bitwise\":true,\"continuation_bitwise\":true,\"expert_accesses\":0,\"live_usm_delta\":0}\n";
    }

    // One bounded device checkpoint, text-only SnapshotPolicy::None. KV and
    // indexer append tails are hidden by restored length/block count. This is
    // a transport/restore experiment, not accepted-prefix state selection.
    void CheckTargetRollback(InferenceEngine& engine,const std::vector<int>& prompt,const std::vector<int>& continuation) {
        Require(continuation.size()>=6,"MTP rollback probe needs six continuation tokens");
        auto& session=Access::Session(engine);
        struct Span {void* runtime;size_t offset,bytes;};
        std::vector<Span> spans;spans.reserve(97);size_t size=0;
        for(auto& cache:session.layerCaches_)for(auto* t:{&cache.linear_state,&cache.linear_conv_state})
            if(t->valid()){spans.push_back({t->data(),size,t->size_bytes()});size+=t->size_bytes();}
        spans.push_back({session.ple_state_.conv_state.data(),size,session.ple_state_.conv_state.size_bytes()});
        size+=session.ple_state_.conv_state.size_bytes();
        auto storage=Tensor::allocate(ctx,{int64_t(size)},dnnl::memory::data_type::u8);
        auto tokens=Tensor::allocate(ctx,{8},dnnl::memory::data_type::s32);
        struct Drain {Context& ctx;~Drain(){try{ctx.synchronize();}catch(...){}}} drain{ctx};
        ctx.memcpy_h2d(tokens.data(),continuation.data(),6*sizeof(int));
        std::vector<int> convHeads(session.layerCaches_.size()),blocks(convHeads.size());
        std::array<int,2> pleTokens{};int pleHead=0,length=0,noMiss=0;
        auto capture=[&] {
            length=session.currentLength_;noMiss=session.consecutive_no_miss_layers_;
            pleTokens=session.ple_state_.prev_tokens;pleHead=session.ple_state_.conv_head;
            for(size_t i=0;i<convHeads.size();++i){convHeads[i]=session.layerCaches_[i].linear_conv_head;blocks[i]=session.layerCaches_[i].num_compressed_blocks;}
            for(const auto& s:spans)ctx.queue().memcpy(storage.data_as<uint8_t>()+s.offset,s.runtime,s.bytes);
        };
        auto restore=[&] {
            for(const auto& s:spans)ctx.queue().memcpy(s.runtime,storage.data_as<uint8_t>()+s.offset,s.bytes);
            session.currentLength_=length;session.consecutive_no_miss_layers_=noMiss;
            session.ple_state_.prev_tokens=pleTokens;session.ple_state_.conv_head=pleHead;
            for(size_t i=0;i<convHeads.size();++i){auto& c=session.layerCaches_[i];c.linear_conv_head=convHeads[i];c.num_compressed_blocks=blocks[i];if(c.linear_state.valid())c.device_state_dirty=true;}
        };
        engine.reset_context();
        engine.forward_prompt_tokens(prompt,0,int(prompt.size()),0,true,{SnapshotPolicy::None,true});
        Access::VerifyRows(engine,tokens.data_as<int>(),1);ctx.synchronize();
        capture();ctx.synchronize();
        Require(length==int(prompt.size())+1,"MTP checkpoint anchor mismatch");
        const auto live=ctx.device_allocation_count(),pinned=ctx.pinned_host_allocation_count();
        std::vector<bf16> reference(4*target.vocab_size()),tail(target.vocab_size()),hidden(10240);
        Access::VerifyRows(engine,tokens.data_as<int>()+1,4);
        ctx.memcpy_d2h(reference.data(),target.buf_.logits.data(),reference.size()*sizeof(bf16));
        Access::VerifyRows(engine,tokens.data_as<int>()+5,1);
        ctx.memcpy_d2h(tail.data(),target.buf_.logits.data(),tail.size()*sizeof(bf16));
        ctx.memcpy_d2h(hidden.data(),target.buf_.hidden4.data(),hidden.size()*sizeof(bf16));
        restore();ctx.synchronize();
        Access::VerifyRows(engine,tokens.data_as<int>()+1,4);
        std::vector<bf16> actual(reference.size());ctx.memcpy_d2h(actual.data(),target.buf_.logits.data(),actual.size()*sizeof(bf16));
        Require(std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(bf16))==0,"Device rollback changed four-row logits");
        Access::VerifyRows(engine,tokens.data_as<int>()+5,1);
        actual.resize(tail.size());ctx.memcpy_d2h(actual.data(),target.buf_.logits.data(),actual.size()*sizeof(bf16));
        Require(std::memcmp(actual.data(),tail.data(),actual.size()*sizeof(bf16))==0,"Device rollback changed continuation logits");
        actual.resize(hidden.size());ctx.memcpy_d2h(actual.data(),target.buf_.hidden4.data(),actual.size()*sizeof(bf16));
        Require(std::memcmp(actual.data(),hidden.data(),actual.size()*sizeof(bf16))==0,"Device rollback changed hidden state");
        restore();ctx.synchronize();std::vector<double> capMs,restoreMs;
        auto now=[] {return std::chrono::steady_clock::now();};
        for(int sample=0;sample<23;++sample){
            auto t=now();capture();ctx.synchronize();double c=std::chrono::duration<double,std::milli>(now()-t).count();
            t=now();restore();ctx.synchronize();double r=std::chrono::duration<double,std::milli>(now()-t).count();
            if(sample>=3){capMs.push_back(c);restoreMs.push_back(r);}
        }
        Require(ctx.device_allocation_count()==live && ctx.pinned_host_allocation_count()==pinned,"Device rollback allocated USM during use");
        auto print=[](const auto& v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';};
        std::cout<<"[MTP STATE] {\"bytes\":"<<size<<",\"copies_per_direction\":"<<spans.size()<<",\"anchor\":"<<length<<",\"capture_ms\":";
        print(capMs);std::cout<<",\"restore_ms\":";print(restoreMs);std::cout<<",\"replay_bitwise\":true,\"transport_only\":true,\"live_usm_delta\":0}\n";
    }
};
