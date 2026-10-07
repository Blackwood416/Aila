#include "ops/QsaIndexerOps.hpp"
#include "Qwen4ExpGreedyProbe.hpp"

// Diagnostic only: checkpoint-backed Q4E MTP, guarded dense-QSA range.
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
        Require(cap>0 && cap<=2048 && cap<=target.cfg_.indexer_budget,
            "MTP probe requires dense-QSA text positions within the model indexer budget");
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
    void RunLive(InferenceEngine& engine, const std::vector<int>& prompt, const std::vector<int>& tokens,
        int maxNew, int liveWidth, Tensor& ids, Tensor& targetHidden, Tensor& draftHidden) {
        auto& draft = *this;
        const int promptCount = int(prompt.size());
        const bool liveProfile = aila::env::read_flag("AILA_Q4E_MTP_LIVE_PROFILE", false);
        const bool selectProfile = liveProfile || aila::env::read_flag("AILA_Q4E_MTP_SELECT_PROFILE", false);
        static const bool verifyGdnOrder = aila::env::read_flag("AILA_Q4E_VERIFY_GDN_ORDER", false);
        const int shadowPosition = aila::env::read_int("AILA_Q4E_MTP_SHADOW_POSITION", -1);
        const bool draftMargins = aila::env::read_flag("AILA_Q4E_MTP_DRAFT_MARGINS", false);
        const bool gpuGreedy = aila::env::read_flag("AILA_Q4E_MTP_GPU_GREEDY", false);
        const bool greedyRows = aila::env::read_flag("AILA_Q4E_MTP_GREEDY_ROWS", false);
        const bool skipDraftTail = aila::env::read_flag("AILA_Q4E_MTP_SKIP_DRAFT_TAIL", false);
        Require(!greedyRows || gpuGreedy, "Batched greedy needs GPU greedy");
        Require(!gpuGreedy || (!draftMargins && shadowPosition < 0), "GPU greedy probe excludes margin and shadow diagnostics");
        double lastDraftMargin = -1;
        double selectCopyMs = 0, selectCpuMs = 0;
        std::vector<bf16> hostLogits(target.vocab_size());
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto elapsed = [&](auto start) { return std::chrono::duration<double,std::milli>(now()-start).count(); };
        auto printInts = [](const auto& v) { std::cout << '['; for(size_t i=0;i<v.size();++i) { if(i)std::cout<<','; std::cout<<v[i]; } std::cout<<']'; };
        Require(liveWidth == 1 || liveWidth == 2 || liveWidth == 4, "MTP live width must be 1, 2 or 4");
        // Bounded end-to-end experiment: real proposals, greedy verification,
        // and snapshot/replay of a rejected suffix. No scheduler integration.
        auto& session = Access::Session(engine);
        struct Span { void* runtime; size_t offset, bytes; };
        std::vector<Span> spans; size_t stateBytes = 0;
        for (auto& cache : session.layerCaches_) for (auto* t : {&cache.linear_state, &cache.linear_conv_state})
            if (t->valid()) {spans.push_back({t->data(),stateBytes,t->size_bytes()});stateBytes+=t->size_bytes();}
        spans.push_back({session.ple_state_.conv_state.data(), stateBytes, session.ple_state_.conv_state.size_bytes()});
        stateBytes += session.ple_state_.conv_state.size_bytes();
        auto checkpoint = Tensor::allocate(ctx, {int64_t(stateBytes)}, dnnl::memory::data_type::u8);
        const bool stateOnly = aila::env::read_flag("AILA_Q4E_MTP_STATE_ONLY_REPLAY", false);
        uint64_t checkpointCopyBytes=0;
        Tensor linearInputs, plePrefix, routerInputs, greedyPartials, greedyResult;
        if (gpuGreedy) {
            greedyPartials = Tensor::allocate(ctx,{4 * mtp_probe::GreedyGroups * int64_t(sizeof(mtp_probe::GreedyResult))},dnnl::memory::data_type::u8);
            greedyResult = Tensor::allocate(ctx,{4 * int64_t(sizeof(mtp_probe::GreedyResult))},dnnl::memory::data_type::u8);
        }
        if (shadowPosition > 0)
            routerInputs = Tensor::allocate(ctx, {int64_t(target.layers_.size()), 2, target.hidden_size_}, dnnl::memory::data_type::bf16);
        if (stateOnly) {
            linearInputs = Tensor::allocate(ctx, {int64_t(target.layers_.size()), 4, target.linear_all_dim_}, dnnl::memory::data_type::bf16);
            plePrefix = Tensor::allocate(ctx, {4, 10240}, dnnl::memory::data_type::f32);
        }
        // Declared after storage: drain queued copies before its destructor.
        struct LiveDrain { Context& ctx; ~LiveDrain() { try { ctx.synchronize(); } catch (...) {} } } liveDrain{ctx};
        Require(!stateOnly || target.gdn_chunk_mode_ == 0, "MTP state-only replay requires the batched GDN path");
        std::vector<int> convHeads(session.layerCaches_.size()), blocks(convHeads.size());
        std::array<int, 2> pleTokens{}; int pleHead = 0, length = 0, noMiss = 0;
        auto capture = [&] {
            length = session.currentLength_; noMiss = session.consecutive_no_miss_layers_;
            pleTokens = session.ple_state_.prev_tokens; pleHead = session.ple_state_.conv_head;
            for (size_t i = 0; i < convHeads.size(); ++i) {
                convHeads[i] = session.layerCaches_[i].linear_conv_head;
                blocks[i] = session.layerCaches_[i].num_compressed_blocks;
            }
            for (const auto& s : spans) ctx.queue().memcpy(checkpoint.data_as<uint8_t>() + s.offset,s.runtime,s.bytes);
            checkpointCopyBytes+=stateBytes;
        };
        auto restore = [&] {
            for (const auto& s : spans) ctx.queue().memcpy(s.runtime, checkpoint.data_as<uint8_t>() + s.offset, s.bytes);
            session.currentLength_ = length; session.consecutive_no_miss_layers_ = noMiss;
            session.ple_state_.prev_tokens = pleTokens; session.ple_state_.conv_head = pleHead;
            for (size_t i = 0; i < convHeads.size(); ++i) {
                auto& c = session.layerCaches_[i]; c.linear_conv_head = convHeads[i]; c.num_compressed_blocks = blocks[i];
                if (c.linear_state.valid()) c.device_state_dirty = true;
            }
        };
        // Same lowest-index tie rule as the diagnostic argmax, without hashing
        // every logit inside the measured generation loop.
        auto select = [&](int row = 0, bool isDraft = false) {
            const auto copyStart = now();
            if (gpuGreedy) {
                mtp_probe::Greedy(ctx, Access::Logits(target).data_as<bf16>()+size_t(row)*target.vocab_size(), target.vocab_size(),
                    static_cast<mtp_probe::GreedyResult*>(greedyPartials.data()), static_cast<mtp_probe::GreedyResult*>(greedyResult.data()));
                mtp_probe::GreedyResult picked{};
                ctx.memcpy_d2h(&picked,greedyResult.data(),sizeof(picked));
                if (selectProfile) selectCopyMs += elapsed(copyStart); // GPU reduce + small result readback
                Require(!picked.nonfinite && picked.index>=0 && picked.index<target.vocab_size(),"Live MTP nonfinite/invalid greedy result");
                return picked.index;
            }
            ctx.memcpy_d2h(hostLogits.data(), Access::Logits(target).data_as<bf16>() + size_t(row) * target.vocab_size(), hostLogits.size() * sizeof(bf16));
            const auto cpuStart = now();
            if (selectProfile) selectCopyMs += std::chrono::duration<double,std::milli>(cpuStart-copyStart).count();
            int best = 0, second = -1;
            for (int i = 0; i < target.vocab_size(); ++i) {
                if (!std::isfinite(float(hostLogits[i]))) Require(false, "Live MTP nonfinite logits");
                if (float(hostLogits[i]) > float(hostLogits[best])) { second = best; best = i; }
                else if (draftMargins && isDraft && i != best &&
                    (second < 0 || float(hostLogits[i]) > float(hostLogits[second]))) second = i;
            }
            if (draftMargins && isDraft) lastDraftMargin = float(hostLogits[best]) - float(hostLogits[second]);
            if (selectProfile) selectCpuMs += elapsed(cpuStart);
            return best;
        };
        const auto requestStart = now();
        engine.reset_context();
        double pairPrepareMs=0;
        const bool pairPrepared=liveWidth==2 && aila::env::read_flag("AILA_Q4E_VERIFY_ATTN_PAIR",false);
        if(pairPrepared) {
            const auto start=now();target.PrepareDecodeVerification(ctx);
            pairPrepareMs=elapsed(start);
        }
        ctx.memcpy_h2d(ids.data(), prompt.data(), prompt.size() * sizeof(int));
        Access::MtpTargetPrompt(engine, ids.data_as<int>(), promptCount);
        ctx.queue().memcpy(targetHidden.data(), Access::FrontHidden(target).data(), size_t(promptCount) * 10240 * sizeof(bf16));
        int liveNext = select();
        const double promptMs = elapsed(requestStart);
        const auto draftInitStart = now();
        if (liveWidth > 1) {
            draft.Reset();
            for (int p = 0; p < promptCount - 1; ++p)
                draft.AppendTargetKv(targetHidden.data_as<bf16>() + size_t(p) * 10240, prompt[p + 1], p);
            ctx.synchronize();
        }
        const double draftInitMs = liveWidth > 1 ? elapsed(draftInitStart) : 0.0;
        const auto draftStatsStart = draft.experts.stats();
        const auto targetStatsStart = target.expert_executor_.stats();
        const auto liveAlloc = ctx.device_allocation_count(), livePinned = ctx.pinned_host_allocation_count();
        std::vector<int> actual = prompt, advances, acceptedLive;
        actual.reserve(promptCount + maxNew); advances.reserve(maxNew); acceptedLive.reserve(maxNew);
        std::vector<double> cycleMs, draftMs, verifyMs, commitMs;
        std::vector<double> firstDraftMargins; firstDraftMargins.reserve(maxNew);
        std::vector<int> verifiedWidths; verifiedWidths.reserve(maxNew);
        std::vector<double> checkpointMs, forwardMs, verifyCopyMs, verifyCpuMs;
        std::vector<double> draftCopyMs, draftCpuMs;
        draftCopyMs.reserve(maxNew); draftCpuMs.reserve(maxNew);
        checkpointMs.reserve(maxNew); forwardMs.reserve(maxNew); verifyCopyMs.reserve(maxNew); verifyCpuMs.reserve(maxNew);
        cycleMs.reserve(maxNew); draftMs.reserve(maxNew); verifyMs.reserve(maxNew); commitMs.reserve(maxNew);
        int lastCorrect = promptCount - 2, rejected = 0;
        uint64_t firstHotset = 0;
        const auto loopStart = now();
        while (int(actual.size()) < promptCount + maxNew) {
            const auto cycleStart = now();
            const int startPos = session.currentLength_, p = startPos - 1;
            const int width = std::min(liveWidth, promptCount + maxNew - int(actual.size()));
            for (int r = lastCorrect + 1; r < p; ++r)
                draft.AppendTargetKv(targetHidden.data_as<bf16>() + size_t(r) * 10240, actual[r + 1], r);
            std::array<int, 4> proposal{}, verified{}; proposal[0] = liveNext;
            const bf16* hidden = targetHidden.data_as<bf16>() + size_t(p) * 10240;
            double firstDraftMargin = -1;
            selectCopyMs = 0; selectCpuMs = 0;
            for (int d = 0; d < width - 1; ++d) {
                draft.Step(hidden, proposal[d], p + d);
                proposal[d + 1] = select(0, true);
                if (d == 0) firstDraftMargin = lastDraftMargin;
                if (!skipDraftTail || d + 1 < width - 1) {
                    ctx.queue().memcpy(draftHidden.data(), Access::FrontHidden(target).data(), 10240 * sizeof(bf16));
                    hidden = draftHidden.data_as<bf16>();
                }
            }
            // The last blocking select has completed all work when no tail copy remains.
            if (!skipDraftTail || width == 1) ctx.synchronize();
            draftMs.push_back(elapsed(cycleStart));
            if (selectProfile) { draftCopyMs.push_back(selectCopyMs); draftCpuMs.push_back(selectCpuMs); }
            if (draftMargins) firstDraftMargins.push_back(firstDraftMargin);
            verifiedWidths.push_back(width);
            lastCorrect = p;
            const auto verifyStart = now();
            const bool shadowWindow = startPos == shadowPosition && (width == 2 || liveWidth == 1) &&
                startPos + 1 < int(tokens.size());
            if (width > 1 || shadowWindow) capture();
            ctx.memcpy_h2d(ids.data(), proposal.data(), size_t(width) * sizeof(int));
            if (shadowWindow) {
                // Diagnostic only: identical incoming checkpoint and token inputs.
                // Re-run the real B2 below after restoring; these timings are polluted.
                if (liveWidth == 1) ctx.memcpy_h2d(ids.data(), tokens.data() + startPos, 2 * sizeof(int));
                std::vector<bf16> batched(size_t(2) * target.vocab_size());
                std::vector<std::vector<float>> batchedState(target.layers_.size());
                {
                    struct RouterCapture {
                        Qwen4ExpBnb4Backend& target;
                        ~RouterCapture() { target.verification_router_inputs_for_test_ = nullptr; }
                    } captureRouter{target};
                    target.verification_router_inputs_for_test_ = routerInputs.data_as<bf16>();
                    Access::VerifyRows(engine, ids.data_as<int>(), 2);
                }
                ctx.memcpy_d2h(batched.data(), Access::Logits(target).data(), batched.size() * sizeof(bf16));
                // Head-only comparison on the exact final B2 mixer input.
                Require(!target.lmHeadQ8Enabled_, "Same-input head shadow requires BF16 head");
                for (int r = 0; r < 2; ++r) {
                    auto x = Tensor::view(ctx, target.buf_.x.data_as<bf16>() + size_t(r) * target.hidden_size_, {1,target.hidden_size_});
                    auto y = Tensor::view(ctx, target.buf_.logits.data_as<bf16>() + size_t(r) * target.vocab_size(), {1,target.vocab_size()});
                    target.lm_head_.forward(ctx,x,y,1);
                    ctx.memcpy_d2h(hostLogits.data(),y.data(),hostLogits.size()*sizeof(bf16));
                    const bf16* b = batched.data() + size_t(r) * target.vocab_size();
                    size_t different=0; float maxAbs=0;
                    for (int i=0;i<target.vocab_size();++i) {
                        different += std::memcmp(b+i,hostLogits.data()+i,sizeof(bf16))!=0;
                        maxAbs=std::max(maxAbs,std::abs(float(b[i])-float(hostLogits[i])));
                    }
                    std::cout << "[MTP HEAD SHADOW] {\"position\":" << startPos+r << ",\"different\":" << different
                        << ",\"max_abs\":" << maxAbs << ",\"b2_1118\":" << float(b[1118]) << ",\"b2_1870\":" << float(b[1870])
                        << ",\"b1_1118\":" << float(hostLogits[1118]) << ",\"b1_1870\":" << float(hostLogits[1870]) << "}\n";
                }
                const int experts=target.cfg_.num_experts, topk=target.cfg_.num_experts_per_tok;
                std::vector<bf16> scores(2*experts), singleScores(2*experts);
                std::vector<int> routeIds(2*topk);
                std::vector<float> routeWeights(2*topk);
                for (size_t l=0;l<target.layers_.size();++l) {
                    auto& layer=target.layers_[l];
                    auto x=Tensor::view(ctx,routerInputs.data_as<bf16>()+l*2*target.hidden_size_,{2,target.hidden_size_});
                    target.DenseLinear(ctx,layer.router_gate,layer.router_w,x,target.buf_.router_scores,2,experts,target.hidden_size_);
                    auto route=[&] { ops::router_softmax_topk_device(ctx,target.buf_.router_scores.data_as<bf16>(),nullptr,
                        target.router_host_ids_,target.router_host_weights_,2,experts,topk,true).wait_and_throw(); };
                    route();
                    ctx.memcpy_d2h(scores.data(),target.buf_.router_scores.data(),scores.size()*sizeof(bf16));
                    std::copy_n(target.router_host_ids_,2*topk,routeIds.data());
                    std::copy_n(target.router_host_weights_,2*topk,routeWeights.data());
                    for (int r=0;r<2;++r) {
                        auto xr=Tensor::view(ctx,x.data_as<bf16>()+size_t(r)*target.hidden_size_,{1,target.hidden_size_});
                        auto yr=Tensor::view(ctx,target.buf_.router_scores.data_as<bf16>()+size_t(r)*experts,{1,experts});
                        target.DenseLinear(ctx,layer.router_gate,layer.router_w,xr,yr,1,experts,target.hidden_size_);
                    }
                    route();
                    ctx.memcpy_d2h(singleScores.data(),target.buf_.router_scores.data(),scores.size()*sizeof(bf16));
                    size_t different=0, idsDifferent=0; float maxAbs=0, weightDiff=0;
                    for (size_t i=0;i<scores.size();++i) {
                        different+=std::memcmp(&scores[i],&singleScores[i],sizeof(bf16))!=0;
                        maxAbs=std::max(maxAbs,std::abs(float(scores[i])-float(singleScores[i])));
                    }
                    for (size_t i=0;i<routeIds.size();++i) {
                        idsDifferent+=routeIds[i]!=target.router_host_ids_[i];
                        weightDiff=std::max(weightDiff,std::abs(routeWeights[i]-target.router_host_weights_[i]));
                    }
                    std::cout << "[MTP ROUTER SHADOW] {\"position\":" << startPos << ",\"layer\":" << l
                        << ",\"scores_different\":" << different << ",\"max_abs\":" << maxAbs
                        << ",\"ids_different\":" << idsDifferent << ",\"weight_max_abs\":" << weightDiff << "}\n";
                }
                for (size_t l = 0; l < batchedState.size(); ++l) {
                    auto& state = session.layerCaches_[l].linear_state;
                    if (!state.valid()) continue;
                    batchedState[l].resize(state.numel());
                    ctx.memcpy_d2h(batchedState[l].data(), state.data(), state.size_bytes());
                }
                restore();
                for (int r = 0; r < 2; ++r) {
                    Access::VerifyRows(engine, ids.data_as<int>() + r, 1);
                    select();
                    const auto* b = batched.data() + size_t(r) * target.vocab_size();
                    auto topTwo = [&](const bf16* x) {
                        std::array<int, 2> top{0, 1};
                        if (float(x[1]) > float(x[0])) std::swap(top[0], top[1]);
                        for (int i = 2; i < target.vocab_size(); ++i) {
                            if (float(x[i]) > float(x[top[0]])) { top[1] = top[0]; top[0] = i; }
                            else if (float(x[i]) > float(x[top[1]])) top[1] = i;
                        }
                        return top;
                    };
                    const auto bt = topTwo(b), st = topTwo(hostLogits.data());
                    size_t different = 0; float maxAbs = 0;
                    for (int i = 0; i < target.vocab_size(); ++i) {
                        different += std::memcmp(b + i, hostLogits.data() + i, sizeof(bf16)) != 0;
                        maxAbs = std::max(maxAbs, std::abs(float(b[i]) - float(hostLogits[i])));
                    }
                    std::cout << "[MTP SHADOW] {\"position\":" << startPos + r << ",\"row\":" << r
                        << ",\"incoming_width\":" << liveWidth
                        << ",\"logit_different\":" << different << ",\"max_abs\":" << maxAbs
                        << ",\"b2_top\":" << bt[0] << ",\"b1_top\":" << st[0]
                        << ",\"b2_margin\":" << float(b[bt[0]]) - float(b[bt[1]])
                        << ",\"b1_margin\":" << float(hostLogits[st[0]]) - float(hostLogits[st[1]])
                        << ",\"b2_1118\":" << float(b[1118]) << ",\"b2_1870\":" << float(b[1870])
                        << ",\"b1_1118\":" << float(hostLogits[1118]) << ",\"b1_1870\":" << float(hostLogits[1870]) << "}\n";
                }
                for (size_t l = 0; l < batchedState.size(); ++l) {
                    if (batchedState[l].empty()) continue;
                    std::vector<float> single(batchedState[l].size());
                    ctx.memcpy_d2h(single.data(), session.layerCaches_[l].linear_state.data(), single.size() * sizeof(float));
                    size_t different = 0; float maxAbs = 0;
                    for (size_t i = 0; i < single.size(); ++i) {
                        different += std::memcmp(&single[i], &batchedState[l][i], sizeof(float)) != 0;
                        maxAbs = std::max(maxAbs, std::abs(single[i] - batchedState[l][i]));
                    }
                    std::cout << "[MTP SHADOW STATE] {\"layer\":" << l << ",\"different\":" << different << ",\"max_abs\":" << maxAbs << "}\n";
                }
                restore();
                ctx.memcpy_h2d(ids.data(), proposal.data(), size_t(width) * sizeof(int));
            }
            // The existing blocking proposal upload already completes checkpoint copies.
            if (selectProfile) checkpointMs.push_back(elapsed(verifyStart));
            const auto forwardStart = now();
            {
                struct CaptureInputs {
                    Qwen4ExpBnb4Backend& target;
                    ~CaptureInputs() {
                        target.verification_linear_inputs_for_test_ = nullptr;
                    }
                } captureInputs{target};
                if (stateOnly && width > 1) target.verification_linear_inputs_for_test_ = linearInputs.data_as<bf16>();
                Access::VerifyRows(engine, ids.data_as<int>(), width);
            }
            if (liveProfile) { ctx.synchronize(); forwardMs.push_back(elapsed(forwardStart)); }
            ctx.queue().memcpy(targetHidden.data_as<bf16>() + size_t(startPos) * 10240,
                Access::FrontHidden(target).data(), size_t(width) * 10240 * sizeof(bf16));
            selectCopyMs = 0; selectCpuMs = 0;
            if (gpuGreedy && greedyRows && width > 1) {
                const auto begin = now();
                mtp_probe::Greedy(ctx,Access::Logits(target).data_as<bf16>(),target.vocab_size(),
                    static_cast<mtp_probe::GreedyResult*>(greedyPartials.data()),static_cast<mtp_probe::GreedyResult*>(greedyResult.data()),width);
                std::array<mtp_probe::GreedyResult,4> picked{};
                ctx.memcpy_d2h(picked.data(),greedyResult.data(),size_t(width)*sizeof(picked[0]));
                if (selectProfile) selectCopyMs += elapsed(begin);
                for (int r=0;r<width;++r) {
                    Require(!picked[r].nonfinite && picked[r].index>=0 && picked[r].index<target.vocab_size(),"Live MTP nonfinite/invalid greedy row");
                    verified[r]=picked[r].index;
                }
            } else for (int r = 0; r < width; ++r) verified[r] = select(r);
            if (selectProfile) { verifyCopyMs.push_back(selectCopyMs); verifyCpuMs.push_back(selectCpuMs); }
            verifyMs.push_back(elapsed(verifyStart));
            int advance = 1;
            while (advance < width && proposal[advance] == verified[advance - 1]) ++advance;
            liveNext = verified[advance - 1];
            const auto commitStart = now();
            if (advance < width) {
                ++rejected;
                if (stateOnly) {
                    // A <=4-token window cannot write the same PLE ring row
                    // twice. Preserve accepted writes before restoring state.
                    for (int r = 0; r < advance; ++r)
                        ctx.queue().memcpy(plePrefix.data_as<float>() + size_t(r) * 10240,
                            session.ple_state_.conv_state.data_as<float>() + size_t((pleHead + r) % 9) * 10240,
                            10240 * sizeof(float));
                    restore();
                    for (size_t l = 0; l < target.layers_.size(); ++l) {
                        auto& layer = target.layers_[l]; auto& cache = session.layerCaches_[l];
                        if (!layer.is_linear) {
                            cache.num_compressed_blocks = (startPos + advance) / std::max(1, target.cfg_.indexer_compress_ratio);
                            continue;
                        }
                        ops::DeltaNetWeights weights{layer.linear_conv1d_weight->data_as<bf16>(),
                            layer.linear_norm_weight->data_as<float>(), layer.linear_A_log->data_as<float>(),
                            layer.linear_dt_bias->data_as<bf16>()};
                        ops::DeltaNetState state{cache.linear_state.data_as<float>(), cache.linear_conv_state.data_as<float>(),
                            cache.linear_conv_head, true};
                        ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, weights, state,
                            linearInputs.data_as<bf16>() + l * 4 * target.linear_all_dim_,
                            target.buf_.attn_out.data_as<bf16>(), advance, verifyGdnOrder);
                        cache.linear_conv_head = state.conv_head; cache.device_state_dirty = state.device_state_dirty;
                    }
                    for (int r = 0; r < advance; ++r)
                        ctx.queue().memcpy(session.ple_state_.conv_state.data_as<float>() + size_t((pleHead + r) % 9) * 10240,
                            plePrefix.data_as<float>() + size_t(r) * 10240, 10240 * sizeof(float));
                    session.ple_state_.conv_head = (pleHead + advance) % 9;
                    std::array<int64_t, 64> ngrams{};
                    std::array<int, 2> prev{};
                    target.ple_.compute_ngram_ids_pure(session.ple_state_, proposal.data(), advance, ngrams.data(), prev);
                    session.ple_state_.prev_tokens = prev;
                    session.currentLength_ = startPos + advance;
                    session.consecutive_no_miss_layers_ = 0;
                } else {
                    restore();
                    Access::VerifyRows(engine, ids.data_as<int>(), advance);
                    ctx.queue().memcpy(targetHidden.data_as<bf16>() + size_t(startPos) * 10240,
                        Access::FrontHidden(target).data(), size_t(advance) * 10240 * sizeof(bf16));
                    liveNext = select(advance - 1);
                }
            }
            bool eos = false;
            for (int r = 0; r < advance; ++r) {
                actual.push_back(proposal[r]);
                if (proposal[r] == 248044 || proposal[r] == 248046) { eos = true; break; }
            }
            ctx.synchronize(); commitMs.push_back(elapsed(commitStart));
            advances.push_back(advance); acceptedLive.push_back(advance - 1); cycleMs.push_back(elapsed(cycleStart));
            if (advances.size() == 1) firstHotset = Access::Cache(engine).FrozenIdentity();
            if (eos) break;
        }
        const double wallMs = elapsed(loopStart);
        const double promptToOutputMs = elapsed(requestStart);
        Require(ctx.device_allocation_count() == liveAlloc && ctx.pinned_host_allocation_count() == livePinned, "Live MTP grew USM allocations");
        std::vector<int> mismatch;
        for (size_t i = promptCount; i < std::min(actual.size(), tokens.size()); ++i)
            if (actual[i] != tokens[i]) mismatch.push_back(int(i) - promptCount);
        std::vector<int> generated(actual.begin() + promptCount, actual.end());
        std::cout << "[MTP LIVE] {\"width\":" << liveWidth << ",\"tokens\":" << generated.size()
                  << ",\"state_only_replay\":" << (stateOnly ? "true" : "false")
                  << ",\"diagnostic_shadow\":" << (shadowPosition > 0 ? "true" : "false")
                  << ",\"gpu_greedy\":" << (gpuGreedy ? "true" : "false")
                  << ",\"greedy_rows\":" << (greedyRows ? "true" : "false")
                  << ",\"skip_draft_tail\":" << (skipDraftTail ? "true" : "false")
                  << ",\"checkpoint_copy_bytes\":" << checkpointCopyBytes
                  << ",\"target_kv_fp8\":" << (session.layerCaches_[3].k.dtype()==dnnl::memory::data_type::f8_e4m3 ? "true" : "false")
                  << ",\"pair_prepare\":" << (pairPrepared ? "true" : "false")
                  << ",\"pair_prepare_ms\":" << pairPrepareMs
                  << ",\"draft_loop_misses\":" << draft.experts.stats().miss_count - draftStatsStart.miss_count
                  << ",\"draft_loop_transfer_bytes\":" << draft.experts.stats().total_transferred_bytes - draftStatsStart.total_transferred_bytes
                  << ",\"target_loop_cpu_tasks\":" << target.expert_executor_.stats().cpu_tasks_completed - targetStatsStart.cpu_tasks_completed
                  << ",\"target_loop_cpu_compute_ns\":" << target.expert_executor_.stats().cpu_compute_ns - targetStatsStart.cpu_compute_ns
                  << ",\"target_loop_cpu_join_ns\":" << target.expert_executor_.stats().cpu_join_wait_ns - targetStatsStart.cpu_join_wait_ns
                  << ",\"wall_ms\":" << wallMs << ",\"tokens_s\":" << generated.size() * 1000.0 / wallMs
                  << ",\"target_prompt_ms\":" << promptMs << ",\"draft_kv_init_ms\":" << draftInitMs
                  << ",\"prompt_to_output_ms\":" << promptToOutputMs
                  << ",\"rejected_cycles\":" << rejected << ",\"advance\":"; printInts(advances);
        std::cout << ",\"accepted_prefix\":"; printInts(acceptedLive);
        std::cout << ",\"verified_width\":"; printInts(verifiedWidths);
        std::cout << ",\"cycle_ms\":"; printInts(cycleMs);
        std::cout << ",\"draft_ms\":"; printInts(draftMs);
        std::cout << ",\"verify_ms\":"; printInts(verifyMs);
        std::cout << ",\"replay_ms\":"; printInts(commitMs);
        if (draftMargins) { std::cout << ",\"first_draft_margin\":"; printInts(firstDraftMargins); }
        if (liveProfile) {
            std::cout << ",\"profile_serialized\":true";
            std::cout << ",\"forward_ms\":"; printInts(forwardMs);
        }
        if (selectProfile) {
            std::cout << ",\"checkpoint_input_ms\":"; printInts(checkpointMs);
            std::cout << ",\"select_copy_ms\":"; printInts(verifyCopyMs);
            std::cout << ",\"select_cpu_ms\":"; printInts(verifyCpuMs);
            std::cout << ",\"draft_select_copy_ms\":"; printInts(draftCopyMs);
            std::cout << ",\"draft_select_cpu_ms\":"; printInts(draftCpuMs);
        }
        std::cout << ",\"generated\":"; printInts(generated);
        std::cout << ",\"hotset_first\":\"" << firstHotset << "\",\"hotset_last\":\"" << Access::Cache(engine).FrozenIdentity() << "\"";
        std::cout << ",\"single_row_target_mismatches\":"; printInts(mismatch);
        std::cout << ",\"live_usm_delta\":0,\"accepted_prefix_commit\":true,\"production\":false}\n";
    }

};
