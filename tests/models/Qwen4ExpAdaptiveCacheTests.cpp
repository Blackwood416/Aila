#include "models/Qwen4ExpAdaptiveCache.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using bf16=sycl::ext::oneapi::bfloat16;
static void require(bool ok,const char* what) { if (!ok) throw std::runtime_error(what); }

struct Weight {
    std::vector<uint8_t> packed,codes;
    std::vector<float> scales,map,qmap;
    int rows,cols,experts;
    Weight(int n,int r,int c):rows(r),cols(c),experts(n) {
        packed.resize(size_t(n)*r*c/2); codes.resize(size_t(n)*r*c/64);
        scales.resize(codes.size()/256); map.resize(256); qmap.resize(16);
        for (size_t i=0;i<packed.size();++i) packed[i]=uint8_t(i*13+i/17+3);
        for (size_t i=0;i<codes.size();++i) codes[i]=uint8_t(i*7+i/29);
        for (size_t i=0;i<scales.size();++i) scales[i]=0.003f+0.00001f*float(i%97)+0.01f*float(i/(size_t(r)*c/64/256));
        for (int i=0;i<256;++i) map[i]=0.5f+i/256.0f;
    }
    Bnb4BitHostWeightRef Ref() const {
        Bnb4BitHostWeightRef r;
        r.name="synthetic";
        r.packed_weight={packed.data(),{experts,rows,cols/2},DT_U8,packed.size()};
        r.absmax={codes.data(),{int64_t(codes.size())},DT_U8,codes.size()};
        r.nested_absmax={reinterpret_cast<const uint8_t*>(scales.data()),{int64_t(scales.size())},DT_F32,scales.size()*4};
        r.nested_quant_map={reinterpret_cast<const uint8_t*>(map.data()),{256},DT_F32,map.size()*4};
        r.quant_map={reinterpret_cast<const uint8_t*>(qmap.data()),{16},DT_F32,qmap.size()*4};
        r.quant_state.quant_type="nf4"; r.quant_state.shape={experts,rows,cols};
        r.quant_state.blocksize=64; r.quant_state.nested=true; r.quant_state.nested_blocksize=256;
        r.quant_state.nested_offset=0.002f;
        return r;
    }
};

static void check_payload(const uint8_t* p,int e,const aila::ExpertSlotGeometry& g,const Weight& gu,const Weight& dn) {
    require(std::memcmp(p,gu.packed.data()+size_t(e)*g.gu_packed_bytes,g.gu_packed_bytes)==0,"gate/up payload");
    require(std::memcmp(p+g.gu_absmax_offset,gu.codes.data()+size_t(e)*g.gu_absmax_count,g.gu_absmax_bytes)==0,"gate/up codes");
    require(std::memcmp(p+g.dn_packed_offset,dn.packed.data()+size_t(e)*g.dn_packed_bytes,g.dn_packed_bytes)==0,"down payload");
    require(std::memcmp(p+g.dn_absmax_offset,dn.codes.data()+size_t(e)*g.dn_absmax_count,g.dn_absmax_bytes)==0,"down codes");
}

static double sample_percentile(const std::vector<double>& values, double fraction) {
    auto sorted=values;
    std::sort(sorted.begin(),sorted.end());
    const size_t index=static_cast<size_t>(fraction*double(sorted.size()-1));
    return sorted[index];
}

static double sample_median(const std::vector<double>& values) {
    auto sorted=values;
    std::sort(sorted.begin(),sorted.end());
    return (sorted[sorted.size()/2-1]+sorted[sorted.size()/2])*0.5;
}

static int run_staging_copy_microbench() {
    _putenv_s("AILA_Q4E_COLD_MODE","1");
    _putenv_s("AILA_Q4E_CPU_THREADS","18");
    _putenv_s("AILA_Q4E_CPU_TILE_POOL","0");
    constexpr int H=2560,I=640,count=16;
    Context ctx;
    aila::HostArenaConfig hc; hc.num_layers=1; hc.num_experts_per_layer=count;
    hc.hidden_size=H; hc.moe_intermediate_size=I;
    aila::Qwen4ExpHostArena arena;
    std::string error;
    require(arena.Allocate(ctx,hc,&error),error.c_str());
    const size_t slot_bytes=arena.SlotBytes();
    require(slot_bytes==2534400,"Q4E staging microbench requires the production compact slot geometry");
    std::array<const uint8_t*,count> sources{};
    for (int e=0;e<count;++e) {
        auto* slot=arena.GetExpertPointer(0,e);
        require(slot!=nullptr,"large-page arena source slot");
        for (size_t j=0;j<slot_bytes;++j)
            slot[j]=uint8_t((j*131u+size_t(e)*47u+(j>>8)*17u)&0xffu);
        sources[e]=slot;
    }
    const size_t bytes=size_t(count)*slot_bytes;
    auto* destination=static_cast<uint8_t*>(ctx.alloc_host(bytes));
    struct HostRelease {
        Context& ctx; void* ptr;
        ~HostRelease() { if (ptr) ctx.free_host(ptr); }
    } release{ctx,destination};

    Qwen4ExpExpertCache cpu_cache;
    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx,cpu_cache,1,1,H,I);
    require(executor.cpu_overlap_enabled(),"bounded CPU worker arena must be enabled");

    auto run=[&](bool parallel) {
        const auto begin=std::chrono::steady_clock::now();
        const bool used_parallel=executor.CopyColdStagingHostSlots(
            sources.data(),destination,slot_bytes,count,parallel);
        const auto end=std::chrono::steady_clock::now();
        require(used_parallel==parallel,"requested staging copy path was not exercised");
        for (int e=0;e<count;++e)
            require(std::memcmp(destination+size_t(e)*slot_bytes,sources[e],slot_bytes)==0,
                    "staging microbench byte comparison");
        return std::chrono::duration<double,std::milli>(end-begin).count();
    };
    (void)run(false); (void)run(true); // warm both paths before interleaved samples
    std::vector<double> serial_ms,parallel_ms;
    serial_ms.reserve(20); parallel_ms.reserve(20);
    for (int batch=0;batch<20;++batch) {
        if ((batch&1)==0) {
            serial_ms.push_back(run(false)); parallel_ms.push_back(run(true));
        } else {
            parallel_ms.push_back(run(true)); serial_ms.push_back(run(false));
        }
    }
    std::cout << "[Q4E staging memcpy microbench] geometry=2560x640 slot_bytes=" << slot_bytes
              << " experts=" << count << " bytes_per_batch=" << bytes
              << " source=native-large-page destination=HostUSM interleaved_batches=20"
              << " serial_median_ms=" << sample_median(serial_ms)
              << " serial_p10_ms=" << sample_percentile(serial_ms,0.10)
              << " serial_p90_ms=" << sample_percentile(serial_ms,0.90)
              << " parallel_median_ms=" << sample_median(parallel_ms)
              << " parallel_p10_ms=" << sample_percentile(parallel_ms,0.10)
              << " parallel_p90_ms=" << sample_percentile(parallel_ms,0.90) << '\n';
    return 0;
}

int main(int argc,char** argv) {
    try {
        if (argc==2 && std::strcmp(argv[1],"--staging-copy-microbench-only")==0)
            return run_staging_copy_microbench();
        if (argc!=1) throw std::invalid_argument("Unknown AdaptiveCacheTests option");
        _putenv_s("AILA_Q4E_STAGE_COPY_PARALLEL","1");
        _putenv_s("AILA_Q4E_CPU_TILE_POOL","0");
        _putenv_s("AILA_Q4E_COLD_MODE","2");
        _putenv_s("AILA_Q4E_CPU_MAX_EXPERTS","16");
        _putenv_s("AILA_Q4E_CPU_THREADS","18");
        _putenv_s("AILA_Q4E_CPU_NF4_Q8","1");
        _putenv_s("AILA_Q4E_CPU_Q8_ACTIVATION64","1");
        _putenv_s("AILA_Q4E_CPU_SCALE_SHIFT","1");
        Context ctx;
        sycl::queue copy(ctx.queue().get_context(),ctx.queue().get_device(),sycl::property::queue::in_order{});
        constexpr int H=2560,I=640,E=4;
        Weight gu(E,2*I,H),dn(E,H,I);
        aila::AdaptiveCacheConfig cfg; cfg.num_layers=1; cfg.num_experts_per_layer=E;
        cfg.num_gpu_slots=2; cfg.staging_slots_per_buffer=2;
        aila::Qwen4ExpAdaptiveCache cache;
        std::string error; require(cache.Allocate(ctx,cfg,&error),error.c_str());
        const auto allocated_residency=cache.ResidencyGeneration();
        const auto allocated_descriptor=cache.DescriptorGeneration();
        require(allocated_residency>0 && allocated_descriptor>0,
                "allocation must publish cache generations");
        const std::vector<std::pair<int,int>> initial={{0,3},{0,1}};
        const std::vector<std::pair<int,int>> duplicate={{0,3},{0,3}};
        bool duplicate_rejected=false;
        try { (void)cache.InitializeResidency(&duplicate); }
        catch (const std::invalid_argument&) { duplicate_rejected=true; }
        require(duplicate_rejected,"duplicate initial resident expert must be rejected");
        require(cache.ResidencyGeneration()>allocated_residency &&
                cache.DescriptorGeneration()>allocated_descriptor,
                "failed residency edit must invalidate graph-facing generations");
        auto owners=cache.InitializeResidency(&initial);
        const auto initialized_residency=cache.ResidencyGeneration();
        const auto initialized_descriptor=cache.DescriptorGeneration();
        require(initialized_residency>allocated_residency &&
                initialized_descriptor>allocated_descriptor,
                "successful residency init must advance generations");
        require(cache.GetResidentEntry(0).layer==0 && cache.GetResidentEntry(0).expert_id==3 &&
                cache.GetResidentEntry(1).layer==0 && cache.GetResidentEntry(1).expert_id==1,
                "ranked initial prefix must preserve slot order");
        aila::HostArenaConfig hc; hc.num_layers=1; hc.num_experts_per_layer=E;
        aila::Qwen4ExpHostArena arena;
        require(arena.Allocate(ctx,hc,&error,owners),error.c_str());
        arena.LoadLayerDirect(0,gu.Ref(),dn.Ref());
        const auto before_publish=cache.DescriptorGeneration();
        cache.FinishInitialLoad(ctx.queue());
        require(cache.DescriptorGeneration()>before_publish,
                "device slot-table publication must advance descriptor generation");
        require(arena.ResidentExperts()==2 && !arena.GetExpertPointer(0,3) && !arena.GetExpertPointer(0,1) &&
                arena.GetExpertPointer(0,0) && arena.GetExpertPointer(0,2),"GPU/RAM complement follows ranked prefix");
        const auto& g=arena.Geometry();
        std::vector<uint8_t> payload(g.slot_bytes);
        for (int e=0;e<E;++e) {
            require((owners[e]!=nullptr)!=(arena.GetExpertPointer(0,e)!=nullptr),"each expert has exactly one initial owner");
            if (owners[e]) ctx.memcpy_d2h(payload.data(),owners[e],payload.size());
            else std::memcpy(payload.data(),arena.GetExpertPointer(0,e),payload.size());
            check_payload(payload.data(),e,g,gu,dn);
        }

        // Keep the nontrivial ranked-prefix check independent from the legacy
        // numerical and swap cases below, which use experts 0/1 as residents.
        arena.Release();
        const std::vector<std::pair<int,int>> baseline={{0,0},{0,1}};
        owners=cache.InitializeResidency(&baseline);
        require(cache.GetResidentEntry(0).expert_id==0 && cache.GetResidentEntry(1).expert_id==1,
                "baseline resident set for subsequent numerical checks");
        require(arena.Allocate(ctx,hc,&error,owners),error.c_str());
        arena.LoadLayerDirect(0,gu.Ref(),dn.Ref());cache.FinishInitialLoad(ctx.queue());

        // Independent device copies of original tensors form the reference;
        // distinct per-expert scales catch an omitted expert index immediately.
        std::vector<void*> allocations;
        auto upload=[&](const void* p,size_t bytes) {
            auto* d=ctx.alloc_device(bytes); allocations.push_back(d); ctx.memcpy_h2d(d,p,bytes); return d;
        };
        auto* gu_w=static_cast<uint8_t*>(upload(gu.packed.data(),gu.packed.size()));
        auto* dn_w=static_cast<uint8_t*>(upload(dn.packed.data(),dn.packed.size()));
        auto* gu_c=static_cast<uint8_t*>(upload(gu.codes.data(),gu.codes.size()));
        auto* dn_c=static_cast<uint8_t*>(upload(dn.codes.data(),dn.codes.size()));
        auto* gu_s=static_cast<float*>(upload(gu.scales.data(),gu.scales.size()*4));
        auto* dn_s=static_cast<float*>(upload(dn.scales.data(),dn.scales.size()*4));
        auto* gu_m=static_cast<float*>(upload(gu.map.data(),gu.map.size()*4));
        auto* dn_m=static_cast<float*>(upload(dn.map.data(),dn.map.size()*4));
        std::array<ops::MoeExpertSlotDev,E> reference{};
        for (int e=0;e<E;++e) {
            auto& d=reference[e];
            d.gate_up_packed=gu_w+size_t(e)*g.gu_packed_bytes; d.down_packed=dn_w+size_t(e)*g.dn_packed_bytes;
            d.gate_up_nested={gu_c+size_t(e)*g.gu_absmax_count,gu_s+size_t(e)*g.gu_absmax_count/256,gu_m,0.002f,256};
            d.down_nested={dn_c+size_t(e)*g.dn_absmax_count,dn_s+size_t(e)*g.dn_absmax_count/256,dn_m,0.002f,256};
        }
        std::vector<bf16> x(H); for (int j=0;j<H;++j) x[j]=bf16((int(j%29)-14)*0.01f);
        auto* xd=static_cast<bf16*>(upload(x.data(),x.size()*2));
        auto* hidden=static_cast<bf16*>(ctx.alloc_device(E*I*2)); allocations.push_back(hidden);
        auto* down=static_cast<float*>(ctx.alloc_device((E*H+16)*4)); allocations.push_back(down);
        ctx.queue().fill(down,0.0f,E*H+16);
        const std::array<int,E> ranks={0,1,2,3};
        auto launch=[&](const std::array<ops::MoeExpertSlotDev,E>& slots,const sycl::event* ready) {
            auto e=ops::moe_decode_gate_up_swiglu_grouped(ctx,slots.data(),nullptr,xd,hidden,E,H,I,64,ready,nullptr,true);
            return ops::moe_decode_down_to_rank_f32(ctx,slots.data(),ranks.data(),hidden,down,E,E,I,H,64,nullptr,nullptr,&e,true);
        };
        launch(reference,nullptr).wait_and_throw();
        std::vector<float> expected(E*H),actual(2*E*H);
        ctx.memcpy_d2h(expected.data(),down,expected.size()*4);
        const auto& meta=arena.GetLayerNestedMetadata(0);
        aila::Qwen4ExpExpertExecutor cpu;
        Qwen4ExpExpertCache retired_cache;
        cpu.init(ctx,retired_cache,1,E,H,I);
        for (int round=0;round<2;++round) {
            const int cold[2]={round ? 3:2,round ? 2:3};
            const auto before_stage_residency=cache.ResidencyGeneration();
            const auto before_stage_descriptor=cache.DescriptorGeneration();
            auto staged=cache.StageColdExpertsAsync(arena,copy,0,cold,2,0,&cpu);
            require(cache.ResidencyGeneration()==before_stage_residency &&
                    cache.DescriptorGeneration()>before_stage_descriptor,
                    "staging changes descriptors but not resident mapping");
            std::array<ops::MoeExpertSlotDev,E> slots{};
            slots[0]=cache.GetResidentSlotDev(cache.GetSlot(0,0),&meta);
            slots[1]=cache.GetResidentSlotDev(cache.GetSlot(0,1),&meta);
            for (int j=0;j<2;++j) slots[cold[j]]=cache.GetStagingSlotDev(0,j,&meta);
            auto done=launch(slots,&staged.ready_event);
            cache.RecordStagingConsumer(0,done); cache.RecordResidentConsumer(done);
            // Leave the first invocation outstanding. The next invocation
            // reuses bank 0 on a different queue and must respect its reader.
            ctx.queue().memcpy(actual.data()+round*E*H,down,expected.size()*4);
            if (round==0) continue;
            ctx.synchronize();
            for (int row=0;row<2;++row)
                require(std::memcmp(actual.data()+row*E*H,expected.data(),expected.size()*4)==0,"mixed resident/staged expert numerical output");
            std::array<float,16> guard{};
            ctx.memcpy_d2h(guard.data(),down+E*H,sizeof(guard));
            for (float value:guard) require(value==0,"expert down guard");
        }
        // Replace the sole GPU owner, then chain a second replacement. No mmap
        // reader exists and the original host payload must survive in the new owner.
        const auto check_cpu=[&](int first) {
            const int ids[2]={first,3}, rank[2]={3,1};
            std::array<ops::cpu::CpuExpertTask,2> original{};
            for (int i=0;i<2;++i) {
                const int e=ids[i]; auto& d=original[i].slot;
                d.gate_up_packed=gu.packed.data()+size_t(e)*g.gu_packed_bytes;
                d.down_packed=dn.packed.data()+size_t(e)*g.dn_packed_bytes;
                d.gate_up_nested={gu.codes.data()+size_t(e)*g.gu_absmax_count,
                    gu.scales.data()+size_t(e)*g.gu_absmax_count/256,gu.map.data(),0.002f,256};
                d.down_nested={dn.codes.data()+size_t(e)*g.dn_absmax_count,
                    dn.scales.data()+size_t(e)*g.dn_absmax_count/256,dn.map.data(),0.002f,256};
            }
            std::vector<bf16> cpu_h(2*I); std::vector<float> cpu_expected(2*H),cpu_actual(E*H);
            ops::cpu::cpu_expert_batch_forward(original.data(),2,x.data(),1,cpu_h.data(),cpu_expected.data(),nullptr,
                ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2,nullptr,true);
            ctx.queue().fill(down,-3.0f,E*H);
            cpu.StageResidentInput(xd);
            require(cpu.StartArenaCold(arena,0,ids,rank,2,xd)==2,"native CPU batch dispatch");
            cpu.JoinResidentCold(down);
            ctx.memcpy_d2h(cpu_actual.data(),down,cpu_actual.size()*4);
            for (int i=0;i<2;++i)
                require(std::memcmp(cpu_actual.data()+rank[i]*H,cpu_expected.data()+i*H,H*4)==0,"native CPU result vs original source");
            for (int r:{0,2}) for (int j=0;j<H;++j) require(cpu_actual[r*H+j]==-3.0f,"CPU rank guard");
        };
        check_cpu(2);
        const auto host_bytes=ctx.current_pinned_host_bytes(),device_bytes=ctx.current_allocated_bytes();
        for (const auto& plan:std::array<aila::AdaptiveSwapPlan,2>{{{0,2,0,0,5},{0,0,2,0,5}}}) {
            const auto before_swap_residency=cache.ResidencyGeneration();
            const auto before_swap_descriptor=cache.DescriptorGeneration();
            cache.ExecuteSwapsAsync(arena,copy,{plan}).wait_and_throw();
            require(cache.ResidencyGeneration()>before_swap_residency &&
                    cache.DescriptorGeneration()>before_swap_descriptor,
                    "resident swap must advance both graph-facing generations");
            require(!arena.GetExpertPointer(0,plan.candidate_expert),"incoming GPU sole owner");
            check_payload(arena.GetExpertPointer(0,plan.victim_expert),plan.victim_expert,g,gu,dn);
            ctx.memcpy_d2h(payload.data(),cache.GetSlotDevicePtr(0),payload.size());
            check_payload(payload.data(),plan.candidate_expert,g,gu,dn);
            check_cpu(plan.victim_expert);
        }
        require(ctx.current_pinned_host_bytes()==host_bytes && ctx.current_allocated_bytes()==device_bytes,"no steady-state USM allocations");
        for (auto* d:allocations) ctx.free_device(d);
        const auto before_release_residency=cache.ResidencyGeneration();
        const auto before_release_descriptor=cache.DescriptorGeneration();
        cache.Release(); arena.Release();
        require(cache.ResidencyGeneration()>before_release_residency &&
                cache.DescriptorGeneration()>before_release_descriptor,
                "release must invalidate graph-facing generations");
        // Full key-space uniqueness without allocating or transferring 58 GiB.
        aila::AdaptiveCacheConfig full; full.hidden_size=256; full.moe_intermediate_size=64;
        require(cache.Allocate(ctx,full,&error),error.c_str());
        auto full_owners=cache.InitializeResidency();
        size_t seen=0;
        for (int l=0;l<48;++l) for (int e=0;e<512;++e) {
            const int slot=cache.GetSlot(l,e); if (slot<0) continue;
            const auto reverse=cache.GetResidentEntry(slot);
            require(reverse.layer==l && reverse.expert_id==e,"initial hotset is not bijective"); ++seen;
        }
        require(seen==2800,"all GPU slots must contain distinct experts");
        std::cout << "PASS: GPU/CPU numerical sources, staging fences, chained RAM/GPU exchange, 2800 unique slots\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
