#include "models/Qwen4ExpHostArena.hpp"
#include "memory/LargePages.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib,"Psapi.lib")
#endif

static void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        Context ctx;
        const size_t initial_host=ctx.current_pinned_host_bytes(), initial_device=ctx.current_allocated_bytes();
        aila::HostArenaConfig cfg;
        cfg.num_layers=2; cfg.num_experts_per_layer=4; cfg.hidden_size=256; cfg.moe_intermediate_size=64;
        aila::Qwen4ExpHostArena arena;
        std::string error;
        require(arena.Allocate(ctx,cfg,&error),error.c_str());
        require(arena.ResidentExperts()==8,"RAM expert count");
        require(arena.TotalAllocatedBytes()==aila::memory::LargePageAllocationSize(8*arena.SlotBytes()),"rounded native accounting");
        require(ctx.current_pinned_host_bytes()==initial_host+arena.TotalAllocatedBytes()+arena.SlotBytes(),"Context accounting");
        auto* source=arena.GetExpertPointer(1,3);
        const auto expected_usm_kind = ctx.is_large_host_registered(source)
            ? sycl::usm::alloc::host
            : sycl::usm::alloc::unknown;
        require(sycl::get_pointer_type(source,ctx.queue().get_context())==expected_usm_kind,"arena USM registration state mismatch");
#ifdef _WIN32
        PSAPI_WORKING_SET_EX_INFORMATION info{}; info.VirtualAddress=source;
        require(QueryWorkingSetEx(GetCurrentProcess(),&info,sizeof(info)) && info.VirtualAttributes.Valid &&
                info.VirtualAttributes.LargePage,"native LargePage bit");
#endif
        for (size_t i=0;i<arena.SlotBytes();++i) source[i]=uint8_t(i*17+91);
        auto* device=ctx.alloc_device(arena.SlotBytes()+16);
        ctx.queue().memset(device,0xCA,arena.SlotBytes()+16).wait_and_throw();
        sycl::queue copy(ctx.queue().get_context(),ctx.queue().get_device(),sycl::property::queue::in_order{});
        auto dma=arena.DmaTransferToGpu(copy,device,1,3);
        // The source is CPU-only; DMA must own an immutable small staging copy.
        std::memset(source,0xEE,arena.SlotBytes());
        dma.wait_and_throw();
        std::vector<uint8_t> result(arena.SlotBytes()+16);
        ctx.memcpy_d2h(result.data(),device,result.size());
        for (size_t i=0;i<arena.SlotBytes();++i) require(result[i]==uint8_t(i*17+91),"full DMA payload");
        for (size_t i=arena.SlotBytes();i<result.size();++i) require(result[i]==0xCA,"DMA guard");
        bool rejected=false;
        try { arena.LoadLayerDirect(0,Bnb4BitHostWeightRef{},Bnb4BitHostWeightRef{}); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"missing nested weights must not become zero weights");
        aila::Qwen4ExpHostArena moved(std::move(arena));
        require(!arena.IsAllocated() && moved.GetExpertPointer(1,3)==source,"move ownership");
        moved.Release(); ctx.free_device(device);
        require(ctx.current_pinned_host_bytes()==initial_host && ctx.current_allocated_bytes()==initial_device,"cleanup accounting");
        std::cout << "PASS: native large pages, rounded accounting, DMA ownership/guard, invalid weights, move/release\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
