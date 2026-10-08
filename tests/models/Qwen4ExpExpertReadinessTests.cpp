#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/SafeTensors.hpp"

#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <future>
#include <cstring>
#include <thread>
#include <limits>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

struct Qwen4ExpResidentTestAccess {
    static void LargePages(Qwen4ExpExpertCache& cache) {
        cache.hotset_swaps_=0;cache.InitializeLargePageResidency();
    }
    static void ResidentHotset(Qwen4ExpExpertCache& cache,const std::vector<int>& keys) {
        cache.hotset_keys_=keys;cache.InstallHotset(keys);
    }
    static uint8_t* ResidentData(Qwen4ExpExpertCache& cache) { return cache.resident_data_; }
    static size_t StagingBytes(const Qwen4ExpExpertCache& cache) {
        return cache.staging_buffer_bytes_ + size_t(cache.prefetch_staging_slots_) * cache.slot_bytes_;
    }
    static void GpuMappedDma(Qwen4ExpExpertCache& cache,bool enabled) {
        cache.gpu_mmap_dma_enabled_=enabled;
        if(enabled) {
            cache.gpu_mmap_dma_first_events_.reserve(cache.num_experts_per_layer_);
            cache.batch_dma_events_.reserve(size_t(cache.num_experts_per_layer_)*2);
        }
    }
    static sycl::event HoldCopyQueue(Qwen4ExpExpertCache& cache,std::shared_future<void> signal) {
        return cache.copy_q_->submit([&](sycl::handler& cgh) {
            cgh.host_task([signal] {signal.wait_for(std::chrono::seconds(10));});
        });
    }
    static sycl::event MaintenanceAcquire(Qwen4ExpExpertCache& cache,int layer,const int* ids,int count,int* slots) {
        return cache.acquire_impl(layer,ids,count,slots,false);
    }
    static void MappedRefs(Qwen4ExpExpertCache& cache, int layer, const Bnb4BitHostWeightRef& gu, const Bnb4BitHostWeightRef& dn) {
        cache.layer_weights_[layer].gate_up=gu; cache.layer_weights_[layer].down=dn; cache.layer_weights_[layer].valid=true;
    }
    static void HostComplement(Qwen4ExpExpertCache& cache, bool enabled, int scan) {
        cache.host_complement_ = enabled; cache.host_complement_scan_ = scan;
    }
    static void ReserveHost(Qwen4ExpExpertCache& cache, int h, bool reserved) {cache.host_slot_reserved_[h] = reserved;}
    static sycl::event HoldHostSource(Qwen4ExpExpertCache& cache, int h, std::shared_future<void> signal, bool source = true) {
        auto event = cache.copy_q_->submit([&](sycl::handler& cgh) {
            cgh.host_task([signal] { signal.wait_for(std::chrono::seconds(10)); });
        });
        if (source) cache.host_slot_records_[h].source_reader_fence = event;
        else cache.host_slot_dma_events_[h] = event;
        return event;
    }
    static uint64_t ChangeGpuKey(Qwen4ExpExpertCache& cache, int g, uint64_t key) {
        const auto old = cache.gpu_slot_records_[g].key; cache.gpu_slot_records_[g].key = key; return old;
    }
    static void BorrowGpuSlot(Qwen4ExpExpertCache& cache, int g) {cache.frozen_slots_[g] = false;}
    static void ColdPolicy(Qwen4ExpExpertCache& cache,int policy) {cache.hotset_cold_policy_=policy;cache.RefreshHotsetColdSlots();}
    static void Adopt(Qwen4ExpExpertCache& cache,int limit) {Prompt(cache);cache.hotset_adopt_ready_=true;cache.hotset_adopt_limit_=limit;cache.hotset_prompt_swaps_=0;}
    static void Prompt(Qwen4ExpExpertCache& cache) {cache.hotset_prompt_=true;cache.hotset_prompt_swaps_=2;}
    static void PromptPrefill(Qwen4ExpExpertCache& cache) {Prompt(cache);cache.hotset_prompt_prefill_=true;}
    static void Adaptive(Qwen4ExpExpertCache& cache,const std::vector<int>& keys) {
        cache.hotset_policy_.Init(cache.num_layers_,cache.num_experts_per_layer_);cache.hotset_keys_=keys;cache.hotset_swaps_=1;
        cache.adaptive_hotset_=true;cache.hotset_borrow_=true;cache.InstallHotset(keys);
    }
    static void LargePagePrompt(Qwen4ExpExpertCache& cache) {
        cache.hotset_prompt_=true;cache.hotset_prompt_swaps_=512;
        cache.hotset_adopt_ready_=true;cache.hotset_adopt_limit_=4096;
        cache.staging_buffer_bytes_=52*cache.slot_bytes_;
    }
    static void HoldCopy(Qwen4ExpExpertCache& cache, int slot, std::shared_future<void> signal) {
        cache.gpu_slot_records_[slot].state = ExpertSlotState::Copying;
        cache.gpu_slot_records_[slot].dma_event = cache.copy_q_->submit([&](sycl::handler& cgh) {
            cgh.host_task([signal] { signal.wait_for(std::chrono::seconds(10)); });
        });
    }
};

namespace {

void expect(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("Test failure: " + msg);
    }
}

struct SyntheticExpertWeights {
    static constexpr int num_layers = 2;
    static constexpr int num_experts_per_layer = 8;
    static constexpr int hidden_size = 256;
    static constexpr int intermediate_size = 64;
    static constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size; // 128
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2); // 16384 bytes
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize); // 512
    const size_t dn_rows = hidden_size; // 256
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2); // 8192 bytes
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize); // 256

    std::vector<uint8_t> gu_packed_host;
    std::vector<uint8_t> gu_absmax_codes;
    std::vector<float> gu_nested_absmax;
    std::vector<float> gu_nested_qmap;

    std::vector<uint8_t> dn_packed_host;
    std::vector<uint8_t> dn_absmax_codes;
    std::vector<float> dn_nested_absmax;
    std::vector<float> dn_nested_qmap;
    std::vector<float> qmap;

    Bnb4BitHostWeightRef gu_ref;
    Bnb4BitHostWeightRef dn_ref;

    explicit SyntheticExpertWeights(int experts = num_experts_per_layer) {
        const size_t total_gu_packed = experts * gu_packed_per_exp;
        const size_t total_gu_absmax = experts * gu_absmax_per_exp;
        const size_t total_dn_packed = experts * dn_packed_per_exp;
        const size_t total_dn_absmax = experts * dn_absmax_per_exp;

        gu_packed_host.resize(total_gu_packed);
        gu_absmax_codes.resize(total_gu_absmax);
        gu_nested_absmax.assign(total_gu_absmax / 256 + 1, 1.25f);
        gu_nested_qmap.resize(256);

        dn_packed_host.resize(total_dn_packed);
        dn_absmax_codes.resize(total_dn_absmax);
        dn_nested_absmax.assign(total_dn_absmax / 256 + 1, 0.75f);
        dn_nested_qmap.resize(256);

        qmap.resize(16);
        for (int i = 0; i < 16; ++i) qmap[i] = static_cast<float>(i - 8) / 8.0f;
        for (int i = 0; i < 256; ++i) {
            gu_nested_qmap[i] = static_cast<float>(i) / 256.0f;
            dn_nested_qmap[i] = static_cast<float>(i) / 256.0f;
        }

        std::mt19937 rng(42);
        for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (size_t i = 0; i < total_gu_absmax; ++i) gu_absmax_codes[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (size_t i = 0; i < total_dn_packed; ++i) dn_packed_host[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (size_t i = 0; i < total_dn_absmax; ++i) dn_absmax_codes[i] = static_cast<uint8_t>(rng() & 0xFF);

        gu_ref.name = "gu";
        gu_ref.packed_weight.ptr = gu_packed_host.data();
        gu_ref.packed_weight.bytes = total_gu_packed;
        gu_ref.packed_weight.shape = {experts * gu_rows, hidden_size / 2};
        gu_ref.packed_weight.dtype = DT_U8;
        gu_ref.absmax.ptr = gu_absmax_codes.data();
        gu_ref.absmax.bytes = total_gu_absmax;
        gu_ref.absmax.shape = {static_cast<int64_t>(total_gu_absmax)};
        gu_ref.absmax.dtype = DT_U8;
        gu_ref.quant_map.ptr = reinterpret_cast<const uint8_t*>(qmap.data());
        gu_ref.quant_map.bytes = 16 * sizeof(float);
        gu_ref.quant_map.shape = {16};
        gu_ref.quant_map.dtype = DT_F32;
        gu_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(gu_nested_absmax.data());
        gu_ref.nested_absmax.bytes = gu_nested_absmax.size() * sizeof(float);
        gu_ref.nested_absmax.dtype = DT_F32;
        gu_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(gu_nested_qmap.data());
        gu_ref.nested_quant_map.bytes = 256 * sizeof(float);
        gu_ref.nested_quant_map.dtype = DT_F32;
        gu_ref.quant_state.quant_type = "nf4";
        gu_ref.quant_state.dtype = "bfloat16";
        gu_ref.quant_state.shape = {experts, gu_rows, hidden_size};
        gu_ref.quant_state.blocksize = blocksize;
        gu_ref.quant_state.nested = true;
        gu_ref.quant_state.nested_blocksize = 256;
        gu_ref.quant_state.nested_offset = 0.05f;

        dn_ref.name = "dn";
        dn_ref.packed_weight.ptr = dn_packed_host.data();
        dn_ref.packed_weight.bytes = total_dn_packed;
        dn_ref.packed_weight.shape = {experts * dn_rows, intermediate_size / 2};
        dn_ref.packed_weight.dtype = DT_U8;
        dn_ref.absmax.ptr = dn_absmax_codes.data();
        dn_ref.absmax.bytes = total_dn_absmax;
        dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
        dn_ref.absmax.dtype = DT_U8;
        dn_ref.quant_map.ptr = reinterpret_cast<const uint8_t*>(qmap.data());
        dn_ref.quant_map.bytes = 16 * sizeof(float);
        dn_ref.quant_map.shape = {16};
        dn_ref.quant_map.dtype = DT_F32;
        dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
        dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
        dn_ref.nested_absmax.dtype = DT_F32;
        dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
        dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
        dn_ref.nested_quant_map.dtype = DT_F32;
        dn_ref.quant_state.quant_type = "nf4";
        dn_ref.quant_state.dtype = "bfloat16";
        dn_ref.quant_state.shape = {experts, dn_rows, intermediate_size};
        dn_ref.quant_state.blocksize = blocksize;
        dn_ref.quant_state.nested = true;
        dn_ref.quant_state.nested_blocksize = 256;
        dn_ref.quant_state.nested_offset = 0.03f;
    }
};

void test_query_residency_and_gpu_tickets(Context& ctx) {
    std::cout << "[Test 1] Testing query_residency, prepare_host, and ensure_gpu tickets...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 4 * slot_bytes; // 4 GPU slots

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    expect(cache.host_pool_enabled(), "host pool should be enabled via env");
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);

    // Initial query on expert (0, 0): should be non-resident on both GPU and host
    ExpertResidency r0 = cache.query_residency(0, 0);
    expect(!r0.gpu_resident, "expected unallocated GPU slot for expert 0");
    expect(!r0.gpu_ready, "expected unready GPU state for expert 0");
    expect(!r0.host_resident, "expected non-resident host state for expert 0");
    expect(!cache.HasGpuRouteSource(0,0), "Missing expert excluded from CPU route");
    expect(!cache.HasGpuRouteSource(-1,0) && !cache.HasGpuRouteSource(0,8), "Invalid GPU route query");

    // Test prepare_host for expert (0, 3)
    FillTicket ft3 = cache.prepare_host(0, 3);
    expect(ft3.ready, "prepare_host should stage expert 3 to ready");
    expect(ft3.hslot >= 0, "prepare_host should assign valid hslot");
    expect(ft3.layer == 0, "prepare_host layer match");
    expect(ft3.expert_id == 3, "prepare_host expert_id match");

    // Query residency for expert 3: host resident, but not yet GPU resident
    ExpertResidency r3 = cache.query_residency(0, 3);
    expect(r3.host_resident, "expert 3 should now be host resident");
    expect(r3.host_ready, "expert 3 should be host ready");
    expect(r3.host_slot == ft3.hslot, "host slot match");
    expect(!r3.gpu_resident, "expert 3 should not yet be GPU resident");
    expect(!cache.HasGpuRouteSource(0,3), "Host-only expert excluded from CPU route");

    // Test ensure_gpu for expert (0, 3)
    GpuExpertTicket gt3 = cache.ensure_gpu(0, 3);
    expect(gt3.gpu_slot >= 0, "ensure_gpu should allocate valid GPU slot");
    expect(gt3.layer == 0, "gpu ticket layer match");
    expect(gt3.expert_id == 3, "gpu ticket expert_id match");
    if (gt3.dma_event != sycl::event{}) {
        gt3.dma_event.wait_and_throw();
    }

    // Now expert 3 must be GPU ready
    ExpertResidency r3_gpu = cache.query_residency(0, 3);
    expect(cache.HasGpuRouteSource(0,3) == (r3_gpu.gpu_ready || r3_gpu.gpu_copying), "Light GPU placement differs from original query");
    expect(r3_gpu.gpu_resident, "expert 3 should be GPU resident");
    expect(r3_gpu.gpu_ready, "expert 3 should be GPU ready");
    expect(r3_gpu.gpu_slot == gt3.gpu_slot, "GPU slot match");

    // Expert 5 was not acquired, should not be GPU ready
    ExpertResidency r5 = cache.query_residency(0, 5);
    expect(!r5.gpu_ready, "expert 5 should not be GPU ready");
    expect(!r5.host_ready, "expert 5 should not be host ready");
    std::cout << "  Passed query_residency and tickets test.\n";
}

void test_host_lease_lifecycle_and_eviction_protection(Context& ctx) {
    std::cout << "[Test 2] Testing HostExpertLease RAII lifecycle & full pool eviction protection...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 4 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    expect(cache.host_pool_enabled(), "host pool should be enabled");
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);

    // Stage expert (0, 0) into host pool
    FillTicket ft0 = cache.prepare_host(0, 0);
    expect(ft0.ready, "expert 0 should be ready in host pool");
    const int hslot0 = ft0.hslot;
    expect(hslot0 >= 0, "valid host slot expected");
    expect(cache.get_host_slot_cpu_readers(hslot0) == 0, "initial cpu_readers should be 0");

    // Acquire lease on expert (0, 0)
    HostExpertLease lease0 = cache.acquire_host_lease(0, 0);
    expect(lease0.is_valid(), "lease0 should be valid");
    expect(lease0.hslot() == hslot0, "lease0 hslot match");
    expect(lease0.layer() == 0, "lease0 layer match");
    expect(lease0.expert_id() == 0, "lease0 expert_id match");
    expect(cache.get_host_slot_cpu_readers(hslot0) == 1, "cpu_readers should be 1 during lease");

    // Verify CPU host pointers in MoeExpertSlotHost
    const auto& sh = lease0.slot_host();
    expect(sh.gate_up_packed != nullptr, "gate_up_packed pointer valid");
    expect(sh.down_packed != nullptr, "down_packed pointer valid");
    expect(sh.gate_up_nested.codes != nullptr, "nested codes valid");
    expect(sh.gate_up_nested.nested_absmax != nullptr, "nested absmax valid");
    expect(sh.gate_up_nested.nested_quant_map != nullptr, "nested qmap valid");

    // Test move semantics
    HostExpertLease lease_moved = std::move(lease0);
    expect(!lease0.is_valid(), "original lease should be invalidated after move");
    expect(lease_moved.is_valid(), "moved lease should remain valid");
    expect(cache.get_host_slot_cpu_readers(hslot0) == 1, "cpu_readers should stay 1 after move");

    // Test second concurrent lease on the same slot
    {
        HostExpertLease lease_b = cache.acquire_host_lease(0, 0);
        expect(lease_b.is_valid(), "second lease should be valid");
        expect(cache.get_host_slot_cpu_readers(hslot0) == 2, "cpu_readers should be 2 for concurrent leases");
    }
    expect(cache.get_host_slot_cpu_readers(hslot0) == 1, "cpu_readers should return to 1 after lease_b destroyed");

    // Test ensure_gpu using HostExpertLease
    GpuExpertTicket gt_from_lease = cache.ensure_gpu(lease_moved);
    expect(gt_from_lease.gpu_slot >= 0, "ensure_gpu from lease should succeed");
    if (gt_from_lease.dma_event != sycl::event{}) {
        gt_from_lease.dma_event.wait_and_throw();
    }
    expect(cache.query_residency(0, 0).gpu_ready, "expert 0 should now be GPU ready via lease");

    // REAL EVICTION PROTECTION TEST:
    // Host pool has 4 slots (0, 1, 2, 3). Slot 0 holds expert (0, 0) with cpu_readers == 1.
    // Stage 3 more experts to fill the remaining 3 slots.
    FillTicket ft1 = cache.prepare_host(0, 1); expect(ft1.ready, "exp 1 ready");
    FillTicket ft2 = cache.prepare_host(0, 2); expect(ft2.ready, "exp 2 ready");
    FillTicket ft3 = cache.prepare_host(0, 3); expect(ft3.ready, "exp 3 ready");

    // Acquire leases on experts 1, 2, 3 as well
    HostExpertLease l1 = cache.acquire_host_lease(0, 1); expect(l1.is_valid(), "l1 valid");
    HostExpertLease l2 = cache.acquire_host_lease(0, 2); expect(l2.is_valid(), "l2 valid");
    HostExpertLease l3 = cache.acquire_host_lease(0, 3); expect(l3.is_valid(), "l3 valid");

    // Now ALL 4 host slots in the pool are actively leased (cpu_readers >= 1).
    // Attempting to allocate expert (0, 4) MUST fail to find an eviction victim!
    FillTicket ft_blocked = cache.prepare_host(0, 4);
    expect(!ft_blocked.ready, "allocation of expert 4 MUST fail when all slots are leased");
    expect(ft_blocked.hslot < 0, "hslot must be negative when no slot can be evicted");

    // Verify expert (0, 0) and expert (0, 1) are STILL host-resident and untouched
    expect(cache.query_residency(0, 0).host_resident, "expert 0 must still be resident and protected");
    expect(cache.query_residency(0, 1).host_resident, "expert 1 must still be resident and protected");

    // Now release lease l1 (expert 1). Now slot for expert 1 has cpu_readers == 0.
    const int hslot1 = l1.hslot();
    l1.release();
    expect(cache.get_host_slot_cpu_readers(hslot1) == 0, "expert 1 readers should be 0");

    // Now expert 4 CAN be prepared into the newly evictable slot 1!
    FillTicket ft_ok = cache.prepare_host(0, 4);
    expect(ft_ok.ready, "expert 4 should now succeed in allocating");
    expect(ft_ok.hslot == hslot1, "expert 4 should evict slot 1");
    expect(!cache.query_residency(0, 1).host_resident, "expert 1 should now be evicted");
    expect(cache.query_residency(0, 4).host_resident, "expert 4 should be resident");

    // Expert 0 in slot 0 was STILL protected throughout!
    expect(cache.query_residency(0, 0).host_resident, "expert 0 was preserved in slot 0");

    // Clean up remaining leases
    lease_moved.release();
    l2.release();
    l3.release();
    expect(cache.get_host_slot_cpu_readers(hslot0) == 0, "all readers released");

    std::cout << "  Passed HostExpertLease lifecycle and eviction protection test.\n";
}

void test_mixed_ready_copying_and_cancellation(Context& ctx) {
    std::cout << "[Test 3] Testing mixed ready/copying states, prefetch cancel & slot reuse...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 4 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);

    // Pre-populate GPU slots with expert 0 and 1
    int pre_req[2] = {0, 1};
    int pre_slots[2] = {-1, -1};
    sycl::event ev = cache.acquire(0, pre_req, 2, pre_slots);
    ev.wait_and_throw();

    ExpertResidency r0 = cache.query_residency(0, 0);
    ExpertResidency r1 = cache.query_residency(0, 1);
    expect(r0.gpu_ready, "expert 0 should be GPU ready");
    expect(r1.gpu_ready, "expert 1 should be GPU ready");

    // Unacquired expert 2: not ready
    ExpertResidency r2 = cache.query_residency(0, 2);
    expect(!r2.gpu_ready && !r2.gpu_resident, "expert 2 should not be resident");

    // Start asynchronous prefetch for experts 2 and 3
    int pf_req[2] = {2, 3};
    cache.prefetch_async(0, pf_req, 2);

    // Test cancellation of in-flight prefetch
    cache.cancel_prefetch_for_testing();

    // Verify slot reuse lifecycle: acquiring new experts after cancellation works seamlessly
    int req_after[2] = {2, 3};
    int slots_after[2] = {-1, -1};
    sycl::event ev_after = cache.acquire(0, req_after, 2, slots_after);
    ev_after.wait_and_throw();

    ExpertResidency r2_after = cache.query_residency(0, 2);
    ExpertResidency r3_after = cache.query_residency(0, 3);
    expect(r2_after.gpu_ready, "expert 2 should be GPU ready after acquire");
    expect(r3_after.gpu_ready, "expert 3 should be GPU ready after acquire");

    std::cout << "  Passed mixed ready/copying and cancellation test.\n";
}

void test_expert_executor_planning_and_ready_subset(Context& ctx) {
    std::cout << "[Test 4] Testing Qwen4ExpExpertExecutor LayerMoePlan & ready subset execution...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 6 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);

    // Pre-populate GPU slots with expert 0 and 1
    int pre_req[2] = {0, 1};
    int pre_slots[2] = {-1, -1};
    sycl::event ev = cache.acquire(0, pre_req, 2, pre_slots);
    ev.wait_and_throw();
    // Fill the cache completely, leaving the two needed ready experts at the LRU tail.
    int filler_ids[4] = {0, 1, 2, 3};
    int filler_slots[4];
    cache.acquire(1, filler_ids, 4, filler_slots).wait_and_throw();
    const auto ready0 = cache.query_residency(0, 0);
    const auto ready1 = cache.query_residency(0, 1);

    // Initialize executor
    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 64, 4);
    expect(executor.is_initialized(), "executor should be initialized");

    // Route for 2 tokens with top_k=4:
    // Token 0 wants experts [0, 1, 2, 3] (0 and 1 are GPU-ready, 2 and 3 are pending)
    // Token 1 wants experts [1, 0, 4, 5] (1 and 0 are GPU-ready, 4 and 5 are pending)
    int route_ids[8] = {0, 1, 2, 3, 1, 0, 4, 5};
    float route_weights[8] = {0.4f, 0.3f, 0.2f, 0.1f, 0.4f, 0.3f, 0.2f, 0.1f};

    aila::LayerMoePlan& plan = executor.plan_layer(0, route_ids, route_weights, 2, 4, false);
    expect(plan.layer == 0, "plan layer mismatch");
    expect(plan.token_count == 2, "token count mismatch");
    expect(plan.total_tasks == 8, "total tasks mismatch");
    expect(plan.ready_gpu_tasks == 4, "expected 4 ready GPU tasks (0, 1 on token 0 and 1, 0 on token 1)");
    expect(plan.pending_gpu_tasks == 4, "expected 4 pending GPU tasks (2, 3 on token 0 and 4, 5 on token 1)");
    expect(plan.ready_gpu_indices.size() == 4, "ready indices count mismatch");
    expect(plan.pending_gpu_indices.size() == 4, "pending indices count mismatch");

    // Verify ready entries have valid gpu_slot
    for (int r_idx : plan.ready_gpu_indices) {
        const auto& entry = plan.entries[r_idx];
        expect(entry.is_gpu, "ready entry must be gpu");
        expect(entry.gpu_slot >= 0, "ready entry must have valid gpu_slot");
    }

    // Now call ensure_gpu_pending to acquire pending experts (2, 3, 4, 5)
    executor.ensure_gpu_pending(plan);
    if (plan.final_dma_event != sycl::event{}) {
        plan.final_dma_event.wait_and_throw();
    }

    // Now all entries should have valid gpu_slot
    for (const auto& entry : plan.entries) {
        expect(entry.gpu_slot >= 0, "all entries must have valid gpu_slot after ensure_gpu_pending");
        const auto res = cache.query_residency(plan.layer, entry.expert_id);
        expect(res.gpu_slot == entry.gpu_slot && res.gpu_generation == entry.gpu_generation,
               "plan must still reference the correct expert generation");
    }
    expect(cache.query_residency(0, 0).gpu_generation == ready0.gpu_generation &&
           cache.query_residency(0, 0).gpu_slot == ready0.gpu_slot, "ready expert 0 was evicted");
    expect(cache.query_residency(0, 1).gpu_generation == ready1.gpu_generation &&
           cache.query_residency(0, 1).gpu_slot == ready1.gpu_slot, "ready expert 1 was evicted");

    // Record execution completion
    executor.record_execution_complete(plan, sycl::event{});
    int overflow_ids[8] = {0, 1, 2, 3, 4, 5, 6, 0};
    auto& overflow = executor.plan_layer(0, overflow_ids, route_weights, 2, 4, false);
    bool rejected = false;
    try { executor.ensure_gpu_pending(overflow); } catch (const std::runtime_error&) { rejected = true; }
    expect(rejected && !cache.is_faulted(), "over-capacity GPU union must reject without poisoning");
    expect(cache.query_residency(0, 0).gpu_generation == ready0.gpu_generation,
           "over-capacity plan must preserve ready bindings");

    // Test edge cases: token_count == 0, null router_ids
    aila::LayerMoePlan& p_empty = executor.plan_layer(0, nullptr, nullptr, 0, 4);
    expect(p_empty.total_tasks == 0, "empty plan for token_count=0");
    aila::LayerMoePlan& p_null = executor.plan_layer(0, nullptr, nullptr, 2, 4);
    expect(p_null.total_tasks == 0, "empty plan for null router_ids");

    // Verify repeated planning on hot path does not throw or corrupt
    for (int iter = 0; iter < 500; ++iter) {
        aila::LayerMoePlan& plan_iter = executor.plan_layer(0, route_ids, route_weights, 2, 4, false);
        expect(plan_iter.total_tasks == 8, "repeated plan total tasks match");
    }

    std::cout << "  Passed Qwen4ExpExpertExecutor planning and ready subset test.\n";
}

void test_cpu_packet_ordered_merge(Context& ctx) {
    for (int H : {257, 2560}) for (bool shared : {false, true}) for (int cpu_count : {0, 3, 16}) {
        constexpr int K = 16;
        std::vector<float> canonical(K * H), scattered(K * H), weights(K);
        std::vector<uint8_t> packet(16 * sizeof(int) + K * H * sizeof(float));
        std::array<int, 16> cpu_ranks;
        std::fill_n(cpu_ranks.data(), 16, -1);
        for (int k = 0; k < K; ++k) {
            weights[k] = float(k - 7) / 31.0f;
            for (int r = 0; r < H; ++r) canonical[k * H + r] = float((k * 37 + r * 13) % 101 - 50) / 19.0f;
        }
        scattered = canonical;
        for (int c = 0; c < cpu_count; ++c) {
            const int rank = (c * 5) % K;
            cpu_ranks[rank] = cpu_count - c - 1;
            std::memcpy(packet.data() + 16 * sizeof(int) + cpu_ranks[rank] * H * sizeof(float), canonical.data() + rank * H, H * sizeof(float));
            std::fill_n(scattered.data() + rank * H, H, std::numeric_limits<float>::quiet_NaN());
        }
        std::memcpy(packet.data(), cpu_ranks.data(), 16 * sizeof(int));
        std::vector<bf16> shared_host(H, bf16(0.25f)), baseline(H), actual(H);
        bf16 gate = bf16(-0.5f);
        auto* down = static_cast<float*>(ctx.alloc_device(K * H * sizeof(float)));
        auto* w = static_cast<float*>(ctx.alloc_device(K * sizeof(float)));
        auto* p = ctx.alloc_device(packet.size());
        auto* s = static_cast<bf16*>(ctx.alloc_device(H * sizeof(bf16)));
        auto* g = static_cast<bf16*>(ctx.alloc_device(sizeof(bf16)));
        auto* y = static_cast<bf16*>(ctx.alloc_device(H * sizeof(bf16)));
        ctx.queue().memcpy(w, weights.data(), K * sizeof(float));
        ctx.queue().memcpy(s, shared_host.data(), H * sizeof(bf16));
        ctx.queue().memcpy(g, &gate, sizeof(bf16));
        ctx.queue().memcpy(down, canonical.data(), K * H * sizeof(float));
        ops::moe_ordered_accumulate_down_rows(ctx, down, w, y, 1, K, H, shared ? s : nullptr, shared ? g : nullptr);
        ctx.queue().memcpy(baseline.data(), y, H * sizeof(bf16)).wait_and_throw();
        ctx.queue().memcpy(down, scattered.data(), K * H * sizeof(float));
        ctx.queue().memcpy(p, packet.data(), packet.size());
        ops::moe_ordered_accumulate_down_rows(ctx, down, w, y, 1, K, H, shared ? s : nullptr, shared ? g : nullptr, nullptr, p);
        ctx.queue().memcpy(actual.data(), y, H * sizeof(bf16)).wait_and_throw();
        expect(std::memcmp(actual.data(), baseline.data(), H * sizeof(bf16)) == 0, "CPU packet merge changed ordered BF16 output");
        bool rejected = false;
        try { ops::moe_ordered_accumulate_down_rows(ctx, down, w, y, 2, K, H, nullptr, nullptr, nullptr, p); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "CPU packet accepted multiple tokens");
        ctx.free_device(down); ctx.free_device(w); ctx.free_device(p); ctx.free_device(s); ctx.free_device(g); ctx.free_device(y);
    }
}

void test_h2b_down_to_rank_and_ordered_merge(Context& ctx) {
    std::cout << "[Test 5] Testing H2b down_to_rank_f32 + ordered rank merge vs fused weighted down...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 4 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    // Acquire 4 experts into GPU slots
    int req[4] = {0, 1, 2, 3};
    int slots[4] = {-1, -1, -1, -1};
    sycl::event ev = cache.acquire(0, req, 4, slots);
    ev.wait_and_throw();

    ops::MoeExpertSlotDev dev_slots[4];
    cache.get_slots_dev(slots, 4, dev_slots);

    // Setup activations
    std::vector<bf16> h_host(4 * synth.intermediate_size);
    for (size_t i = 0; i < h_host.size(); ++i) {
        h_host[i] = bf16(static_cast<float>((i % 17) - 8) / 10.0f);
    }
    bf16* h_dev = static_cast<bf16*>(ctx.alloc_device(h_host.size() * sizeof(bf16)));
    ctx.queue().memcpy(h_dev, h_host.data(), h_host.size() * sizeof(bf16)).wait();

    std::vector<float> weights_host = {0.4f, 0.3f, 0.2f, 0.1f};
    float* weights_dev = static_cast<float*>(ctx.alloc_device(4 * sizeof(float)));
    ctx.queue().memcpy(weights_dev, weights_host.data(), 4 * sizeof(float)).wait();

    // 1. Run baseline fused decode down
    bf16* y_fused_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));
    ops::moe_decode_down_weighted_accumulate(
        ctx, dev_slots, nullptr, weights_host.data(), h_dev, nullptr, y_fused_dev,
        4, synth.intermediate_size, synth.hidden_size, synth.blocksize);

    // 2. Run H2b down_to_rank_f32 + ordered rank merge
    float* down_acc_dev = static_cast<float*>(ctx.alloc_device(4 * synth.hidden_size * sizeof(float)));
    bf16* y_merged_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));

    int ranks_host[4] = {0, 1, 2, 3};

    sycl::event down_ev = ops::moe_decode_down_to_rank_f32(
        ctx, dev_slots, ranks_host, h_dev, down_acc_dev, 4, 4,
        synth.intermediate_size, synth.hidden_size, synth.blocksize,
        nullptr, nullptr, nullptr, cache.uses_nested_codes());

    sycl::event merge_ev = ops::moe_ordered_accumulate_down_rows(
        ctx, down_acc_dev, weights_dev, y_merged_dev, 1, 4, synth.hidden_size,
        nullptr, nullptr, &down_ev);
    merge_ev.wait_and_throw();

    // Read back and compare
    std::vector<bf16> y_fused_host(synth.hidden_size);
    std::vector<bf16> y_merged_host(synth.hidden_size);
    ctx.queue().memcpy(y_fused_host.data(), y_fused_dev, synth.hidden_size * sizeof(bf16)).wait();
    ctx.queue().memcpy(y_merged_host.data(), y_merged_dev, synth.hidden_size * sizeof(bf16)).wait();

    float max_diff = 0.0f;
    for (int i = 0; i < synth.hidden_size; ++i) {
        float f = static_cast<float>(y_fused_host[i]);
        float m = static_cast<float>(y_merged_host[i]);
        float diff = std::abs(f - m);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "  max diff between fused down and down_to_rank_f32+ordered_merge: " << max_diff << "\n";
    expect(max_diff < 1e-3f, "down_to_rank_f32 + ordered merge must match fused down output");

    ctx.free_device(h_dev);
    ctx.free_device(weights_dev);
    ctx.free_device(y_fused_dev);
    ctx.free_device(down_acc_dev);
    ctx.free_device(y_merged_dev);

    std::cout << "  Passed H2b down_to_rank_f32 and ordered merge equivalence test.\n";
}

void test_h2b_h3_cpu_expert_dispatch_and_credit_bounds(Context& ctx) {
    std::cout << "[Test 6] Testing H2b/H3 CPU expert dispatch, bounded credits, and execution...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 4 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    // 1. GPU pre-population: acquire expert 0 into GPU
    int pre_req[1] = {0};
    int pre_slots[1] = {-1};
    sycl::event ev = cache.acquire(0, pre_req, 1, pre_slots);
    ev.wait_and_throw();

    // 2. Host pool pre-population: prepare expert 1 and 2 in host pool (host ready, GPU miss)
    FillTicket t1 = cache.prepare_host(0, 1);
    FillTicket t2 = cache.prepare_host(0, 2);
    expect(t1.ready, "ticket 1 must be ready");
    expect(t2.ready, "ticket 2 must be ready");

    // Initialize executor with test shapes
    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 64, 4, synth.hidden_size, synth.intermediate_size);
    executor.set_max_cpu_credits(2);

    // Plan layer: route wants [0, 1, 2, 3]
    // 0: GPU-ready
    // 1: Host-ready miss -> should go to CPU (credit 1)
    // 2: Host-ready miss -> should go to CPU (credit 2)
    // 3: Cold miss -> should go to GPU pending
    int route_ids[4] = {0, 1, 2, 3};
    float route_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};

    aila::LayerMoePlan& plan = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(plan.token_count == 1, "token count mismatch");
    expect(plan.total_tasks == 4, "total tasks mismatch");
    expect(plan.ready_gpu_tasks == 1, "expected 1 ready GPU task (expert 0)");
    expect(plan.cpu_tasks == 2, "expected 2 CPU tasks (expert 1 and 2)");
    expect(plan.pending_gpu_tasks == 1, "expected 1 pending GPU task (expert 3)");

    // Test bounded admission credit: set max_cpu_credits to 1
    executor.set_max_cpu_credits(1);
    aila::LayerMoePlan& plan_bounded = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(plan_bounded.cpu_tasks == 1, "CPU tasks must be strictly bounded to max_cpu_credits=1");
    expect(plan_bounded.pending_gpu_tasks == 2, "excess host-ready misses must fall back to GPU pending");

    // Re-plan with 2 credits for execution test
    executor.set_max_cpu_credits(2);
    aila::LayerMoePlan& plan_exec = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(plan_exec.cpu_tasks == 2, "must have 2 CPU tasks");

    // Ensure pending GPU task (expert 3) is acquired
    executor.ensure_gpu_pending(plan_exec);
    if (plan_exec.final_dma_event != sycl::event{}) {
        plan_exec.final_dma_event.wait_and_throw();
    }

    // Allocate device buffers for execution test
    bf16* x_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));
    std::vector<bf16> x_host(synth.hidden_size, bf16(0.5f));
    ctx.queue().memcpy(x_dev, x_host.data(), synth.hidden_size * sizeof(bf16)).wait();

    bf16* moe_h_dev = static_cast<bf16*>(ctx.alloc_device(4 * synth.intermediate_size * sizeof(bf16)));
    bf16* y_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));

    sycl::event hetero_ev = executor.execute_hetero_decode(
        ctx, plan_exec, x_dev, moe_h_dev, route_weights, y_dev, 1, 4,
        nullptr, nullptr, cache.uses_nested_codes());
    hetero_ev.wait_and_throw();

    // Check stats
    const auto& stats = executor.stats();
    expect(stats.cpu_tasks_dispatched == 2, "stats: expected 2 cpu tasks dispatched");
    expect(stats.cpu_tasks_completed == 2, "stats: expected 2 cpu tasks completed");
    expect(stats.gpu_tasks_dispatched == 2, "stats: expected 2 gpu tasks dispatched");
    expect(stats.layers_hetero_executed == 1, "stats: expected 1 layer hetero executed");

    // Verify y_dev output contains valid finite numbers (not NaN or Inf)
    std::vector<bf16> y_host(synth.hidden_size);
    ctx.queue().memcpy(y_host.data(), y_dev, synth.hidden_size * sizeof(bf16)).wait();
    bool all_finite = true;
    for (int i = 0; i < synth.hidden_size; ++i) {
        float val = static_cast<float>(y_host[i]);
        if (std::isnan(val) || std::isinf(val)) {
            all_finite = false;
            break;
        }
    }
    expect(all_finite, "hetero decode output must be finite numbers");

    ctx.free_device(x_dev);
    ctx.free_device(moe_h_dev);
    ctx.free_device(y_dev);

    std::cout << "  Passed H2b/H3 CPU expert dispatch and credit bounds test.\n";
}

void test_h3_straggler_monitoring_and_penalty(Context& ctx) {
    std::cout << "[Test 7] Testing H3 straggler monitoring, penalty throttling, and recovery...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 4 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    cache.prepare_host(0, 0);
    cache.prepare_host(0, 1);
    cache.prepare_host(0, 2);

    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 64, 4, synth.hidden_size, synth.intermediate_size);
    executor.set_max_cpu_credits(2);

    int route_ids[4] = {0, 1, 2, 3};
    float route_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};

    // Without penalty: 2 CPU tasks allowed
    expect(!executor.is_straggler_penalty_active(), "initially straggler penalty must be inactive");
    aila::LayerMoePlan& p1 = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(p1.cpu_tasks == 2, "expected 2 cpu tasks with 2 credits");

    // Manually trigger straggler penalty
    executor.set_straggler_penalty(true);
    expect(executor.is_straggler_penalty_active(), "straggler penalty must be active");

    // With penalty active: credits shrunk by 1 (effective credits = 1)
    aila::LayerMoePlan& p2 = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(p2.cpu_tasks == 1, "expected 1 cpu task under straggler penalty (throttled by 1)");

    // Restore penalty
    executor.set_straggler_penalty(false);
    aila::LayerMoePlan& p3 = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(p3.cpu_tasks == 2, "credits must recover when penalty is cleared");

    // Edge case: token_count > max_cpu_tokens (e.g. 5 tokens) -> CPU tasks must be 0
    std::vector<int> large_route(5 * 4, 0);
    std::vector<float> large_weights(5 * 4, 0.25f);
    aila::LayerMoePlan& p_large = executor.plan_layer(0, large_route.data(), large_weights.data(), 5, 4, true);
    expect(p_large.cpu_tasks == 0, "CPU dispatch must be disabled for token count > max_cpu_tokens (m <= 4)");

    executor.set_max_cpu_credits(1);
    executor.set_straggler_penalty(true);
    for (int layer = 0; layer < 8; ++layer) {
        auto& cooling = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
        expect(cooling.cpu_tasks == 0, "single-credit straggler policy must cool down");
    }
    auto& retry = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
    expect(retry.cpu_tasks == 1, "single-credit policy must eventually retry after warmup");
    executor.set_max_cpu_credits(0);
    for (int layer = 0; layer < 10; ++layer) {
        auto& disabled = executor.plan_layer(0, route_ids, route_weights, 1, 4, true);
        expect(disabled.cpu_tasks == 0, "explicit zero credits must remain disabled");
    }

    std::cout << "  Passed H3 straggler monitoring and penalty test.\n";
}

void test_h2b_multi_token_and_large_task_dispatch(Context& ctx) {
    std::cout << "[Test 8] Testing H2b multi-token (m=2, top_k=10 => 20 tasks, >16 GPU tasks) dispatch...\n";
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=8");
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 20 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    // Populate GPU slots with expert 0
    int pre_req[1] = {0};
    int pre_slots[1] = {-1};
    sycl::event ev = cache.acquire(0, pre_req, 1, pre_slots);
    ev.wait_and_throw();

    // Host-ready experts: 1, 2
    cache.prepare_host(0, 1);
    cache.prepare_host(0, 2);

    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 64, 10, synth.hidden_size, synth.intermediate_size);
    executor.set_max_cpu_credits(2);
    executor.set_max_cpu_tokens(4);

    // 2 tokens, top_k = 10 -> total = 20 tasks!
    // Token 0: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1] (0: GPU, 1: CPU, 2: CPU, others: GPU pending)
    // Token 1: [0, 3, 4, 5, 6, 7, 0, 1, 2, 3] (0: GPU, others: GPU pending since credits full)
    std::vector<int> route_ids = {
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
        0, 3, 4, 5, 6, 7, 0, 1, 2, 3
    };
    std::vector<float> route_weights(20, 0.1f);

    try {
        aila::LayerMoePlan& plan = executor.plan_layer(0, route_ids.data(), route_weights.data(), 2, 10, true);
        expect(plan.token_count == 2, "token count must be 2");
        expect(plan.total_tasks == 20, "total tasks must be 20");
        expect(plan.cpu_tasks == 2, "cpu tasks must be 2 (within effective_credits=2)");
        expect(plan.gpu_tasks == 18, "gpu tasks must be 18 (> 16, exercising dev pointer kernel dispatch)");

        executor.ensure_gpu_pending(plan);
        if (plan.final_dma_event != sycl::event{}) {
            plan.final_dma_event.wait_and_throw();
        }

        // Allocate device buffers
        bf16* x_dev = static_cast<bf16*>(ctx.alloc_device(2 * synth.hidden_size * sizeof(bf16)));
        std::vector<bf16> x_host(2 * synth.hidden_size, bf16(0.25f));
        ctx.queue().memcpy(x_dev, x_host.data(), 2 * synth.hidden_size * sizeof(bf16)).wait();

        bf16* moe_h_dev = static_cast<bf16*>(ctx.alloc_device(20 * synth.intermediate_size * sizeof(bf16)));
        bf16* y_dev = static_cast<bf16*>(ctx.alloc_device(2 * synth.hidden_size * sizeof(bf16)));

        sycl::event hetero_ev = executor.execute_hetero_decode(
            ctx, plan, x_dev, moe_h_dev, route_weights.data(), y_dev, 2, 10,
            nullptr, nullptr, cache.uses_nested_codes());
        hetero_ev.wait_and_throw();

        // Verify stats
        const auto& stats = executor.stats();
        expect(stats.cpu_tasks_dispatched == 2, "expected 2 cpu tasks dispatched");
        expect(stats.cpu_tasks_completed == 2, "expected 2 cpu tasks completed");
        expect(stats.gpu_tasks_dispatched == 18, "expected 18 gpu tasks dispatched (>16)");
        expect(stats.layers_hetero_executed == 1, "expected 1 layer hetero executed");

        // Check output values are finite
        std::vector<bf16> y_host(2 * synth.hidden_size);
        ctx.queue().memcpy(y_host.data(), y_dev, 2 * synth.hidden_size * sizeof(bf16)).wait();
        for (size_t i = 0; i < y_host.size(); ++i) {
            float val = static_cast<float>(y_host[i]);
            expect(!std::isnan(val) && !std::isinf(val), "output values must be finite numbers");
        }

        ctx.free_device(x_dev);
        ctx.free_device(moe_h_dev);
        ctx.free_device(y_dev);

        std::cout << "  Passed H2b multi-token (>16 tasks) dispatch and execution test.\n";
    } catch (const std::exception& e) {
        std::cout << "  [Test 8 Exception caught]: " << e.what() << "\n";
        throw;
    }
}

void test_active_gpu_experts_greater_than_16(Context& ctx) {
    std::cout << "[Test 9] Testing gate_up and down_to_rank_f32 with >16 active GPU experts...\n";
    SyntheticExpertWeights synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 24 * slot_bytes;

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    // Acquire all 8 unique experts into cache slots
    constexpr int kUniqueExp = 8;
    std::vector<int> req(kUniqueExp);
    for (int i = 0; i < kUniqueExp; ++i) req[i] = i;
    std::vector<int> slots(kUniqueExp, -1);
    sycl::event ev = cache.acquire(0, req.data(), kUniqueExp, slots.data());
    ev.wait_and_throw();

    constexpr int kNumExp = 20;
    std::vector<ops::MoeExpertSlotDev> dev_slots(kNumExp);
    for (int i = 0; i < kNumExp; ++i) {
        dev_slots[i] = cache.get_slot_dev(slots[i % kUniqueExp]);
    }

    ops::MoeExpertSlotDev* dev_slots_dev = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(kNumExp * sizeof(ops::MoeExpertSlotDev)));
    ctx.queue().memcpy(dev_slots_dev, dev_slots.data(), kNumExp * sizeof(ops::MoeExpertSlotDev)).wait();

    std::vector<int> token_ids(kNumExp, 0);
    int* token_ids_dev = static_cast<int*>(ctx.alloc_device(kNumExp * sizeof(int)));
    ctx.queue().memcpy(token_ids_dev, token_ids.data(), kNumExp * sizeof(int)).wait();

    std::vector<int> ranks(kNumExp);
    for (int i = 0; i < kNumExp; ++i) ranks[i] = i % 10;
    int* ranks_dev = static_cast<int*>(ctx.alloc_device(kNumExp * sizeof(int)));
    ctx.queue().memcpy(ranks_dev, ranks.data(), kNumExp * sizeof(int)).wait();

    bf16* x_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));
    std::vector<bf16> x_host(synth.hidden_size, bf16(0.3f));
    ctx.queue().memcpy(x_dev, x_host.data(), synth.hidden_size * sizeof(bf16)).wait();

    bf16* moe_h_dev = static_cast<bf16*>(ctx.alloc_device(kNumExp * synth.intermediate_size * sizeof(bf16)));
    float* down_acc_dev = static_cast<float*>(ctx.alloc_device(kNumExp * synth.hidden_size * sizeof(float)));

    // Launch grouped gate_up with 20 experts (>16)
    sycl::event gu_ev = ops::moe_decode_gate_up_swiglu_grouped(
        ctx, dev_slots_dev, nullptr, x_dev, moe_h_dev, kNumExp,
        synth.hidden_size, synth.intermediate_size, synth.blocksize,
        nullptr, token_ids_dev, cache.uses_nested_codes());

    // Launch down_to_rank_f32 with 20 experts (>16)
    sycl::event down_ev = ops::moe_decode_down_to_rank_f32(
        ctx, dev_slots_dev, ranks_dev, moe_h_dev, down_acc_dev,
        kNumExp, 10, synth.intermediate_size, synth.hidden_size, synth.blocksize,
        token_ids_dev, nullptr, &gu_ev, cache.uses_nested_codes());
    down_ev.wait_and_throw();

    // Verify down_acc_dev has valid non-zero values
    std::vector<float> acc_host(kNumExp * synth.hidden_size);
    ctx.queue().memcpy(acc_host.data(), down_acc_dev, acc_host.size() * sizeof(float)).wait();
    bool has_nonzero = false;
    for (float f : acc_host) {
        expect(!std::isnan(f) && !std::isinf(f), "acc values must be finite");
        if (std::abs(f) > 1e-6f) has_nonzero = true;
    }
    expect(has_nonzero, "down_to_rank_f32 for 20 experts must compute non-zero activations");

    ctx.free_device(dev_slots_dev);
    ctx.free_device(token_ids_dev);
    ctx.free_device(ranks_dev);
    ctx.free_device(x_dev);
    ctx.free_device(moe_h_dev);
    ctx.free_device(down_acc_dev);

    std::cout << "  Passed >16 active GPU experts gate_up and down_to_rank_f32 test.\n";
}

void test_direct_ready_lease(Context& ctx) {
    std::cout << "[Hotset H1] Ready leases, cold-only supply, original rank ownership...\n";
    SyntheticExpertWeights synth;
    ModelWeights weights;
    constexpr int H = SyntheticExpertWeights::hidden_size, I = SyntheticExpertWeights::intermediate_size;
    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp;
    auto init = [&](Qwen4ExpExpertCache& cache) {
        std::string error;
        expect(cache.init(ctx, weights, 2, 4 * slot_bytes, 8, H, I, 64, &error, true), error);
        cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
        cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);
    };
    auto x = Tensor::allocate(ctx, {2, H}, dnnl::memory::data_type::bf16);
    auto h = Tensor::allocate(ctx, {4, I}, dnnl::memory::data_type::bf16);
    auto down = Tensor::allocate(ctx, {4, H}, dnnl::memory::data_type::f32);
    auto out = Tensor::allocate(ctx, {2, H}, dnnl::memory::data_type::bf16);
    auto route_weights = Tensor::allocate(ctx, {4}, dnnl::memory::data_type::f32);
    std::vector<bf16> input(2 * H), reference(2 * H), result(2 * H);
    for (int j=0;j<2*H;++j) input[j]=bf16(float((j*37)%97-48)*.003f);
    const int ids[4] = {0,1,0,3}, tokens[4] = {0,0,1,1}, ranks[4] = {0,1,0,1};
    const float ws[4] = {.75f,.25f,.125f,.875f};
    ctx.memcpy_h2d(x.data(),input.data(),input.size()*2);
    ctx.memcpy_h2d(route_weights.data(),ws,sizeof(ws));
    {
        Qwen4ExpExpertCache ref; init(ref);
        int slots[4]; ref.acquire(0,ids,4,slots).wait_and_throw();
        ops::MoeExpertSlotDev desc[4]; ref.get_slots_dev(slots,4,desc);
        auto gu = ops::moe_decode_gate_up_swiglu_grouped(ctx,desc,nullptr,x.data_as<bf16>(),h.data_as<bf16>(),4,H,I,64,nullptr,tokens,true);
        auto dn = ops::moe_decode_down_to_rank_f32(ctx,desc,ranks,h.data_as<bf16>(),down.data_as<float>(),4,2,I,H,64,tokens,nullptr,&gu,true);
        auto ev = ops::moe_ordered_accumulate_down_rows(ctx,down.data_as<float>(),route_weights.data_as<float>(),out.data_as<bf16>(),2,2,H,nullptr,nullptr,&dn);
        ref.record_consumer_reader_fences(slots,4,ev);
        ctx.memcpy_d2h(reference.data(),out.data(),reference.size()*2);
    }
    for (int kind=0;kind<3;++kind) { // all cold / mixed / all Ready
        Qwen4ExpExpertCache cache; init(cache);
        int slots[4];
        if(kind==1) cache.acquire(0,ids,1,slots).wait_and_throw();
        if(kind==2) cache.acquire(0,ids,4,slots).wait_and_throw();
        cache.reset_stats();
        auto lease=cache.BindReady(0,ids,4);
        expect(lease.ready_count==(kind==0?0:kind==1?2:4),"Ready route coverage");
        expect(lease.cold_count==4-lease.ready_count,"Cold route coverage");
        ctx.queue().memset(down.data(),0xff,4*H*sizeof(float)); // stale/unwritten ranks must fail
        auto launch=[&](bool ready,const sycl::event& dma) {
            const int n=ready?lease.ready_count:lease.cold_count;
            if(!n)return sycl::event{};
            ops::MoeExpertSlotDev desc[4];int ts[4],rs[4];
            for(int j=0;j<n;++j){int flat=ready?lease.ready_ranks[j]:lease.cold_ranks[j];
                ts[j]=flat/2;rs[j]=flat%2;desc[j]=cache.get_slot_dev(ready?lease.ready_slots[j]:slots[j]);}
            auto gu=ops::moe_decode_gate_up_swiglu_grouped(ctx,desc,nullptr,x.data_as<bf16>(),h.data_as<bf16>(),n,H,I,64,&dma,ts,true);
            return ops::moe_decode_down_to_rank_f32(ctx,desc,rs,h.data_as<bf16>(),down.data_as<float>(),n,2,I,H,64,ts,nullptr,&gu,true);
        };
        auto ev=launch(true,{});
        // Force the leased Ready slot to the eviction end while cold supply runs.
        if(kind==1)cache.demote(0,ids,1);
        auto dma=cache.AcquireCold(lease,slots);
        if(lease.cold_count)ev=launch(false,dma);
        for(int j=0;j<lease.ready_count;++j)
            expect(cache.ValidateReadyBinding(0,lease.ready_ids[j],lease.ready_slots[j],lease.ready_generations[j]),"Cold acquire evicted a leased Ready slot");
        auto merged=ops::moe_ordered_accumulate_down_rows(ctx,down.data_as<float>(),route_weights.data_as<float>(),out.data_as<bf16>(),2,2,H,nullptr,nullptr,&ev);
        cache.record_consumer_reader_fences(slots,lease.cold_count,merged);
        lease.Finish(merged);
        ctx.memcpy_d2h(result.data(),out.data(),result.size()*2);
        expect(std::memcmp(reference.data(),result.data(),result.size()*2)==0,"Split route duplicated or lost a rank/weight");
        expect(cache.stats().hit_count==uint64_t(lease.ready_count),"Bypass hit accounting");
        expect(cache.stats().acquire_calls==uint64_t(kind!=2),"All-ready called acquire");
        expect(cache.stats().direct_entries+cache.stats().cold_request_entries==4,"Accounting route partition");
    }
    auto ids_dev=Tensor::allocate(ctx,{2},dnnl::memory::data_type::s32);
    auto selected_dev=Tensor::allocate(ctx,{2},dnnl::memory::data_type::s32);
    int* mailbox=static_cast<int*>(ctx.alloc_host(4*sizeof(int)));
    float* mailbox_weights=static_cast<float*>(ctx.alloc_host(2*sizeof(float)));
    struct MailboxDrain {Context& ctx;int* ids;float* weights;~MailboxDrain(){ctx.synchronize();ctx.free_host(ids);ctx.free_host(weights);}} mailbox_drain{ctx,mailbox,mailbox_weights};
    for(int kind=0;kind<3;++kind){
        Qwen4ExpExpertCache cache;init(cache);cache.EnableResidentDeviceTables();
        int slots[4];
        if(kind==1)cache.acquire(0,ids,1,slots).wait_and_throw();
        if(kind==2)cache.acquire(0,ids,4,slots).wait_and_throw();
        ctx.queue().memset(down.data(),0xff,4*H*sizeof(float));
        for(int token=0;token<2;++token){
            cache.BeginResidentDeviceEpoch();auto epoch=cache.BeginResidentLayer(0);
            ctx.memcpy_h2d(ids_dev.data(),ids+token*2,2*sizeof(int));
            auto route=ops::moe_resident_classify(ctx,ids_dev.data_as<int>(),route_weights.data_as<float>()+token*2,
                cache.resident_layer_table(0),selected_dev.data_as<int>(),mailbox,mailbox_weights,mailbox+2,2,8);
            ops::moe_resident_gate_up(ctx,cache.resident_descriptors(),selected_dev.data_as<int>(),x.data_as<bf16>()+token*H,h.data_as<bf16>(),2,H,I,true);
            auto last=ops::moe_resident_down(ctx,cache.resident_descriptors(),selected_dev.data_as<int>(),h.data_as<bf16>(),down.data_as<float>()+token*2*H,2,H,I,true);
            route.wait_and_throw();
            expect(mailbox[0]==ids[token*2] && mailbox[1]==ids[token*2+1] && mailbox_weights[0]==ws[token*2] && mailbox_weights[1]==ws[token*2+1],"GPU mailbox changed router IDs/weights");
            auto lease=epoch.BindRoute(mailbox,mailbox+2,2);
            if(token==0)expect(lease.ready_count==(kind==0?0:kind==1?1:2),"GPU resident mask wrong");
            if(lease.cold_count){
                const auto dma=cache.AcquireCold(lease,slots);ops::MoeExpertSlotDev desc[2];
                cache.get_slots_dev(slots,lease.cold_count,desc);
                last=ops::moe_decode_gate_up_swiglu_grouped(ctx,desc,nullptr,x.data_as<bf16>()+token*H,h.data_as<bf16>(),lease.cold_count,H,I,64,&dma,nullptr,true);
                last=ops::moe_decode_down_to_rank_f32(ctx,desc,lease.cold_ranks.data(),h.data_as<bf16>(),down.data_as<float>()+token*2*H,lease.cold_count,2,I,H,64,nullptr,nullptr,&last,true);
                cache.record_consumer_reader_fences(slots,lease.cold_count,last);
            }
            auto merge=ops::moe_ordered_accumulate_down_rows(ctx,down.data_as<float>()+token*2*H,route_weights.data_as<float>()+token*2,out.data_as<bf16>()+token*H,1,2,H,nullptr,nullptr,&last);
            lease.Finish(merge);
        }
        ctx.memcpy_d2h(result.data(),out.data(),result.size()*2);
        expect(std::memcmp(reference.data(),result.data(),result.size()*2)==0,"Device resident hot/cold rank mismatch");
    }
    {
        Qwen4ExpExpertCache cache;init(cache);cache.EnableResidentDeviceTables();
        int slots[4];cache.acquire(0,ids,4,slots).wait_and_throw();
        ctx.memcpy_h2d(ids_dev.data(),ids,2*sizeof(int));
        struct Cancelled {};
        sycl::event reader;
        for(int mode=1;mode<=2;++mode){
            try{
                cache.BeginResidentDeviceEpoch();
                // Hold the in-order queue briefly so cancellation happens with
                // real weight readers submitted, not after they already retired.
                ctx.queue().submit([](sycl::handler& cgh){cgh.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(25));});});
                if(mode==1){
                    auto lease=cache.BindReady(0,ids,2);ops::MoeExpertSlotDev desc[2];cache.get_slots_dev(lease.ready_slots.data(),2,desc);
                    reader=ops::moe_decode_gate_up_swiglu_grouped(ctx,desc,nullptr,x.data_as<bf16>(),h.data_as<bf16>(),2,H,I,64,nullptr,nullptr,true);
                    throw Cancelled{};
                }else{
                    auto epoch=cache.BeginResidentLayer(0);
                    ops::moe_resident_classify(ctx,ids_dev.data_as<int>(),route_weights.data_as<float>(),cache.resident_layer_table(0),selected_dev.data_as<int>(),mailbox,mailbox_weights,mailbox+2,2,8);
                    reader=ops::moe_resident_gate_up(ctx,cache.resident_descriptors(),selected_dev.data_as<int>(),x.data_as<bf16>(),h.data_as<bf16>(),2,H,I,true);
                    throw Cancelled{};
                }
            }catch(const Cancelled&){}
            expect(reader.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete,"Cancelled resident lease released before GPU readers drained");
            expect(!cache.is_faulted(),"Cancellation poisoned a healthy cache");
        }
        int replacement[4]={4,5,6,7};cache.acquire(0,replacement,4,slots).wait_and_throw();
        expect(cache.contains(0,7),"Cancelled epoch leaked slot reservations");
    }
    {
        Qwen4ExpExpertCache cache;init(cache);
        int hot_ids[4]={0,1,2,3},hot_slots[4];cache.acquire(0,hot_ids,4,hot_slots).wait_and_throw();
        const auto old_gen=cache.get_slot_generation(hot_slots[0]);
        {
            auto lease=cache.BindReady(0,hot_ids,4);
            int cold_id=7,slot;
            bool rejected=false;try{cache.acquire(0,&cold_id,1,&slot);}catch(const std::runtime_error&){rejected=true;}
            expect(rejected && !cache.is_faulted(),"Full lease capacity must reject before mutation");
            expect(cache.ValidateReadyBinding(0,0,hot_slots[0],old_gen),"Lease identity changed");
        } // exceptional/no-submit release drains, then frees reservations
        int replacement[4]={4,5,6,7};cache.acquire(0,replacement,4,hot_slots).wait_and_throw();
        expect(!cache.ValidateReadyBinding(0,0,hot_slots[0],old_gen),"Stale generation accepted");
        int id=4;auto snapshot=cache.query_residency(0,id);
        expect(cache.ValidateReadyBinding(0,id,snapshot.gpu_slot,snapshot.gpu_generation),"Current generation rejected");
        expect(!cache.ValidateReadyBinding(0,id,snapshot.gpu_slot,old_gen),"Old generation accepted for a rebound slot");
        std::promise<void> release;auto signal=release.get_future().share();
        Qwen4ExpResidentTestAccess::HoldCopy(cache,snapshot.gpu_slot,signal);
        expect(cache.HasGpuRouteSource(0,id), "Copying expert admitted to CPU route");
        auto lease=cache.BindReady(0,&id,1);
        const bool copying_rejected=lease.ready_count==0 && lease.cold_count==1;
        release.set_value();
        expect(copying_rejected,"Copying was treated as Ready");
        cache.AcquireCold(lease,hot_slots).wait_and_throw();lease.Finish({});
        cache.EnableResidentDeviceTables();cache.BeginResidentDeviceEpoch();
        cache.acquire(1,replacement,4,hot_slots).wait_and_throw(); // changes every descriptor generation after the snapshot
        auto epoch=cache.BeginResidentLayer(1);
        int mapped[8];ctx.memcpy_d2h(mapped,cache.resident_layer_table(1),sizeof(mapped));
        for(int e:replacement)expect(mapped[e]==-1,"Stale descriptor generation published to device");
        int missing[4]={-1,-1,-1,-1};auto fresh=epoch.BindRoute(replacement,missing,4);
        expect(fresh.ready_count==0 && fresh.cold_count==4,"Old epoch silently rebound to new payload");
        cache.AcquireCold(fresh,hot_slots).wait_and_throw();fresh.Finish({});
    }
    std::cout << "[Hotset H1 PASS] all_hot=1 mixed=1 all_cold=1 copying=1 repeat_expert=1 stale_generation=1 lease_release=1 rank_exact=1\n";
    std::cout << "[Hotset H2 PASS] device_select=1 mailbox_exact=1 masked_hot=1 cold_merge=1 epoch_generation=1 rank_exact=1 cancellation_drain=1\n";
}

void test_frozen_transient(Context& ctx) {
    SyntheticExpertWeights synth;ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
    const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    expect(cache.init(ctx,weights,2,4*bytes,8,256,64,64,&error,true),error);
    for(int l=0;l<2;++l)cache.set_layer_expert_ref(l,synth.gu_ref,synth.dn_ref);
    int initial[2]={0,1},slots[4];cache.acquire(0,initial,2,slots).wait_and_throw();
    cache.FreezeReady(2);const auto identity=cache.FrozenIdentity();
    expect(cache.frozen_slots()==2,"Frozen resident count");
    cache.EnableResidentDeviceTables();cache.BeginResidentDeviceEpoch();
    {auto epoch=cache.BeginResidentLayer(0);}
    const auto metadata=cache.stats().resident_metadata_bytes;
    cache.BeginResidentDeviceEpoch();{auto epoch=cache.BeginResidentLayer(0);}
    expect(cache.stats().resident_metadata_bytes==metadata,"Immutable frozen tables were republished");
    for(int repeat=0;repeat<3;++repeat) {
        int route[3]={0,2,3};auto binding=cache.BindReady(0,route,3);
        expect(binding.ready_count==1 && binding.cold_count==2,"Cold staging became a direct hit");
        cache.AcquireCold(binding,slots).wait_and_throw();
        const auto generation=cache.get_slot_generation(slots[0]);
        // A pending real payload reader must survive immediate logical invalidation.
        auto scratch=Tensor::allocate(ctx,{32},dnnl::memory::data_type::u8);
        auto source=cache.get_slot_dev(slots[0]).gate_up_packed;
        auto reader=ctx.queue().memcpy(scratch.data(),source,32);
        cache.record_consumer_reader_fences(slots,2,reader);cache.DiscardTransient(slots,2);
        expect(!cache.contains(0,2) && !cache.ValidateReadyBinding(0,2,slots[0],generation),"Transient remained admitted");
        binding.Finish(reader);reader.wait_and_throw();
        expect(cache.FrozenIdentity()==identity && cache.IsDirectReady(0,1),"Cold fill evicted frozen hot weights");
    }
    int too_many[3]={4,5,6};bool rejected=false;
    try{cache.acquire(0,too_many,3,slots);}catch(const std::runtime_error&){rejected=true;}
    expect(rejected && !cache.is_faulted() && cache.FrozenIdentity()==identity,"Oversized staging mutated frozen cache");
    cache.ThawReady();cache.acquire(1,too_many,3,slots).wait_and_throw();
    expect(!cache.frozen(),"Thaw failed");
    std::cout<<"[C1 CACHE PASS] frozen_identity=1 transient_no_admission=1 reader_fence=1 capacity_reject=1 thaw=1\n";
}

void test_adaptive_hotset(Context& ctx) {
    SyntheticExpertWeights synth;ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
    const size_t slot_bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    expect(cache.init(ctx,weights,2,4*slot_bytes,8,256,64,64,&error,true),error);
    cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);cache.set_layer_expert_ref(1,synth.gu_ref,synth.dn_ref);
    Qwen4ExpResidentTestAccess::Adaptive(cache,{0,8});
    expect(cache.stats().hit_count==0 && cache.stats().miss_count==0 && cache.stats().acquire_calls==0,
        "Profile maintenance polluted routed demand/prefetch hit rate");
    cache.EnableResidentDeviceTables();cache.BeginResidentDeviceEpoch();
    const int incoming=2;const auto old=cache.query_residency(0,0);
    cache.ObserveHotsetRoute(0,&incoming,1,1);cache.ObserveHotsetRoute(0,&incoming,1,1);
    {auto lease=cache.BindReady(0,&incoming,1);expect(!lease.ready_count,"Incoming must start cold");}
    cache.RebalanceHotset();
    expect(!cache.IsDirectReady(0,0) && !cache.IsDirectReady(0,2),"Pending refill exposed old/new identity");
    expect(!cache.ValidateReadyBinding(0,0,old.gpu_slot,old.gpu_generation),"Retired generation remained valid");
    cache.PublishHotsetPending(true);cache.BeginResidentDeviceEpoch();
    expect(cache.IsDirectReady(0,2) && cache.IsDirectReady(1,0),"Wrong layer or missing adaptive admission");
    std::array<uint8_t,32> payload{};auto slot=cache.query_residency(0,2).gpu_slot;
    ctx.memcpy_d2h(payload.data(),cache.get_slot_dev(slot).gate_up_packed,payload.size());
    expect(!std::memcmp(payload.data(),synth.gu_packed_host.data()+2*synth.gu_packed_per_exp,payload.size()),"Published payload mismatches incoming expert");
    const auto identity=cache.FrozenIdentity();
    cache.BeginHotsetForward(0,12);expect(cache.frozen_slots()==0,"Prefill did not borrow tier");
    int ids[4]={3,4,5,6},slots[4];cache.acquire(1,ids,4,slots).wait_and_throw();
    cache.BeginHotsetForward(12,2,-1,true);
    expect(cache.FrozenIdentity()==identity && cache.IsDirectReady(0,2),"Prefill handoff failed to restore evolved tier");
    const auto& policy=cache.hotset_policy();
    expect(policy.decode[2]==2 && policy.prefill[2]==0 && policy.unique[2]==2,"Frequency accounting changed with placement");
    expect(cache.stats().epoch_swaps==1 && cache.stats().epoch_refill_bytes>0,"Refill costs omitted");
    Qwen4ExpResidentTestAccess::Prompt(cache);
    cache.BeginHotsetForward(0,4);
    int prompt_ids[4]={5,5,5,5};
    cache.ObserveHotsetRoute(0,prompt_ids,4,1);cache.ObserveHotsetRoute(1,prompt_ids,4,1);
    int preserved_slot;cache.acquire(1,prompt_ids,1,&preserved_slot).wait_and_throw();
    const auto preserved_generation=cache.query_residency(1,5).gpu_generation;
    cache.BeginHotsetForward(4,1);
    expect(cache.IsDirectReady(0,5) && cache.IsDirectReady(1,5),"Prompt handoff restored stale profile");
    expect(cache.ValidateReadyBinding(1,5,preserved_slot,preserved_generation),"Prompt handoff reloaded surviving target");
    expect(!cache.IsDirectReady(0,2) && cache.frozen_slots()==2,"Prompt handoff changed tier capacity");
    cache.BeginResidentDeviceEpoch();
    slot=cache.query_residency(0,5).gpu_slot;
    ctx.memcpy_d2h(payload.data(),cache.get_slot_dev(slot).gate_up_packed,payload.size());
    expect(!std::memcmp(payload.data(),synth.gu_packed_host.data()+5*synth.gu_packed_per_exp,payload.size()),"Prompt payload mismatches key");
    std::cout<<"[H4 CACHE PASS] retired_generation=1 publish_after_dma=1 payload=1 same_layer=1 borrow_restore=1\n";
}

void test_prompt_prefill(Context& ctx) {
    SyntheticExpertWeights synth;ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
    const size_t slot_bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    expect(cache.init(ctx,weights,2,12*slot_bytes,8,256,64,64,&error,true),error);
    cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);cache.set_layer_expert_ref(1,synth.gu_ref,synth.dn_ref);
    Qwen4ExpResidentTestAccess::Adaptive(cache,{0,1,2,3,8,9,10,11});
    Qwen4ExpResidentTestAccess::PromptPrefill(cache);
    cache.BeginHotsetForward(0,4);
    int prompt_ids[4]={5,6,5,6};
    cache.ObserveHotsetRoute(0,prompt_ids,4,1);cache.ObserveHotsetRoute(1,prompt_ids,4,1);
    cache.EndHotsetForward(4,128);
    expect(cache.frozen_slots()==4 && cache.IsDirectReady(0,6),"Prefill did not adapt borrowed tier");
    const auto refill=cache.stats().epoch_refill_bytes;
    cache.EndHotsetForward(4,4);
    expect(cache.stats().epoch_refill_bytes==refill,"Final prefill performed unused refresh");
    cache.BeginHotsetForward(4,1,0);
    cache.ObserveHotsetRoute(0,prompt_ids,1,1);cache.ObserveHotsetRoute(1,prompt_ids,1,1);
    cache.EndHotsetForward(1,0);
    expect(cache.frozen_slots()==4 && cache.hotset_policy().decode[5]==0 && cache.hotset_policy().prompt_rows==5,
        "Single-row prefill tail was misclassified as decode");
    // The first decode operation can verify several tokens at once. It must
    // finish prompt handoff, then retain the decode tier on later windows.
    cache.BeginHotsetForward(5,4,-1,true);
    // Expert 0 is a valid victim in both layers; expert 1 is outside this
    // two-swap budget and must survive the temporary prefill tier.
    expect(cache.frozen_slots()==8 && cache.IsDirectReady(1,1),"Prefill overwrote logical decode tier");
    const auto decode_refill=cache.stats().epoch_refill_bytes;
    const auto prompt_rows=cache.hotset_policy().prompt_rows;
    for (int rows : {2,3,4,1}) {
        cache.BeginHotsetForward(9,rows,-1,true);
        expect(cache.frozen_slots()==8 && !cache.hotset_policy().collecting_prompt &&
            cache.hotset_policy().prompt_rows==prompt_rows && cache.stats().epoch_refill_bytes==decode_refill,
            "Decode verification borrowed slots or started a new prompt");
    }
    cache.BeginHotsetForward(9,4,0);
    expect(cache.frozen_slots()==4 && cache.hotset_policy().collecting_prompt,
        "An explicit follow-up prefill no longer borrows its tier");
    std::cout<<"[H5 PREFILL PASS] chunk_admission=1 final_chunk_skip=1 decode_tier_preserved=1\n";
}

void test_ready_adoption(Context& ctx) {
    for(int cold_policy=0;cold_policy<=2;++cold_policy) {
        SyntheticExpertWeights synth;ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
        const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
        expect(cache.init(ctx,weights,2,12*bytes,8,256,64,64,&error,true),error);
        for(int l=0;l<2;++l)cache.set_layer_expert_ref(l,synth.gu_ref,synth.dn_ref);
        Qwen4ExpResidentTestAccess::Adaptive(cache,{0,1,2,3,8,9,10,11});
        Qwen4ExpResidentTestAccess::Adopt(cache,1);
        // Exercise all three physical cold-victim structures.
        Qwen4ExpResidentTestAccess::ColdPolicy(cache,cold_policy);
        cache.BeginHotsetForward(0,8,128);
        cache.EnableResidentDeviceTables();cache.BeginResidentDeviceEpoch();
        const int ids[8]={5,5,5,5,6,6,6,6};cache.ObserveHotsetRoute(0,ids,8,1);
        int pending_id=6,pending_slot;cache.acquire(0,&pending_id,1,&pending_slot).wait_and_throw();
        std::promise<void> release;auto signal=release.get_future().share();
        Qwen4ExpResidentTestAccess::HoldCopy(cache,pending_slot,signal);
        cache.RetirePrefillSlots(&pending_slot,1);
        const bool rejected_copy=cache.stats().ready_adoptions==0 && !cache.IsDirectReady(0,6);
        release.set_value();
        expect(rejected_copy,"Pending DMA was admitted or waited on");
        int protected_ids[4]={0,1,2,3};
        {auto protected_tier=cache.BindReady(0,protected_ids,4);
            expect(protected_tier.ready_count==4,"Protected fixture tier");
            int id=5,slot;cache.acquire(0,&id,1,&slot).wait_and_throw();cache.RetirePrefillSlots(&slot,1);
            expect(cache.stats().ready_adoptions==0 && !cache.contains(0,5),"Admission demoted an active lease");
        }
        const auto old=cache.query_residency(0,0);
        int cold[2]={5,6},slots[2];cache.acquire(0,cold,2,slots).wait_and_throw();
        const auto incoming=cache.query_residency(0,5);
        // Demoted payload must remain protected by its outstanding real reader.
        auto scratch=Tensor::allocate(ctx,{32},dnnl::memory::data_type::u8);
        auto reader=ctx.queue().memcpy(scratch.data(),cache.get_slot_dev(old.gpu_slot).gate_up_packed,32);
        cache.record_consumer_reader_fences(&old.gpu_slot,1,reader);
        const auto before=cache.stats();cache.RetirePrefillSlots(slots,2);
        expect(cache.stats().ready_adoptions==1 && cache.frozen_slots()==4,"Ready admission exceeded budget/capacity");
        expect(cache.IsDirectReady(0,5) && !cache.IsDirectReady(0,0) && !cache.contains(0,6),"Ready admission/discard partition");
        expect(cache.ValidateReadyBinding(0,5,incoming.gpu_slot,incoming.gpu_generation),"Admission moved incoming payload");
        expect(cache.ValidateReadyBinding(0,0,old.gpu_slot,old.gpu_generation),"Demotion invalidated live payload");
        expect(cache.stats().total_transferred_bytes==before.total_transferred_bytes && cache.stats().epoch_refill_bytes==before.epoch_refill_bytes,"Admission copied payload");
        // Descriptor snapshot predating the demand fill cannot publish the new key.
        {auto epoch=cache.BeginResidentLayer(0);int table[8];ctx.memcpy_d2h(table,cache.resident_layer_table(0),sizeof(table));expect(table[5]==-1,"Stale descriptor exposed promoted payload");}
        cache.BeginResidentDeviceEpoch();
        {auto epoch=cache.BeginResidentLayer(0);int table[8];ctx.memcpy_d2h(table,cache.resident_layer_table(0),sizeof(table));expect(table[5]==incoming.gpu_slot && table[0]==-1,"Updated device tier mismatches ownership");}
        int replace[7]={0,1,2,3,4,6,7},replacement_slots[7];cache.acquire(1,replace,7,replacement_slots).wait_and_throw();
        reader.wait_and_throw();std::array<uint8_t,32> readback{};ctx.memcpy_d2h(readback.data(),scratch.data(),32);
        expect(!std::memcmp(readback.data(),synth.gu_packed_host.data(),32),"Demoted reader observed overwritten payload");
        cache.BeginHotsetForward(8,1);
        expect(cache.IsDirectReady(0,5) && cache.frozen_slots()==8,"Borrow restore undid ready adoption");
    }
    std::cout<<"[H6 READY PASS] no_copy=1 stable_generation=1 bounded_same_layer=1 cold_topology=1 descriptor_generation=1 reader_fence=1 borrow_restore=1\n";
}

void test_host_ram_zero(Context& ctx) {
    struct EnvScope {
        std::array<std::string,3> previous;
        const std::array<const char*,3> keys={"AILA_Q4E_HOT_RAM_MB","AILA_TEST_Q4E_HOST_POOL","AILA_TEST_Q4E_HOST_SLOTS"};
        EnvScope() {
            for(size_t i=0;i<keys.size();++i)if(const char* value=std::getenv(keys[i]))previous[i]=value;
            _putenv_s(keys[1],"1");_putenv_s(keys[2],"0");
        }
        ~EnvScope(){for(size_t i=0;i<keys.size();++i)_putenv_s(keys[i],previous[i].c_str());}
    } env_scope;
    SyntheticExpertWeights synth;ModelWeights weights;
    const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    // The test gate exercises the full-model host-budget branch without a 1 GiB GPU allocation.
    // Zero test slots prevents the independent test-slot override from enabling the pool.
    const char* settings[]={"0","-1","1",""};
    for(int mode=0;mode<4;++mode) {
        _putenv_s("AILA_Q4E_HOT_RAM_MB",settings[mode]);
        const bool enabled=mode>=2;
        Qwen4ExpExpertCache cache;std::string err;
        expect(cache.init(ctx,weights,2,2*bytes,8,256,64,64,&err,true),"Host RAM fixture init: "+err);
        cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);
        cache.set_reader_type(aila::ExpertReaderType::Mmap);
        expect(cache.host_pool_enabled()==enabled,"Host RAM explicit zero/negative/default enablement");
        expect(cache.num_host_slots()==(enabled ? 16 : 0),"Host RAM must use budget rather than zero test-slot override");
        const int id=3;
        const auto source=cache.MappedExpertHostSource(0,id);
        expect(source.gate_up_packed==synth.gu_packed_host.data()+id*synth.gu_packed_per_exp &&
            source.down_packed==synth.dn_packed_host.data()+id*synth.dn_packed_per_exp &&
            source.gate_up_nested.codes==synth.gu_absmax_codes.data()+id*synth.gu_absmax_per_exp &&
            source.down_nested.codes==synth.dn_absmax_codes.data()+id*synth.dn_absmax_per_exp,
            "Host RAM disabled invalidated mapped source");
        const auto ticket=cache.prepare_host(0,id);
        expect(ticket.ready==enabled && (ticket.hslot>=0)==enabled,"Host RAM preparation must follow pool enablement");
        {
            auto lease=cache.acquire_host_lease(0,id);
            expect(lease.is_valid()==enabled,"Host RAM lease must follow pool enablement");
            if(enabled) {
                expect(!std::memcmp(lease.slot_host().gate_up_packed,source.gate_up_packed,synth.gu_packed_per_exp) &&
                    !std::memcmp(lease.slot_host().down_packed,source.down_packed,synth.dn_packed_per_exp),
                    "Positive Host RAM budget changed host payload");
            }
        }
        int slot=-1;cache.acquire(0,&id,1,&slot).wait_and_throw();
        std::vector<uint8_t> expected(bytes),actual(bytes);size_t offset=0;
        auto append=[&](const std::vector<uint8_t>& data,size_t per) {
            std::memcpy(expected.data()+offset,data.data()+size_t(id)*per,per);offset+=per;
        };
        append(synth.gu_packed_host,synth.gu_packed_per_exp);append(synth.gu_absmax_codes,synth.gu_absmax_per_exp);
        append(synth.dn_packed_host,synth.dn_packed_per_exp);append(synth.dn_absmax_codes,synth.dn_absmax_per_exp);
        ctx.memcpy_d2h(actual.data(),cache.get_slot_dev(slot).gate_up_packed,bytes);
        expect(actual==expected,"Host RAM enablement changed acquire payload");
        expect(cache.host_contains(0,id)==enabled,"Disabled Host RAM acquire populated host pool");
    }
    std::cout<<"[HOST RAM ZERO PASS] zero_disabled=1 negative_disabled=1 positive_enabled=1 default_enabled=1 mapped_source=1 exact_payload=1\n";
}

void test_gpu_mapped_cold_dma(Context& ctx) {
    struct EnvScope {
        const std::array<const char*,3> keys={"AILA_Q4E_HOT_RAM_MB","AILA_TEST_Q4E_HOST_POOL","AILA_TEST_Q4E_HOST_SLOTS"};
        std::array<std::string,3> previous;
        EnvScope() {
            for(size_t i=0;i<keys.size();++i) {
                if(const char* value=std::getenv(keys[i]))previous[i]=value;
                _putenv_s(keys[i],"0");
            }
        }
        ~EnvScope(){for(size_t i=0;i<keys.size();++i)_putenv_s(keys[i],previous[i].c_str());}
    } env_scope;
    SyntheticExpertWeights synth;ModelWeights weights;
    const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    auto init=[&](Qwen4ExpExpertCache& cache,int slots,bool enabled) {
        std::string err;
        expect(cache.init(ctx,weights,2,size_t(slots)*bytes,8,256,64,64,&err,true),"GPU mapped fixture init: "+err);
        expect(!cache.host_pool_enabled(),"GPU mapped fixture requires HostRam0");
        for(int l=0;l<2;++l)cache.set_layer_expert_ref(l,synth.gu_ref,synth.dn_ref);
        cache.set_reader_type(aila::ExpertReaderType::Mmap);
        Qwen4ExpResidentTestAccess::GpuMappedDma(cache,enabled);
        int hot=0,slot=-1;cache.acquire(0,&hot,1,&slot).wait_and_throw();cache.FreezeReady(1);
        cache.SetDemandForwardRows(1);
    };
    auto payload=[&](Qwen4ExpExpertCache& cache,int slot,int id) {
        const auto desc=cache.get_slot_dev(slot);
        std::vector<uint8_t> expected(bytes),actual(bytes);size_t offset=0;
        auto append=[&](const std::vector<uint8_t>& source,size_t per) {
            std::memcpy(expected.data()+offset,source.data()+size_t(id)*per,per);offset+=per;
        };
        append(synth.gu_packed_host,synth.gu_packed_per_exp);append(synth.gu_absmax_codes,synth.gu_absmax_per_exp);
        append(synth.dn_packed_host,synth.dn_packed_per_exp);append(synth.dn_absmax_codes,synth.dn_absmax_per_exp);
        ctx.memcpy_d2h(actual.data(),desc.gate_up_packed,bytes);
        expect(actual==expected,"GPU mapped four-segment payload changed expert identity");
        expect(desc.gate_up_nested.codes==desc.gate_up_packed+synth.gu_packed_per_exp &&
            desc.down_packed==desc.gate_up_nested.codes+synth.gu_absmax_per_exp &&
            desc.down_nested.codes==desc.down_packed+synth.dn_packed_per_exp,
            "GPU mapped descriptor offsets changed");
        expect(desc.gate_up_nested.blocksize==256 && desc.down_nested.blocksize==256 &&
            desc.gate_up_nested.offset==synth.gu_ref.quant_state.nested_offset &&
            desc.down_nested.offset==synth.dn_ref.quant_state.nested_offset,"GPU mapped descriptor quantization changed");
        float gu_scale=0,dn_scale=0;
        ctx.memcpy_d2h(&gu_scale,desc.gate_up_nested.nested_absmax,sizeof(float));
        ctx.memcpy_d2h(&dn_scale,desc.down_nested.nested_absmax,sizeof(float));
        expect(gu_scale==synth.gu_nested_absmax[size_t(id)*synth.gu_absmax_per_exp/256] &&
            dn_scale==synth.dn_nested_absmax[size_t(id)*synth.dn_absmax_per_exp/256],"GPU mapped nested expert metadata changed");
    };
    for(bool enabled:{false,true}) {
        Qwen4ExpExpertCache cache;init(cache,3,enabled);const auto identity=cache.FrozenIdentity();
        const auto before=cache.stats();const auto reads=cache.reader()->stats().read_count;
        int ids[3]={0,3,6},slots[2]={-1,-1};auto route=cache.BindReady(0,ids,3);
        expect(route.ready_count==1 && route.cold_count==2,"GPU mapped route rank changed");
        auto done=cache.AcquireCold(route,slots);done.wait_and_throw();
        for(int c=0;c<2;++c)payload(cache,slots[c],ids[route.cold_ranks[c]]);
        const auto a=cache.get_slot_dev(slots[0]),b=cache.get_slot_dev(slots[1]);
        expect(b.gate_up_nested.nested_absmax-a.gate_up_nested.nested_absmax==
            (6-3)*ptrdiff_t(synth.gu_absmax_per_exp/256),"GPU mapped GU descriptor lost expert identity");
        expect(b.down_nested.nested_absmax-a.down_nested.nested_absmax==
            (6-3)*ptrdiff_t(synth.dn_absmax_per_exp/256),"GPU mapped down descriptor lost expert identity");
        expect(cache.stats().gpu_mmap_dma_experts-before.gpu_mmap_dma_experts==uint64_t(enabled?2:0) &&
            cache.stats().gpu_mmap_dma_copies-before.gpu_mmap_dma_copies==uint64_t(enabled?8:0) &&
            cache.stats().gpu_mmap_dma_bytes-before.gpu_mmap_dma_bytes==uint64_t(enabled?2*bytes:0),"GPU mapped DMA counters changed");
        expect(cache.stats().total_transferred_bytes-before.total_transferred_bytes==2*bytes &&
            cache.dma_timing_stats().last_memcpy_count==uint32_t(enabled?8:2) &&
            cache.dma_timing_stats().last_batch_bytes==2*bytes,"GPU mapped DMA count multiplied payload bytes");
        expect(enabled ? cache.reader()->stats().read_count==reads : cache.reader()->stats().read_count>reads,
            "GPU mapped candidate failed to bypass reader or changed opt-out");
        expect(cache.FrozenIdentity()==identity,"GPU mapped DMA changed hot placement");
        cache.DiscardTransient(slots,2);route.Finish(done);
        // Ordinary acquire and multi-row cold service retain the old reader.
        const auto mapped=cache.stats().gpu_mmap_dma_experts;
        const auto ordinary_reads=cache.reader()->stats().read_count;
        int id=2,slot=-1;cache.acquire(0,&id,1,&slot).wait_and_throw();
        expect(cache.stats().gpu_mmap_dma_experts==mapped && cache.reader()->stats().read_count>ordinary_reads,
            "GPU mapped DMA escaped explicit cold service");
        const auto old_reads=cache.reader()->stats().read_count;
        cache.SetDemandForwardRows(2);id=7;
        {auto fallback=cache.BindReady(0,&id,1);auto ev=cache.AcquireCold(fallback,&slot);ev.wait_and_throw();fallback.Finish(ev);}
        expect(cache.stats().gpu_mmap_dma_experts==mapped && cache.reader()->stats().read_count>old_reads,
            "GPU mapped DMA escaped single-row cold service");
        cache.SetDemandForwardRows(1);
        auto invalid=synth.gu_ref;invalid.quant_state.shape={7,128,256};
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,invalid,synth.dn_ref);
        id=4;
        {auto fallback=cache.BindReady(0,&id,1);auto ev=cache.AcquireCold(fallback,&slot);ev.wait_and_throw();fallback.Finish(ev);}
        expect(cache.stats().gpu_mmap_dma_experts==mapped,"GPU mapped invalid original geometry failed to fall back");
        payload(cache,slot,id);
    }
    {
        Qwen4ExpExpertCache cache;init(cache,2,true);const auto identity=cache.FrozenIdentity();
        std::promise<void> release;auto signal=release.get_future().share();
        auto held=Qwen4ExpResidentTestAccess::HoldCopyQueue(cache,signal);
        int id=3,slot=-1;auto route=cache.BindReady(0,&id,1);
        auto done=cache.AcquireCold(route,&slot);
        const auto resident=cache.query_residency(0,id);
        const bool pending=resident.gpu_copying && !resident.gpu_ready &&
            done.get_info<sycl::info::event::command_execution_status>()!=sycl::info::event_command_status::complete;
        const auto generation=cache.get_slot_generation(slot);
        cache.DiscardTransient(&slot,1);route.Finish({});
        int next_id=6,next_slot=-1;auto next=cache.BindReady(0,&next_id,1);
        auto reuse=std::async(std::launch::async,[&] {return cache.AcquireCold(next,&next_slot);});
        const bool dma_blocked=reuse.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout;
        release.set_value();held.wait_and_throw();auto next_done=reuse.get();next_done.wait_and_throw();
        expect(pending && dma_blocked,"GPU mapped slot was consumed or overwritten before four-segment DMA completion");
        expect(next_slot==slot && cache.get_slot_generation(next_slot)>generation &&
            !cache.ValidateReadyBinding(0,3,slot,generation),"GPU mapped reuse accepted stale generation");
        payload(cache,next_slot,next_id);next.Finish(next_done);
        // A pending GPU consumer survives logical invalidation and blocks the
        // next physical write, independently of the already completed DMA.
        std::promise<void> reader_release;auto reader_signal=reader_release.get_future().share();
        auto reader=ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.host_task([reader_signal]{reader_signal.wait_for(std::chrono::seconds(10));});
        });
        cache.record_consumer_reader_fences(&next_slot,1,reader);cache.DiscardTransient(&next_slot,1);
        int final_id=7,final_slot=-1;auto final_route=cache.BindReady(0,&final_id,1);
        auto writer=std::async(std::launch::async,[&] {return cache.AcquireCold(final_route,&final_slot);});
        const bool reader_blocked=writer.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout;
        reader_release.set_value();reader.wait_and_throw();auto final_done=writer.get();final_done.wait_and_throw();
        expect(reader_blocked && final_slot==next_slot,"GPU mapped reuse dropped prior consumer fence");
        payload(cache,final_slot,final_id);final_route.Finish(final_done);
        expect(cache.FrozenIdentity()==identity && !cache.is_faulted(),"GPU mapped fenced reuse poisoned hot identity");
    }
    {
        auto cache=std::make_unique<Qwen4ExpExpertCache>();init(*cache,2,true);
        std::promise<void> release;auto signal=release.get_future().share();
        auto held=Qwen4ExpResidentTestAccess::HoldCopyQueue(*cache,signal);
        int id=5,slot=-1;auto route=cache->BindReady(0,&id,1);
        auto done=cache->AcquireCold(route,&slot);route.Finish({});
        auto cleanup=std::async(std::launch::async,[owned=std::move(cache)]()mutable {owned.reset();});
        const bool blocked=cleanup.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout;
        release.set_value();held.wait_and_throw();cleanup.get();
        expect(blocked && done.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete,
            "GPU mapped cleanup released allocations before copy queue drained");
    }
    std::cout<<"[GPU MAPPED DMA PASS] four_segments=1 original_nf4=1 descriptors=1 expert_identity=1 opt_out=1 reader_bypass=1 count_bytes=1 cold_single_row_only=1 invalid_fallback=1 pending_not_ready=1 dma_reuse_fence=1 consumer_fence=1 generation=1 frozen_identity=1 cleanup_drain=1\n";
}

void test_short_demand_parts(Context& ctx) {
    struct EnvScope {
        std::array<std::string,3> previous;
        const std::array<const char*,3> keys={"AILA_Q4E_FILL_PARTS","AILA_Q4E_FILL_SHORT_PARTS","AILA_TEST_Q4E_HOST_POOL"};
        EnvScope(){for(size_t i=0;i<keys.size();++i)if(const char* value=std::getenv(keys[i]))previous[i]=value;_putenv_s(keys[0],"8");}
        ~EnvScope(){for(size_t i=0;i<keys.size();++i)_putenv_s(keys[i],previous[i].c_str());}
    } env_scope;
    SyntheticExpertWeights synth;ModelWeights weights;
    const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    for(bool host_pool : {false,true})for(int override_parts : {0,1,3,32}) {
        _putenv_s("AILA_TEST_Q4E_HOST_POOL",host_pool ? "1" : "0");
        const auto setting=std::to_string(override_parts);_putenv_s("AILA_Q4E_FILL_SHORT_PARTS",setting.c_str());
        Qwen4ExpExpertCache cache;std::string err;
        expect(cache.init(ctx,weights,2,2*bytes,8,256,64,64,&err,true),"Short fill fixture init: "+err);
        for(int l=0;l<2;++l)cache.set_layer_expert_ref(l,synth.gu_ref,synth.dn_ref);
        cache.set_reader_type(aila::ExpertReaderType::Mmap);
        expect(cache.host_pool_enabled() == host_pool,"Short fill fixture host mode");
        cache.SetDemandForwardRows(41);
        Qwen4ExpExpertCache moved(std::move(cache));cache=std::move(moved);
        auto verify=[&](int e,int slot) {
            std::vector<uint8_t> expected(bytes),actual(bytes);
            size_t off=0;
            auto append=[&](const std::vector<uint8_t>& src,size_t per) {std::memcpy(expected.data()+off,src.data()+size_t(e)*per,per);off+=per;};
            append(synth.gu_packed_host,synth.gu_packed_per_exp);append(synth.gu_absmax_codes,synth.gu_absmax_per_exp);
            append(synth.dn_packed_host,synth.dn_packed_per_exp);append(synth.dn_absmax_codes,synth.dn_absmax_per_exp);
            ctx.memcpy_d2h(actual.data(),cache.get_slot_dev(slot).gate_up_packed,bytes);
            expect(actual == expected,"Short fill changed partition payload");
        };
        // The first operation after move must retain the recorded short context.
        int id=0,slot=-1;auto reads=cache.reader()->stats().read_count;auto short_fills=cache.stats().demand_short_part_fills;
        cache.acquire(0,&id,1,&slot).wait_and_throw();verify(id,slot);
        expect(cache.reader()->stats().read_count-reads == uint64_t(override_parts ? override_parts : 8),"Moved short fill context lost");
        expect(cache.stats().demand_short_part_fills-short_fills == uint64_t(override_parts != 0),"Short fill accounting");
        const int original_slot=slot;const auto original_generation=cache.get_slot_generation(slot);
        auto snapshot=Tensor::allocate(ctx,{static_cast<int64_t>(synth.gu_packed_per_exp)},dnnl::memory::data_type::u8);
        ctx.queue().submit([](sycl::handler& cgh){cgh.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(5));});});
        auto reader=ctx.queue().memcpy(snapshot.data(),cache.get_slot_dev(slot).gate_up_packed,synth.gu_packed_per_exp);
        cache.record_consumer_reader_fences(&slot,1,reader);
        const int rows[4]={1,64,65,0};
        for(int n=0;n<4;++n) {
            id=n+1;cache.SetDemandForwardRows(rows[n]);reads=cache.reader()->stats().read_count;
            cache.acquire(0,&id,1,&slot).wait_and_throw();verify(id,slot);
            const int expected=override_parts && rows[n]>0 && rows[n]<=64 ? override_parts : 8;
            expect(cache.reader()->stats().read_count-reads == uint64_t(expected),"Short parts threshold/disabled default changed");
        }
        reader.wait_and_throw();std::vector<uint8_t> saved(synth.gu_packed_per_exp);ctx.memcpy_d2h(saved.data(),snapshot.data(),saved.size());
        expect(!std::memcmp(saved.data(),synth.gu_packed_host.data(),saved.size()),"Short fill overwrote live GPU reader");
        expect(cache.get_slot_generation(original_slot)>original_generation && !cache.contains(0,0),"Short fill reused stale generation");
        cache.SetDemandForwardRows(41);id=5;reads=cache.reader()->stats().read_count;short_fills=cache.stats().demand_short_part_fills;
        Qwen4ExpResidentTestAccess::MaintenanceAcquire(cache,0,&id,1,&slot).wait_and_throw();verify(id,slot);
        expect(cache.reader()->stats().read_count-reads == 8 && cache.stats().demand_short_part_fills == short_fills,"Profile/maintenance used short override");
        id=6;reads=cache.reader()->stats().read_count;cache.prefetch_async(0,&id,1,1);cache.wait_prefetch();
        expect(cache.reader()->stats().read_count-reads == 8,"Prefetch changed global partition");
        cache.acquire(0,&id,1,&slot).wait_and_throw();verify(id,slot);
    }
    std::cout<<"[SHORT FILL PARTS PASS] exact_payload=1 odd_partitions=1 threshold=1 disabled_default=1 maintenance_global=1 prefetch_global=1 move=1 generation=1 reader_fence=1 host_and_staging=1\n";
}

void test_cpu_packed_source_cache(Context& ctx) {
    SyntheticExpertWeights synth;ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
    expect(cache.init(ctx,weights,2,4*32768,8,256,64,64,&error,true),"CPU packed fixture init: "+error);
    cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);
    const size_t bytes=synth.gu_packed_per_exp+synth.dn_packed_per_exp;
    const auto device_count=ctx.device_allocation_count(),host_count=ctx.pinned_host_allocation_count();
    bool cached=true;auto original=cache.MappedExpertHostSource(0,3);
    cache.CpuExpertHostSource(0,3,&cached);expect(!cached,"Disabled CPU source cache reported a hit");
    cache.BuildCpuPackedSourceCache(bytes+bytes/2,{3,1});
    expect(cache.cpu_packed_source_bytes()==bytes && cache.cpu_packed_source_slots()==1,"CPU packed budget floor");
    auto source=cache.CpuExpertHostSource(0,3,&cached);expect(cached,"Selected CPU source not cached");
    expect(source.gate_up_packed!=original.gate_up_packed && source.down_packed!=original.down_packed,"CPU packed source must own copies");
    expect(!std::memcmp(source.gate_up_packed,original.gate_up_packed,synth.gu_packed_per_exp) &&
        !std::memcmp(source.down_packed,original.down_packed,synth.dn_packed_per_exp),"CPU packed byte identity");
    expect(source.gate_up_nested.codes==original.gate_up_nested.codes && source.down_nested.nested_absmax==original.down_nested.nested_absmax &&
        source.gate_up_nested.nested_quant_map==original.gate_up_nested.nested_quant_map && source.down_nested.offset==original.down_nested.offset,"CPU packed metadata must retain original nested layout");
    const auto stored=source.gate_up_packed;
    for(const auto& keys:std::vector<std::vector<int>>{{3,3},{-1},{16},{8}}) {
        bool rejected=false;try{cache.BuildCpuPackedSourceCache(2*bytes,keys);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && cache.CpuExpertHostSource(0,3).gate_up_packed==stored,"Invalid replacement changed prior CPU cache");
    }
    cache.CpuExpertHostSource(0,1,&cached);expect(!cached,"Budget-excluded source reported cached");
    expect(!cache.CpuExpertHostSource(-1,0,&cached).gate_up_packed && !cached,"Invalid CPU source lookup");
    expect(ctx.device_allocation_count()==device_count && ctx.pinned_host_allocation_count()==host_count,"CPU packed initial build allocated USM");
    Qwen4ExpExpertCache moved(std::move(cache));cache=std::move(moved);
    expect(cache.CpuExpertHostSource(0,3,&cached).gate_up_packed==stored && cached,"CPU cache move lost immutable byte ownership");
    auto rebound=synth.gu_ref;auto changed=synth.gu_packed_host;changed[3*synth.gu_packed_per_exp]^=0x81;
    rebound.packed_weight.ptr=changed.data();cache.set_layer_expert_ref(0,rebound,synth.dn_ref);
    source=cache.CpuExpertHostSource(0,3,&cached);
    expect(!cached && source.gate_up_packed==changed.data()+3*synth.gu_packed_per_exp,"Ref replacement consumed stale copied weights");
    cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);
    const auto stable_device_count=ctx.device_allocation_count(),stable_host_count=ctx.pinned_host_allocation_count();
    cache.BuildCpuPackedSourceCache(2*bytes,{1,3});
    for(int id:{1,3}) {
        original=cache.MappedExpertHostSource(0,id);source=cache.CpuExpertHostSource(0,id,&cached);
        expect(cached && !std::memcmp(source.gate_up_packed,original.gate_up_packed,synth.gu_packed_per_exp) &&
            !std::memcmp(source.down_packed,original.down_packed,synth.dn_packed_per_exp),"Multi-slot CPU packed offsets");
        std::vector<bf16> x(256),h0(64),h1(64);std::vector<float> y0(256),y1(256);
        for(int i=0;i<256;++i)x[i]=bf16(float(i%23-11)*.03125f);
        ops::cpu::cpu_expert_forward(original,x.data(),y0.data(),1,h0.data(),nullptr,1,false,256,64,256,64,ops::cpu::CpuKernelBackend::Avx2);
        ops::cpu::cpu_expert_forward(source,x.data(),y1.data(),1,h1.data(),nullptr,1,false,256,64,256,64,ops::cpu::CpuKernelBackend::Avx2);
        expect(!std::memcmp(h0.data(),h1.data(),64*sizeof(bf16)) && !std::memcmp(y0.data(),y1.data(),256*sizeof(float)),"CPU packed compute bitwise");
    }
    const auto retained=cache.CpuExpertHostSource(0,3).gate_up_packed;
    expect(cache.RefreshCpuPackedSourceCache({3,1})==0 && cache.CpuExpertHostSource(0,3).gate_up_packed==retained,"CPU packed refresh must reuse retained keys");
    expect(cache.RefreshCpuPackedSourceCache({3,5})==bytes && cache.CpuExpertHostSource(0,3).gate_up_packed==retained,"CPU packed refresh copied retained source");
    source=cache.CpuExpertHostSource(0,5,&cached);original=cache.MappedExpertHostSource(0,5);
    expect(cached && !std::memcmp(source.gate_up_packed,original.gate_up_packed,synth.gu_packed_per_exp) &&
        !std::memcmp(source.down_packed,original.down_packed,synth.dn_packed_per_exp),"CPU packed refresh source bytes");
    cache.CpuExpertHostSource(0,1,&cached);expect(!cached,"CPU packed refresh retained unselected key");
    for(const auto& keys:std::vector<std::vector<int>>{{5,5},{-1},{16},{8}}) {
        bool rejected=false;try{cache.RefreshCpuPackedSourceCache(keys);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && cache.CpuExpertHostSource(0,3).gate_up_packed==retained,"Invalid CPU refresh changed published source");
    }
    expect(cache.RefreshCpuPackedSourceCache({3})==0 && cache.cpu_packed_source_bytes()==2*bytes,"CPU packed refresh changed arena budget");
    cache.CpuExpertHostSource(0,5,&cached);expect(!cached,"CPU packed short refresh retained old key");
    expect(cache.RefreshCpuPackedSourceCache({})==0,"CPU packed empty refresh copied bytes");
    cache.CpuExpertHostSource(0,3,&cached);expect(!cached,"CPU packed empty refresh did not clear keys");
    expect(cache.RefreshCpuPackedSourceCache({1,3})==2*bytes,"CPU packed refresh after empty selection");
    cache.BuildCpuPackedSourceCache(0,{});
    expect(cache.cpu_packed_source_bytes()==0 && cache.cpu_packed_source_slots()==0 &&
        cache.CpuExpertHostSource(0,3,&cached).gate_up_packed==cache.MappedExpertHostSource(0,3).gate_up_packed && !cached,"CPU packed disable must release/fall back");
    expect(ctx.device_allocation_count()==stable_device_count && ctx.pinned_host_allocation_count()==stable_host_count,"CPU packed cache allocated USM");
    std::cout<<"[CPU PACKED SOURCE CACHE PASS] original_bytes=1 nested_metadata=1 budget_floor=1 invalid_atomic=1 move=1 ref_rebind=1 exact_compute=1 no_usm=1 disabled_release=1\n";
}

void test_mapped_expert_source(Context& ctx) {
    SyntheticExpertWeights synth; ModelWeights weights;
    for (bool compact : {false,true}) {
        Qwen4ExpExpertCache cache;std::string err;
        expect(cache.init(ctx,weights,2,4*32768,8,256,64,64,&err,compact),"Mapped fixture init: "+err);
        cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);
        const auto before=cache.stats();
        auto source=cache.MappedExpertHostSource(0,3);
        expect(source.gate_up_packed == synth.gu_packed_host.data()+3*synth.gu_packed_per_exp &&
            source.down_packed == synth.dn_packed_host.data()+3*synth.dn_packed_per_exp,"Mapped packed expert offsets");
        expect(!source.gate_up_absmax && !source.down_absmax &&
            source.gate_up_nested.codes == synth.gu_absmax_codes.data()+3*synth.gu_absmax_per_exp &&
            source.down_nested.codes == synth.dn_absmax_codes.data()+3*synth.dn_absmax_per_exp &&
            source.gate_up_nested.nested_absmax == synth.gu_nested_absmax.data()+3*(synth.gu_absmax_per_exp/256) &&
            source.down_nested.nested_absmax == synth.dn_nested_absmax.data()+3*(synth.dn_absmax_per_exp/256),"Mapped nested view must use original codes even for expanded cache");
        expect(!cache.host_contains(0,3) && cache.stats().host_pool_misses == before.host_pool_misses &&
            cache.stats().total_transferred_bytes == before.total_transferred_bytes,"Mapped view performed host fill/H2D");
        expect(!cache.MappedExpertHostSource(1,0).gate_up_packed &&
            !cache.MappedExpertHostSource(-1,0).gate_up_packed && !cache.MappedExpertHostSource(0,8).gate_up_packed,"Mapped invalid refs/id accepted");
        auto bad=synth.gu_ref;bad.packed_weight.bytes=3*synth.gu_packed_per_exp;
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,bad,synth.dn_ref);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped truncated packed source accepted");
        bad=synth.gu_ref;bad.nested_absmax.bytes=sizeof(float);
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,bad,synth.dn_ref);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped truncated nested scale source accepted");
        bad=synth.gu_ref;bad.quant_state.nested_blocksize=257;
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,bad,synth.dn_ref);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped unaligned nested expert boundary accepted");
        auto gu_flat=synth.gu_ref,dn_flat=synth.dn_ref;
        gu_flat.quant_state.shape={8*128,256};dn_flat.quant_state.shape={8*256,64};
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,gu_flat,dn_flat);
        auto flat=cache.MappedExpertHostSource(0,3);
        expect(flat.gate_up_packed == source.gate_up_packed && flat.down_packed == source.down_packed &&
            flat.gate_up_nested.codes == source.gate_up_nested.codes,"Mapped valid flattened geometry rejected");
        // Same numel and byte spans, but a different logical row/column stride.
        bad=synth.gu_ref;bad.quant_state.shape={8,256,128};
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,bad,synth.dn_ref);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped rank3 wrong expert geometry accepted");
        bad=gu_flat;bad.quant_state.shape={8*256,128};
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,bad,dn_flat);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped flattened wrong expert geometry accepted");
        bad=synth.dn_ref;bad.quant_state.shape={4,512,64};
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,synth.gu_ref,bad);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped wrong expert count accepted");
        std::vector<float> gu_scale(synth.gu_absmax_codes.size(),.5f),dn_scale(synth.dn_absmax_codes.size(),.25f);
        auto gu=synth.gu_ref,dn=synth.dn_ref;gu.quant_state.nested=false;dn.quant_state.nested=false;
        gu.absmax.ptr=reinterpret_cast<const uint8_t*>(gu_scale.data());gu.absmax.bytes=gu_scale.size()*sizeof(float);gu.absmax.dtype=DT_F32;
        dn.absmax.ptr=reinterpret_cast<const uint8_t*>(dn_scale.data());dn.absmax.bytes=dn_scale.size()*sizeof(float);dn.absmax.dtype=DT_F32;
        Qwen4ExpResidentTestAccess::MappedRefs(cache,0,gu,dn);
        source=cache.MappedExpertHostSource(0,3);
        expect(!source.gate_up_nested.codes && !source.down_nested.codes &&
            source.gate_up_absmax == gu_scale.data()+3*synth.gu_absmax_per_exp &&
            source.down_absmax == dn_scale.data()+3*synth.dn_absmax_per_exp,"Mapped unnested f32 offsets");
        gu.absmax.dtype=DT_U8;Qwen4ExpResidentTestAccess::MappedRefs(cache,0,gu,dn);
        expect(!cache.MappedExpertHostSource(0,3).gate_up_packed,"Mapped unnested u8 source miscast to float");
    }
    std::cout<<"[MAPPED SOURCE PASS] no_staging=1 offsets=1 compact_nested=1 expanded_nested=1 unnested=1 metadata_bounds=1 logical_geometry=1 invalid_refs=1\n";
}

void test_mapped_resident_cold(Context& ctx) {
    struct EnvScope {
        std::array<std::string,6> previous;
        const std::array<const char*,6> keys={"AILA_Q4E_CPU_MMAP_COLD","AILA_Q4E_COLD_MODE","AILA_Q4E_COLD_POLICY","AILA_Q4E_CPU_MMAP_PREFETCH","AILA_Q4E_CPU_MMAP_TOUCH","AILA_Q4E_CPU_MAX_EXPERTS"};
        EnvScope() {
            for(size_t i=0;i<keys.size();++i)if(const char* value=std::getenv(keys[i]))previous[i]=value;
            _putenv_s(keys[0],"1");_putenv_s(keys[1],"2");_putenv_s(keys[2],"1");
            _putenv_s(keys[3],"1");
            _putenv_s(keys[4],"1");
        }
        ~EnvScope(){for(size_t i=0;i<keys.size();++i)_putenv_s(keys[i],previous[i].c_str());}
    } env_scope;
    constexpr int H=2560,I=640,E=4;
    constexpr size_t gu_bytes=size_t(2*I)*H/2,dn_bytes=size_t(H)*I/2;
    constexpr size_t gu_scales=size_t(2*I)*H/64,dn_scales=size_t(H)*I/64;
    std::vector<uint8_t> gu_packed(E*gu_bytes,0x88),dn_packed(E*dn_bytes,0x88);
    std::vector<uint8_t> gu_codes(E*gu_scales,128),dn_codes(E*dn_scales,128);
    std::vector<float> gu_nested(E*gu_scales/256,1.f),dn_nested(E*dn_scales/256,1.f);
    SyntheticExpertWeights tiny;
    auto gu=tiny.gu_ref,dn=tiny.dn_ref;
    gu.packed_weight.ptr=gu_packed.data();gu.packed_weight.bytes=gu_packed.size();gu.quant_state.shape={E,2*I,H};
    dn.packed_weight.ptr=dn_packed.data();dn.packed_weight.bytes=dn_packed.size();dn.quant_state.shape={E,H,I};
    gu.absmax.ptr=gu_codes.data();gu.absmax.bytes=gu_codes.size();dn.absmax.ptr=dn_codes.data();dn.absmax.bytes=dn_codes.size();
    gu.nested_absmax.ptr=reinterpret_cast<const uint8_t*>(gu_nested.data());gu.nested_absmax.bytes=gu_nested.size()*sizeof(float);
    dn.nested_absmax.ptr=reinterpret_cast<const uint8_t*>(dn_nested.data());dn.nested_absmax.bytes=dn_nested.size()*sizeof(float);
    ModelWeights weights;Qwen4ExpExpertCache cache;std::string err;
    expect(cache.init(ctx,weights,1,4*(gu_bytes+dn_bytes+gu_scales+dn_scales),E,H,I,64,&err,true),"Mapped resident fixture init: "+err);
    cache.set_layer_expert_ref(0,gu,dn);
    int hot=0,slot=-1;cache.acquire(0,&hot,1,&slot).wait_and_throw();cache.FreezeReady(1);
    expect(cache.prepare_host(0,1).ready && !cache.host_contains(0,2),"Mapped mixed fixture residency");
    aila::Qwen4ExpExpertExecutor executor;executor.init(ctx,cache,4,3,H,I);executor.set_max_cpu_credits(2);
    std::vector<bf16> input(H,bf16(.01f));
    auto x=Tensor::allocate(ctx,{H},dnnl::memory::data_type::bf16);ctx.memcpy_h2d(x.data(),input.data(),H*sizeof(bf16));
    auto output=Tensor::allocate(ctx,{3,H},dnnl::memory::data_type::f32);
    struct Observer {const uint8_t* host=nullptr;const uint8_t* mapped=nullptr;int calls=0;} observer;
    observer.host=cache.get_expert_slot_host(0,1).gate_up_packed;observer.mapped=gu_packed.data()+2*gu_bytes;
    executor.set_cpu_observer_for_testing([](void* state,Context&,int,const aila::RouteRankEntry& entry,
        const ops::MoeExpertSlotHost& source,const bf16*,const bf16*,const float*) {
        auto& o=*static_cast<Observer*>(state);
        expect(source.gate_up_packed == (entry.expert_id==1 ? o.host : o.mapped),"CPU observer source confused mixed leases/maps");++o.calls;
    },&observer);
    const auto before=cache.stats();
    {
        int ids[3]={0,1,2};auto route=cache.BindReady(0,ids,3);
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 2 && route.cold_count == 0,"Mapped resident admission failed");
        expect(cache.query_residency(0,1).host_cpu_readers == 1 && !cache.host_contains(0,2),"Mapped source consumed fake host lease or released real lease early");
        executor.JoinResidentCold(output.data_as<float>()).wait_and_throw();route.Finish({});
    }
    expect(observer.calls == 2 && executor.stats().cpu_mapped_tasks_dispatched == 1 &&
        executor.stats().cpu_mapped_tasks_completed == 1 && cache.query_residency(0,1).host_cpu_readers == 0,"Mapped completion/lease accounting");
#ifdef _WIN32
    const auto& prefetch_stats=executor.stats();
    const size_t source_bytes=gu_bytes+dn_bytes+gu_scales+dn_scales+
        (gu_scales/256+dn_scales/256)*sizeof(float)+2*256*sizeof(float);
    expect(prefetch_stats.cpu_mmap_prefetch_calls == 1 && prefetch_stats.cpu_mmap_prefetch_ranges == 8 &&
        prefetch_stats.cpu_mmap_prefetch_bytes == source_bytes,"Mapped prefetch must batch exact original source ranges, excluding host-staged source");
    expect(prefetch_stats.cpu_mmap_prefetch_failures <= 1,"Mapped advisory prefetch failure accounting");
    SYSTEM_INFO page_info{};GetSystemInfo(&page_info);
    const size_t page_size=page_info.dwPageSize;
    expect(page_size > 0,"Mapped touch fixture requires OS page size");
    auto span_pages=[&](const void* data,size_t bytes) {
        const auto start=reinterpret_cast<uintptr_t>(data);
        return (start+bytes-1)/page_size-start/page_size+1;
    };
    const auto mapped_source=cache.MappedExpertHostSource(0,2);
    const uint64_t expected_pages=span_pages(mapped_source.gate_up_packed,gu_bytes)+span_pages(mapped_source.down_packed,dn_bytes)+
        span_pages(mapped_source.gate_up_nested.codes,gu_scales)+span_pages(mapped_source.down_nested.codes,dn_scales)+
        span_pages(mapped_source.gate_up_nested.nested_absmax,(gu_scales/256)*sizeof(float))+
        span_pages(mapped_source.down_nested.nested_absmax,(dn_scales/256)*sizeof(float))+
        span_pages(mapped_source.gate_up_nested.nested_quant_map,256*sizeof(float))+
        span_pages(mapped_source.down_nested.nested_quant_map,256*sizeof(float));
    expect(prefetch_stats.cpu_mmap_touch_pages == expected_pages,
        "Mapped touch must cover only each source span's leading and internal pages");
#endif
    expect(cache.stats().host_pool_misses == before.host_pool_misses &&
        cache.stats().total_transferred_bytes == before.total_transferred_bytes && !cache.host_contains(0,2),"Mapped resident performed weight fill/H2D");
    std::vector<float> rows(3*H);ctx.memcpy_d2h(rows.data(),output.data(),rows.size()*sizeof(float));
    expect(!std::memcmp(rows.data()+H,rows.data()+2*H,H*sizeof(float)),"Mapped and host-staged identical expert outputs differ");
    executor.set_cpu_observer_for_testing(nullptr,nullptr);
    executor.set_max_cpu_credits(1);
    {
        int ids[3]={0,1,2};auto route=cache.BindReady(0,ids,3);
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 1 && route.cold_count == 1 && route.cold_ids[0] == 2,"Mapped cold bypassed CPU credits");
        executor.AbortResidentCold();route.Finish({});
    }
    expect(cache.query_residency(0,1).host_cpu_readers == 0,"Mapped mixed cancellation leaked host lease");
    _putenv_s("AILA_Q4E_CPU_MMAP_PREFETCH","0");
    _putenv_s("AILA_Q4E_CPU_MMAP_TOUCH","0");
    executor.init(ctx,cache,4,3,H,I);executor.set_max_cpu_credits(2);
    {
        int ids[3]={0,1,2};auto route=cache.BindReady(0,ids,3);
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 2 && route.cold_count == 0,"Disabled prefetch changed mapped CPU admission");
        executor.JoinResidentCold(output.data_as<float>()).wait_and_throw();route.Finish({});
    }
    expect(executor.stats().cpu_mmap_prefetch_calls == 0 && executor.stats().cpu_mmap_prefetch_bytes == 0 &&
        executor.stats().cpu_mmap_touch_pages == 0,"Disabled mapped preparation must not issue source hints/touches");
    std::vector<float> without_prefetch(3*H);ctx.memcpy_d2h(without_prefetch.data(),output.data(),without_prefetch.size()*sizeof(float));
    expect(!std::memcmp(rows.data()+H,without_prefetch.data()+H,2*H*sizeof(float)),"Advisory prefetch changed cold expert arithmetic");
    _putenv_s("AILA_Q4E_CPU_MMAP_TOUCH","1");
    executor.init(ctx,cache,4,3,H,I);executor.set_max_cpu_credits(2);
    {
        int ids[3]={0,1,2};auto route=cache.BindReady(0,ids,3);
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 2 && route.cold_count == 0,"Touch-only changed mapped CPU admission");
        executor.JoinResidentCold(output.data_as<float>()).wait_and_throw();route.Finish({});
    }
    expect(executor.stats().cpu_mmap_prefetch_calls == 0 && executor.stats().cpu_mmap_prefetch_bytes == 0,
        "Touch-only must not report advisory PrefetchVirtualMemory hints");
#ifdef _WIN32
    expect(executor.stats().cpu_mmap_touch_pages == expected_pages,"Touch-only exact page coverage");
#endif
    ctx.memcpy_d2h(without_prefetch.data(),output.data(),without_prefetch.size()*sizeof(float));
    expect(!std::memcmp(rows.data()+H,without_prefetch.data()+H,2*H*sizeof(float)),"Source touch changed cold expert arithmetic");
    _putenv_s("AILA_Q4E_CPU_MMAP_PREFETCH","1");
    _putenv_s("AILA_Q4E_CPU_MMAP_COLD","0");
    executor.init(ctx,cache,4,3,H,I);executor.set_max_cpu_credits(2);
    {
        int ids[3]={0,1,2};auto route=cache.BindReady(0,ids,3);
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 1 && route.cold_count == 1 && route.cold_ids[0] == 2,"Disabled mapped cold changed dispatch");
        executor.AbortResidentCold();route.Finish({});
    }
    expect(executor.stats().cpu_mapped_tasks_dispatched == 0 && executor.stats().cpu_mmap_prefetch_calls == 0 &&
        executor.stats().cpu_mmap_touch_pages == 0,"Disabled mapped task/prefetch/touch accounting");
    _putenv_s("AILA_Q4E_CPU_MMAP_COLD","1");executor.init(ctx,cache,4,3,H,I);
    int gpu_id=3,gpu_slot=-1;cache.acquire(0,&gpu_id,1,&gpu_slot).wait_and_throw();
    {
        auto route=cache.BindReady(0,&gpu_id,1);
        // An already Ready GPU source remains on GPU even if classified cold
        // because it is outside the immutable resident tier.
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 0 && route.cold_count == 1,"Mapped cold stole GPU Ready work");
        route.Finish({});
    }
    std::promise<void> release;auto signal=release.get_future().share();
    Qwen4ExpResidentTestAccess::HoldCopy(cache,gpu_slot,signal);
    {
        auto route=cache.BindReady(0,&gpu_id,1);
        const int admitted=executor.StartResidentCold(route,x.data_as<bf16>());
        release.set_value(); // drain the test source before assertions/cleanup
        expect(admitted == 0 && route.cold_count == 1,"Mapped cold stole GPU Copying work");route.Finish({});
    }
    expect(executor.stats().cpu_mapped_tasks_dispatched == 0,"GPU-ineligible mapped task accounting");
    _putenv_s("AILA_Q4E_CPU_MAX_EXPERTS","0");
    executor.init(ctx,cache,4,3,H,I); // Intentionally no setter: exercise env admission.
    {
        int ids[3]={0,1,2};auto route=cache.BindReady(0,ids,3);
        expect(executor.StartResidentCold(route,x.data_as<bf16>()) == 0 && route.cold_count == 2 &&
            route.cold_ids[0] == 1 && route.cold_ids[1] == 2,"Explicit zero CPU credits must leave every cold rank on GPU");
        route.Finish({});
    }
    expect(executor.stats().cpu_tasks_dispatched == 0 && executor.stats().cpu_mmap_prefetch_calls == 0 &&
        executor.stats().cpu_mmap_touch_pages == 0 && cache.query_residency(0,1).host_cpu_readers == 0,
        "Zero CPU credits must not dispatch, warm sources, or acquire host leases");
    std::cout<<"[MAPPED COLD PASS] host_lease_first=1 mixed_sources=1 rank_output=1 observer=1 no_weight_staging=1 prefetch_ranges=1 prefetch_advisory=1 touch_pages=1 touch_exact=1 credits=1 credits_zero_env=1 cancellation=1 opt_in=1 gpu_ready=1 gpu_copying=1\n";
}

void test_host_complement(Context& ctx) {
    struct HostSlotsScope {
        std::string previous;
        HostSlotsScope() {
            if (const char* value=std::getenv("AILA_TEST_Q4E_HOST_SLOTS")) previous=value;
            if (_putenv_s("AILA_TEST_Q4E_HOST_SLOTS","4") != 0)
                throw std::runtime_error("Cannot configure complement host fixture");
        }
        ~HostSlotsScope() {_putenv_s("AILA_TEST_Q4E_HOST_SLOTS",previous.c_str());}
    } host_slots_scope;
    SyntheticExpertWeights synth; ModelWeights weights;
    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp;
    // Modes cover the successful preference and every ownership exclusion. A
    // non-GPU expert is older than the duplicate, so ordinary LRU picks it.
    for (int mode = 0; mode <= 9; ++mode) {
        Qwen4ExpExpertCache cache; std::string err;
        expect(cache.init(ctx,weights,2,4*slot_bytes,8,256,64,64,&err,true),"Complement fixture init: "+err);
        expect(cache.num_host_slots() == 4,"Complement fixture requires four host slots");
        cache.set_layer_expert_ref(0,synth.gu_ref,synth.dn_ref);
        cache.set_layer_expert_ref(1,synth.gu_ref,synth.dn_ref);
        Qwen4ExpResidentTestAccess::HostComplement(cache,mode != 0,2);
        const auto old_cold = cache.prepare_host(0,0);
        int hot_id=1, hot_slot=-1;cache.acquire(0,&hot_id,1,&hot_slot).wait_and_throw();
        Qwen4ExpResidentTestAccess::Adaptive(cache,{1});
        const auto duplicate = cache.prepare_host(0,1); // promote to protected host tier
        expect(cache.prepare_host(0,2).ready && cache.prepare_host(0,3).ready,"Complement pool not full");
        expect(cache.host_is_protected(0,1),"Duplicate must exercise protected tier");
        if (mode == 1) {
            // Cache moves preserve configuration, lists, slot identities and stats.
            Qwen4ExpExpertCache moved(std::move(cache)); cache = std::move(moved);
        }
        HostExpertLease reader;
        std::promise<void> signal; auto ready=signal.get_future().share();
        sycl::event held;
        uint64_t old_gpu_key=0;
        if (mode == 2) reader=cache.acquire_host_lease(0,1);
        if (mode == 3) Qwen4ExpResidentTestAccess::ReserveHost(cache,duplicate.hslot,true);
        if (mode == 4) held=Qwen4ExpResidentTestAccess::HoldHostSource(cache,duplicate.hslot,ready);
        if (mode == 5) Qwen4ExpResidentTestAccess::HoldCopy(cache,hot_slot,ready);
        if (mode == 6) old_gpu_key=Qwen4ExpResidentTestAccess::ChangeGpuKey(cache,hot_slot,UINT64_MAX);
        if (mode == 7) cache.ThawReady();
        if (mode == 8) held=Qwen4ExpResidentTestAccess::HoldHostSource(cache,duplicate.hslot,ready,false);
        if (mode == 9) Qwen4ExpResidentTestAccess::BorrowGpuSlot(cache,hot_slot);
        const auto before=cache.stats();
        const auto incoming=cache.prepare_host(0,4);
        // Release the submitted test fence before checking assertions/cleanup.
        signal.set_value();
        if (held != sycl::event{}) held.wait_and_throw();
        if (mode == 3) Qwen4ExpResidentTestAccess::ReserveHost(cache,duplicate.hslot,false);
        if (mode == 6) Qwen4ExpResidentTestAccess::ChangeGpuKey(cache,hot_slot,old_gpu_key);
        expect(incoming.ready,"Complement fill failed");
        const bool selected_duplicate = mode == 1;
        expect(incoming.hslot == (selected_duplicate ? duplicate.hslot : old_cold.hslot),"Complement violated ownership exclusion mode "+std::to_string(mode));
        expect(cache.stats().host_complement_evictions-before.host_complement_evictions == uint64_t(selected_duplicate),"Complement eviction accounting");
        expect(cache.stats().host_complement_scans-before.host_complement_scans <= 2,"Complement exceeded scan bound");
        expect(cache.stats().host_complement_searches-before.host_complement_searches == uint64_t(mode != 0 && mode != 7),"Disabled/thawed complement searched");
        const auto stale = selected_duplicate ? duplicate : old_cold;
        expect(incoming.generation > stale.generation && !cache.host_contains(0,selected_duplicate ? 1 : 0),"Complement reused stale host generation/key");
        auto lease=cache.acquire_host_lease(0,4);
        expect(lease.is_valid() && !std::memcmp(lease.slot_host().gate_up_packed,
            synth.gu_packed_host.data()+4*synth.gu_packed_per_exp,synth.gu_packed_per_exp),"Complement fill payload changed");
        if (mode == 2) expect(reader.is_valid() && !std::memcmp(reader.slot_host().gate_up_packed,
            synth.gu_packed_host.data()+synth.gu_packed_per_exp,synth.gu_packed_per_exp),"Complement overwrote active CPU reader");
    }
    std::cout<<"[HOST COMPLEMENT PASS] bounded_scan=1 default_compatibility=1 cpu_lease=1 reservation=1 source_fence=1 dma_fence=1 copying=1 key_identity=1 thaw=1 borrowed_slot=1 generation=1 move=1 payload=1\n";
}

void test_large_page_residency(Context& ctx) {
    _putenv_s("AILA_TEST_Q4E_HOST_SLOTS","0");
    _putenv_s("AILA_Q4E_HOT_RAM_MB","0");
    const auto host_bytes=ctx.current_pinned_host_bytes(),host_count=ctx.pinned_host_allocation_count();
    {
        SyntheticExpertWeights synth;ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
        const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
        std::vector<std::vector<uint8_t>> expected(8,std::vector<uint8_t>(bytes));
        for(int expert=0;expert<8;++expert) {
            size_t at=0;
            auto append=[&](const std::vector<uint8_t>& data,size_t n){
                std::memcpy(expected[expert].data()+at,data.data()+expert*n,n);at+=n;
            };
            append(synth.gu_packed_host,synth.gu_packed_per_exp);append(synth.gu_absmax_codes,synth.gu_absmax_per_exp);
            append(synth.dn_packed_host,synth.dn_packed_per_exp);append(synth.dn_absmax_codes,synth.dn_absmax_per_exp);
        }
        const auto scales=synth.gu_nested_absmax,qmap=synth.gu_nested_qmap;
        expect(cache.init(ctx,weights,2,12*bytes,8,256,64,64,&error,true),error);
        for(int layer=0;layer<2;++layer)cache.set_layer_expert_ref(layer,synth.gu_ref,synth.dn_ref);
        Qwen4ExpResidentTestAccess::Adaptive(cache,{0,1,2,3,8,9,10,11});
        const auto before=ctx.current_pinned_host_bytes();
        const auto staging_before=Qwen4ExpResidentTestAccess::StagingBytes(cache);
        Qwen4ExpResidentTestAccess::LargePages(cache);
        expect(ctx.current_pinned_host_bytes()==before-staging_before+
            Qwen4ExpResidentTestAccess::StagingBytes(cache)+cache.resident_bytes(),"Large-page ledger omitted rounded bytes");
        auto* arena=Qwen4ExpResidentTestAccess::ResidentData(cache);
        PSAPI_WORKING_SET_EX_INFORMATION page{};page.VirtualAddress=arena;
        expect(QueryWorkingSetEx(GetCurrentProcess(),&page,sizeof(page)) && page.VirtualAttributes.Valid &&
            page.VirtualAttributes.LargePage,"Arena is not backed by actual Windows large pages");
        MEMORY_BASIC_INFORMATION region{};
        expect(VirtualQuery(arena,&region,sizeof(region))==sizeof(region),"Cannot query large-page region");
        auto* guard=arena+region.RegionSize-64;std::memset(guard,0xa7,64);
        // Poison every original source. Subsequent CPU reads, GPU uploads and
        // ownership exchanges must all use the two resident tiers.
        for(auto* data:{&synth.gu_packed_host,&synth.gu_absmax_codes,&synth.dn_packed_host,&synth.dn_absmax_codes})
            std::fill(data->begin(),data->end(),0);
        std::fill(synth.gu_nested_absmax.begin(),synth.gu_nested_absmax.end(),0);
        std::fill(synth.gu_nested_qmap.begin(),synth.gu_nested_qmap.end(),0);
        const auto allocations=ctx.pinned_host_allocation_count();
        auto check=[&](Qwen4ExpExpertCache& current){
            std::vector<uint8_t> payload(bytes);int gpu_only=0;
            for(int layer=0;layer<2;++layer)for(int expert=0;expert<8;++expert) {
                const auto source=current.CpuExpertHostSource(layer,expert);
                if(source.gate_up_packed) {
                    expect(!std::memcmp(source.gate_up_packed,expected[expert].data(),bytes),"Resident CPU payload changed");
                    expect(source.gate_up_nested.nested_absmax[0]==scales[expert*2] &&
                        !std::memcmp(source.gate_up_nested.nested_quant_map,qmap.data(),qmap.size()*sizeof(float)),
                        "CPU nested metadata still reads original storage");
                } else {
                    ++gpu_only;expect(current.IsDirectReady(layer,expert),"GPU-only expert lost its ready owner");
                }
                const auto placement=current.query_residency(layer,expert);
                if(placement.gpu_resident) {
                    ctx.memcpy_d2h(payload.data(),current.get_slot_dev(placement.gpu_slot).gate_up_packed,bytes);
                    expect(payload==expected[expert],"Resident GPU payload changed");
                }
            }
            expect(gpu_only==4,"Complement capacity changed");
            for(int i=0;i<64;++i)expect(guard[i]==0xa7,"Large-page tail guard changed");
            expect(ctx.pinned_host_allocation_count()==allocations,"Residency allocated in a hotset transition");
        };
        check(cache);
        Qwen4ExpResidentTestAccess::ResidentHotset(cache,{4,12,13,14,5,6,7,15});
        check(cache);
        Qwen4ExpResidentTestAccess::Adopt(cache,4);
        cache.BeginHotsetForward(0,128,0);
        // Only one layer-0 core slot remains after lending. Increasing scores
        // make each new adoption evict the preceding one inside the same batch.
        for(int expert=0;expert<4;++expert) {
            int route[16];std::fill_n(route,4*(expert+1),expert);
            cache.ObserveHotsetRoute(0,route,4*(expert+1),1);
        }
        const int ids[]={0,1,2,3};int slots[4];
        cache.acquire(0,ids,4,slots).wait_and_throw();
        cache.RetirePrefillSlots(slots,4);
        expect(cache.stats().ready_adoptions==4,"Resident chained adoption was not exercised");
        check(cache);
        cache.BeginHotsetForward(128,1);check(cache);
        expect(cache.resident_exchange_bytes()>0,"Resident exchange was not exercised");
        Qwen4ExpExpertCache moved=std::move(cache);check(moved);
    }
    expect(ctx.current_pinned_host_bytes()==host_bytes && ctx.pinned_host_allocation_count()==host_count,
        "Large-page cleanup leaked host allocation accounting");
    std::cout<<"[LARGE PAGES PASS] native_backing=1 rounded_accounting=1 cpu_gpu_payload=1 no_mmap_reads=1 metadata=1 exchange=1 adoption=1 borrow_restore=1 move_cleanup=1 guard=1\n";
}

void test_large_page_handoff(Context& ctx) {
    _putenv_s("AILA_TEST_Q4E_HOST_SLOTS","0");_putenv_s("AILA_Q4E_HOT_RAM_MB","0");
    SyntheticExpertWeights synth(512);ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
    const size_t bytes=synth.gu_packed_per_exp+synth.gu_absmax_per_exp+synth.dn_packed_per_exp+synth.dn_absmax_per_exp;
    expect(cache.init(ctx,weights,48,2824*bytes,512,256,64,64,&error,true),error);
    for(int layer=0;layer<48;++layer)cache.set_layer_expert_ref(layer,synth.gu_ref,synth.dn_ref);
    std::vector<int> keys;
    for(int expert=0;keys.size()<2808;++expert)for(int layer=0;layer<48 && keys.size()<2808;++layer)
        keys.push_back(layer*512+expert);
    Qwen4ExpResidentTestAccess::Adaptive(cache,keys);
    Qwen4ExpResidentTestAccess::LargePages(cache);
    Qwen4ExpResidentTestAccess::LargePagePrompt(cache);
    for(int phase=0;phase<2;++phase) {
        cache.BeginHotsetForward(phase*4096,phase?1872:4096,phase?0:1872);
        std::vector<int> route;
        for(int expert=0;expert<512;++expert)for(int n=0;n<2+((expert+phase*32)%512)/16;++n)route.push_back(expert);
        for(int layer=0;layer<48;++layer) {
            cache.ObserveHotsetRoute(layer,route.data(),int(route.size()),1);
            for(int first=0;first<512;first+=32) {
                int ids[32],slots[32];for(int i=0;i<32;++i)ids[i]=first+i;
                cache.acquire(layer,ids,32,slots).wait_and_throw();
                cache.RetirePrefillSlots(slots,32);
            }
        }
        cache.EndHotsetForward(phase?1872:4096,phase?0:1872);
    }
    std::cout<<"[LARGE PAGE HANDOFF] prefill complete adoptions="<<cache.stats().ready_adoptions<<std::endl;
    cache.BeginHotsetForward(5968,1);
    for(int layer=0;layer<48;++layer)for(int expert=0;expert<512;++expert) {
        const auto source=cache.CpuExpertHostSource(layer,expert);
        expect(source.gate_up_packed || cache.IsDirectReady(layer,expert),
            "Handoff lost expert "+std::to_string(layer)+"/"+std::to_string(expert));
        if(source.gate_up_packed)
            expect(!std::memcmp(source.gate_up_packed,synth.gu_packed_host.data()+expert*synth.gu_packed_per_exp,synth.gu_packed_per_exp) &&
                   !std::memcmp(source.gate_up_nested.codes,synth.gu_absmax_codes.data()+expert*synth.gu_absmax_per_exp,synth.gu_absmax_per_exp) &&
                   !std::memcmp(source.down_packed,synth.dn_packed_host.data()+expert*synth.dn_packed_per_exp,synth.dn_packed_per_exp) &&
                   !std::memcmp(source.down_nested.codes,synth.dn_absmax_codes.data()+expert*synth.dn_absmax_per_exp,synth.dn_absmax_per_exp),
                   "Batched handoff changed resident expert payload");
    }
    std::cout<<"[LARGE PAGE HANDOFF PASS] full_key_space=1 real_slot_counts=1 staged_batches=1 prompt_swap=1\n";
}

void test_large_page_async_cpu(Context& ctx) {
    _putenv_s("AILA_TEST_Q4E_HOST_SLOTS","0");_putenv_s("AILA_Q4E_HOT_RAM_MB","0");
    _putenv_s("AILA_Q4E_CPU_MMAP_COLD","1");_putenv_s("AILA_Q4E_CPU_ASYNC_ROUTE","1");
    _putenv_s("AILA_Q4E_COLD_MODE","2");_putenv_s("AILA_Q4E_COLD_POLICY","1");
    _putenv_s("AILA_Q4E_CPU_THREADS","18");_putenv_s("AILA_Q4E_CPU_MAX_EXPERTS","10");
    _putenv_s("AILA_Q4E_CPU_NF4_Q8","1");_putenv_s("AILA_Q4E_CPU_Q8_ACTIVATION64","1");
    using bf16=sycl::ext::oneapi::bfloat16;
    constexpr int H=2560,I=640,E=512;
    const size_t gu_bytes=H*I,dn_bytes=H*I/2,gu_scales=2*H*I/64,dn_scales=H*I/64;
    std::vector<uint8_t> gu_packed(E*gu_bytes,0x89),dn_packed(E*dn_bytes,0x89);
    std::vector<uint8_t> gu_codes(E*gu_scales,100),dn_codes(E*dn_scales,100);
    std::vector<float> gu_nested(E*gu_scales/256,1.25f),dn_nested(E*dn_scales/256,.75f);
    SyntheticExpertWeights tiny;auto gu=tiny.gu_ref,dn=tiny.dn_ref;
    gu.packed_weight.ptr=gu_packed.data();gu.packed_weight.bytes=gu_packed.size();gu.quant_state.shape={E,2*I,H};
    dn.packed_weight.ptr=dn_packed.data();dn.packed_weight.bytes=dn_packed.size();dn.quant_state.shape={E,H,I};
    gu.absmax.ptr=gu_codes.data();gu.absmax.bytes=gu_codes.size();dn.absmax.ptr=dn_codes.data();dn.absmax.bytes=dn_codes.size();
    gu.nested_absmax.ptr=reinterpret_cast<const uint8_t*>(gu_nested.data());gu.nested_absmax.bytes=gu_nested.size()*sizeof(float);
    dn.nested_absmax.ptr=reinterpret_cast<const uint8_t*>(dn_nested.data());dn.nested_absmax.bytes=dn_nested.size()*sizeof(float);
    const size_t bytes=gu_bytes+dn_bytes+gu_scales+dn_scales;
    ModelWeights weights;Qwen4ExpExpertCache cache;std::string error;
    expect(cache.init(ctx,weights,1,64*bytes,E,H,I,64,&error,true),error);cache.set_layer_expert_ref(0,gu,dn);
    std::vector<int> keys(32);for(int i=0;i<32;++i)keys[i]=i;
    Qwen4ExpResidentTestAccess::Adaptive(cache,keys);
    std::vector<bf16> input(H,bf16(.01f));
    auto x=Tensor::allocate(ctx,{H},dnnl::memory::data_type::bf16);ctx.memcpy_h2d(x.data(),input.data(),H*sizeof(bf16));
    auto output=Tensor::allocate(ctx,{10,H},dnnl::memory::data_type::f32);
    auto run=[&]{
        aila::Qwen4ExpExpertExecutor executor;executor.init(ctx,cache,4,10,H,I);
        executor.PrepareAsyncResidentSources(1,E);
        int ids[10],selected[10];for(int i=0;i<10;++i){ids[i]=100+i;selected[i]=-1;}
        executor.StageResidentInput(x.data_as<bf16>());
        auto route=cache.BindReady(0,ids,10);
        executor.StartAsyncResidentRoute(0,ids,selected,ctx.queue().ext_oneapi_submit_barrier());
        auto done=executor.JoinResidentCold(output.data_as<float>());
        executor.RetireAsyncResidentRoute(route);expect(route.cold_count==0,"Async resident CPU lost a cold expert");
        done.wait_and_throw();route.Finish(done);
        std::vector<float> result(10*H);ctx.memcpy_d2h(result.data(),output.data(),result.size()*sizeof(float));
        return result;
    };
    const auto expected=run();
    Qwen4ExpResidentTestAccess::LargePages(cache);
    const auto actual=run();
    expect(!std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)),"Large-page async CPU output differs from original NF4 source");
    std::cout<<"[LARGE PAGE ASYNC CPU PASS] full_geometry=1 fresh_executor=1 activation64=1 bitwise_equal=1\n";
}

void test_prompt_policy() {
    Qwen4ExpHotsetPolicy full,chunked;full.Init(1,8);chunked.Init(1,8);
    full.BeginPrompt();chunked.BeginPrompt();
    const int ids[]={0,1,0,2,2,3,4,5,4,6,4,7};
    full.ObservePrompt(0,ids,6,2);chunked.ObservePrompt(0,ids,2,2);chunked.ObservePrompt(0,ids+4,4,2);
    full.SeedFromPrompt();chunked.SeedFromPrompt();
    for(int i=0;i<8;++i)expect(std::abs(full.usage[i]-chunked.usage[i])<1e-5f,"Prompt depends on chunk geometry");
    full.usage[7]=20;const auto learned=full.usage;
    full.BeginPrompt();full.ObservePrompt(0,ids,6,2);const auto similarity=full.SeedFromPrompt();
    expect(similarity>.99999f && std::abs(full.usage[7]-learned[7])<1e-4f,"Repeated prompt erased decode learning");
    Qwen4ExpHotsetPolicy switched;switched.Init(1,8);switched.BeginPrompt();
    const int a[]={0,0,0},b[]={7,7,7};
    switched.ObservePrompt(0,a,3,1);switched.SeedFromPrompt();
    switched.BeginPrompt();switched.ObservePrompt(0,b,3,1);
    expect(switched.SeedFromPrompt()==0 && switched.usage[0]==0 && switched.usage[7]>13,"Domain switch retained stale usage");
    std::vector<int> resident(8,-1);resident[0]=0;
    const auto swaps=switched.Select(resident,1,true);
    expect(swaps.size()==1 && swaps[0].incoming==7 && swaps[0].outgoing==0,"Prompt selection violated budget or direction");
    expect(switched.Select(resident,0,true).empty(),"Zero prompt budget admitted weights");
    std::cout<<"[H5 POLICY PASS] chunk_invariance=1 repeat_learning=1 domain_switch=1 bounded_admission=1\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        std::cout << "========================================\n";
        std::cout << " Aila Qwen4Exp Expert Readiness (H1/H2b/H3) Tests\n";
        std::cout << "========================================\n";

        // Enable hot host RAM pool for readiness unit tests
        _putenv("AILA_TEST_Q4E_HOST_POOL=1");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

        Context ctx;
        if(argc==2 && std::string(argv[1])=="--large-page-async-cpu-only") {
            test_large_page_async_cpu(ctx);return 0;
        }
        if(argc==2 && std::string(argv[1])=="--large-page-handoff-only") {
            test_large_page_handoff(ctx);return 0;
        }
        if(argc==2 && std::string(argv[1])=="--large-pages-only") {
            test_large_page_residency(ctx);return 0;
        }
        test_query_residency_and_gpu_tickets(ctx);
        test_host_lease_lifecycle_and_eviction_protection(ctx);
        test_mixed_ready_copying_and_cancellation(ctx);
        test_expert_executor_planning_and_ready_subset(ctx);
        test_h2b_down_to_rank_and_ordered_merge(ctx);
        test_cpu_packet_ordered_merge(ctx);
        test_h2b_h3_cpu_expert_dispatch_and_credit_bounds(ctx);
        test_h3_straggler_monitoring_and_penalty(ctx);
        test_h2b_multi_token_and_large_task_dispatch(ctx);
        test_active_gpu_experts_greater_than_16(ctx);
        test_direct_ready_lease(ctx);
        test_frozen_transient(ctx);
        test_adaptive_hotset(ctx);
        test_prompt_policy();
        test_prompt_prefill(ctx);
        test_ready_adoption(ctx);
        test_host_complement(ctx);
        test_mapped_expert_source(ctx);
        test_cpu_packed_source_cache(ctx);
        test_host_ram_zero(ctx);
        test_gpu_mapped_cold_dma(ctx);
        test_short_demand_parts(ctx);
        test_mapped_resident_cold(ctx);

        _putenv("AILA_TEST_Q4E_HOST_POOL=");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=");

        std::cout << "All H1/H2b/H3 expert readiness tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "FATAL ERROR in readiness tests: " << e.what() << "\n";
        _putenv("AILA_TEST_Q4E_HOST_POOL=");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
        return 1;
    }
}
