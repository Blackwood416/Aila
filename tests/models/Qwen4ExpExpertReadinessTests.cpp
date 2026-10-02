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

struct Qwen4ExpResidentTestAccess {
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

    SyntheticExpertWeights() {
        const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
        const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
        const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
        const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

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
        gu_ref.packed_weight.shape = {num_experts_per_layer * gu_rows, hidden_size / 2};
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
        gu_ref.quant_state.shape = {num_experts_per_layer, gu_rows, hidden_size};
        gu_ref.quant_state.blocksize = blocksize;
        gu_ref.quant_state.nested = true;
        gu_ref.quant_state.nested_blocksize = 256;
        gu_ref.quant_state.nested_offset = 0.05f;

        dn_ref.name = "dn";
        dn_ref.packed_weight.ptr = dn_packed_host.data();
        dn_ref.packed_weight.bytes = total_dn_packed;
        dn_ref.packed_weight.shape = {num_experts_per_layer * dn_rows, intermediate_size / 2};
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
        dn_ref.quant_state.shape = {num_experts_per_layer, dn_rows, intermediate_size};
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

} // namespace

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << " Aila Qwen4Exp Expert Readiness (H1/H2b/H3) Tests\n";
        std::cout << "========================================\n";

        // Enable hot host RAM pool for readiness unit tests
        _putenv("AILA_TEST_Q4E_HOST_POOL=1");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

        Context ctx;
        test_query_residency_and_gpu_tickets(ctx);
        test_host_lease_lifecycle_and_eviction_protection(ctx);
        test_mixed_ready_copying_and_cancellation(ctx);
        test_expert_executor_planning_and_ready_subset(ctx);
        test_h2b_down_to_rank_and_ordered_merge(ctx);
        test_h2b_h3_cpu_expert_dispatch_and_credit_bounds(ctx);
        test_h3_straggler_monitoring_and_penalty(ctx);
        test_h2b_multi_token_and_large_task_dispatch(ctx);
        test_active_gpu_experts_greater_than_16(ctx);
        test_direct_ready_lease(ctx);

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
