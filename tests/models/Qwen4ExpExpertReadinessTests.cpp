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
    }

    // Record execution completion
    executor.record_execution_complete(plan, sycl::event{});

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

} // namespace

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << " Aila Qwen4Exp Expert Readiness (H1) Tests\n";
        std::cout << "========================================\n";

        // Enable hot host RAM pool for readiness unit tests
        _putenv("AILA_TEST_Q4E_HOST_POOL=1");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

        Context ctx;
        test_query_residency_and_gpu_tickets(ctx);
        test_host_lease_lifecycle_and_eviction_protection(ctx);
        test_mixed_ready_copying_and_cancellation(ctx);
        test_expert_executor_planning_and_ready_subset(ctx);

        _putenv("AILA_TEST_Q4E_HOST_POOL=");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=");

        std::cout << "All H1 expert readiness tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in readiness tests: " << e.what() << "\n";
        _putenv("AILA_TEST_Q4E_HOST_POOL=");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
        return 1;
    }
}
