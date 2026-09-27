#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/SafeTensors.hpp"

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
    std::cout << "[Test 1] Testing query_residency and ticket states...\n";
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
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);

    // Initial query on expert 0: should be non-resident on both GPU and host
    ExpertResidency r0 = cache.query_residency(0, 0);
    expect(!r0.gpu_resident, "expected unallocated GPU slot for expert 0");
    expect(!r0.gpu_ready, "expected unready GPU state for expert 0");

    // Acquire experts 0, 1 into GPU cache
    int req[2] = {0, 1};
    int slots[2] = {-1, -1};
    sycl::event ev = cache.acquire(0, req, 2, slots);
    ev.wait_and_throw();

    // Now experts 0 and 1 must be GPU resident and GPU ready
    ExpertResidency r_after0 = cache.query_residency(0, 0);
    expect(r_after0.gpu_resident, "expert 0 should be GPU resident");
    expect(r_after0.gpu_ready, "expert 0 should be GPU ready");
    expect(r_after0.gpu_slot == slots[0], "slot mismatch for expert 0");

    ExpertResidency r_after1 = cache.query_residency(0, 1);
    expect(r_after1.gpu_resident, "expert 1 should be GPU resident");
    expect(r_after1.gpu_ready, "expert 1 should be GPU ready");
    expect(r_after1.gpu_slot == slots[1], "slot mismatch for expert 1");

    // Expert 2 was not acquired, should not be GPU ready
    ExpertResidency r_after2 = cache.query_residency(0, 2);
    expect(!r_after2.gpu_ready, "expert 2 should not be GPU ready");
    std::cout << "  Passed query_residency test.\n";
}

void test_host_lease_lifecycle_and_eviction_protection(Context& ctx) {
    std::cout << "[Test 2] Testing HostExpertLease RAII lifecycle & eviction protection...\n";
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

    // If host pool is enabled, test lease mechanics
    if (cache.host_pool_enabled()) {
        // Stage expert (0, 3) through acquire
        int req[1] = {3};
        int slots[1] = {-1};
        sycl::event ev = cache.acquire(0, req, 1, slots);
        ev.wait_and_throw();

        // Check host residency
        ExpertResidency r3 = cache.query_residency(0, 3);
        if (r3.host_ready) {
            const int hslot = r3.host_slot;
            expect(hslot >= 0, "valid host slot expected for ready expert");
            expect(cache.get_host_slot_cpu_readers(hslot) == 0, "initial cpu_readers should be 0");

            // Acquire lease
            {
                HostExpertLease lease = cache.acquire_host_lease(0, 3);
                expect(lease.is_valid(), "lease should be valid");
                expect(lease.hslot() == hslot, "lease hslot match");
                expect(lease.layer() == 0, "lease layer match");
                expect(lease.expert_id() == 3, "lease expert_id match");
                expect(cache.get_host_slot_cpu_readers(hslot) == 1, "cpu_readers should be 1 during lease");

                // Check host pointers in MoeExpertSlotHost
                const auto& sh = lease.slot_host();
                expect(sh.gate_up_packed != nullptr, "gate_up_packed pointer valid");
                expect(sh.down_packed != nullptr, "down_packed pointer valid");
                expect(sh.gate_up_nested.codes != nullptr, "nested codes valid");
                expect(sh.gate_up_nested.nested_absmax != nullptr, "nested absmax valid");
                expect(sh.gate_up_nested.nested_quant_map != nullptr, "nested qmap valid");

                // Test move semantics
                HostExpertLease lease_moved = std::move(lease);
                expect(!lease.is_valid(), "original lease should be invalidated after move");
                expect(lease_moved.is_valid(), "moved lease should remain valid");
                expect(cache.get_host_slot_cpu_readers(hslot) == 1, "cpu_readers should stay 1 after move");

                // Test second concurrent lease on the same slot
                {
                    HostExpertLease lease2 = cache.acquire_host_lease(0, 3);
                    expect(lease2.is_valid(), "second lease should be valid");
                    expect(cache.get_host_slot_cpu_readers(hslot) == 2, "cpu_readers should be 2 for concurrent leases");
                }
                expect(cache.get_host_slot_cpu_readers(hslot) == 1, "cpu_readers should return to 1 after lease2 destroyed");
            }
            // After lease scope ends:
            expect(cache.get_host_slot_cpu_readers(hslot) == 0, "cpu_readers should be 0 after all leases released");
        }
    }
    std::cout << "  Passed HostExpertLease lifecycle test.\n";
}

void test_expert_executor_planning(Context& ctx) {
    std::cout << "[Test 3] Testing Qwen4ExpExpertExecutor LayerMoePlan & ready subset...\n";
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

    // Verify repeated planning on hot path does not throw or corrupt
    for (int iter = 0; iter < 100; ++iter) {
        aila::LayerMoePlan& plan_iter = executor.plan_layer(0, route_ids, route_weights, 2, 4, false);
        expect(plan_iter.ready_gpu_tasks == 4, "repeated plan ready tasks match");
    }

    std::cout << "  Passed Qwen4ExpExpertExecutor planning test.\n";
}

} // namespace

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << " Aila Qwen4Exp Expert Readiness (H1) Tests\n";
        std::cout << "========================================\n";

        Context ctx;
        test_query_residency_and_gpu_tickets(ctx);
        test_host_lease_lifecycle_and_eviction_protection(ctx);
        test_expert_executor_planning(ctx);

        std::cout << "All H1 expert readiness tests PASSED successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in readiness tests: " << e.what() << "\n";
        return 1;
    }
}
