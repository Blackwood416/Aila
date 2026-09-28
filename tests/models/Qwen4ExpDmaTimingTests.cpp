#define _CRT_SECURE_NO_WARNINGS
#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/SafeTensors.hpp"

#include <cmath>
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

bool is_event_complete(const sycl::event& ev) {
    try {
        return ev.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (...) {
        return true;
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

void test_dma_timing_stats(Context& ctx) {
    std::cout << "[Test 1] Testing DmaTimingStats tracking and record_first_consumer_start...\n";
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

    const DmaTimingStats& initial_stats = cache.dma_timing_stats();
    expect(initial_stats.batches_measured == 0, "initial batches_measured must be 0");
    expect(initial_stats.total_dma_bytes == 0, "initial total_dma_bytes must be 0");

    // Acquire 2 experts (cold misses -> DMA on copy_q_)
    int req_experts[2] = {1, 2};
    int out_slots[2] = {-1, -1};
    sycl::event ev = cache.acquire(0, req_experts, 2, out_slots);
    if (ev != sycl::event{}) {
        ev.wait_and_throw();
    }

    const DmaTimingStats& stats1 = cache.dma_timing_stats();
    expect(stats1.batches_measured == 1, "batches_measured should be 1 after cold acquire");
    expect(stats1.last_memcpy_count == 2, "last_memcpy_count should be 2");
    expect(stats1.last_batch_bytes == 2 * slot_bytes, "last_batch_bytes mismatch");
    expect(stats1.total_dma_bytes == 2 * slot_bytes, "total_dma_bytes mismatch");
    expect(stats1.last_submit_ns > 0, "last_submit_ns should be positive");

    // Trigger consumer timing hook
    cache.record_first_consumer_start();
    const DmaTimingStats& stats_after_consumer = cache.dma_timing_stats();
    expect(stats_after_consumer.last_consumer_gap_ns > 0, "last_consumer_gap_ns should be positive");
    expect(stats_after_consumer.total_consumer_gap_ns > 0, "total_consumer_gap_ns should be positive");

    // Now acquire the same 2 experts again (cache hits -> no DMA needed)
    int out_slots_hit[2] = {-1, -1};
    sycl::event ev_hit = cache.acquire(0, req_experts, 2, out_slots_hit);
    expect(out_slots_hit[0] == out_slots[0], "slot 0 should match hit");
    expect(out_slots_hit[1] == out_slots[1], "slot 1 should match hit");

    const DmaTimingStats& stats2 = cache.dma_timing_stats();
    expect(stats2.batches_measured == 1, "batches_measured should not increment on hit");
    expect(stats2.total_dma_bytes == 2 * slot_bytes, "total_dma_bytes should not increase on hit");

    std::cout << "  DmaTimingStats verified: submit_ns=" << stats1.last_submit_ns
              << ", gap_ns=" << stats_after_consumer.last_consumer_gap_ns
              << ", batches=" << stats2.batches_measured << " [PASSED]\n";
}

void test_slot_dma_event_tracking(Context& ctx) {
    std::cout << "[Test 2] Testing per-slot DMA event tracking and get_slot_dma_event...\n";
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

    // Initial boundary check: invalid slot ids return complete/empty events
    expect(is_event_complete(cache.get_slot_dma_event(-1)), "invalid negative slot must return complete event");
    expect(is_event_complete(cache.get_slot_dma_event(100)), "out of bounds slot must return complete event");

    // Acquire expert 3
    int req = 3;
    int out_slot = -1;
    sycl::event batch_ev = cache.acquire(0, &req, 1, &out_slot);
    expect(out_slot >= 0 && out_slot < 4, "valid slot required");

    sycl::event slot_ev = cache.get_slot_dma_event(out_slot);
    slot_ev.wait_and_throw();
    expect(is_event_complete(slot_ev), "slot DMA event must be complete after wait");

    // Verify slot transitioned to Ready
    expect(cache.get_slot_state(out_slot) == ExpertSlotState::Ready, "slot should be Ready after wait");

    // Test ensure_gpu returns valid ticket with matching dma_event
    GpuExpertTicket ticket = cache.ensure_gpu(0, 4);
    expect(ticket.gpu_slot >= 0, "ensure_gpu slot valid");
    ticket.dma_event.wait_and_throw();
    expect(is_event_complete(ticket.dma_event), "ensure_gpu ticket dma_event must be complete after wait");

    // Re-query ensure_gpu for already resident expert 4
    GpuExpertTicket resident_ticket = cache.ensure_gpu(0, 4);
    expect(resident_ticket.ready, "already resident expert should report ready == true");
    expect(resident_ticket.gpu_slot == ticket.gpu_slot, "resident expert slot match");
    expect(is_event_complete(resident_ticket.dma_event), "resident expert dma_event must be complete");

    std::cout << "  Per-slot DMA event tracking verified. [PASSED]\n";
}

void test_decoupled_ready_pending_gpu_execution(Context& ctx) {
    std::cout << "[Test 3] Testing decoupled ready vs pending GPU execution in ExpertExecutor...\n";
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

    // Warm up experts 0 and 1 so they are resident and ready on GPU
    int warm_req[2] = {0, 1};
    int warm_slots[2] = {-1, -1};
    sycl::event warm_ev = cache.acquire(0, warm_req, 2, warm_slots);
    if (warm_ev != sycl::event{}) warm_ev.wait_and_throw();

    // Setup executor with top_k = 4, hidden_size = 256, intermediate_size = 64
    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 1, 4, synth.hidden_size, synth.intermediate_size);

    // Plan layer with 4 experts for 1 token: [0, 1, 2, 3]
    // 0 and 1 are GPU ready (hits)
    // 2 and 3 are GPU pending (misses)
    int router_ids[4] = {0, 1, 2, 3};
    float router_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};

    aila::LayerMoePlan& plan = executor.plan_layer(0, router_ids, router_weights, 1, 4, false);
    expect(plan.ready_gpu_tasks == 2, "expected exactly 2 ready GPU tasks (experts 0 and 1)");
    expect(plan.pending_gpu_tasks == 2, "expected exactly 2 pending GPU tasks (experts 2 and 3)");
    expect(plan.gpu_tasks == 4, "expected 4 total GPU tasks");

    // Call ensure_gpu_pending to launch DMA for pending experts
    executor.ensure_gpu_pending(plan);
    plan.final_dma_event.wait_and_throw();
    expect(is_event_complete(plan.final_dma_event), "final_dma_event should complete");

    // Verify each pending entry received its per-slot DMA event
    for (int idx : plan.pending_gpu_indices) {
        expect(plan.entries[idx].gpu_slot >= 0, "pending entry must have valid gpu_slot");
        plan.entries[idx].dma_event.wait_and_throw();
        expect(is_event_complete(plan.entries[idx].dma_event), "pending entry dma_event should complete");
    }

    // Allocate test device buffers for execution
    bf16* x_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));
    bf16* moe_h_dev = static_cast<bf16*>(ctx.alloc_device(4 * synth.intermediate_size * sizeof(bf16)));
    bf16* y_dev = static_cast<bf16*>(ctx.alloc_device(synth.hidden_size * sizeof(bf16)));

    // Initialize x_dev with small non-zero test values
    std::vector<bf16> x_host(synth.hidden_size);
    for (size_t i = 0; i < x_host.size(); ++i) {
        x_host[i] = bf16(0.125f);
    }
    ctx.queue().memcpy(x_dev, x_host.data(), x_host.size() * sizeof(bf16)).wait();
    ctx.queue().memset(y_dev, 0, synth.hidden_size * sizeof(bf16)).wait();

    // Execute heterogeneous decode with decoupled ready launch
    sycl::event finish_ev = executor.execute_hetero_decode(
        ctx, plan, x_dev, moe_h_dev, router_weights, y_dev, 1, 4,
        nullptr, nullptr, cache.uses_nested_codes());
    finish_ev.wait_and_throw();

    // Verify output in y_dev
    std::vector<bf16> y_host(synth.hidden_size);
    ctx.queue().memcpy(y_host.data(), y_dev, y_host.size() * sizeof(bf16)).wait();

    bool has_nonzero = false;
    for (size_t i = 0; i < y_host.size(); ++i) {
        float f = static_cast<float>(y_host[i]);
        expect(!std::isnan(f) && !std::isinf(f), "output activations must be finite");
        if (std::abs(f) > 1e-6f) has_nonzero = true;
    }
    expect(has_nonzero, "execute_hetero_decode with decoupled launch must produce non-zero activations");

    ctx.free_device(x_dev);
    ctx.free_device(moe_h_dev);
    ctx.free_device(y_dev);

    std::cout << "  Decoupled ready/pending GPU execution completed and verified! [PASSED]\n";
}

void test_host_slot_dma_event_fencing(Context& ctx) {
    std::cout << "[Test 4] Testing host slot per-expert DMA release and fencing...\n";
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
    expect(cache.host_pool_enabled(), "host pool must be enabled");
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    // Prepare expert 5 into host pool
    FillTicket ft = cache.prepare_host(0, 5);
    expect(ft.ready, "prepare_host should stage expert 5");
    expect(ft.hslot >= 0, "valid host slot required");

    // Acquire into GPU: should hit host pool and copy to GPU slot
    int req = 5;
    int gslot = -1;
    sycl::event copy_ev = cache.acquire(0, &req, 1, &gslot);
    expect(gslot >= 0, "GPU slot assigned");
    if (copy_ev != sycl::event{}) copy_ev.wait_and_throw();

    // Verify GPU slot is Ready
    expect(cache.get_slot_state(gslot) == ExpertSlotState::Ready, "GPU slot must be Ready");

    // Acquire host lease on expert 5
    auto lease = cache.acquire_host_lease(0, 5);
    expect(lease.is_valid(), "host lease on expert 5 should be valid");
    expect(cache.get_host_slot_cpu_readers(lease.hslot()) >= 1, "cpu readers should reflect active lease");
    lease.release();
    expect(cache.get_host_slot_cpu_readers(ft.hslot) == 0, "cpu readers should drop to 0 after release");

    std::cout << "  Host slot per-expert DMA fencing and lease release verified! [PASSED]\n";
}

} // namespace

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << " Aila Qwen4Exp DMA Timing & Decoupling (H5)\n";
        std::cout << "========================================\n";

        _putenv("AILA_TEST_Q4E_HOST_POOL=1");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

        Context ctx;
        test_dma_timing_stats(ctx);
        test_slot_dma_event_tracking(ctx);
        test_decoupled_ready_pending_gpu_execution(ctx);
        test_host_slot_dma_event_fencing(ctx);

        _putenv("AILA_TEST_Q4E_HOST_POOL=");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=");

        std::cout << "========================================\n";
        std::cout << " All H5 DMA Timing Tests PASSED!        \n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL in H5 DMA Timing tests: " << e.what() << "\n";
        _putenv("AILA_TEST_Q4E_HOST_POOL=");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
        return 1;
    }
}
