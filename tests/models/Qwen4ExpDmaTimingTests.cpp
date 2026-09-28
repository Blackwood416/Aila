#define _CRT_SECURE_NO_WARNINGS
#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/SafeTensors.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <array>
#include <algorithm>
#include <iomanip>

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

template <int NumLayers = 2, int NumExperts = 8, int Hidden = 256, int Intermediate = 64>
struct SyntheticExpertWeightsT {
    static constexpr int num_layers = NumLayers;
    static constexpr int num_experts_per_layer = NumExperts;
    static constexpr int hidden_size = Hidden;
    static constexpr int intermediate_size = Intermediate;
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

    SyntheticExpertWeightsT() {
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

using SyntheticExpertWeights = SyntheticExpertWeightsT<2, 8>;

class ReadyFirstReader : public aila::IExpertReader {
public:
    ReadyFirstReader(std::unique_ptr<aila::IExpertReader> reader, const aila::LayerMoePlan& plan)
        : reader_(std::move(reader)), plan_(plan) {}
    aila::ExpertReaderType type() const override { return reader_->type(); }
    const char* name() const override { return "ready_first_test"; }
    void read_expert_slice(int layer, int expert, uint8_t* slot, int part, int parts, uint64_t gen) override {
        if (plan_.ready_subset_event == sycl::event{}) {
            missing_ready.store(true);
        } else {
            auto ready = plan_.ready_subset_event;
            ready.wait_and_throw();
        }
        observed_parts.fetch_add(1);
        reader_->read_expert_slice(layer, expert, slot, part, parts, gen);
    }
    void set_layer_source(int layer, const aila::ExpertLayerSource& source) override {
        reader_->set_layer_source(layer, source);
    }
    aila::ExpertReaderStats stats() const override { return reader_->stats(); }
    void reset_stats() override { reader_->reset_stats(); }
    std::atomic<int> observed_parts{0};
    std::atomic<bool> missing_ready{false};
private:
    std::unique_ptr<aila::IExpertReader> reader_;
    const aila::LayerMoePlan& plan_;
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

    uint64_t previous_copy_ns = stats2.total_copy_ns;
    expect(stats2.batches_profiled == 1 && previous_copy_ns > 0, "first copy batch must be profiled");
    for (int pass = 0; pass < 16; ++pass) {
        const int first = (pass % 2 == 0) ? 4 : 0;
        int ids[4] = {first, first + 1, first + 2, first + 3};
        int slots[4];
        cache.acquire(0, ids, 4, slots).wait_and_throw();
        const auto sample = cache.dma_timing_stats();
        expect(sample.last_memcpy_count == 4 && sample.last_batch_bytes == 4 * slot_bytes,
               "parallel completion must account for every copy");
        expect(sample.batches_profiled == static_cast<uint64_t>(pass + 2),
               "each consecutive copy batch must refresh profiling");
        expect(sample.last_copy_ns > 0 && sample.total_copy_ns == previous_copy_ns + sample.last_copy_ns,
               "copy duration must belong to the new batch");
        expect(sample.total_dma_bytes == (2 + 4ULL * (pass + 1)) * slot_bytes,
               "parallel completion must not lose transferred bytes");
        const double expected_rate = static_cast<double>(sample.last_batch_bytes) / sample.last_copy_ns;
        expect(std::abs(sample.last_effective_gbps - expected_rate) < 1e-9, "stale DMA rate");
        expect(cache.dma_timing_stats().total_copy_ns == sample.total_copy_ns, "profiling must be idempotent");
        previous_copy_ns = sample.total_copy_ns;
    }

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

    cache.set_layer_expert_ref(1, synth.gu_ref, synth.dn_ref);
    int fillers[4] = {0, 1, 2, 3};
    int filler_slots[4];
    cache.acquire(1, fillers, 4, filler_slots).wait_and_throw();
    cache.acquire(0, warm_req, 2, warm_slots).wait_and_throw();
    auto& deferred = executor.plan_layer(0, router_ids, router_weights, 1, 4, false);
    expect(deferred.ready_gpu_tasks == 2 && deferred.pending_gpu_tasks == 2, "mixed deferred plan required");
    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = synth.gu_packed_per_exp;
    geom.gu_absmax_count = geom.gu_absmax_bytes = synth.gu_absmax_per_exp;
    geom.dn_packed_bytes = synth.dn_packed_per_exp;
    geom.dn_absmax_count = geom.dn_absmax_bytes = synth.dn_absmax_per_exp;
    geom.gu_absmax_offset = geom.gu_packed_bytes;
    geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
    geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
    geom.slot_bytes = slot_bytes;
    geom.compact_nested = true;
    auto observer = std::make_unique<ReadyFirstReader>(aila::CreateMmapExpertReader(
        geom, {{synth.gu_ref, synth.dn_ref, true}}, false), deferred);
    auto* observer_ptr = observer.get();
    cache.set_reader(std::move(observer));
    executor.execute_hetero_decode(ctx, deferred, x_dev, moe_h_dev, router_weights, y_dev,
        1, 4, nullptr, nullptr, true, true).wait_and_throw();
    expect(observer_ptr->observed_parts.load() > 0 && !observer_ptr->missing_ready.load(),
           "ready GPU work must be submitted before any demand fill callback");
    expect(executor.stats().ready_first_layers == 1, "ready-first execution must be counted");
    std::vector<bf16> deferred_output(synth.hidden_size);
    ctx.queue().memcpy(deferred_output.data(), y_dev, deferred_output.size() * sizeof(bf16)).wait_and_throw();
    expect(std::memcmp(y_host.data(), deferred_output.data(), y_host.size() * sizeof(bf16)) == 0,
           "deferred acquire must preserve every output BF16 bit");

    cache.set_reader(nullptr);
    cache.acquire(1, fillers, 4, filler_slots).wait_and_throw();
    bool stale_rejected = false;
    try {
        executor.execute_hetero_decode(ctx, deferred, x_dev, moe_h_dev, router_weights, y_dev, 1, 4);
    } catch (const std::logic_error&) { stale_rejected = true; }
    expect(stale_rejected, "evicted plan must be rejected before dispatch");

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

void test_multi_token_decoupled_execution_and_hit_protection(Context& ctx) {
    std::cout << "[Test 5] Testing multi-token (>16 tasks) decoupled execution & hit protection...\n";
    SyntheticExpertWeightsT<1, 24> synth;
    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t slot_bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp * sizeof(uint8_t) +
                              synth.dn_packed_per_exp + synth.dn_absmax_per_exp * sizeof(uint8_t);
    const size_t cache_bytes = 24 * slot_bytes; // 24 GPU slots

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, synth.num_layers, cache_bytes,
                         synth.num_experts_per_layer, synth.hidden_size,
                         synth.intermediate_size, synth.blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);

    // 1. Warm up 10 experts (0..9) on GPU so they are resident and Ready
    std::vector<int> warm_req(10);
    std::vector<int> warm_slots(10, -1);
    for (int i = 0; i < 10; ++i) warm_req[i] = i;
    sycl::event warm_ev = cache.acquire(0, warm_req.data(), 10, warm_slots.data());
    if (warm_ev != sycl::event{}) warm_ev.wait_and_throw();

    // Setup executor with token_count = 2, top_k = 10 -> total 20 tasks (> 16 tasks!)
    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 2, 10, synth.hidden_size, synth.intermediate_size);
    executor.set_max_cpu_tokens(1);

    // Token 0 uses experts 0..9 (hits, ready = 10)
    // Token 1 uses experts 10..19 (misses, pending = 10)
    std::vector<int> router_ids(20);
    std::vector<float> router_weights(20);
    for (int i = 0; i < 10; ++i) {
        router_ids[i] = i;
        router_weights[i] = 0.1f;
    }
    for (int i = 10; i < 20; ++i) {
        router_ids[i] = i;
        router_weights[i] = 0.1f;
    }

    aila::LayerMoePlan& plan = executor.plan_layer(0, router_ids.data(), router_weights.data(), 2, 10, false);
    expect(plan.gpu_tasks == 20, "expected 20 total GPU tasks (> 16 tasks)");
    expect(plan.ready_gpu_tasks == 10, "expected 10 ready GPU tasks");
    expect(plan.pending_gpu_tasks == 10, "expected 10 pending GPU tasks");
    expect(plan.cpu_tasks == 0, "GPU ready-first must not depend on CPU token admission");

    // Allocate test buffers for m=2 tokens
    bf16* x_dev = static_cast<bf16*>(ctx.alloc_device(2 * synth.hidden_size * sizeof(bf16)));
    bf16* moe_h_dev = static_cast<bf16*>(ctx.alloc_device(20 * synth.intermediate_size * sizeof(bf16)));
    bf16* y_dev = static_cast<bf16*>(ctx.alloc_device(2 * synth.hidden_size * sizeof(bf16)));

    std::vector<bf16> x_host(2 * synth.hidden_size, bf16(0.125f));
    ctx.queue().memcpy(x_dev, x_host.data(), x_host.size() * sizeof(bf16)).wait();
    ctx.queue().memset(y_dev, 0, 2 * synth.hidden_size * sizeof(bf16)).wait();

    // Execute multi-token decoupled execution (> 16 tasks path)
    sycl::event finish_ev = executor.execute_hetero_decode(
        ctx, plan, x_dev, moe_h_dev, router_weights.data(), y_dev, 2, 10,
        nullptr, nullptr, cache.uses_nested_codes(), true);
    finish_ev.wait_and_throw();
    expect(plan.ready_subset_event != sycl::event{}, "multi-token ready subset must be submitted");

    // Verify output in y_dev
    std::vector<bf16> y_host(2 * synth.hidden_size);
    ctx.queue().memcpy(y_host.data(), y_dev, y_host.size() * sizeof(bf16)).wait();
    for (size_t i = 0; i < y_host.size(); ++i) {
        float f = static_cast<float>(y_host[i]);
        expect(!std::isnan(f) && !std::isinf(f), "multi-token output activations must be finite");
    }

    // 2. Timing and Hit Protection Verification
    const auto stats_after_first = cache.dma_timing_stats();
    expect(stats_after_first.batches_measured >= 1, "batches_measured should be >= 1 after first acquire with misses");
    uint64_t measured_batches_prev = stats_after_first.batches_measured;
    uint64_t gap_prev = stats_after_first.total_consumer_gap_ns;

    // Execute a second pass with the exact same 20 experts (now 100% hits in cache!)
    aila::LayerMoePlan& plan_hits = executor.plan_layer(0, router_ids.data(), router_weights.data(), 2, 10, false);
    expect(plan_hits.ready_gpu_tasks == 20, "all 20 tasks should be hits");
    expect(plan_hits.pending_gpu_tasks == 0, "0 pending tasks on hit");

    executor.ensure_gpu_pending(plan_hits);
    sycl::event hit_finish_ev = executor.execute_hetero_decode(
        ctx, plan_hits, x_dev, moe_h_dev, router_weights.data(), y_dev, 2, 10,
        nullptr, nullptr, cache.uses_nested_codes());
    hit_finish_ev.wait_and_throw();
    std::vector<bf16> hit_output(y_host.size());
    ctx.queue().memcpy(hit_output.data(), y_dev, hit_output.size() * sizeof(bf16)).wait_and_throw();
    expect(std::memcmp(y_host.data(), hit_output.data(), y_host.size() * sizeof(bf16)) == 0,
           "multi-token deferred path must match all-resident execution bitwise");

    // Call record_first_consumer_start() again: should be a NO-OP since no new DMA was issued!
    cache.record_first_consumer_start();
    const auto stats_after_hit = cache.dma_timing_stats();
    expect(stats_after_hit.batches_measured == measured_batches_prev,
           "batches_measured must NOT increment on 100% hit batch");
    expect(stats_after_hit.total_consumer_gap_ns == gap_prev,
           "total_consumer_gap_ns must NOT accumulate on 100% hit batch");

    ctx.free_device(x_dev);
    ctx.free_device(moe_h_dev);
    ctx.free_device(y_dev);

    std::cout << "  Multi-token decoupled execution and hit protection verified! [PASSED]\n";
}

class CpuOverlapReader : public aila::IExpertReader {
public:
    CpuOverlapReader(std::unique_ptr<aila::IExpertReader> reader, aila::Qwen4ExpExpertExecutor& executor)
        : reader_(std::move(reader)), executor_(executor) {}
    aila::ExpertReaderType type() const override { return reader_->type(); }
    const char* name() const override { return "cpu_overlap_test"; }
    void read_expert_slice(int layer, int expert, uint8_t* slot, int part, int parts, uint64_t gen) override {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!executor_.cpu_work_complete_for_testing() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!executor_.cpu_work_complete_for_testing()) timed_out.store(true);
        observed.fetch_add(1);
        reader_->read_expert_slice(layer, expert, slot, part, parts, gen);
    }
    void set_layer_source(int layer, const aila::ExpertLayerSource& source) override { reader_->set_layer_source(layer, source); }
    aila::ExpertReaderStats stats() const override { return reader_->stats(); }
    void reset_stats() override { reader_->reset_stats(); }
    std::atomic<bool> timed_out{false};
    std::atomic<int> observed{0};
private:
    std::unique_ptr<aila::IExpertReader> reader_;
    aila::Qwen4ExpExpertExecutor& executor_;
};

void test_cpu_fill_overlap(Context& ctx) {
    _putenv("AILA_Q4E_CPU_OVERLAP=1");
    struct EnvReset { ~EnvReset() { _putenv("AILA_Q4E_CPU_OVERLAP="); } } reset;
    SyntheticExpertWeights synth;
    ModelWeights weights;
    Qwen4ExpExpertCache cache;
    const size_t bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp + synth.dn_packed_per_exp + synth.dn_absmax_per_exp;
    std::string error;
    expect(cache.init(ctx, weights, 2, 4 * bytes, 8, synth.hidden_size, synth.intermediate_size, 64, &error, true), error);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    int ready = 0, slot;
    cache.acquire(0, &ready, 1, &slot).wait_and_throw();
    cache.prepare_host(0, 1);
    aila::Qwen4ExpExpertExecutor executor;
    executor.init(ctx, cache, 4, 6, synth.hidden_size, synth.intermediate_size);
    executor.set_max_cpu_credits(1);
    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = synth.gu_packed_per_exp;
    geom.gu_absmax_count = geom.gu_absmax_bytes = synth.gu_absmax_per_exp;
    geom.dn_packed_bytes = synth.dn_packed_per_exp;
    geom.dn_absmax_count = geom.dn_absmax_bytes = synth.dn_absmax_per_exp;
    geom.gu_absmax_offset = geom.gu_packed_bytes;
    geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
    geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
    geom.slot_bytes = bytes; geom.compact_nested = true;
    auto observer = std::make_unique<CpuOverlapReader>(aila::CreateMmapExpertReader(
        geom, {{synth.gu_ref, synth.dn_ref, true}}, false), executor);
    auto* observed = observer.get();
    cache.set_reader(std::move(observer));
    int ids[4] = {0, 1, 2, 3};
    float rw[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    auto x = Tensor::allocate(ctx, {synth.hidden_size}, dnnl::memory::data_type::bf16);
    auto h = Tensor::allocate(ctx, {4, synth.intermediate_size}, dnnl::memory::data_type::bf16);
    auto y = Tensor::allocate(ctx, {synth.hidden_size}, dnnl::memory::data_type::bf16);
    std::vector<bf16> host_x(synth.hidden_size, bf16(0.25f)), first(synth.hidden_size), second(synth.hidden_size);
    ctx.memcpy_h2d(x.data(), host_x.data(), x.size_bytes());
    auto& plan = executor.plan_layer(0, ids, rw, 1, 4, true);
    expect(plan.cpu_tasks == 1 && plan.pending_gpu_tasks == 2, "mixed CPU/fill fixture");
    executor.execute_hetero_decode(ctx, plan, x.data_as<bf16>(), h.data_as<bf16>(), rw, y.data_as<bf16>(),
        1, 4, nullptr, nullptr, true, true).wait_and_throw();
    expect(observed->observed.load() > 0 && !observed->timed_out.load(), "CPU must complete while demand fill is blocked");
    expect(executor.stats().cpu_async_layers == 1, "async layer accounting");
    ctx.memcpy_d2h(first.data(), y.data(), y.size_bytes());
    cache.set_reader(nullptr);
    for (int pass = 0; pass < 4; ++pass) {
        executor.set_straggler_penalty(false);
        auto& repeat = executor.plan_layer(0, ids, rw, 1, 4, true);
        expect(repeat.cpu_tasks == 1, "CPU host lease remains reusable");
        executor.execute_hetero_decode(ctx, repeat, x.data_as<bf16>(), h.data_as<bf16>(), rw, y.data_as<bf16>(),
            1, 4, nullptr, nullptr, true, true).wait_and_throw();
        ctx.memcpy_d2h(second.data(), y.data(), y.size_bytes());
        expect(std::memcmp(first.data(), second.data(), y.size_bytes()) == 0, "async result buffers must be reusable bitwise");
    }
    executor.set_cpu_observer_for_testing([](void*, Context&, int, const aila::RouteRankEntry&,
        const ops::MoeExpertSlotHost&, const bf16*, const bf16*, const float*) { throw std::runtime_error("injected coordinator failure"); }, nullptr);
    executor.set_straggler_penalty(false);
    bool caught = false;
    try {
        auto& failing = executor.plan_layer(0, ids, rw, 1, 4, true);
        executor.execute_hetero_decode(ctx, failing, x.data_as<bf16>(), h.data_as<bf16>(), rw, y.data_as<bf16>(),
            1, 4, nullptr, nullptr, true, true);
    } catch (const std::runtime_error&) { caught = true; }
    expect(caught, "coordinator failure must propagate");
    const auto residency = cache.query_residency(0, 1);
    expect(cache.get_host_slot_cpu_readers(residency.host_slot) == 0, "failure must release CPU readers after worker drain");
    executor.set_cpu_observer_for_testing(nullptr, nullptr);
    auto& recovered = executor.plan_layer(0, ids, rw, 1, 4, true);
    executor.execute_hetero_decode(ctx, recovered, x.data_as<bf16>(), h.data_as<bf16>(), rw, y.data_as<bf16>(),
        1, 4, nullptr, nullptr, true, true).wait_and_throw();
    ctx.memcpy_d2h(second.data(), y.data(), y.size_bytes());
    expect(std::memcmp(first.data(), second.data(), y.size_bytes()) == 0, "recovery output differs");
    // Failure after CPU launch but before acquire completes must join the worker.
    int oversized[6] = {0, 1, 2, 3, 4, 5};
    float oversized_weights[6] = {0.25f, 0.25f, 0.125f, 0.125f, 0.125f, 0.125f};
    executor.set_straggler_penalty(false);
    caught = false;
    try {
        auto& failing = executor.plan_layer(0, oversized, oversized_weights, 1, 6, true);
        expect(failing.cpu_tasks == 1 && failing.gpu_tasks == 5, "over-capacity fixture");
        executor.execute_hetero_decode(ctx, failing, x.data_as<bf16>(), h.data_as<bf16>(), oversized_weights, y.data_as<bf16>(),
            1, 6, nullptr, nullptr, true, true);
    } catch (const std::runtime_error&) { caught = true; }
    expect(caught && !executor.cpu_work_complete_for_testing(), "failed acquire must drain the CPU job");
    expect(cache.get_host_slot_cpu_readers(residency.host_slot) == 0, "failed acquire must release CPU lease after join");
    _putenv("AILA_Q4E_CPU_OVERLAP=");
    aila::Qwen4ExpExpertExecutor synchronous;
    synchronous.init(ctx, cache, 4, 4, synth.hidden_size, synth.intermediate_size);
    synchronous.set_max_cpu_credits(1);
    auto& sync_plan = synchronous.plan_layer(0, ids, rw, 1, 4, true);
    synchronous.ensure_gpu_pending(sync_plan);
    synchronous.execute_hetero_decode(ctx, sync_plan, x.data_as<bf16>(), h.data_as<bf16>(), rw, y.data_as<bf16>(),
        1, 4, nullptr, nullptr, true).wait_and_throw();
    ctx.memcpy_d2h(second.data(), y.data(), y.size_bytes());
    expect(std::memcmp(first.data(), second.data(), y.size_bytes()) == 0, "async and synchronous output must match bitwise");
    std::cout << "[PASS] CPU/fill overlap, repeated buffers and exception drain\n";
}

void test_large_cpu_placement(Context& ctx) {
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=40");
    _putenv("AILA_Q4E_CPU_THREADS=12");
    _putenv("AILA_Q4E_CPU_FIXED_CREDITS=1");
    struct Reset {
        ~Reset() {
            _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");
            _putenv("AILA_Q4E_CPU_THREADS=");
            _putenv("AILA_Q4E_CPU_FIXED_CREDITS=");
            _putenv("AILA_Q4E_CPU_PLACEMENT=");
            _putenv("AILA_Q4E_CPU_BATCH=");
            _putenv("AILA_Q4E_CPU_OVERLAP=");
            _putenv("AILA_Q4E_CPU_BATCH_PREPARE=");
            _putenv("AILA_Q4E_CPU_COALESCE_RESULTS=");
            _putenv("AILA_Q4E_CPU_WAIT_PREFETCH=");
        }
    } reset;
    SyntheticExpertWeightsT<1, 40, 2560, 640> synth;
    ModelWeights weights;
    Qwen4ExpExpertCache cache;
    const size_t bytes = synth.gu_packed_per_exp + synth.gu_absmax_per_exp + synth.dn_packed_per_exp + synth.dn_absmax_per_exp;
    std::string error;
    expect(cache.init(ctx, weights, 1, 40 * bytes, 40, synth.hidden_size, synth.intermediate_size, 64, &error, true), error);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    auto x = Tensor::allocate(ctx, {4, synth.hidden_size}, dnnl::memory::data_type::bf16);
    auto h = Tensor::allocate(ctx, {40, synth.intermediate_size}, dnnl::memory::data_type::bf16);
    auto y = Tensor::allocate(ctx, {4, synth.hidden_size}, dnnl::memory::data_type::bf16);
    std::vector<bf16> input(4 * synth.hidden_size), expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) input[i] = bf16(0.001f * (static_cast<int>(i % 37) - 18));
    ctx.memcpy_h2d(x.data(), input.data(), x.size_bytes());
    int ids[40]; float rw[40];
    for (int c = 0; c < 40; ++c) { ids[c] = c; rw[c] = (c % 10 + 1) / 55.0f; }
    for (int tokens : {1, 4}) {
        for (int credits : {5, 40}) {
            for (int mode = 0; mode < 6; ++mode) {
                _putenv(mode ? "AILA_Q4E_CPU_OVERLAP=1" : "AILA_Q4E_CPU_OVERLAP=0");
                _putenv(mode ? "AILA_Q4E_CPU_BATCH=1" : "AILA_Q4E_CPU_BATCH=0");
                _putenv(mode == 1 ? "AILA_Q4E_CPU_PLACEMENT=1" : "AILA_Q4E_CPU_PLACEMENT=2");
                _putenv(mode == 2 || mode >= 4 ? "AILA_Q4E_CPU_BATCH_PREPARE=1" : "AILA_Q4E_CPU_BATCH_PREPARE=0");
                _putenv(mode >= 3 ? "AILA_Q4E_CPU_COALESCE_RESULTS=1" : "AILA_Q4E_CPU_COALESCE_RESULTS=0");
                _putenv(mode == 5 ? "AILA_Q4E_CPU_WAIT_PREFETCH=1" : "AILA_Q4E_CPU_WAIT_PREFETCH=0");
                aila::Qwen4ExpExpertExecutor executor;
                executor.init(ctx, cache, 4, 10, synth.hidden_size, synth.intermediate_size);
                executor.set_max_cpu_credits(credits);
                executor.set_straggler_penalty(true);
                for (int pass = 0; pass < 2; ++pass) {
                    auto& plan = executor.plan_layer(0, ids, rw, tokens, 10, true);
                    expect(plan.cpu_tasks == std::min(credits, tokens * 10), "large fixed credits must not be throttled");
                    executor.execute_hetero_decode(ctx, plan, x.data_as<bf16>(), h.data_as<bf16>(), rw,
                        y.data_as<bf16>(), tokens, 10, nullptr, nullptr, true, true).wait_and_throw();
                    ctx.memcpy_d2h(actual.data(), y.data(), tokens * synth.hidden_size * sizeof(bf16));
                    if (mode == 0 && pass == 0) expected = actual;
                    else expect(std::memcmp(expected.data(), actual.data(), tokens * synth.hidden_size * sizeof(bf16)) == 0,
                        "large sync/async CPU plans must preserve ordered merge bits");
                    for (int expert = 0; expert < tokens * 10; ++expert) {
                        const auto residency = cache.query_residency(0, expert);
                        expect(cache.get_host_slot_cpu_readers(residency.host_slot) == 0, "large plan must release every lease");
                    }
                }
                expect(executor.stats().cpu_max_batch_tasks == std::min(credits, tokens * 10), "large batch counter");
                expect(executor.stats().cpu_result_copies == (mode >= 3 ? 2 : 2 * std::min(credits, tokens * 10)), "contiguous CPU ranks must share one result copy");
                expect(executor.stats().cpu_result_bytes == 2ULL * std::min(credits, tokens * 10) * synth.hidden_size * sizeof(float), "coalescing must not change transferred result bytes");
            }
        }
    }
    int slots[10];
    cache.acquire(0, ids, 10, slots).wait_and_throw();
    _putenv("AILA_Q4E_CPU_PLACEMENT=0");
    aila::Qwen4ExpExpertExecutor miss_only;
    miss_only.init(ctx, cache, 4, 10, synth.hidden_size, synth.intermediate_size);
    miss_only.set_max_cpu_credits(40);
    expect(miss_only.plan_layer(0, ids, rw, 1, 10, true).cpu_tasks == 0, "miss-only policy must retain GPU hits");
    _putenv("AILA_Q4E_CPU_PLACEMENT=1");
    aila::Qwen4ExpExpertExecutor host_ready;
    host_ready.init(ctx, cache, 4, 10, synth.hidden_size, synth.intermediate_size);
    host_ready.set_max_cpu_credits(40);
    expect(host_ready.plan_layer(0, ids, rw, 1, 10, true).cpu_tasks == 10, "host-ready policy must accept GPU hits");
    std::cout << "[PASS] 12-thread CPU placement, 5/10/40 tasks, mixed/all-CPU, sync/batch exact and lease reuse\n";
}

class BatchPrepareReader : public aila::IExpertReader {
public:
    explicit BatchPrepareReader(std::unique_ptr<aila::IExpertReader> inner) : inner_(std::move(inner)) {}
    aila::ExpertReaderType type() const override { return inner_->type(); }
    const char* name() const override { return "batch_prepare_test"; }
    void read_expert_slice(int layer, int expert, uint8_t* slot, int part, int parts, uint64_t gen) override {
        const int concurrency = active.fetch_add(1) + 1;
        int observed = peak.load();
        while (observed < concurrency && !peak.compare_exchange_weak(observed, concurrency)) {}
        struct Finish { BatchPrepareReader* self; ~Finish() { self->active.fetch_sub(1); self->finished.fetch_add(1); } } finish{this};
        while (hold.load()) std::this_thread::yield();
        if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (expert == fail_expert) throw std::runtime_error("injected host preparation read failure");
        inner_->read_expert_slice(layer, expert, slot, part, parts, gen);
    }
    void set_layer_source(int layer, const aila::ExpertLayerSource& source) override { inner_->set_layer_source(layer, source); }
    aila::ExpertReaderStats stats() const override { return inner_->stats(); }
    void reset_stats() override { inner_->reset_stats(); }
    std::atomic<int> active{0}, peak{0}, finished{0};
    std::atomic<bool> hold{false};
    int fail_expert = -1;
    bool delay = true;
private:
    std::unique_ptr<aila::IExpertReader> inner_;
};

template <typename Weights>
aila::ExpertSlotGeometry compact_geometry(const Weights& synth) {
    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = synth.gu_packed_per_exp;
    geom.gu_absmax_count = geom.gu_absmax_bytes = synth.gu_absmax_per_exp;
    geom.dn_packed_bytes = synth.dn_packed_per_exp;
    geom.dn_absmax_count = geom.dn_absmax_bytes = synth.dn_absmax_per_exp;
    geom.gu_absmax_offset = geom.gu_packed_bytes;
    geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
    geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
    geom.slot_bytes = geom.dn_absmax_offset + geom.dn_absmax_bytes;
    geom.compact_nested = true;
    return geom;
}

void test_host_batch_prepare(Context& ctx) {
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=40");
    struct Reset { ~Reset() { _putenv("AILA_TEST_Q4E_HOST_SLOTS=4"); } } reset;
    SyntheticExpertWeightsT<1, 48> synth;
    const auto geom = compact_geometry(synth);
    ModelWeights weights;
    Qwen4ExpExpertCache cache;
    std::string error;
    expect(cache.init(ctx, weights, 1, 4 * geom.slot_bytes, 48, synth.hidden_size, synth.intermediate_size, 64, &error, true), error);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    cache.prepare_host(0, 0);
    auto pinned = cache.acquire_host_lease(0, 0);
    int ready = 1, gpu_slot = -1;
    cache.acquire(0, &ready, 1, &gpu_slot).wait_and_throw();
    const auto transferred = cache.stats().total_transferred_bytes;
    auto reader = std::make_unique<BatchPrepareReader>(aila::CreateMmapExpertReader(geom, {{synth.gu_ref, synth.dn_ref, true}}, false));
    auto* observed = reader.get();
    cache.set_reader(std::move(reader));
    std::array<HostExpertLease, 40> leases, warm;
    int ids[40];
    for (int c = 0; c < 40; ++c) ids[c] = c == 39 ? 1 : c;
    expect(cache.prepare_host_batch(0, ids, 40, leases.data()) == 37, "only distinct cold experts should be filled");
    expect(observed->finished == 37 && observed->active == 0 && observed->peak > 1, "host preparation must use multiple pool workers and drain");
    for (int c = 0; c < 40; ++c) {
        expect(leases[c].is_valid(), "every requested task must retain a lease");
        expect(std::memcmp(leases[c].slot_host().gate_up_packed, synth.gu_packed_host.data() + ids[c] * synth.gu_packed_per_exp, synth.gu_packed_per_exp) == 0, "batched host bytes differ");
    }
    expect(cache.prepare_host_batch(0, ids, 40, warm.data()) == 0 && observed->finished == 37, "warm batch must not re-read weights");
    expect(cache.stats().total_transferred_bytes == transferred, "CPU host preparation must not submit GPU weight copies");
    for (auto& lease : leases) lease.release();
    for (auto& lease : warm) lease.release();
    int cold[4] = {40, 41, 42, 43};
    observed->fail_expert = 41;
    bool caught = false;
    try { cache.prepare_host_batch(0, cold, 4, leases.data()); }
    catch (const std::runtime_error&) { caught = true; }
    expect(caught && observed->active == 0 && observed->finished == 41, "failed batch must finish all admitted callbacks");
    for (int c = 0; c < 4; ++c)
        expect(!leases[c] && !cache.query_residency(0, cold[c]).host_resident, "failed batch must discard partial slots and leases");
    observed->fail_expert = -1;
    expect(cache.prepare_host_batch(0, cold, 4, leases.data()) == 4, "cache must recover after reader failure");
    for (auto& lease : leases) lease.release();
    for (int c = 0; c < 40; ++c) ids[c] = c + 8;
    cache.prepare_host_batch(0, ids, 40, leases.data());
    expect(std::count_if(leases.begin(), leases.end(), [](const auto& lease) { return lease.is_valid(); }) == 39, "pinned external expert must force one capacity fallback");
    expect(cache.query_residency(0, 0).host_generation == pinned.generation(), "batch replacement must not evict an active reader");
    for (auto& lease : leases) lease.release();
    int waiting[4] = {4, 5, 6, 7};
    observed->hold = true;
    struct Unhold { BatchPrepareReader* reader; ~Unhold() { reader->hold = false; } } unhold{observed};
    cache.prefetch_async(0, waiting, 4, 4);
    const bool was_filling = cache.query_residency(0, waiting[3]).host_filling;
    observed->hold = false;
    expect(was_filling, "prefetch must still be filling for the wait fixture");
    expect(cache.prepare_host_batch(0, waiting, 4, leases.data(), true) == 0, "CPU must join existing fills instead of refilling");
    cache.prefetch_async(0, waiting, 4, 4);
    for (int c = 0; c < 4; ++c) {
        expect(leases[c] && std::memcmp(leases[c].slot_host().gate_up_packed,
            synth.gu_packed_host.data() + waiting[c] * synth.gu_packed_per_exp, synth.gu_packed_per_exp) == 0,
            "prefetch cancellation must retain completed CPU-leased input");
    }
    for (auto& lease : leases) lease.release();
    caught = false;
    try { cache.prepare_host_batch(0, ids, 41, leases.data()); } catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "oversized batch must be rejected before touching outputs");
    expect(cache.prepare_host_batch(0, nullptr, 0, nullptr) == 0, "empty preparation must be a no-op");
    std::cout << "[PASS] Host batch preparation: parallel fills, dedup, pinning, capacity, exception drain and retry\n";
}

void benchmark_host_preparation(Context& ctx) {
    _putenv("AILA_Q4E_FILL_THREADS=4");
    struct Reset { ~Reset() { _putenv("AILA_TEST_Q4E_HOST_SLOTS=4"); _putenv("AILA_Q4E_FILL_THREADS="); } } reset;
    SyntheticExpertWeightsT<1, 80, 2560, 640> synth;
    const auto geom = compact_geometry(synth);
    for (int count : {10, 40}) {
        _putenv(count == 10 ? "AILA_TEST_Q4E_HOST_SLOTS=10" : "AILA_TEST_Q4E_HOST_SLOTS=40");
        ModelWeights weights;
        Qwen4ExpExpertCache cache;
        std::string error;
        expect(cache.init(ctx, weights, 1, 4 * geom.slot_bytes, 80, synth.hidden_size, synth.intermediate_size, 64, &error, true), error);
        cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
        std::array<HostExpertLease, 40> leases;
        int ids[40], calls = 0;
        auto run = [&](bool batch) {
            for (auto& lease : leases) lease.release();
            const int base = (calls++ % 2) * count;
            for (int c = 0; c < count; ++c) ids[c] = base + c;
            const auto begin = std::chrono::steady_clock::now();
            if (batch) expect(cache.prepare_host_batch(0, ids, count, leases.data()) == count, "benchmark must force cold host misses");
            else for (int c = 0; c < count; ++c) {
                cache.prepare_host(0, ids[c]);
                leases[c] = cache.acquire_host_lease(0, ids[c]);
            }
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            for (int c = 0; c < count; ++c) {
                expect(leases[c].is_valid(), "benchmark requires complete host batch");
                expect(std::memcmp(leases[c].slot_host().gate_up_packed,
                    synth.gu_packed_host.data() + ids[c] * synth.gu_packed_per_exp, synth.gu_packed_per_exp) == 0, "prepared bytes mismatch");
            }
            return elapsed;
        };
        for (int warm = 0; warm < 6; ++warm) run(warm % 2);
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<double> times[2];
            for (int sample = 0; sample < 20; ++sample)
                for (int step = 0; step < 2; ++step) {
                    const int mode = (sample + step + rep) % 2;
                    times[mode].push_back(run(mode != 0));
                }
            for (int mode = 0; mode < 2; ++mode) {
                auto& values = times[mode];
                double mean = 0, variance = 0;
                for (double value : values) mean += value / values.size();
                for (double value : values) variance += (value - mean) * (value - mean) / values.size();
                const double cv = std::sqrt(variance) / mean;
                std::sort(values.begin(), values.end());
                std::cout << std::fixed << std::setprecision(4) << "[Host prepare] tasks=" << count << " batch=" << mode
                    << " rep=" << rep << " samples=20 median_ms=" << (values[9] + values[10]) * 0.5
                    << " p10_ms=" << values[1] << " p90_ms=" << values[17] << " cv=" << cv << " noisy=" << (cv > 0.1) << '\n';
            }
        }
    }
}

void test_sparse_result_copies(Context& ctx) {
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=8");
    struct Reset { ~Reset() { _putenv("AILA_TEST_Q4E_HOST_SLOTS=4"); _putenv("AILA_Q4E_CPU_COALESCE_RESULTS="); } } reset;
    SyntheticExpertWeights synth;
    ModelWeights weights;
    Qwen4ExpExpertCache cache;
    const auto geom = compact_geometry(synth);
    std::string error;
    expect(cache.init(ctx, weights, 1, 8 * geom.slot_bytes, 8, synth.hidden_size, synth.intermediate_size, 64, &error, true), error);
    cache.set_layer_expert_ref(0, synth.gu_ref, synth.dn_ref);
    int gpu[4] = {0, 3, 4, 7}, slots[4];
    cache.acquire(0, gpu, 4, slots).wait_and_throw();
    for (int expert : {1, 2, 5, 6}) cache.prepare_host(0, expert);
    auto x = Tensor::allocate(ctx, {synth.hidden_size}, dnnl::memory::data_type::bf16);
    auto h = Tensor::allocate(ctx, {8, synth.intermediate_size}, dnnl::memory::data_type::bf16);
    auto y = Tensor::allocate(ctx, {synth.hidden_size}, dnnl::memory::data_type::bf16);
    std::vector<bf16> input(synth.hidden_size, bf16(0.125f)), expected(input.size()), actual(input.size());
    ctx.memcpy_h2d(x.data(), input.data(), x.size_bytes());
    int ids[8] = {0, 1, 2, 3, 4, 5, 6, 7}; float rw[8] = {0.05f, 0.1f, 0.15f, 0.2f, 0.05f, 0.1f, 0.15f, 0.2f};
    for (int mode = 0; mode < 2; ++mode) {
        _putenv(mode ? "AILA_Q4E_CPU_COALESCE_RESULTS=1" : "AILA_Q4E_CPU_COALESCE_RESULTS=0");
        aila::Qwen4ExpExpertExecutor executor;
        executor.init(ctx, cache, 1, 8, synth.hidden_size, synth.intermediate_size);
        executor.set_max_cpu_credits(8);
        auto& plan = executor.plan_layer(0, ids, rw, 1, 8, true);
        expect(plan.cpu_indices == std::vector<int>({1, 2, 5, 6}), "sparse CPU rank fixture");
        executor.execute_hetero_decode(ctx, plan, x.data_as<bf16>(), h.data_as<bf16>(), rw, y.data_as<bf16>(), 1, 8, nullptr, nullptr, true, true).wait_and_throw();
        ctx.memcpy_d2h(actual.data(), y.data(), y.size_bytes());
        if (!mode) expected = actual;
        else expect(std::memcmp(expected.data(), actual.data(), y.size_bytes()) == 0, "coalesced copies must not overwrite intervening GPU ranks");
        expect(executor.stats().cpu_result_copies == (mode ? 2 : 4), "sparse CPU ranks require exactly two runs");
    }
    std::cout << "[PASS] Sparse result coalescing preserves GPU rank holes bitwise\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << "========================================\n";
        std::cout << " Aila Qwen4Exp DMA Timing & Decoupling (H5)\n";
        std::cout << "========================================\n";

        _putenv("AILA_TEST_Q4E_HOST_POOL=1");
        _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

        Context ctx;
        if (argc == 2 && std::string(argv[1]) == "--host-batch-bench") {
            benchmark_host_preparation(ctx);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--large-cpu-ordered") {
            _putenv("AILA_Q4E_CPU_GPU_ORDER=1");
            _putenv("AILA_Q4E_CPU_PREPARED_INPUT=1");
            test_large_cpu_placement(ctx);
            return 0;
        }
        test_dma_timing_stats(ctx);
        test_slot_dma_event_tracking(ctx);
        test_decoupled_ready_pending_gpu_execution(ctx);
        test_host_slot_dma_event_fencing(ctx);
        test_multi_token_decoupled_execution_and_hit_protection(ctx);
        test_cpu_fill_overlap(ctx);
        test_host_batch_prepare(ctx);
        test_sparse_result_copies(ctx);
        test_large_cpu_placement(ctx);

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
