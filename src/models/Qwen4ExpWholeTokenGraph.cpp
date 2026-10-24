#include "Qwen4ExpBnb4Backend.hpp"

#include "../ops/GraphControl.hpp"
#include "../profile/Profiling.hpp"
#include "Qwen4ExpTokenGraphKernels.hpp"
#include "Qwen4ExpTokenGraphStageProf.hpp"
#include "../utils/EnvUtils.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace ex = sycl::ext::oneapi::experimental;
using bf16 = sycl::ext::oneapi::bfloat16;
using aila::kTokenGraphPacketSlots;

namespace {

constexpr int kWholeTokenGraphHidden = 2560;
constexpr int kWholeTokenGraphLayers = 48;
constexpr int kWholeTokenGraphTopK = 10;
constexpr int kWholeTokenGraphExperts = 512;
constexpr int kWholeTokenGraphIntermediate = 640;
constexpr int kWholeTokenGraphFirstSparsePosition = 2052;
constexpr uint32_t kServiceTimeoutMs = 5000;
constexpr uint32_t kServiceSpinLimit = 20'000'000u;

bool whole_token_graph_requested() {
    static const bool enabled =
        aila::env::read_flag("AILA_Q4E_TOKEN_GRAPH", false);
    return enabled;
}

bool whole_token_graph_trace_requested() {
    static const bool enabled =
        aila::env::read_flag("AILA_Q4E_TOKEN_GRAPH_TRACE", false);
    return enabled;
}

bool same_token_graph_key(
    const aila::TokenGraphKey& a,
    const aila::TokenGraphKey& b) {
    return a.backend == b.backend && a.session == b.session &&
           a.session_instance_id == b.session_instance_id &&
           a.session_generation == b.session_generation &&
           a.session_state_generation == b.session_state_generation &&
           a.runtime_generation == b.runtime_generation &&
           a.runtime_capacity == b.runtime_capacity &&
           a.residency_generation == b.residency_generation &&
           a.descriptor_generation == b.descriptor_generation &&
           a.ple_workspace_generation == b.ple_workspace_generation &&
           a.ple_graph_view_generation == b.ple_graph_view_generation &&
           a.topology_generation == b.topology_generation &&
           a.topology_flags == b.topology_flags &&
           a.body == b.body && a.attention == b.attention;
}

uint64_t whole_token_schema_flags() {
    return (uint64_t{1}
            << static_cast<unsigned>(aila::TokenGraphBody::WholeToken)) |
           (uint64_t{1}
            << (8u + static_cast<unsigned>(
                        aila::TokenGraphAttention::QsaSparse512BlockNative)));
}

void submit_repeat_hidden4(sycl::queue& queue, const bf16* source,
                           bf16* destination) {
    using vec8 = sycl::vec<bf16, 8>;
    constexpr int dim8 = kWholeTokenGraphHidden / 8;
    constexpr int work_group = 256;
    constexpr int work_groups =
        (dim8 + work_group - 1) / work_group;
    queue.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_groups * work_group),
            sycl::range<1>(work_group)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(16)]] {
            const int index = static_cast<int>(item.get_global_id(0));
            if (index >= dim8) return;
            const vec8 value =
                *reinterpret_cast<const vec8*>(source + index * 8);
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                *reinterpret_cast<vec8*>(
                    destination + group * kWholeTokenGraphHidden +
                    index * 8) = value;
            }
        });
}

}  // namespace

bool Qwen4ExpBnb4Backend::WholeTokenGraphEnabled() {
    return whole_token_graph_requested();
}

bool Qwen4ExpBnb4Backend::WholeTokenGraphEligible(
    Context& ctx, Session& session, int count,
    const ForwardOptions& options) {
    const bool trace = whole_token_graph_trace_requested();
    const auto reject = [&](const char* reason) {
        if (trace) {
            AILA_LOG_INFO(
                "[Q4E TokenGraph] reject=%s pos=%d count=%d",
                reason, session.CurrentLength(), count);
        }
        return false;
    };
    if (!WholeTokenGraphEnabled() || count != 1 ||
        options.decode_verification || options.prefill_remaining >= 0 ||
        options.checkpoint_after != 0 || !session.graph_gdn_state_enabled_) {
        return reject("base");
    }
    const auto device = ctx.queue().get_device();
    if (!device.has(sycl::aspect::ext_oneapi_limited_graph) &&
        !device.has(sycl::aspect::ext_oneapi_graph)) {
        return reject("device_graph");
    }

    const int start_pos = session.CurrentLength();
    if (start_pos < kWholeTokenGraphFirstSparsePosition ||
        start_pos + 1 > session.Capacity() ||
        session.token_graph_poisoned()) {
        return reject("position");
    }
    if (session.mropePosT_.valid() || session.mropePosH_.valid() ||
        session.mropePosW_.valid() || session.mropePromptLength_ != 0 ||
        session.mropeTextPositionDelta_ != 0 ||
        !session.embeddingOverridePositions_.empty() ||
        !session.embeddingOverrideValues_.empty() ||
        session.embeddingOverrideHiddenSize_ != 0) {
        return reject("overrides");
    }
    if ((!embed_host_.valid() &&
         (!embed_weight_ || !embed_weight_->valid())) ||
        hidden_size_ != kWholeTokenGraphHidden ||
        cfg_.num_hidden_layers != kWholeTokenGraphLayers ||
        cfg_.num_experts != kWholeTokenGraphExperts ||
        cfg_.num_experts_per_tok != kWholeTokenGraphTopK ||
        cfg_.moe_intermediate_size != kWholeTokenGraphIntermediate ||
        cfg_.shared_expert_intermediate_size !=
            kWholeTokenGraphIntermediate ||
        full_q_heads_ != 24 || full_kv_heads_ != 2 ||
        full_head_dim_ != 256 || full_q_dim_ != 6144 ||
        full_kv_dim_ != 512 || full_q_proj_dim_ != 12288 ||
        full_fused_qkv_dim_ != 13312 ||
        cfg_.indexer_n_heads != 4 || cfg_.indexer_kv_heads != 1 ||
        cfg_.indexer_head_dim != 128 ||
        (cfg_.indexer_compress_ratio > 0
             ? cfg_.indexer_compress_ratio
             : 4) != 4 ||
        (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / 4 !=
            512) {
        return reject("geometry_or_embedding");
    }
    if (verify_rows_alignment_ || gpuNf4Q8ReferenceForTest_ ||
        decodeHcReferenceForTest_ || attentionNf4Q8Enabled_ ||
        gpuNf4Q8Supported_ || !use_adaptive_pipeline_ ||
        !adaptive_cache_.IsAllocated() ||
        adaptive_cache_.NumLayers() < kWholeTokenGraphLayers ||
        !adaptive_cache_.DeviceSlotTable() ||
        !host_arena_.IsAllocated() ||
        host_arena_.NumLayers() < kWholeTokenGraphLayers ||
        !resident_down_.valid() || !resident_weights_.valid() ||
        !buf_.moe_h.valid() || !buf_.shared_gate_buf.valid() ||
        !buf_.shared_inter_buf.valid() || !buf_.shared_out_buf.valid()) {
        return reject("runtime_adaptive");
    }
    if (!expert_executor_.cpu_overlap_enabled() ||
        expert_executor_.max_cpu_credits() < kWholeTokenGraphTopK) {
        return reject("cpu_service");
    }
    const bool compressed_k =
        aila::env::read_flag("AILA_QSA_COMPRESSED_K", true);
    const bool block_native =
        aila::env::read_flag("AILA_QSA_BLOCK_NATIVE", true);
    const bool dense_gemv =
        aila::env::read_flag("AILA_Q4E_DENSE_GEMV", true);
    if (!compressed_k || !block_native || !dense_gemv ||
        !ple_.key_proj().supports_decode_graph() ||
        !ple_.value_proj().supports_decode_graph()) {
        return reject("env_or_ple");
    }

    const int target_blocks = (start_pos + 1) / 4;
    for (int layer_index = 0; layer_index < kWholeTokenGraphLayers;
         ++layer_index) {
        const auto& layer = layers_[layer_index];
        if (layer.is_linear) {
            if (!layer.linear_all_proj.supports_decode_graph() ||
                !layer.linear_o_proj.supports_decode_graph()) {
                return reject("linear_projection");
            }
            continue;
        }
        const auto& cache = session.layerCaches_[layer_index];
        if (!layer.qkv_proj.supports_decode_graph() ||
            !layer.o_proj.supports_decode_graph() ||
            !layer.q_norm_weight || !layer.k_norm_weight ||
            !layer.index_qk_weight || !layer.index_q_norm_weight ||
            !layer.index_k_norm_weight ||
            layer.q_norm_weight->dtype() != dnnl::memory::data_type::bf16 ||
            layer.k_norm_weight->dtype() != dnnl::memory::data_type::bf16 ||
            layer.q_norm_weight->shape(0) != 256 ||
            layer.k_norm_weight->shape(0) != 256 ||
            layer.index_qk_weight->dtype() !=
                dnnl::memory::data_type::bf16 ||
            layer.index_qk_weight->ndim() != 2 ||
            layer.index_qk_weight->shape(0) != 640 ||
            layer.index_qk_weight->shape(1) != 2560 ||
            layer.index_q_norm_weight->dtype() !=
                dnnl::memory::data_type::bf16 ||
            layer.index_k_norm_weight->dtype() !=
                dnnl::memory::data_type::bf16 ||
            layer.index_q_norm_weight->shape(0) != 128 ||
            layer.index_k_norm_weight->shape(0) != 128 ||
            cache.k.dtype() != dnnl::memory::data_type::bf16 ||
            cache.v.dtype() != dnnl::memory::data_type::bf16 ||
            cache.indexer_k.dtype() != dnnl::memory::data_type::bf16 ||
            cache.indexer_compressed_k.dtype() !=
                dnnl::memory::data_type::bf16 ||
            cache.k_scale.valid() || cache.v_scale.valid() ||
            cache.num_compressed_blocks < target_blocks - 1 ||
            cache.num_compressed_blocks > target_blocks) {
            return reject("full_attention_layout");
        }
    }
    return true;
}

namespace {

#include <thread>

// ---- debug-only per-node device timing for the whole-token graph body ----
// Enabled by AILA_Q4E_TOKEN_GRAPH_NODE_PROF=1 on a profiling queue
// (AILA_PROFILE_Q4E_COLD_DEVICE=1). When enabled the body runs un-captured on
// the live queue, so every mark is a real barrier whose profiling timestamps
// can be queried; a captured graph yields no per-node events at all. Marks go
// after the last submission of a node group and carry that group's label, so
// the interval between two consecutive marks is the device span of the group
// that sits between them. Barriers serialise the stream, so absolute totals
// carry barrier overhead; the per-group shares do not.
enum WholeTokenNodeLabel : int {
    kNodeAnchor = 0,
    kNodeAttnHc,
    kNodeAttention,
    kNodeMlpHc,
    kNodeRoute,
    kNodeResident,
    kNodeShared,
    kNodeTail,
    // QSA layer-body sub-stages, declared in Qwen4ExpTokenGraphStageProf.hpp so
    // the body in Qwen4ExpPrefill.cpp can mark them too. Keep contiguous.
    kNodeQkvProj,
    kNodeIndexerQk,
    kNodePrepareQkv,
    kNodePrepareIndexerQ,
    kNodeCompressBlock,
    kNodeBlockScores,
    kNodeSelectBlocks,
    kNodeValidateBlocks,
    kNodeSparseAttend,
    kNodeOProj,
    kNodeHcNorm,
    kNodeHcDown,
    kNodeHcUp,
    kNodeEmpty,
    kNodeLabelCount,
};

static_assert(static_cast<int>(aila::token_graph_stage_prof::kStageEnd) ==
                  static_cast<int>(kNodeLabelCount),
              "stage labels drifted from the whole-token profiler labels");

constexpr const char* kNodeLabelNames[kNodeLabelCount] = {
    "anchor", "attn_hc", "attention", "mlp_hc",
    "route", "resident", "shared", "tail",
    "qkv_proj", "indexer_qk", "prepare_qkv", "prepare_idx_q", "compress",
    "blk_scores", "select", "validate", "attend", "o_proj",
    "hc_norm", "hc_down", "hc_up", "empty"};

struct WholeTokenNodeProfiler {
    bool enabled = false;
    sycl::queue* queue = nullptr;
    std::vector<std::pair<int, sycl::event>> marks;

    static WholeTokenNodeProfiler& instance() {
        static WholeTokenNodeProfiler profiler;
        return profiler;
    }

    void mark(int label) {
        if (!enabled || queue == nullptr) return;
        marks.emplace_back(label, queue->ext_oneapi_submit_barrier());
    }

    void drain_and_log(const char* where) {
        if (!enabled || marks.size() < 2) return;
        try {
            marks.back().second.wait_and_throw();
        } catch (...) {
            marks.clear();
            return;
        }
        double span[kNodeLabelCount] = {};
        int count[kNodeLabelCount] = {};
        for (std::size_t i = 1; i < marks.size(); ++i) {
            try {
                const auto prev = marks[i - 1].second.get_profiling_info<
                    sycl::info::event_profiling::command_end>();
                const auto cur = marks[i].second.get_profiling_info<
                    sycl::info::event_profiling::command_end>();
                if (cur > prev) {
                    const int label = marks[i].first;
                    span[label] += static_cast<double>(cur - prev) * 1e-6;
                    ++count[label];
                }
            } catch (...) {
            }
        }
        double total = 0.0;
        for (int i = 0; i < kNodeLabelCount; ++i) total += span[i];
        std::string line;
        char cell[40];
        for (int i = 0; i < kNodeLabelCount; ++i) {
            // Keep two decimals: QSA sub-stages are far below 1 ms each and an
            // integer format silently reads as zero.
            std::snprintf(cell, sizeof(cell), " %s=%.2f/%d",
                          kNodeLabelNames[i], span[i], count[i]);
            line += cell;
        }
        AILA_LOG_INFO("[Q4E NodeProf] %s marks=%zu total_ms=%.2f%s",
                      where, marks.size(), total, line.c_str());
        if (total <= 0.0) {
            AILA_LOG_WARN(
                "[Q4E NodeProf] no profiling timestamps; queue is not "
                "profiling-enabled (set AILA_PROFILE_Q4E_COLD_DEVICE=1)");
        }
        marks.clear();
    }
};

WholeTokenNodeProfiler& whole_token_node_profiler() {
    return WholeTokenNodeProfiler::instance();
}

bool whole_token_node_prof_requested() {
    static const bool requested = [] {
        const char* raw = std::getenv("AILA_Q4E_TOKEN_GRAPH_NODE_PROF");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    return requested;
}

// Submits the recorded node sequence straight to the in-order queue instead of
// replaying it through a sycl command graph. Same nodes, same doorbell/packet
// handoff, no graph executable. Exists to separate graph-replay cost from the
// cost of the nodes themselves; measurements showed the two differ by a wide
// margin, so this is also a candidate production path, not only a probe.
// Collapses the three element-wise tail kernels (ordered merge, shared-expert
// add, final bridge) into a single launch. At the measured 21 us launch floor
// that removes 96 launches per token, about 2 ms.
// AILA_Q4E_TG_ROUTE_FUSED=0 restores the split router. The fused form uses
// launch_router_softmax_topk<true>, which performs the resident slot
// classification inside the top-k kernel, so the separate classify submission
// disappears: one submission saved per layer, 48 per token, about 2.5 ms at the
// measured ~53 us per submit.
bool whole_token_route_fused_requested() {
    static const bool requested = [] {
        // Off by default. The mechanism is verified by reading the code -
        // launch_router_softmax_topk<true> submits one kernel where this path
        // used to submit two - and wiring the graph tensors in place of the
        // Direct path's host arrays removed the 0xC0000409 crash. But a single
        // A/B cannot resolve it: submit_ms is 35.4-38.6 ms across recent runs,
        // so a 2.5 ms saving is inside its noise, and route_wait (device side)
        // should not move at all because this only removes a host submission.
        // Land several roadmap fusions and measure the aggregate instead.
        const char* raw = std::getenv("AILA_Q4E_TG_ROUTE_FUSED");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    return requested;
}

bool whole_token_tail_fused_requested() {
    static const bool requested = [] {
        const char* raw = std::getenv("AILA_Q4E_TG_TAIL_FUSED");
        // On by default: bit-exact, and two A/B pairs taken in opposite order
        // both show route_wait improving by 0.85-0.97 ms (96 launches removed,
        // i.e. about 9.4 us per launch in the real non-profiling path).
        if (raw == nullptr || raw[0] == '\0') return true;
        return raw[0] != '0';
    }();
    return requested;
}

// Records the token as one command graph per layer instead of one graph for
// the whole token. Experimental: see section 20.4 of the round-1 document.
bool whole_token_subgraphs_requested() {
    static const bool requested = [] {
        const char* raw = std::getenv("AILA_Q4E_TG_SUBGRAPHS");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    return requested;
}

bool whole_token_unrolled_requested() {
    static const bool requested = [] {
        const char* raw = std::getenv("AILA_Q4E_TOKEN_GRAPH_UNROLLED");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    return requested;
}

}  // namespace

namespace aila::token_graph_stage_prof {

void mark(int stage) { whole_token_node_profiler().mark(stage); }

}  // namespace aila::token_graph_stage_prof

std::size_t Qwen4ExpBnb4Backend::RecordWholeTokenGraphBody(
    Context& ctx, Session& session, int* control) {
    auto& graph = whole_token_graph_;
    if (!graph.service.allocated()) {
        throw std::runtime_error("whole-token graph service is not allocated");
    }
    auto* control_ring = graph.service.control();
    auto* route_ring = graph.service.route();
    auto* packet = graph.service.packet();
    auto* epoch = reinterpret_cast<aila::TokenGraphDeviceEpoch*>(
        graph.device_epoch.data());
    auto* packet_dev = graph.packet_dev.data_as<int>();
    const int top_k = cfg_.num_experts_per_tok;
    const int experts = cfg_.num_experts;
    const int intermediate = cfg_.moe_intermediate_size;
    std::size_t nodes = 0;
    auto node = [&nodes] { ++nodes; };

    aila::token_graph_service::submit_begin_replay(ctx.queue(), epoch);
    node();
    // Launch-cost calibration for the profiler: eight empty kernels, each
    // bracketed by marks, so the average `empty` span is the floor that every
    // real stage span contains on top of its own execution time. Without this
    // number a 40 us span cannot be attributed to either launch or work.
    if (whole_token_node_profiler().enabled) {
        for (int sample = 0; sample < 8; ++sample) {
            ctx.queue().parallel_for(
                sycl::range<1>(32),
                [=](sycl::item<1> item) { (void)item; });
            whole_token_node_profiler().mark(kNodeEmpty);
        }
    }
    submit_repeat_hidden4(
        ctx.queue(), buf_.x.data_as<bf16>(),
        buf_.hidden4.data_as<bf16>());
    node();
    whole_token_node_profiler().mark(kNodeAnchor);

    int full_ordinal = 0;
    for (int layer_index = 0; layer_index < cfg_.num_hidden_layers;
         ++layer_index) {
        auto& layer = layers_[layer_index];
        auto& cache = session.layerCaches_[layer_index];
        if (layer_index == 1) {
            ple_.consume_graph(ctx, graph.ple_view);
            node();
        }

        RunHyperConnection(ctx, layer.attn_hc, 1, true);
        node();
        whole_token_node_profiler().mark(kNodeAttnHc);
        if (layer.is_linear) {
            RunAttentionLayerTile(
                ctx, session, layer_index, buf_.x, buf_.y, 1,
                session.CurrentLength(), nullptr, 0, false);
            node();
        } else {
            nodes += RecordSingleFullAttentionLayerGraphBody(
                ctx, session, layer_index, full_ordinal, buf_.x,
                buf_.y, control);
            ++full_ordinal;
        }
        whole_token_node_profiler().mark(kNodeAttention);

        RunHyperConnectionCombined(
            ctx, layer.mlp_hc, 1, true, buf_.y.data_as<bf16>(),
            buf_.hc_inject_f32.data_as<float>(), nullptr);
        node();
        whole_token_node_profiler().mark(kNodeMlpHc);

        ops::bf16_moe_front_fused(
            ctx, layer.router_w->data_as<bf16>(),
            layer.shared_expert_gate_w->data_as<bf16>(),
            buf_.x.data_as<bf16>(),
            buf_.router_scores.data_as<bf16>(),
            buf_.shared_gate_buf.data_as<bf16>(), nullptr, nullptr,
            hidden_size_, experts);
        if (whole_token_route_fused_requested()) {
            ops::router_softmax_topk_resident(
                ctx, buf_.router_scores.data_as<bf16>(), nullptr,
                graph.route_ids.data_as<int>(),
                graph.route_weights.data_as<float>(),
                adaptive_cache_.DeviceSlotTable() + layer_index * experts,
                // The three host-side route arrays are the graph's own tensors:
                // router_softmax_topk_resident writes host_ids/weights/selected
                // unconditionally, and the Direct path's buffers are sized for a
                // different call shape (cap * top_k), which crashed this path
                // with 0xC0000409. Writing the same values into the tensors this
                // path already owns is idempotent.
                graph.selected.data_as<int>(),
                graph.route_ids.data_as<int>(),
                graph.route_weights.data_as<float>(),
                graph.selected.data_as<int>(), experts, top_k, true);
        } else {
            ops::router_softmax_topk_device(
                ctx, buf_.router_scores.data_as<bf16>(), nullptr,
                graph.route_ids.data_as<int>(),
                graph.route_weights.data_as<float>(), 1, experts, top_k,
                true);
        }
        aila::token_graph_service::submit_input_and_route_publish(
            ctx.queue(),
            reinterpret_cast<const std::uint16_t*>(
                buf_.x.data_as<bf16>()),
            control_ring, route_ring, packet, epoch,
            graph.route_ids.data_as<int>(),
            graph.route_weights.data_as<float>(),
            static_cast<uint32_t>(top_k), layer_index == 0);
        if (!whole_token_route_fused_requested()) {
            ops::moe_resident_classify_device(
                ctx, graph.route_ids.data_as<int>(),
                adaptive_cache_.DeviceSlotTable() + layer_index * experts,
                graph.selected.data_as<int>(), top_k, experts);
        }
        node();
        whole_token_node_profiler().mark(kNodeRoute);

        ops::moe_resident_gate_up(
            ctx, graph.resident_descriptors.data_as<ops::MoeExpertSlotDev>(),
            graph.selected.data_as<int>(), buf_.x.data_as<bf16>(),
            buf_.moe_h.data_as<bf16>(), top_k, hidden_size_,
            intermediate, true, true);
        ops::moe_resident_down(
            ctx, graph.resident_descriptors.data_as<ops::MoeExpertSlotDev>(),
            graph.selected.data_as<int>(), buf_.moe_h.data_as<bf16>(),
            resident_down_.data_as<float>(), top_k, hidden_size_,
            intermediate, true, true);
        whole_token_node_profiler().mark(kNodeResident);

        if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                ctx, layer.shared_gate_up_proj, buf_.x,
                buf_.shared_inter_buf,
                cfg_.shared_expert_intermediate_size)) {
            layer.shared_gate_up_proj.ForwardDecodeRows(
                ctx, linear_scratch_, buf_.x,
                buf_.shared_gate_up_buf, 1);
            ops::moe_silu_gate_up_mul_flat(
                ctx, buf_.shared_gate_up_buf.data_as<bf16>(),
                buf_.shared_inter_buf.data_as<bf16>(), 1,
                cfg_.shared_expert_intermediate_size);
        }
        layer.shared_down_proj.ForwardDecodeRows(
            ctx, linear_scratch_, buf_.shared_inter_buf,
            buf_.shared_out_buf, 1);
        node();
        whole_token_node_profiler().mark(kNodeShared);

        aila::token_graph_service::submit_packet_handoff(
            ctx.queue(), control_ring, packet, epoch, packet_dev,
            static_cast<uint32_t>(top_k), kServiceSpinLimit);
        if (whole_token_tail_fused_requested()) {
            aila::token_graph_kernels::submit_gated_fused_tail(
                ctx.queue(), resident_down_.data_as<float>(),
                graph.route_weights.data_as<float>(),
                graph.selected.data_as<int>(), packet_dev,
                buf_.shared_out_buf.data_as<bf16>(),
                buf_.shared_gate_buf.data_as<bf16>(),
                buf_.hc_inject_f32.data_as<float>(),
                buf_.y.data_as<bf16>(), buf_.hidden4.data_as<bf16>(),
                control_ring, epoch, static_cast<uint32_t>(top_k));
        } else {
            aila::token_graph_kernels::submit_gated_ordered_merge(
                ctx.queue(), resident_down_.data_as<float>(),
                graph.route_weights.data_as<float>(),
                graph.selected.data_as<int>(), packet_dev,
                buf_.y.data_as<bf16>(), control_ring, epoch,
                static_cast<uint32_t>(top_k));
            aila::token_graph_kernels::submit_add_shared_expert_to_y(
                ctx.queue(), buf_.y.data_as<bf16>(),
                buf_.shared_out_buf.data_as<bf16>(),
                buf_.shared_gate_buf.data_as<bf16>(), epoch);
            aila::token_graph_kernels::submit_gated_final_bridge(
                ctx.queue(), buf_.hidden4.data_as<bf16>(),
                buf_.y.data_as<bf16>(),
                buf_.hc_inject_f32.data_as<float>(), epoch);
        }
        if (layer_index + 1 == cfg_.num_hidden_layers) {
            aila::token_graph_service::submit_publish_done(
                ctx.queue(), control_ring, epoch);
            node();
        }
        whole_token_node_profiler().mark(kNodeTail);
    }
    return nodes;
}

std::size_t Qwen4ExpBnb4Backend::RecordWholeTokenGraphPrologue(
    Context& ctx, Session& session) {
    auto& graph = whole_token_graph_;
    if (!graph.service.allocated()) {
        throw std::runtime_error("whole-token graph service is not allocated");
    }
    (void)session;
    auto* epoch = reinterpret_cast<aila::TokenGraphDeviceEpoch*>(
        graph.device_epoch.data());
    std::size_t nodes = 0;
    aila::token_graph_service::submit_begin_replay(ctx.queue(), epoch);
    ++nodes;
    if (whole_token_node_profiler().enabled) {
        for (int sample = 0; sample < 8; ++sample) {
            ctx.queue().parallel_for(
                sycl::range<1>(32),
                [=](sycl::item<1> item) { (void)item; });
            whole_token_node_profiler().mark(kNodeEmpty);
        }
    }
    submit_repeat_hidden4(
        ctx.queue(), buf_.x.data_as<bf16>(),
        buf_.hidden4.data_as<bf16>());
    ++nodes;
    whole_token_node_profiler().mark(kNodeAnchor);
    return nodes;
}

std::size_t Qwen4ExpBnb4Backend::RecordWholeTokenGraphEpilogue(
    Context& ctx, Session& session) {
    auto& graph = whole_token_graph_;
    (void)session;
    auto* control_ring = graph.service.control();
    auto* epoch = reinterpret_cast<aila::TokenGraphDeviceEpoch*>(
        graph.device_epoch.data());
    aila::token_graph_service::submit_publish_done(
        ctx.queue(), control_ring, epoch);
    return 1;
}

std::size_t Qwen4ExpBnb4Backend::RecordWholeTokenGraphLayer(
    Context& ctx, Session& session, int layer_index, int full_ordinal,
    int* control) {
    auto& graph = whole_token_graph_;
    if (!graph.service.allocated()) {
        throw std::runtime_error("whole-token graph service is not allocated");
    }
    auto* control_ring = graph.service.control();
    auto* route_ring = graph.service.route();
    auto* packet = graph.service.packet();
    auto* epoch = reinterpret_cast<aila::TokenGraphDeviceEpoch*>(
        graph.device_epoch.data());
    auto* packet_dev = graph.packet_dev.data_as<int>();
    const int top_k = cfg_.num_experts_per_tok;
    const int experts = cfg_.num_experts;
    const int intermediate = cfg_.moe_intermediate_size;
    std::size_t nodes = 0;
    auto node = [&nodes] { ++nodes; };

    auto& layer = layers_[layer_index];
    if (layer_index == 1) {
        ple_.consume_graph(ctx, graph.ple_view);
        node();
    }

    RunHyperConnection(ctx, layer.attn_hc, 1, true);
    node();
    whole_token_node_profiler().mark(kNodeAttnHc);
    if (layer.is_linear) {
        RunAttentionLayerTile(
            ctx, session, layer_index, buf_.x, buf_.y, 1,
            session.CurrentLength(), nullptr, 0, false);
        node();
    } else {
        nodes += RecordSingleFullAttentionLayerGraphBody(
            ctx, session, layer_index, full_ordinal, buf_.x,
            buf_.y, control);
    }
    whole_token_node_profiler().mark(kNodeAttention);

    RunHyperConnectionCombined(
        ctx, layer.mlp_hc, 1, true, buf_.y.data_as<bf16>(),
        buf_.hc_inject_f32.data_as<float>(), nullptr);
    node();
    whole_token_node_profiler().mark(kNodeMlpHc);

    ops::bf16_moe_front_fused(
        ctx, layer.router_w->data_as<bf16>(),
        layer.shared_expert_gate_w->data_as<bf16>(),
        buf_.x.data_as<bf16>(),
        buf_.router_scores.data_as<bf16>(),
        buf_.shared_gate_buf.data_as<bf16>(), nullptr, nullptr,
        hidden_size_, experts);
    ops::router_softmax_topk_device(
        ctx, buf_.router_scores.data_as<bf16>(), nullptr,
        graph.route_ids.data_as<int>(),
        graph.route_weights.data_as<float>(), 1, experts, top_k,
        true);
    aila::token_graph_service::submit_input_and_route_publish(
        ctx.queue(),
        reinterpret_cast<const std::uint16_t*>(buf_.x.data_as<bf16>()),
        control_ring, route_ring, packet, epoch,
        graph.route_ids.data_as<int>(),
        graph.route_weights.data_as<float>(),
        static_cast<uint32_t>(top_k), layer_index == 0);
    ops::moe_resident_classify_device(
        ctx, graph.route_ids.data_as<int>(),
        adaptive_cache_.DeviceSlotTable() + layer_index * experts,
        graph.selected.data_as<int>(), top_k, experts);
    node();
    whole_token_node_profiler().mark(kNodeRoute);

    ops::moe_resident_gate_up(
        ctx, graph.resident_descriptors.data_as<ops::MoeExpertSlotDev>(),
        graph.selected.data_as<int>(), buf_.x.data_as<bf16>(),
        buf_.moe_h.data_as<bf16>(), top_k, hidden_size_,
        intermediate, true, true);
    ops::moe_resident_down(
        ctx, graph.resident_descriptors.data_as<ops::MoeExpertSlotDev>(),
        graph.selected.data_as<int>(), buf_.moe_h.data_as<bf16>(),
        resident_down_.data_as<float>(), top_k, hidden_size_,
        intermediate, true, true);
    whole_token_node_profiler().mark(kNodeResident);

    if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
            ctx, layer.shared_gate_up_proj, buf_.x, buf_.shared_inter_buf,
            cfg_.shared_expert_intermediate_size)) {
        layer.shared_gate_up_proj.ForwardDecodeRows(
            ctx, linear_scratch_, buf_.x, buf_.shared_gate_up_buf, 1);
        ops::moe_silu_gate_up_mul_flat(
            ctx, buf_.shared_gate_up_buf.data_as<bf16>(),
            buf_.shared_inter_buf.data_as<bf16>(), 1,
            cfg_.shared_expert_intermediate_size);
    }
    layer.shared_down_proj.ForwardDecodeRows(
        ctx, linear_scratch_, buf_.shared_inter_buf,
        buf_.shared_out_buf, 1);
    node();
    whole_token_node_profiler().mark(kNodeShared);

    aila::token_graph_service::submit_packet_handoff(
        ctx.queue(), control_ring, packet, epoch, packet_dev,
        static_cast<uint32_t>(top_k), kServiceSpinLimit);
    if (whole_token_tail_fused_requested()) {
        aila::token_graph_kernels::submit_gated_fused_tail(
            ctx.queue(), resident_down_.data_as<float>(),
            graph.route_weights.data_as<float>(),
            graph.selected.data_as<int>(), packet_dev,
            buf_.shared_out_buf.data_as<bf16>(),
            buf_.shared_gate_buf.data_as<bf16>(),
            buf_.hc_inject_f32.data_as<float>(),
            buf_.y.data_as<bf16>(), buf_.hidden4.data_as<bf16>(),
            control_ring, epoch, static_cast<uint32_t>(top_k));
    } else {
        aila::token_graph_kernels::submit_gated_ordered_merge(
            ctx.queue(), resident_down_.data_as<float>(),
            graph.route_weights.data_as<float>(),
            graph.selected.data_as<int>(), packet_dev,
            buf_.y.data_as<bf16>(), control_ring, epoch,
            static_cast<uint32_t>(top_k));
        aila::token_graph_kernels::submit_add_shared_expert_to_y(
            ctx.queue(), buf_.y.data_as<bf16>(),
            buf_.shared_out_buf.data_as<bf16>(),
            buf_.shared_gate_buf.data_as<bf16>(), epoch);
        aila::token_graph_kernels::submit_gated_final_bridge(
            ctx.queue(), buf_.hidden4.data_as<bf16>(),
            buf_.y.data_as<bf16>(),
            buf_.hc_inject_f32.data_as<float>(), epoch);
    }
    whole_token_node_profiler().mark(kNodeTail);
    return nodes;
}

bool Qwen4ExpBnb4Backend::ServiceWholeTokenGraphReplay(
    Context& ctx, Session& session, sycl::event& event,
    uint32_t token_sequence, uint32_t& gate_done_sequence) {
    gate_done_sequence = 0;
    const bool trace = whole_token_graph_trace_requested();
    const auto service_begin = trace
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    double route_wait_ms = 0.0;
    double cold_ms = 0.0;
    double publish_ms = 0.0;
    auto& graph = whole_token_graph_;
    auto& service = graph.service;
    struct RouteRecord {
        std::array<int, kWholeTokenGraphTopK> ids{};
        std::array<float, kWholeTokenGraphTopK> weights{};
    };
    std::array<RouteRecord, kWholeTokenGraphLayers> routes{};
    std::array<int, kWholeTokenGraphTopK> cold_ids{};
    std::array<int, kWholeTokenGraphTopK> cold_ranks{};

    for (int layer_index = 0; layer_index < cfg_.num_hidden_layers;
         ++layer_index) {
        uint32_t sequence = 0;
        const auto route_begin = trace
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        if (service.wait_request(
                kServiceTimeoutMs, sequence) !=
            aila::Qwen4ExpTokenGraphService::WaitResult::Reached) {
            return false;
        }
        if (trace) {
            route_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - route_begin).count();
        }
        aila::token_graph_service::RouteView route;
        if (!service.read_route(
                sequence, cfg_.num_experts_per_tok, cfg_.num_experts,
                route)) {
            return false;
        }
        for (int rank = 0; rank < cfg_.num_experts_per_tok; ++rank) {
            routes[layer_index].ids[rank] = route.ids[rank];
            routes[layer_index].weights[rank] = route.weights[rank];
        }

        int cold_count = 0;
        for (int rank = 0; rank < cfg_.num_experts_per_tok; ++rank) {
            const int expert = route.ids[rank];
            if (adaptive_cache_.GetSlot(layer_index, expert) >= 0) {
                continue;
            }
            cold_ids[cold_count] = expert;
            cold_ranks[cold_count] = rank;
            ++cold_count;
        }
        auto* packet = service.packet();
        std::fill_n(
            packet->rank_to_row, kTokenGraphPacketSlots, -1);
        int rows = 0;
        if (cold_count > 0) {
            const auto cold_begin = trace
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            rows = expert_executor_.RunGraphColdPacket(
                host_arena_, layer_index, cold_ids.data(),
                cold_ranks.data(), cold_count,
                reinterpret_cast<const bf16*>(
                    service.packet_input_bf16()),
                packet->rank_to_row, packet->values);
            if (trace) {
                cold_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - cold_begin).count();
            }
        }
        packet->rows = static_cast<uint32_t>(rows);
        aila::Qwen4ExpTokenGraphService::PacketView view;
        view.gate = aila::TokenGraphGateState::Valid;
        view.top_k = static_cast<uint32_t>(cfg_.num_experts_per_tok);
        view.rows = static_cast<uint32_t>(rows);
        view.rank_to_row = packet->rank_to_row;
        view.values = packet->values;
        const auto publish_begin = trace
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        if (!service.publish_packet(sequence, view)) {
            return false;
        }
        if (trace) {
            publish_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - publish_begin).count();
        }
        adaptive_cache_.ObserveRouting(
            layer_index, route.ids, cfg_.num_experts_per_tok);
        if (rows > 0) {
            session.consecutive_no_miss_layers_ = 0;
        } else {
            ++session.consecutive_no_miss_layers_;
        }
        gate_done_sequence = sequence;
        (void)token_sequence;
    }

    if (!service.wait_done(gate_done_sequence, kServiceTimeoutMs)) {
        return false;
    }
    try {
        event.wait_and_throw();
    } catch (...) {
        return false;
    }
    if (trace) {
        const double service_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - service_begin).count();
        AILA_LOG_INFO(
            "[Q4E TokenGraphTiming] pos=%d route_wait_ms=%.3f "
            "cold_ms=%.3f publish_ms=%.3f service_ms=%.3f "
            "spin_iters=%llu",
            session.CurrentLength(), route_wait_ms, cold_ms, publish_ms,
            service_ms,
            static_cast<unsigned long long>(
                service.control()->reserved[0] & 0xffffffffull));
    }
    return true;
}

Tensor& Qwen4ExpBnb4Backend::ForwardWholeTokenGraph(
    Context& ctx, Session& session, const int* tokenIdsDevice,
    int count, const ForwardOptions& options) {
    if (!WholeTokenGraphEligible(ctx, session, count, options)) {
        throw std::logic_error(
            "Qwen4Exp whole-token graph entered ineligible decode state");
    }
    const int start_pos = session.CurrentLength();
    const bool trace = whole_token_graph_trace_requested();
    if (trace) {
        AILA_LOG_INFO(
            "[Q4E TokenGraph] eligible=1 pos=%d capacity=%d",
            start_pos, session.Capacity());
    }
    const int cached_len = start_pos + count;
    auto& graph = whole_token_graph_;
    ensure_runtime_buffers(ctx, 1);

    int token_id = 0;
    ctx.memcpy_d2h(&token_id, tokenIdsDevice, sizeof(token_id));
    if (token_id < 0 || token_id >= cfg_.vocab_size) {
        throw std::runtime_error(
            "Qwen4Exp whole-token graph received an invalid token id");
    }

    EmbedTokens(ctx, tokenIdsDevice, 1, &token_id);

    auto prepare_ple = [&] {
        ple_ticket_ = ple_.prepare_fixed_slot(
            ctx, session.ple_state_, &token_id, 1, &session,
            session.generation(), start_pos,
            session.ple_ticket_epoch());
        ple_.wait_prepared(ctx, ple_ticket_);
    };

    const bool have_graph =
        graph.executable && token_graph_runtime_.phase() ==
                                aila::TokenGraphPhase::Ready;
    bool reuse = false;
    if (have_graph) {
        const auto key = MakeTokenGraphKey(
            session, aila::TokenGraphBody::WholeToken,
            aila::TokenGraphAttention::QsaSparse512BlockNative,
            whole_token_schema_flags());
        reuse = same_token_graph_key(token_graph_runtime_.key(), key);
    }

    if (!reuse) {
        if (!ReleaseWholeTokenGraph(&ctx, &session, false)) {
            throw std::runtime_error(
                "Qwen4Exp whole-token graph could not release stale executable");
        }
        prepare_ple();
        graph.ple_view = ple_.create_graph_view(
            ctx, session.ple_state_, buf_.hidden4.data_as<bf16>(),
            nullptr, buf_.hidden4.data_as<bf16>());
        graph.ple_view_valid = true;
        PrepareGdnGraphState(ctx, session);

        if (!graph.control.valid()) {
            graph.control = Tensor::allocate(
                ctx, {ops::graph_control::words},
                dnnl::memory::data_type::s32);
        }
        if (!graph.device_epoch.valid()) {
            graph.device_epoch = Tensor::allocate(
                ctx, {sizeof(aila::TokenGraphDeviceEpoch)},
                dnnl::memory::data_type::u8);
        }
        if (!graph.packet_dev.valid()) {
            graph.packet_dev = Tensor::allocate(
                ctx, {int64_t(kTokenGraphPacketSlots) +
                          int64_t(kTokenGraphPacketSlots) *
                              kWholeTokenGraphHidden},
                dnnl::memory::data_type::s32);
        }
        if (!graph.route_ids.valid()) {
            graph.route_ids = Tensor::allocate(
                ctx, {cfg_.num_experts_per_tok},
                dnnl::memory::data_type::s32);
        }
        if (!graph.route_weights.valid()) {
            graph.route_weights = Tensor::allocate(
                ctx, {cfg_.num_experts_per_tok},
                dnnl::memory::data_type::f32);
        }
        if (!graph.selected.valid()) {
            graph.selected = Tensor::allocate(
                ctx, {cfg_.num_experts_per_tok},
                dnnl::memory::data_type::s32);
        }
        if (!graph.resident_descriptors.valid()) {
            std::vector<ops::MoeExpertSlotDev> descriptors(
                static_cast<size_t>(adaptive_cache_.NumGpuSlots()));
            for (int slot = 0;
                 slot < adaptive_cache_.NumGpuSlots(); ++slot) {
                const auto entry = adaptive_cache_.GetResidentEntry(slot);
                if (entry.layer < 0 || entry.expert_id < 0) continue;
                const auto& nested =
                    host_arena_.GetLayerNestedMetadata(entry.layer);
                descriptors[static_cast<size_t>(slot)] =
                    adaptive_cache_.GetResidentSlotDev(slot, &nested);
            }
            graph.resident_descriptors = Tensor::allocate(
                ctx, {static_cast<int64_t>(
                         descriptors.size() *
                         sizeof(ops::MoeExpertSlotDev))},
                dnnl::memory::data_type::u8);
            ctx.memcpy_h2d(
                graph.resident_descriptors.data(), descriptors.data(),
                descriptors.size() * sizeof(ops::MoeExpertSlotDev));
        }
        if (!graph.service.allocated()) {
            graph.service.allocate(ctx.queue().get_context());
        }
        graph.service.reset();
        ctx.queue().memset(
            graph.device_epoch.data(), 0,
            graph.device_epoch.size_bytes());
        graph.control_host.fill(0);
        ctx.memcpy_h2d(
            graph.control.data(), graph.control_host.data(),
            graph.control.size_bytes());
        const auto key = MakeTokenGraphKey(
            session, aila::TokenGraphBody::WholeToken,
            aila::TokenGraphAttention::QsaSparse512BlockNative,
            whole_token_schema_flags());
        if (!BeginCapture(session, aila::TokenGraphBody::WholeToken,
                          aila::TokenGraphAttention::QsaSparse512BlockNative,
                          whole_token_schema_flags())) {
            throw std::runtime_error(
                "Qwen4Exp whole-token graph failed to enter capture");
        }
        try {
            ex::command_graph<ex::graph_state::modifiable> recording(
                ctx.queue().get_context(), ctx.queue().get_device());
            recording.begin_recording(ctx.queue());
            std::size_t nodes = 0;
            try {
                nodes = RecordWholeTokenGraphBody(
                    ctx, session, graph.control.data_as<int>());
                if (nodes == 0) {
                    throw std::runtime_error(
                        "Qwen4Exp whole-token graph recorded no nodes");
                }
                recording.end_recording(ctx.queue());
            } catch (...) {
                try {
                    recording.end_recording(ctx.queue());
                } catch (...) {
                }
                throw;
            }
            graph.executable = std::make_unique<
                WholeTokenGraphState::Executable>(recording.finalize());
            graph.node_count = nodes;
            if (!MarkReady()) {
                throw std::runtime_error(
                    "Qwen4Exp whole-token graph failed to publish executable");
            }
            if (trace) {
                AILA_LOG_INFO(
                    "[Q4E TokenGraph] capture_ready pos=%d nodes=%zu",
                    start_pos, nodes);
            }
            (void)key;
        } catch (...) {
            (void)AbortAndPoisonTokenGraph(session);
            (void)ReleaseWholeTokenGraph(&ctx, &session, false);
            throw;
        }
        if (whole_token_subgraphs_requested()) {
            // Record the same node sequence as fifty command graphs: prologue,
            // one per layer, epilogue. The single whole-token graph above is
            // still built so the existing reuse and phase machinery is
            // unchanged; only the replay path chooses between them.
            graph.subgraphs.clear();
            auto record_piece = [&](auto&& submit) {
                sycl::ext::oneapi::experimental::command_graph<
                    sycl::ext::oneapi::experimental::graph_state::modifiable>
                    rec(ctx.queue().get_context(), ctx.queue().get_device());
                rec.begin_recording(ctx.queue());
                submit();
                rec.end_recording(ctx.queue());
                graph.subgraphs.push_back(std::make_unique<
                    WholeTokenGraphState::Executable>(rec.finalize()));
            };
            record_piece(
                [&] { (void)RecordWholeTokenGraphPrologue(ctx, session); });
            int ordinal = 0;
            for (int layer_index = 0;
                 layer_index < cfg_.num_hidden_layers; ++layer_index) {
                const int layer_ordinal = ordinal;
                record_piece([&] {
                    (void)RecordWholeTokenGraphLayer(
                        ctx, session, layer_index, layer_ordinal,
                        graph.control.data_as<int>());
                });
                if (!layers_[layer_index].is_linear) ++ordinal;
            }
            record_piece(
                [&] { (void)RecordWholeTokenGraphEpilogue(ctx, session); });
            if (trace) {
                AILA_LOG_INFO(
                    "[Q4E TokenGraph] subgraph_ready pos=%d graphs=%zu",
                    start_pos, graph.subgraphs.size());
            }
        }
    } else {
        prepare_ple();
    }

    graph.control_host.fill(0);
    graph.control_host[ops::graph_control::start_pos] = start_pos;
    graph.control_host[ops::graph_control::abort] = 0;
    int full_ordinal = 0;
    for (int layer_index = 0; layer_index < cfg_.num_hidden_layers;
         ++layer_index) {
        if (!layers_[layer_index].is_linear) {
            graph.control_host[
                ops::graph_control_layer_blocks_slot(full_ordinal)] =
                session.layerCaches_[layer_index].num_compressed_blocks;
            ++full_ordinal;
        }
    }
    (void)ctx.memcpy_h2d_async(
        graph.control.data(), graph.control_host.data(),
        graph.control.size_bytes());
    const auto replay_begin = std::chrono::steady_clock::now();
    const uint32_t token_sequence = token_graph_runtime_.next_sequence();
    const bool unrolled = whole_token_unrolled_requested() ||
                          whole_token_node_prof_requested();
    sycl::event event;
    uint32_t done_sequence = 0;
    bool service_ok = true;
    std::thread service_thread;
    try {
        if (unrolled) {
            // The un-captured body is submitted node by node, which costs the
            // host tens of milliseconds, while the device stalls at layer 0's
            // packet wait until the host starts servicing. Service
            // concurrently with the submission pass instead of after it.
            // `event` is consumed only by the service's final wait, and the
            // in-order memcpy_d2h that follows re-establishes the ordering, so
            // an event that is not assigned yet cannot be observed early.
            service_thread = std::thread([&] {
                try {
                    service_ok = ServiceWholeTokenGraphReplay(
                        ctx, session, event, token_sequence, done_sequence);
                } catch (...) {
                    service_ok = false;
                }
            });
            auto& node_profiler = whole_token_node_profiler();
            if (whole_token_node_prof_requested() &&
                !node_profiler.enabled) {
                node_profiler.queue = &ctx.queue();
                node_profiler.enabled = true;
            }
            node_profiler.drain_and_log("prev");
            const auto submit_begin = std::chrono::steady_clock::now();
            const std::size_t submitted_nodes = RecordWholeTokenGraphBody(
                ctx, session, graph.control.data_as<int>());
            event = ctx.queue().ext_oneapi_submit_barrier();
            const double submit_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - submit_begin)
                    .count();
            if (whole_token_graph_trace_requested()) {
                AILA_LOG_INFO(
                    "[Q4E TokenGraph] submit pos=%d nodes=%zu "
                    "submit_ms=%.3f",
                    start_pos, submitted_nodes, submit_ms);
            }
        } else if (!graph.subgraphs.empty()) {
            // One ext_oneapi_graph per layer is one WDDM submission per layer,
            // so a layer's doorbell reaches the device as soon as that graph is
            // submitted rather than at a batch boundary. Inside each graph the
            // layer's nodes run back to back from one closed command list.
            for (auto& sub : graph.subgraphs) {
                event = ctx.queue().ext_oneapi_graph(*sub);
            }
        } else {
            event = ctx.queue().ext_oneapi_graph(*graph.executable);
        }
        if (!Launch(session, token_sequence, event)) {
            throw std::runtime_error(
                "Qwen4Exp whole-token graph launch was rejected");
        }
        const auto status = token_graph_runtime_.event_status();
        if (status == aila::TokenGraphEventStatus::NoEvent ||
            status == aila::TokenGraphEventStatus::Failed) {
            throw std::runtime_error(
                "Qwen4Exp whole-token graph event status failed after launch");
        }
        if (unrolled) {
            if (service_thread.joinable()) {
                service_thread.join();
            }
            if (!service_ok) {
                graph.service.abort();
                try {
                    event.wait();
                } catch (...) {
                }
                throw std::runtime_error(
                    "Qwen4Exp whole-token host service failed");
            }
        } else if (!ServiceWholeTokenGraphReplay(
                       ctx, session, event, token_sequence, done_sequence)) {
            graph.service.abort();
            try {
                event.wait();
            } catch (...) {
            }
            throw std::runtime_error(
                "Qwen4Exp whole-token host service failed");
        }
        ctx.memcpy_d2h(
            graph.control_host.data(), graph.control.data(),
            graph.control.size_bytes());
        aila::TokenGraphDeviceEpoch epoch_host{};
        ctx.memcpy_d2h(
            &epoch_host, graph.device_epoch.data(),
            graph.device_epoch.size_bytes());
        const int target_blocks = (start_pos + 1) / 4;
        bool gate_valid =
            graph.control_host[ops::graph_control::abort] == 0 &&
            graph.control_host[ops::graph_control::start_pos] == start_pos &&
            !epoch_host.sticky_abort &&
            epoch_host.gate_state ==
                static_cast<uint32_t>(
                    aila::TokenGraphGateState::Valid);
        full_ordinal = 0;
        for (int layer_index = 0;
             gate_valid && layer_index < cfg_.num_hidden_layers;
             ++layer_index) {
            if (layers_[layer_index].is_linear) continue;
            gate_valid =
                graph.control_host[
                    ops::graph_control_layer_blocks_slot(full_ordinal)] ==
                target_blocks;
            ++full_ordinal;
        }
        if (!Complete(token_sequence, gate_valid)) {
            throw std::runtime_error(
                "Qwen4Exp whole-token graph completion gate rejected");
        }
        if (trace) {
            const double replay_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - replay_begin).count();
            AILA_LOG_INFO(
                "[Q4E TokenGraph] replay pos=%d gate=%d sequence=%u "
                "service_done=%u replay_ms=%.3f",
                start_pos, gate_valid ? 1 : 0, token_sequence,
                done_sequence, replay_ms);
        }
        full_ordinal = 0;
        for (int layer_index = 0; layer_index < cfg_.num_hidden_layers;
             ++layer_index) {
            if (layers_[layer_index].is_linear) continue;
            session.layerCaches_[layer_index].num_compressed_blocks =
                graph.control_host[
                    ops::graph_control_layer_blocks_slot(full_ordinal)];
            ++full_ordinal;
        }
        ple_.commit(ple_ticket_, session.ple_state_);
        session.currentLength_ = cached_len;

        if (options.snapshot_policy == SnapshotPolicy::ReusablePrefix) {
            CaptureSnapshot(ctx, session, cached_len, count,
                            options.capture_snapshot,
                            /*preserve_token_graph=*/true);
        }

        if (options.need_logits) {
            RunHyperConnection(ctx, mixer_, 1, false);
            if (lmHeadQ8Enabled_) {
                ops::bf16_q8_head_forward(
                    ctx, lm_head_q8_weight_, lm_head_q8_scale_, buf_.x,
                    buf_.logits, lm_head_q8_input_,
                    lm_head_q8_input_scale_, 1, hidden_size_,
                    cfg_.vocab_size);
                ++lmHeadQ8ForwardCount_;
                ++lmHeadQ8RowCount_;
            } else {
                const bf16* last_x =
                    buf_.x.data_as<bf16>() +
                    static_cast<size_t>(count - 1) * hidden_size_;
                Tensor x_last = Tensor::view(
                    ctx, const_cast<bf16*>(last_x),
                    {1, hidden_size_});
                lm_head_.forward(ctx, x_last, buf_.logits, 1);
            }
            lastTokenLogits_ = Tensor::view(
                ctx, buf_.logits.data(),
                {1, static_cast<int64_t>(cfg_.vocab_size)},
                buf_.logits.dtype());
        }
        return buf_.logits;
    } catch (...) {
        (void)AbortAndPoisonTokenGraph(session);
        (void)ReleaseWholeTokenGraph(&ctx, &session, false);
        throw;
    }
}

bool Qwen4ExpBnb4Backend::ReleaseWholeTokenGraph(
    Context* ctx, Session* session, bool idle_confirmed) noexcept {
    auto& graph = whole_token_graph_;
    if (!graph.executable && !graph.ple_view_valid &&
        !graph.service.allocated()) {
        return true;
    }

    bool safe = idle_confirmed;
    if (!safe && ctx) {
        graph.service.abort();
        (void)token_graph_runtime_.Abort();
        safe = ctx->drain_noexcept();
        if (safe) {
            safe = token_graph_runtime_.wait_drained_noexcept();
        }
    }
    if (!safe) return false;

    graph.executable.reset();
    graph.subgraphs.clear();
    graph.node_count = 0;
    if (graph.ple_view_valid && ctx && session) {
        try {
            ple_.detach_graph_view(*ctx, session->ple_state_);
        } catch (...) {
            return false;
        }
    }
    graph.ple_view_valid = false;
    graph.ple_view = {};
    if (graph.service.allocated()) graph.service.reset();
    return true;
}
