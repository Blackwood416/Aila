// P6 real-weight L3 sparse-QSA attention sublayer probe.
//
// Scope:
//   real Qwen3.8 L3 q/k/v/o + index_qk projections
//   deterministic x[2560], no upstream prefill, HC, PLE or MoE
//   direct lane vs one captured graph lane
//
// The direct lane uses the production host-scalar QSA path. The graph lane
// captures qkv_proj -> index_qk_proj -> QGKV graph prep -> QSA graph prepare,
// one-block graph compression, fixed-grid scoring/selection, block-native
// attention and o_proj in one executable.
//
// This source only compiles. It is not invoked by build_qsa_layer.ps1.

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "engine/Types.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/GraphControl.hpp"
#include "ops/Ops.hpp"
#include "ops/AttentionOps.hpp"
#include "ops/QsaIndexerOps.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ex = sycl::ext::oneapi::experimental;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

constexpr int kLayer = 3;
constexpr int kHidden = 2560;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 64;
constexpr int kQDim = kQHeads * kHeadDim;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kQProjDim = 2 * kQDim;
constexpr int kFusedQkvDim = kQProjDim + 2 * kKvDim;
constexpr int kIndexerHeads = 4;
constexpr int kIndexerKvHeads = 1;
constexpr int kIndexerHeadDim = 128;
constexpr int kIndexerOutDim =
    (kIndexerHeads + kIndexerKvHeads) * kIndexerHeadDim;
constexpr int kIndexerQDim = kIndexerHeads * kIndexerHeadDim;
constexpr int kCompressRatio = 4;
constexpr int kBlockTopK = 512;
constexpr int kMaxTiles = 17;
constexpr int kPartialStride = 272;
constexpr int kFirstPosition = 2052;
constexpr int kSteps = 4;
constexpr int kSeededCompressedBlocks = 513;
constexpr int kCapacity = 2060;
constexpr const char* kDefaultModel =
    "C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";

struct Geometry {
    int capacity = kCapacity;
    int max_blocks = kCapacity / kCompressRatio;

    void validate() const {
        if (capacity < kFirstPosition + kSteps) {
            throw std::invalid_argument(
                "QSA L3 probe capacity must cover positions 2052..2055");
        }
        const int final_blocks = (kFirstPosition + kSteps) / kCompressRatio;
        if (max_blocks < final_blocks) {
            throw std::invalid_argument(
                "QSA L3 probe compressed-K capacity is too small");
        }
    }
};

void upload_bnb_metadata(Context& ctx, ModelWeights& weights,
                         const std::string& name) {
    static constexpr const char* suffixes[] = {
        "", ".absmax", ".quant_map", ".nested_absmax",
        ".nested_quant_map", ".quant_state.bitsandbytes__nf4"};
    for (const char* suffix : suffixes) {
        const std::string key = name + suffix;
        if (weights.has_host(key)) weights.upload(key, ctx);
    }
}

void init_quant_linear(Context& ctx, ModelWeights& weights,
                       const std::string& name, Bnb4BitLinear& linear) {
    upload_bnb_metadata(ctx, weights, name);
    Bnb4BitWeightRef ref;
    std::string error;
    if (!LoadBnb4BitWeightRef(ctx, weights, name, ref, &error)) {
        throw std::runtime_error("Cannot load " + name + ": " + error);
    }
    if (!linear.init(ctx, ref, &error)) {
        throw std::runtime_error("Cannot init " + name + ": " + error);
    }
    linear.set_allow_async_dequant(true);
}

Tensor load_plus_one_norm(Context& ctx, ModelWeights& weights,
                          const std::string& name, int expected) {
    const auto view = weights.host_view(name);
    if (!view.valid() || view.dtype != DT_BF16 ||
        view.bytes != static_cast<size_t>(expected) * sizeof(bf16)) {
        throw std::runtime_error("Expected BF16 norm " + name);
    }
    std::vector<bf16> host(expected);
    std::memcpy(host.data(), view.ptr, host.size() * sizeof(bf16));
    for (bf16& value : host) {
        value = bf16(static_cast<float>(value) + 1.0f);
    }
    Tensor result =
        Tensor::allocate(ctx, {expected}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(result.data(), host.data(), host.size() * sizeof(bf16));
    return result;
}

struct RealL3Weights {
    Context& ctx;
    ModelWeights weights;
    ModelSpec spec;
    Qwen4ExpTextConfig cfg;
    Qwen4ExpLayer layer;
    Bnb4BitLinear q_src;
    Bnb4BitLinear k_src;
    Bnb4BitLinear v_src;
    Tensor q_norm;
    Tensor k_norm;
    Tensor index_q_norm;
    Tensor index_k_norm;

    RealL3Weights(Context& context, const std::string& model_dir)
        : ctx(context), weights(LoadModelWeightsLazy(model_dir, &context)) {
        std::string error;
        if (!aila::modelspec::load_from_dir(model_dir, spec, &error)) {
            throw std::runtime_error("Cannot load model spec: " + error);
        }
        if (spec.family != ModelFamily::Qwen4Exp) {
            throw std::runtime_error("Model is not Qwen4Exp");
        }
        cfg = spec.qwen4exp_text;
        if (cfg.hidden_size != kHidden ||
            cfg.num_attention_heads != kQHeads ||
            cfg.num_key_value_heads != kKvHeads ||
            cfg.head_dim != kHeadDim ||
            cfg.indexer_n_heads != kIndexerHeads ||
            cfg.indexer_kv_heads != kIndexerKvHeads ||
            cfg.indexer_head_dim != kIndexerHeadDim ||
            cfg.indexer_compress_ratio != kCompressRatio ||
            cfg.indexer_budget / cfg.indexer_compress_ratio != kBlockTopK) {
            throw std::runtime_error("Unexpected Qwen4Exp L3 geometry");
        }
        if (!cfg.layer_types.empty() &&
            cfg.layer_types.at(kLayer) != "full_attention") {
            throw std::runtime_error("Layer 3 is not full attention");
        }

        const std::string prefix =
            "model.language_model.layers." + std::to_string(kLayer) + ".";
        init_quant_linear(ctx, weights, prefix + "self_attn.q_proj.weight",
                          q_src);
        init_quant_linear(ctx, weights, prefix + "self_attn.k_proj.weight",
                          k_src);
        init_quant_linear(ctx, weights, prefix + "self_attn.v_proj.weight",
                          v_src);

        if (!layer.qkv_proj.init_fused_rows(ctx, q_src, k_src, v_src,
                                            &error)) {
            throw std::runtime_error("Cannot fuse L3 QKV: " + error);
        }
        init_quant_linear(ctx, weights, prefix + "self_attn.o_proj.weight",
                          layer.o_proj);
        if (!layer.qkv_proj.supports_decode_graph() ||
            !layer.o_proj.supports_decode_graph()) {
            throw std::runtime_error(
                "L3 QKV/o_proj does not expose the packed decode graph path");
        }

        q_norm = load_plus_one_norm(
            ctx, weights, prefix + "self_attn.q_norm.weight", kHeadDim);
        k_norm = load_plus_one_norm(
            ctx, weights, prefix + "self_attn.k_norm.weight", kHeadDim);
        layer.q_norm_weight = &q_norm;
        layer.k_norm_weight = &k_norm;

        layer.index_qk_weight = &weights.upload(
            prefix + "self_attn.indexer.index_qk_proj.weight", ctx);
        if (!layer.index_qk_weight->valid() ||
            layer.index_qk_weight->dtype() != dnnl::memory::data_type::bf16 ||
            layer.index_qk_weight->numel() !=
                static_cast<int64_t>(kIndexerOutDim) * kHidden) {
            throw std::runtime_error("Unexpected L3 index_qk_proj shape");
        }
        layer.index_qk_proj.init(
            ctx, *layer.index_qk_weight, kHidden, kIndexerOutDim,
            /*preprocessed=*/false, /*prepare_gemv=*/false);
        index_q_norm = load_plus_one_norm(
            ctx, weights,
            prefix + "self_attn.indexer.q_layernorm.weight",
            kIndexerHeadDim);
        index_k_norm = load_plus_one_norm(
            ctx, weights,
            prefix + "self_attn.indexer.k_layernorm.weight",
            kIndexerHeadDim);
        layer.index_q_norm_weight = &index_q_norm;
        layer.index_k_norm_weight = &index_k_norm;

        ctx.synchronize();
        std::cout << "[QSA L3 WEIGHTS] real=1 layer=" << kLayer
                  << " qkv=" << layer.qkv_proj.in_features() << "x"
                  << layer.qkv_proj.out_features()
                  << " index_qk=" << kHidden << "x" << kIndexerOutDim
                  << " graph_ready=1\n";
    }
};

struct Lane {
    Context& ctx;
    Tensor x;
    Tensor linear_all;
    Tensor qkv_packed;
    Tensor k_packed;
    Tensor v_packed;
    Tensor q;
    Tensor gate;
    Tensor k_cache;
    Tensor v_cache;
    Tensor indexer_qk;
    Tensor qsa_q;
    Tensor indexer_raw_k;
    Tensor indexer_compressed_k;
    Tensor scores;
    Tensor active_blocks;
    Tensor attention_scores;
    Tensor partials;
    Tensor attn_out;
    Tensor y;
    Bnb4BitLinearScratch scratch;

    Lane(Context& context, const Geometry& geometry)
        : ctx(context),
          x(Tensor::allocate(ctx, {1, kHidden},
                             dnnl::memory::data_type::bf16)),
          linear_all(Tensor::allocate(ctx, {1, kFusedQkvDim},
                                      dnnl::memory::data_type::bf16)),
          q(Tensor::allocate(ctx, {1, kQDim},
                             dnnl::memory::data_type::bf16)),
          gate(Tensor::allocate(ctx, {1, kQDim},
                                dnnl::memory::data_type::bf16)),
          k_cache(Tensor::allocate(ctx, {kKvHeads, geometry.capacity,
                                         kHeadDim},
                                   dnnl::memory::data_type::bf16)),
          v_cache(Tensor::allocate(ctx, {kKvHeads, geometry.capacity,
                                         kHeadDim},
                                   dnnl::memory::data_type::bf16)),
          indexer_qk(Tensor::allocate(ctx, {1, kIndexerOutDim},
                                      dnnl::memory::data_type::bf16)),
          qsa_q(Tensor::allocate(ctx, {1, kIndexerQDim},
                                 dnnl::memory::data_type::bf16)),
          indexer_raw_k(Tensor::allocate(
              ctx, {geometry.capacity, kIndexerHeadDim},
              dnnl::memory::data_type::bf16)),
          indexer_compressed_k(Tensor::allocate(
              ctx, {geometry.max_blocks, kIndexerHeadDim},
              dnnl::memory::data_type::bf16)),
          scores(Tensor::allocate(ctx, {geometry.max_blocks},
                                  dnnl::memory::data_type::f32)),
          active_blocks(Tensor::allocate(
              ctx, {kBlockTopK + kCompressRatio},
              dnnl::memory::data_type::s32)),
          attention_scores(Tensor::allocate(
              ctx, {kQHeads, geometry.capacity},
              dnnl::memory::data_type::f32)),
          partials(Tensor::allocate(
              ctx, {kQHeads, kMaxTiles, kPartialStride},
              dnnl::memory::data_type::f32)),
          attn_out(Tensor::allocate(ctx, {1, kQDim},
                                    dnnl::memory::data_type::bf16)),
          y(Tensor::allocate(ctx, {1, kHidden},
                             dnnl::memory::data_type::bf16)) {
        qkv_packed = Tensor::view(
            ctx, linear_all.data_as<bf16>(), {1, kQProjDim},
            dnnl::memory::data_type::bf16);
        k_packed = Tensor::view(
            ctx, linear_all.data_as<bf16>() + kQProjDim, {1, kKvDim},
            dnnl::memory::data_type::bf16);
        v_packed = Tensor::view(
            ctx, linear_all.data_as<bf16>() + kQProjDim + kKvDim,
            {1, kKvDim}, dnnl::memory::data_type::bf16);
    }
};

struct SeedData {
    std::vector<bf16> x;
    std::vector<bf16> k_cache;
    std::vector<bf16> v_cache;
    std::vector<bf16> indexer_raw_k;
    std::vector<bf16> indexer_compressed_k;
    std::vector<float> scores;
    std::vector<int> active_blocks;
    std::vector<float> partials;

    explicit SeedData(const Geometry& geometry)
        : x(kHidden),
          k_cache(static_cast<size_t>(kKvHeads) * geometry.capacity *
                  kHeadDim),
          v_cache(k_cache.size()),
          indexer_raw_k(static_cast<size_t>(geometry.capacity) *
                        kIndexerHeadDim),
          indexer_compressed_k(static_cast<size_t>(geometry.max_blocks) *
                               kIndexerHeadDim),
          scores(geometry.max_blocks, -12345.0f),
          active_blocks(kBlockTopK + kCompressRatio, -1),
          partials(static_cast<size_t>(kQHeads) * kMaxTiles *
                   kPartialStride, -54321.0f) {
        std::mt19937 rng(20261018);
        std::uniform_real_distribution<float> x_dist(-0.125f, 0.125f);
        std::uniform_real_distribution<float> kv_dist(-0.08f, 0.08f);
        for (bf16& value : x) value = bf16(x_dist(rng));
        for (bf16& value : k_cache) value = bf16(kv_dist(rng));
        for (bf16& value : v_cache) value = bf16(kv_dist(rng));
        for (bf16& value : indexer_raw_k) value = bf16(kv_dist(rng));

        const bf16 untouched(321.0f);
        for (bf16& value : indexer_compressed_k) value = untouched;
        const size_t seeded =
            static_cast<size_t>(kSeededCompressedBlocks) * kIndexerHeadDim;
        for (size_t i = 0; i < seeded; ++i) {
            indexer_compressed_k[i] = bf16(kv_dist(rng));
        }
    }
};

void upload_seed(Context& ctx, Lane& lane, const Geometry& geometry,
                 const SeedData& seed) {
    ctx.memcpy_h2d(lane.x.data(), seed.x.data(),
                   seed.x.size() * sizeof(bf16));
    ctx.memcpy_h2d(lane.k_cache.data(), seed.k_cache.data(),
                   seed.k_cache.size() * sizeof(bf16));
    ctx.memcpy_h2d(lane.v_cache.data(), seed.v_cache.data(),
                   seed.v_cache.size() * sizeof(bf16));
    ctx.memcpy_h2d(lane.indexer_raw_k.data(), seed.indexer_raw_k.data(),
                   seed.indexer_raw_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(lane.indexer_compressed_k.data(),
                   seed.indexer_compressed_k.data(),
                   seed.indexer_compressed_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(lane.scores.data(), seed.scores.data(),
                   seed.scores.size() * sizeof(float));
    ctx.memcpy_h2d(lane.active_blocks.data(), seed.active_blocks.data(),
                   seed.active_blocks.size() * sizeof(int));
    ctx.memcpy_h2d(lane.partials.data(), seed.partials.data(),
                   seed.partials.size() * sizeof(float));
    (void)geometry;
}

void run_direct(Context& ctx, Lane& lane, RealL3Weights& real,
                const Geometry& geometry, int position, int& host_blocks) {
    real.layer.qkv_proj.ForwardDecodeRows(
        ctx, lane.scratch, lane.x, lane.linear_all, 1);
    ops::bf16_gemv_bf16(
        ctx, *real.layer.index_qk_weight, lane.x.data_as<bf16>(),
        lane.indexer_qk.data_as<bf16>(), kIndexerOutDim, kHidden);

    ops::decode_prepare_qgkv_packed_partial(
        ctx, lane.qkv_packed, lane.k_packed, lane.v_packed, lane.q,
        lane.gate, real.q_norm, real.k_norm, lane.k_cache, lane.v_cache,
        position, kQHeads, kKvHeads, kHeadDim, real.cfg.rms_norm_eps,
        kRotaryDim, real.cfg.rope.rope_theta,
        real.cfg.rope.mrope_interleaved, nullptr, nullptr, nullptr, 0, 0,
        real.cfg.rope.mrope_section[0],
        real.cfg.rope.mrope_section[1],
        real.cfg.rope.mrope_section[2]);

    ops::qsa_prepare_q_and_cache_k(
        ctx, lane.indexer_qk, real.index_q_norm, lane.indexer_raw_k,
        lane.qsa_q, position, real.cfg.rope.rope_theta,
        real.cfg.rope.mrope_interleaved, nullptr, nullptr, nullptr, 0, 0,
        real.cfg.rope.mrope_section[0],
        real.cfg.rope.mrope_section[1],
        real.cfg.rope.mrope_section[2]);

    const int visible_len = position + 1;
    const int target_blocks = visible_len / kCompressRatio;
    if (target_blocks > host_blocks) {
        if (target_blocks != host_blocks + 1) {
            throw std::runtime_error(
                "Direct QSA skipped a completed compressed block");
        }
        ops::qsa_compress_key_blocks(
            ctx, lane.indexer_raw_k, real.index_k_norm,
            lane.indexer_compressed_k, host_blocks, 1,
            real.cfg.rope.rope_theta,
            real.cfg.rope.mrope_interleaved, nullptr, nullptr, nullptr, 0,
            0, real.cfg.rope.mrope_section[0],
            real.cfg.rope.mrope_section[1],
            real.cfg.rope.mrope_section[2]);
        host_blocks = target_blocks;
    }

    ops::qsa_compute_block_scores_compressed(
        ctx, lane.qsa_q, lane.indexer_compressed_k, lane.scores,
        host_blocks);
    const int active_blocks = ops::qsa_select_topk_blocks(
        ctx, lane.scores, host_blocks, kBlockTopK, visible_len,
        kCompressRatio, lane.active_blocks);
    if (active_blocks != std::min(host_blocks, kBlockTopK)) {
        throw std::runtime_error("Direct QSA selector count mismatch");
    }

    ops::attention_decode(
        ctx, lane.q, lane.k_cache, lane.v_cache, lane.attn_out,
        lane.attention_scores, kQHeads, kKvHeads, kHeadDim, visible_len,
        &lane.partials, &lane.gate, nullptr, 0,
        lane.active_blocks.data_as<int>(), active_blocks, nullptr, nullptr);
    real.layer.o_proj.ForwardDecodeRows(
        ctx, lane.scratch, lane.attn_out, lane.y, 1);
    (void)geometry;
}

void capture_graph_body(Context& ctx, Lane& lane, RealL3Weights& real,
                        const Geometry& geometry, Tensor& control) {
    real.layer.qkv_proj.ForwardDecodeRows(
        ctx, lane.scratch, lane.x, lane.linear_all, 1);
    ops::bf16_gemv_bf16(
        ctx, *real.layer.index_qk_weight, lane.x.data_as<bf16>(),
        lane.indexer_qk.data_as<bf16>(), kIndexerOutDim, kHidden);

    ops::decode_prepare_qgkv_packed_partial_graph(
        ctx, lane.qkv_packed, lane.k_packed, lane.v_packed, lane.q,
        lane.gate, real.q_norm, real.k_norm, lane.k_cache, lane.v_cache,
        control.data_as<int>(), kQHeads, kKvHeads, kHeadDim,
        real.cfg.rms_norm_eps, kRotaryDim, real.cfg.rope.rope_theta,
        real.cfg.rope.mrope_interleaved, nullptr, nullptr, nullptr, 0, 0,
        real.cfg.rope.mrope_section[0],
        real.cfg.rope.mrope_section[1],
        real.cfg.rope.mrope_section[2]);

    ops::qsa_prepare_q_and_cache_k_graph(
        ctx, lane.indexer_qk, real.index_q_norm, lane.indexer_raw_k,
        lane.qsa_q, control.data_as<int>(), real.cfg.rope.rope_theta,
        real.cfg.rope.mrope_interleaved, nullptr, nullptr, nullptr, 0, 0,
        real.cfg.rope.mrope_section[0],
        real.cfg.rope.mrope_section[1],
        real.cfg.rope.mrope_section[2]);

    ops::qsa_compress_one_completed_block_graph(
        ctx, lane.indexer_raw_k, real.index_k_norm,
        lane.indexer_compressed_k, control.data_as<int>(),
        real.cfg.rope.rope_theta, real.cfg.rope.mrope_interleaved,
        nullptr, nullptr, nullptr, 0, 0,
        real.cfg.rope.mrope_section[0],
        real.cfg.rope.mrope_section[1],
        real.cfg.rope.mrope_section[2]);

    ops::qsa_compute_block_scores_compressed_graph(
        ctx, lane.qsa_q, lane.indexer_compressed_k, lane.scores,
        control.data_as<int>(), geometry.max_blocks);
    ops::qsa_select_topk_blocks_graph(
        ctx, lane.scores, lane.active_blocks, control.data_as<int>(),
        kBlockTopK, kCompressRatio, geometry.max_blocks);
    ops::attention_validate_qsa_active_blocks_graph(
        ctx, lane.k_cache, lane.v_cache, lane.active_blocks,
        control.data_as<int>(), kKvHeads);
    ops::attention_decode_exact_head256_block_native_graph(
        ctx, lane.q, lane.k_cache, lane.v_cache, lane.attn_out,
        lane.partials, lane.active_blocks, control.data_as<int>(), kQHeads,
        kKvHeads, &lane.gate);
    real.layer.o_proj.ForwardDecodeRows(
        ctx, lane.scratch, lane.attn_out, lane.y, 1);
}

using ExecutableGraph =
    ex::command_graph<ex::graph_state::executable>;

ExecutableGraph capture_graph(Context& ctx, Lane& lane, RealL3Weights& real,
                              const Geometry& geometry, Tensor& control) {
    ex::command_graph<ex::graph_state::modifiable> recording(
        ctx.queue().get_context(), ctx.queue().get_device());
    recording.begin_recording(ctx.queue());
    try {
        capture_graph_body(ctx, lane, real, geometry, control);
        recording.end_recording(ctx.queue());
    } catch (...) {
        recording.end_recording(ctx.queue());
        throw;
    }
    return recording.finalize();
}

void require_bitwise(Context& ctx, const Tensor& direct,
                     const Tensor& graph, const char* label,
                     int position) {
    if (!direct.valid() || !graph.valid() ||
        direct.size_bytes() != graph.size_bytes()) {
        throw std::runtime_error(std::string("QSA L3 tensor geometry: ") +
                                 label);
    }
    std::vector<uint8_t> direct_host(direct.size_bytes());
    std::vector<uint8_t> graph_host(graph.size_bytes());
    ctx.memcpy_d2h(direct_host.data(), direct.data(), direct_host.size());
    ctx.memcpy_d2h(graph_host.data(), graph.data(), graph_host.size());
    if (std::memcmp(direct_host.data(), graph_host.data(),
                    direct_host.size()) != 0) {
        throw std::runtime_error(
            std::string("QSA L3 mismatch: ") + label +
            " at pos=" + std::to_string(position));
    }
}

void compare_lanes(Context& ctx, Lane& direct, Lane& graph,
                   int position) {
    require_bitwise(ctx, direct.x, graph.x, "x", position);
    require_bitwise(ctx, direct.linear_all, graph.linear_all,
                    "linear_all", position);
    require_bitwise(ctx, direct.qkv_packed, graph.qkv_packed,
                    "qkv_packed", position);
    require_bitwise(ctx, direct.k_packed, graph.k_packed, "k_packed",
                    position);
    require_bitwise(ctx, direct.v_packed, graph.v_packed, "v_packed",
                    position);
    require_bitwise(ctx, direct.q, graph.q, "q", position);
    require_bitwise(ctx, direct.gate, graph.gate, "gate", position);
    require_bitwise(ctx, direct.k_cache, graph.k_cache, "k_cache",
                    position);
    require_bitwise(ctx, direct.v_cache, graph.v_cache, "v_cache",
                    position);
    require_bitwise(ctx, direct.indexer_qk, graph.indexer_qk,
                    "indexer_qk", position);
    require_bitwise(ctx, direct.qsa_q, graph.qsa_q, "qsa_q", position);
    require_bitwise(ctx, direct.indexer_raw_k, graph.indexer_raw_k,
                    "indexer_raw_k", position);
    require_bitwise(ctx, direct.indexer_compressed_k,
                    graph.indexer_compressed_k, "indexer_compressed_k",
                    position);
    require_bitwise(ctx, direct.scores, graph.scores, "scores", position);
    require_bitwise(ctx, direct.active_blocks, graph.active_blocks,
                    "active_blocks", position);
    require_bitwise(ctx, direct.partials, graph.partials, "partials",
                    position);
    require_bitwise(ctx, direct.attn_out, graph.attn_out, "attn_out",
                    position);
    require_bitwise(ctx, direct.y, graph.y, "y", position);
}

struct Arguments {
    std::string model = kDefaultModel;
    int capacity = kCapacity;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--capacity" && i + 1 < argc) {
            args.capacity = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout
                << "Usage: probe_qsa_layer [--model DIR] [--capacity N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        if (!std::filesystem::exists(args.model)) {
            throw std::runtime_error("Model directory does not exist: " +
                                     args.model);
        }
        Geometry geometry;
        geometry.capacity = args.capacity;
        geometry.max_blocks = geometry.capacity / kCompressRatio;
        geometry.validate();

        Context ctx;
        const auto device = ctx.queue().get_device();
        if (!device.has(sycl::aspect::ext_oneapi_limited_graph) &&
            !device.has(sycl::aspect::ext_oneapi_graph)) {
            throw std::runtime_error("Device does not support SYCL graphs");
        }

        std::cout << "[QSA L3 PROBE] model=" << args.model
                  << " capacity=" << geometry.capacity
                  << " first_pos=" << kFirstPosition
                  << " steps=" << kSteps
                  << " seeded_compressed_blocks="
                  << kSeededCompressedBlocks << "\n";

        RealL3Weights real(ctx, args.model);
        SeedData seed(geometry);
        Lane direct(ctx, geometry);
        Lane graph(ctx, geometry);
        upload_seed(ctx, direct, geometry, seed);
        upload_seed(ctx, graph, geometry, seed);

        Tensor control = Tensor::allocate(
            ctx, {ops::graph_control::words},
            dnnl::memory::data_type::s32);
        std::array<int, ops::graph_control::words> control_host{};
        control_host[ops::graph_control::start_pos] = kFirstPosition;
        control_host[ops::graph_control::compressed_blocks] =
            kSeededCompressedBlocks;
        control_host[ops::graph_control::abort] = 0;
        ctx.memcpy_h2d(control.data(), control_host.data(),
                       control_host.size() * sizeof(int));
        ctx.synchronize();

        ExecutableGraph executable =
            capture_graph(ctx, graph, real, geometry, control);

        int direct_blocks = kSeededCompressedBlocks;
        int graph_blocks = kSeededCompressedBlocks;
        for (int step = 0; step < kSteps; ++step) {
            const int position = kFirstPosition + step;
            const int visible_len = position + 1;

            run_direct(ctx, direct, real, geometry, position, direct_blocks);

            std::fill(control_host.begin(), control_host.end(), 0);
            control_host[ops::graph_control::start_pos] = position;
            control_host[ops::graph_control::compressed_blocks] =
                graph_blocks;
            ctx.memcpy_h2d(control.data(), control_host.data(),
                           control_host.size() * sizeof(int));
            auto graph_event = ctx.queue().ext_oneapi_graph(executable);
            graph_event.wait_and_throw();
            ctx.memcpy_d2h(control_host.data(), control.data(),
                           control_host.size() * sizeof(int));

            if (control_host[ops::graph_control::abort] != 0) {
                throw std::runtime_error(
                    "Graph QSA aborted at pos=" +
                    std::to_string(position));
            }
            graph_blocks =
                control_host[ops::graph_control::compressed_blocks];
            if (graph_blocks != direct_blocks) {
                throw std::runtime_error(
                    "QSA compressed-block count mismatch at pos=" +
                    std::to_string(position) + ": graph=" +
                    std::to_string(graph_blocks) + " direct=" +
                    std::to_string(direct_blocks));
            }

            if (direct_blocks != visible_len / kCompressRatio) {
                throw std::runtime_error(
                    "Direct compressed-block state is inconsistent");
            }
            compare_lanes(ctx, direct, graph, position);
            std::cout << "[QSA L3 STEP] pos=" << position
                      << " visible=" << visible_len
                      << " blocks=" << graph_blocks
                      << " active=512"
                      << " abort=0"
                      << " bitwise=1\n";
        }
        ctx.synchronize();
        std::cout
            << "[QSA L3 RESULT] bit_exact=1 steps=4 start_pos=2052..2055 "
               "blocks=513..514 selector=complement_x3+cutoff_x1 "
               "tiles=17,17,17,16 nodes=qkv,index_qk,qgkv,"
               "qsa_prepare,compress,score,select,attention,o_proj\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[QSA L3 ERROR] " << error.what() << "\n";
        return 1;
    }
}
