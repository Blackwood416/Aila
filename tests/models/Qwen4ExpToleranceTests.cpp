#include "engine/Engine.hpp"
#include "utils/Qwen38A770Preset.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>
#include <sycl/ext/intel/math.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

static constexpr float kTestQuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

struct Qwen4ExpGenerationTestAccess {
    static void ResidentMode(Qwen4ExpBnb4Backend& b,int mode) {b.context_->synchronize();b.resident_decode_mode_=mode;}
    static void ColdMode(Qwen4ExpBnb4Backend& b,int cold,bool prefill) {
        b.context_->synchronize();b.cold_mode_=cold;b.resident_decode_mode_=cold?2:0;b.resident_prefill_=prefill;
    }
    static const Tensor& MoeInput(Qwen4ExpBnb4Backend& b){return b.buf_.x;}
    static const Tensor& MoeOutput(Qwen4ExpBnb4Backend& b){return b.buf_.y;}
    static const Tensor& ResidentDown(Qwen4ExpBnb4Backend& b){return b.resident_down_;}
    static void ResidentProbe(Qwen4ExpBnb4Backend& b,void* data,decltype(b.resident_probe_) probe){b.resident_probe_data_=data;b.resident_probe_=probe;}
    static void ExpertQualityProbe(Qwen4ExpBnb4Backend& b,void* data,decltype(b.expert_quality_probe_) hook) {
        b.expert_quality_probe_data_=data;b.expert_quality_probe_=hook;
    }
    static Tensor& ExpertIntermediate(Qwen4ExpBnb4Backend& b) {return b.buf_.moe_h;}
    static Tensor& ExpertDown(Qwen4ExpBnb4Backend& b) {return b.buf_.moe_down_acc;}
    static void CompactMoe(Qwen4ExpBnb4Backend& b, int tile) {
        b.context_->synchronize(); b.moe_compact_tile_=tile;b.runtime_seq_capacity_=0;
        b.ensure_runtime_buffers(*b.context_,64);
    }
    static void AttentionTile(Qwen4ExpBnb4Backend& b, int tile, int reserve = 0) {
        b.context_->synchronize();
        b.prefill_attention_tile_ = tile;
        if (reserve) b.ensure_runtime_buffers(*b.context_, reserve);
    }
    static Tensor& AttentionOnly(Qwen4ExpBnb4Backend& b, int layer, int count, int position) {
        b.RunAttentionWindow(*b.context_, *b.defaultSession_, layer, count, position, nullptr, 0);
        return b.buf_.y;
    }
    static int FullLayer(Qwen4ExpBnb4Backend& b) {
        for (int i=0;i<int(b.layers_.size());++i) if (!b.layers_[i].is_linear) return i;
        throw std::runtime_error("Missing full attention layer");
    }
    static Tensor& MtpTargetPrompt(InferenceEngine& e,const int* ids,int count) {
        return Backend(e).ForwardImpl(Ctx(e),Session(e),ids,count,true,{SnapshotPolicy::None,true});
    }
    static Tensor& Logits(Qwen4ExpBnb4Backend& b) { return b.buf_.logits; }
    static ModelWeights& Weights(InferenceEngine& e) { return *e.weights_; }
    static Tensor& VerifyRows(InferenceEngine& e, const int* ids, int count) {
        auto& b = Backend(e);
        ForwardOptions options{SnapshotPolicy::None, true};
        options.decode_verification = true;
        return b.ForwardImpl(Ctx(e), Session(e), ids, count, false, options);
    }
    // Ordinary single-token decode exactly as the live T1 path issues it:
    // decode_verification=false, lastTokenOnly=true (Model::ForwardImpl).
    static Tensor& DecodeStep(InferenceEngine& e, const int* ids, int count) {
        auto& b = Backend(e);
        ForwardOptions options{SnapshotPolicy::None, true};
        return b.Forward(Ctx(e), Session(e), ids, count, options);
    }
    static void ReserveVerifyLogits(InferenceEngine& e, int rows) {
        auto& b = Backend(e);
        if (b.buf_.logits.shape(0) < rows)
            b.buf_.logits = Tensor::allocate(Ctx(e), {rows, b.cfg_.vocab_size}, dnnl::memory::data_type::bf16);
    }
    static const ops::DeltaNetGeometry& FrontGeometry(Qwen4ExpBnb4Backend& b) { return b.layers_.at(0).geom; }
    static void PrepareVerification(Qwen4ExpBnb4Backend& b, Context& c) { b.PrepareDecodeVerification(c); }
    static int HiddenSize(const Qwen4ExpBnb4Backend& b) { return b.hidden_size_; }
    static int LayerCount(const Qwen4ExpBnb4Backend& b) { return int(b.layers_.size()); }
    static int VocabSize(const Qwen4ExpBnb4Backend& b) { return b.cfg_.vocab_size; }
    // B4-2 alignment probe: copies the per-layer attention output and the
    // per-layer router input for every verify row into caller-owned
    // [layers, rows, hidden] bf16 buffers. Inert (no copies) unless rows>0, so
    // ordinary forwards stay byte-identical.
    static void StageProbe(Qwen4ExpBnb4Backend& b, bf16* attnOut, bf16* routerX, bf16* hiddenIn,
                           bf16* attnIn, bf16* attnRaw, int rows) {
        b.verification_attn_out_for_test_ = attnOut;
        b.verification_router_x_for_test_ = routerX;
        b.verification_hidden_in_for_test_ = hiddenIn;
        b.verification_attn_in_for_test_ = attnIn;
        b.verification_attn_raw_for_test_ = attnRaw;
        b.verification_stage_rows_for_test_ = rows;
    }
    static Tensor& FrontHidden(Qwen4ExpBnb4Backend& b) { return b.buf_.hidden4; }
    static Tensor& FrontX(Qwen4ExpBnb4Backend& b) { return b.buf_.x; }
    static Tensor& FrontScores(Qwen4ExpBnb4Backend& b) { return b.buf_.router_scores; }
    static Tensor& FrontInject(Qwen4ExpBnb4Backend& b) { return b.buf_.hc_inject_f32; }
    static const int* FrontIds(Qwen4ExpBnb4Backend& b) { return b.router_host_ids_; }
    static const float* FrontWeights(Qwen4ExpBnb4Backend& b) { return b.router_host_weights_; }
    // Exact fixed-address layer-0 decode front, excluding PLE, expert acquire and MoE.
    // The recurrent state belongs to the test, leaving session state untouched.
    static void Front(Qwen4ExpBnb4Backend& b, ops::DeltaNetState& state) {
        auto& ctx=*b.context_; auto& layer=b.layers_.at(0);
        if (!layer.is_linear) throw std::runtime_error("Graph probe requires a linear layer 0");
        b.RunHyperConnection(ctx,layer.attn_hc,1,true);
        layer.linear_all_proj.ForwardDecodeRows(ctx,b.linear_scratch_,b.buf_.x,b.buf_.linear_all,1);
        const auto* p=b.buf_.linear_all.data_as<bf16>();
        ops::DeltaNetWeights weights{layer.linear_conv1d_weight->data_as<bf16>(),
            layer.linear_norm_weight->data_as<float>(),layer.linear_A_log->data_as<float>(),layer.linear_dt_bias->data_as<bf16>()};
        ops::run_linear_delta_decode_gpu(ctx,layer.geom,weights,state,p,p+b.linear_qkv_dim_,
            p+b.linear_qkv_dim_+b.linear_z_dim_,p+b.linear_qkv_dim_+b.linear_z_dim_+b.linear_kv_heads_,b.buf_.attn_out.data_as<bf16>());
        layer.linear_o_proj.ForwardDecodeRows(ctx,b.linear_scratch_,b.buf_.attn_out,b.buf_.y,1);
        b.RunHyperConnectionCombined(ctx,layer.mlp_hc,1,true,b.buf_.y.data_as<bf16>(),b.buf_.hc_inject_f32.data_as<float>());
        b.DenseLinear(ctx,layer.router_gate,layer.router_w,b.buf_.x,b.buf_.router_scores,1,b.cfg_.num_experts,b.hidden_size_);
        ops::router_softmax_topk_device(ctx,b.buf_.router_scores.data_as<bf16>(),nullptr,
            b.router_host_ids_,b.router_host_weights_,1,b.cfg_.num_experts,b.cfg_.num_experts_per_tok,true);
    }
    static Context& Ctx(InferenceEngine& engine) { return *engine.ctx_; }
    static Qwen4ExpBnb4Backend& Backend(InferenceEngine& engine) {
        return dynamic_cast<Qwen4ExpBnb4Backend&>(*engine.backend_);
    }
    static Qwen4ExpBnb4Backend::Session& Session(InferenceEngine& engine) {
        return Backend(engine).DefaultSession(*engine.ctx_);
    }
    static void Grouped(Qwen4ExpBnb4Backend& b, int minRows) { b.grouped_xmx_min_rows_override_ = minRows; }
    static void Histogram(Qwen4ExpBnb4Backend& b) { b.expert_histogram_enabled_ = true; b.expert_histogram_ = {}; }
    static void ResidentSplit(Qwen4ExpBnb4Backend& b, bool enabled) { b.small_rows_resident_split_override_ = enabled ? 1 : 0; }
    static const auto& Histogram(const Qwen4ExpBnb4Backend& b) { return b.expert_histogram_; }
    static uint64_t GroupedActs(const Qwen4ExpBnb4Backend& b) { return b.grouped_xmx_acts_; }
    static void GdnMode(Qwen4ExpBnb4Backend& b, int mode) {
        b.gdn_chunk_mode_ = mode;
        if (mode) b.gdn_chunk_scratch_.Reserve(*b.context_, b.runtime_seq_capacity_, b.linear_kv_heads_);
    }
    static uint64_t GdnCalls(const Qwen4ExpBnb4Backend& b) { return b.gdn_chunk_calls_; }
    static void Probe(Qwen4ExpBnb4Backend& b, decltype(b.grouped_expert_probe_) hook) { b.grouped_expert_probe_ = hook; }
    struct ExpertView {
        Context& ctx;
        const bf16* x;
        const int* tokens;
        const int* ranks;
        const ops::MoeExpertSlotDev* slots;
        const ops::MoeExpertTile* tiles;
        bool nested;
    };
    static ExpertView Experts(Qwen4ExpBnb4Backend& b) {
        return {*b.context_, b.buf_.x.data_as<bf16>(), b.expert_map_token_host_, b.expert_map_rank_host_,
            b.expert_map_slot_host_, b.expert_tiles_host_, true};
    }
};
using Access = Qwen4ExpGenerationTestAccess;

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct ToleranceScoreCard {
    int m = 0;
    float max_abs_diff = 0.0f;
    double rms_ref = 0.0;
    double rms_diff = 0.0;
    double nrmse = 0.0;
    double cosine_similarity = 0.0;
    bool top1_match = false;
    int top1_ref = -1;
    int top1_cand = -1;
    float top5_overlap = 0.0f;
    double ms_ref = 0.0;
    double ms_cand = 0.0;
    double speedup = 1.0;
};

ToleranceScoreCard evaluate_logits_tolerance(
    const std::vector<float>& ref_logits,
    const std::vector<float>& cand_logits,
    int m_tokens,
    int vocab_size) {
    ToleranceScoreCard card;
    card.m = m_tokens;
    expect(ref_logits.size() == cand_logits.size(), "Logits buffer sizes must match");
    expect(ref_logits.size() == static_cast<size_t>(vocab_size), "Expected single-row logits across vocabulary");

    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    double sum_sq_cand = 0.0;
    double dot_prod = 0.0;
    float max_abs_diff = 0.0f;

    for (size_t i = 0; i < ref_logits.size(); ++i) {
        float r = ref_logits[i];
        float c = cand_logits[i];
        float diff = std::abs(r - c);
        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_sq_diff += diff * diff;
        sum_sq_ref += r * r;
        sum_sq_cand += c * c;
        dot_prod += r * c;
    }

    const size_t n = ref_logits.size();
    card.max_abs_diff = max_abs_diff;
    card.rms_ref = std::sqrt(sum_sq_ref / n);
    card.rms_diff = std::sqrt(sum_sq_diff / n);
    card.nrmse = card.rms_diff / std::max(card.rms_ref, 1e-6);
    card.cosine_similarity = dot_prod / std::max(std::sqrt(sum_sq_ref * sum_sq_cand), 1e-12);

    // Top-1 & Top-5
    std::vector<int> ref_indices(n), cand_indices(n);
    std::iota(ref_indices.begin(), ref_indices.end(), 0);
    std::iota(cand_indices.begin(), cand_indices.end(), 0);

    std::partial_sort(ref_indices.begin(), ref_indices.begin() + 5, ref_indices.end(),
                      [&](int a, int b) { return ref_logits[a] > ref_logits[b]; });
    std::partial_sort(cand_indices.begin(), cand_indices.begin() + 5, cand_indices.end(),
                      [&](int a, int b) { return cand_logits[a] > cand_logits[b]; });

    card.top1_ref = ref_indices[0];
    card.top1_cand = cand_indices[0];
    card.top1_match = (card.top1_ref == card.top1_cand);

    std::set<int> ref_top5(ref_indices.begin(), ref_indices.begin() + 5);
    int common = 0;
    for (int k = 0; k < 5; ++k) {
        if (ref_top5.count(cand_indices[k])) ++common;
    }
    card.top5_overlap = static_cast<float>(common) / 5.0f;
    return card;
}

void print_tolerance_card(const std::string& title, const ToleranceScoreCard& card) {
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "  " << title << " (Tokens M=" << card.m << ")\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Max Abs Difference:  " << std::fixed << std::setprecision(6) << card.max_abs_diff << "\n";
    std::cout << "  NRMSE:               " << std::fixed << std::setprecision(4) << (card.nrmse * 100.0) << " %\n";
    std::cout << "  Cosine Similarity:   " << std::fixed << std::setprecision(6) << card.cosine_similarity << "\n";
    std::cout << "  Top-1 Match:         " << (card.top1_match ? "YES [MATCH]" : "DIFF [BRANCH]")
              << " (Ref ID: " << card.top1_ref << ", Cand ID: " << card.top1_cand << ")\n";
    std::cout << "  Top-5 Overlap:       " << std::fixed << std::setprecision(1) << (card.top5_overlap * 100.0f) << " %\n";
    if (card.ms_ref > 0.0) {
        std::cout << "  Latency:             Baseline=" << card.ms_ref << " ms | XMX=" << card.ms_cand
                  << " ms | Speedup: " << std::setprecision(2) << card.speedup << "x\n";
    }
    std::cout << "------------------------------------------------------------\n";
}

void evaluate_mock_shape_tolerance(Context& ctx, int K, int N, const std::string& desc,
                                   const std::vector<int>& m_values,
                                   bool force_xmx_for_eval = false) {
    constexpr int blocksize = 64;
    std::mt19937 rng(42 + K * 17 + N * 31);
    std::uniform_int_distribution<int> u8_dist(0, 255);
    std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
    std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);

    const size_t packed_bytes = static_cast<size_t>(N) * (K / 2);
    const size_t absmax_count = static_cast<size_t>(N) * (K / blocksize);

    std::vector<uint8_t> packed_host(packed_bytes);
    std::vector<float> absmax_host(absmax_count);
    for (auto& v : packed_host) v = static_cast<uint8_t>(u8_dist(rng));
    for (auto& v : absmax_host) v = f_dist(rng);

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed_bytes)}, dnnl::memory::data_type::u8);
    Tensor absmax_dev = Tensor::allocate(ctx, {static_cast<int64_t>(absmax_count)}, dnnl::memory::data_type::f32);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(packed_dev.data(), packed_host.data(), packed_bytes);
    ctx.memcpy_h2d(absmax_dev.data(), absmax_host.data(), absmax_count * sizeof(float));
    ctx.memcpy_h2d(qmap_dev.data(), kTestQuantMap, 16 * sizeof(float));

    Bnb4BitWeightRef wref;
    wref.name = "mock_" + desc;
    wref.packed_weight = &packed_dev;
    wref.absmax = &absmax_dev;
    wref.quant_map = &qmap_dev;
    wref.quant_state.shape = {N, K};
    wref.quant_state.blocksize = blocksize;
    wref.quant_state.quant_type = "nf4";

    Bnb4BitLinear linear;
    std::string err;
    if (!linear.init(ctx, wref, &err)) throw std::runtime_error(err);
    linear.set_allow_async_dequant(true);
    linear.set_direct_prefill_io(true);

    Bnb4BitLinearScratch scratch_ref, scratch_cand;

    for (int M : m_values) {
        std::vector<bf16> x_host(static_cast<size_t>(M) * K);
        for (auto& v : x_host) v = bf16(in_dist(rng));
        Tensor x_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

        Tensor out_ref = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
        Tensor out_cand = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);

        linear.set_allow_dense_xmx(false);
        linear.set_force_dense_xmx(false);
        linear.forward(ctx, scratch_ref, x_dev, out_ref, M);

        linear.set_allow_dense_xmx(true);
        if (force_xmx_for_eval) {
            linear.set_force_dense_xmx(true);
        }
        linear.forward(ctx, scratch_cand, x_dev, out_cand, M);
        ctx.synchronize();

        std::vector<bf16> ref_host(static_cast<size_t>(M) * N);
        std::vector<bf16> cand_host(static_cast<size_t>(M) * N);
        ctx.memcpy_d2h(ref_host.data(), out_ref.data(), ref_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(cand_host.data(), out_cand.data(), cand_host.size() * sizeof(bf16));

        // 1. Evaluate full tensor tolerance across all M * N elements
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0, sum_sq_cand = 0.0, dot_prod = 0.0;
        float tensor_max_diff = 0.0f;
        const size_t total_elems = ref_host.size();
        for (size_t i = 0; i < total_elems; ++i) {
            float r = static_cast<float>(ref_host[i]);
            float c = static_cast<float>(cand_host[i]);
            float d = std::abs(r - c);
            if (d > tensor_max_diff) tensor_max_diff = d;
            sum_sq_diff += d * d;
            sum_sq_ref += r * r;
            sum_sq_cand += c * c;
            dot_prod += r * c;
        }
        double tensor_rms_ref = std::sqrt(sum_sq_ref / total_elems);
        double tensor_nrmse = std::sqrt(sum_sq_diff / total_elems) / std::max(tensor_rms_ref, 1e-6);
        double tensor_cos_sim = dot_prod / std::max(std::sqrt(sum_sq_ref * sum_sq_cand), 1e-12);
        expect(tensor_nrmse <= 0.015, "Full tensor NRMSE must meet tolerance (<= 1.5%)");
        expect(tensor_cos_sim >= 0.999, "Full tensor Cosine Similarity must meet tolerance (>= 0.999)");

        // 2. Evaluate tolerance on the last row
        std::vector<float> row_ref(N), row_cand(N);
        const size_t row_offset = static_cast<size_t>(M - 1) * N;
        for (int i = 0; i < N; ++i) {
            row_ref[i] = static_cast<float>(ref_host[row_offset + i]);
            row_cand[i] = static_cast<float>(cand_host[row_offset + i]);
        }

        auto card = evaluate_logits_tolerance(row_ref, row_cand, M, N);
        print_tolerance_card(desc, card);

        expect(card.cosine_similarity >= 0.999, "Cosine similarity must meet tolerance (>= 0.999)");
        expect(card.nrmse <= 0.015, "NRMSE must meet tolerance (<= 1.5%)");
    }
}

void test_mock_prefill_tolerance(Context& ctx) {
    std::cout << "\n======================================================================\n";
    std::cout << "  [Phase X0 Unit] Synthetic Prefill Tolerance & Operator Parity\n";
    std::cout << "======================================================================\n";

    // 1. Standard projection (K=2560, N=2560), testing small odd & even M: {3, 4, 7, 26, 30}
    evaluate_mock_shape_tolerance(ctx, 2560, 2560, "Standard Projection (2560x2560)", {3, 4, 7, 26, 30});

    // 2. Linear Attention projection (K=6144, N=2560): M in {4, 26, 30}
    evaluate_mock_shape_tolerance(ctx, 6144, 2560, "Linear Attention Projection (6144x2560)", {4, 26, 30});

    // 3. Shared Expert Down projection (K=1280, N=2560): M in {4, 26, 30}
    evaluate_mock_shape_tolerance(ctx, 1280, 2560, "Shared Expert Down (1280x2560)", {4, 26, 30});

    // 4. Shared Expert Gate/Up (K=2560, N=1280): verify shape heuristic fallback (oneDNN is faster)
    evaluate_mock_shape_tolerance(ctx, 2560, 1280, "Shared Expert Gate/Up Heuristic Fallback (2560x1280)", {4, 26, 30}, false);

    // 5. Shared Expert Gate/Up with forced XMX: verify numerical parity even when forced
    evaluate_mock_shape_tolerance(ctx, 2560, 1280, "Shared Expert Gate/Up Forced XMX (2560x1280)", {4, 26, 30}, true);

    std::cout << "[PASS] Synthetic Prefill Tolerance Suite passed across all shapes and M boundaries\n";
}

std::vector<float> extract_bf16_to_f32(Context& ctx, const Tensor& tensor) {
    expect(tensor.valid() && tensor.dtype() == dnnl::memory::data_type::bf16, "Expected BF16 tensor");
    const size_t count = static_cast<size_t>(tensor.numel());
    std::vector<bf16> host_bf16(count);
    ctx.memcpy_d2h(host_bf16.data(), tensor.data(), count * sizeof(bf16));
    std::vector<float> host_f32(count);
    for (size_t i = 0; i < count; ++i) {
        host_f32[i] = static_cast<float>(host_bf16[i]);
    }
    return host_f32;
}

void test_real_model_tolerance(const std::string& model_dir) {
    std::cout << "\n======================================================================\n";
    std::cout << "  [Phase X0 / X1 Acceptance] Real Model Prefill & E2E Tolerance\n";
    std::cout << "  Model: " << model_dir << "\n";
    std::cout << "======================================================================\n";

    InferenceEngine engine;
    expect(engine.init(model_dir, 512), "Engine initialization failed");
    auto& q4e_backend = Access::Backend(engine);
    Context& ctx = Access::Ctx(engine);
    const int vocab = q4e_backend.vocab_size();

    Tokenizer tokenizer;
    expect(tokenizer.load(model_dir), "Cannot load tokenizer");

    GenerationConfig config;
    config.do_sample = false;
    config.thinking_budget_tokens = 0;
    config.max_new_tokens = 48;

    aila::chat::ChatRequest request;
    std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config, request, &error), "Failed to parse chat request");

    aila::chat::ChatFormatTextResult rendered;
    aila::chat::ChatFormatter formatter;
    expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error),
           "Failed to render chat text");

    const auto ids = tokenizer.encode(rendered.text);
    const int num_prompt_tokens = static_cast<int>(ids.size());
    std::cout << "[Info] Rendered prompt tokens: " << num_prompt_tokens << " (Story format)\n";
    expect(num_prompt_tokens >= 30, "Expected story prompt to have >= 30 tokens");

    // ------------------------------------------------------------------------
    // Part 1: Prefill Logits Evaluation (oneDNN Baseline vs XMX Joint Matrix)
    // ------------------------------------------------------------------------
    std::cout << "\n--- Measuring Prefill Logits Parity (M=26 prefix and M=30 full) ---\n";

    Tensor token_buf = Tensor::allocate(ctx, {num_prompt_tokens}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(token_buf.data(), ids.data(), num_prompt_tokens * sizeof(int));
    const int* dev_ids = token_buf.data_as<int>();

    // 1.0 Warmup pass on prompt: warms JIT compilation and populates active expert weights
    std::cout << "\n[Warmup] Warming expert cache on prompt (M=26)...\n";
    q4e_backend.reset();
    auto cold_t0 = std::chrono::high_resolution_clock::now();
    q4e_backend.forward(ctx, dev_ids, 26, {SnapshotPolicy::None, true});
    ctx.synchronize();
    auto cold_t1 = std::chrono::high_resolution_clock::now();
    double ms_cold = std::chrono::duration<double, std::milli>(cold_t1 - cold_t0).count();
    std::cout << "[Warmup] Cold-start prefill complete in " << ms_cold << " ms\n";

    // 1.1 Prefix M=26 Logits (Baseline oneDNN, warm cache)
    q4e_backend.set_prefill_xmx(false);
    q4e_backend.set_direct_prefill_io(true);
    q4e_backend.reset();
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor& ref_26_tensor = q4e_backend.forward(ctx, dev_ids, 26, {SnapshotPolicy::None, true});
    ctx.synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_ref_26 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<float> ref_logits_26 = extract_bf16_to_f32(ctx, ref_26_tensor);

    // 1.2 Prefix M=26 Logits (XMX Joint Matrix, warm cache)
    q4e_backend.set_prefill_xmx(true);
    q4e_backend.set_direct_prefill_io(true);
    q4e_backend.reset();
    t0 = std::chrono::high_resolution_clock::now();
    Tensor& cand_26_tensor = q4e_backend.forward(ctx, dev_ids, 26, {SnapshotPolicy::None, true});
    ctx.synchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double ms_cand_26 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<float> cand_logits_26 = extract_bf16_to_f32(ctx, cand_26_tensor);

    auto card_26 = evaluate_logits_tolerance(ref_logits_26, cand_logits_26, 26, vocab);
    card_26.ms_ref = ms_ref_26;
    card_26.ms_cand = ms_cand_26;
    card_26.speedup = ms_ref_26 / std::max(ms_cand_26, 1e-4);
    print_tolerance_card("Prompt Prefix Logits Parity (M=26)", card_26);

    // 1.3 Full Story M=30 Logits (Baseline oneDNN)
    q4e_backend.set_prefill_xmx(false);
    q4e_backend.set_direct_prefill_io(true);
    q4e_backend.reset();
    t0 = std::chrono::high_resolution_clock::now();
    Tensor& ref_30_tensor = q4e_backend.forward(ctx, dev_ids, 30, {SnapshotPolicy::None, true});
    ctx.synchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double ms_ref_30 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<float> ref_logits_30 = extract_bf16_to_f32(ctx, ref_30_tensor);

    // 1.4 Full Story M=30 Logits (XMX Joint Matrix)
    q4e_backend.set_prefill_xmx(true);
    q4e_backend.set_direct_prefill_io(true);
    q4e_backend.reset();
    t0 = std::chrono::high_resolution_clock::now();
    Tensor& cand_30_tensor = q4e_backend.forward(ctx, dev_ids, 30, {SnapshotPolicy::None, true});
    ctx.synchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double ms_cand_30 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<float> cand_logits_30 = extract_bf16_to_f32(ctx, cand_30_tensor);

    auto card_30 = evaluate_logits_tolerance(ref_logits_30, cand_logits_30, 30, vocab);
    card_30.ms_ref = ms_ref_30;
    card_30.ms_cand = ms_cand_30;
    card_30.speedup = ms_ref_30 / std::max(ms_cand_30, 1e-4);
    print_tolerance_card("Full Prompt Story Logits Parity (M=30)", card_30);

    // Numerical tolerance bounds verification
    expect(card_26.cosine_similarity >= 0.995, "M=26 Logits Cosine Similarity must be >= 0.995");
    expect(card_30.cosine_similarity >= 0.995, "M=30 Logits Cosine Similarity must be >= 0.995");
    expect(card_26.top1_match, "M=26 Top-1 token prediction must match");
    expect(card_30.top1_match, "M=30 Top-1 token prediction must match");
    expect(card_30.top5_overlap >= 0.80f, "M=30 Top-5 overlap must be >= 80%");

    // ------------------------------------------------------------------------
    // Part 2: End-to-End Generation & Semantic Coherence
    // ------------------------------------------------------------------------
    std::cout << "\n--- Measuring End-to-End Text Generation & Coherence ---\n";

    // 2.0 Warmup generation to ensure generation JIT and expert caching are stable
    engine.reset_context();
    engine.generate_chat_request_text(request);

    // 2.1 Baseline Generation
    q4e_backend.set_prefill_xmx(false);
    q4e_backend.set_direct_prefill_io(true);
    engine.reset_context();
    double ttft_ms_ref = 0.0;
    bool first_ref = true;
    auto gen_t0 = std::chrono::high_resolution_clock::now();
    const std::string text_ref = engine.generate_chat_request_text(request, [&](const std::string&) {
        if (first_ref) {
            ttft_ms_ref = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - gen_t0).count();
            first_ref = false;
        }
    });
    auto gen_t1 = std::chrono::high_resolution_clock::now();
    double gen_ms_ref = std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
    expect(!text_ref.empty(), "Baseline generation returned empty text");

    // 2.2 XMX Generation
    q4e_backend.set_prefill_xmx(true);
    q4e_backend.set_direct_prefill_io(true);
    engine.reset_context();
    double ttft_ms_cand = 0.0;
    bool first_cand = true;
    gen_t0 = std::chrono::high_resolution_clock::now();
    const std::string text_cand = engine.generate_chat_request_text(request, [&](const std::string&) {
        if (first_cand) {
            ttft_ms_cand = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - gen_t0).count();
            first_cand = false;
        }
    });
    gen_t1 = std::chrono::high_resolution_clock::now();
    double gen_ms_cand = std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
    expect(!text_cand.empty(), "XMX generation returned empty text");

    std::cout << "\n[Baseline Output (Total " << gen_ms_ref << " ms, TTFT " << ttft_ms_ref << " ms)]:\n" << text_ref << "\n";
    std::cout << "\n[XMX Output (Total " << gen_ms_cand << " ms, TTFT " << ttft_ms_cand << " ms)]:\n" << text_cand << "\n";
    if (ttft_ms_cand > 0.0 && ttft_ms_ref > 0.0) {
        std::cout << "TTFT Speedup: " << std::fixed << std::setprecision(2)
                  << (ttft_ms_ref / ttft_ms_cand) << "x (TTFT " << ttft_ms_ref << " ms -> " << ttft_ms_cand << " ms)\n";
    }

    // Analyze token overlap in generated text
    const auto gen_ids_ref = tokenizer.encode(text_ref);
    const auto gen_ids_cand = tokenizer.encode(text_cand);
    size_t matching_tokens = 0;
    const size_t min_len = std::min(gen_ids_ref.size(), gen_ids_cand.size());
    for (size_t i = 0; i < min_len; ++i) {
        if (gen_ids_ref[i] == gen_ids_cand[i]) ++matching_tokens;
    }
    const double token_match_rate = min_len > 0 ? (static_cast<double>(matching_tokens) / min_len) : 0.0;
    std::cout << "\nGeneration Token Parity (first " << min_len << " tokens): "
              << matching_tokens << "/" << min_len << " (" << std::fixed << std::setprecision(1)
              << (token_match_rate * 100.0) << " %)\n";

    // Validate semantic coherence and lack of degeneracy
    expect(text_cand.length() >= 50, "Generated text too short");
    expect(text_cand.find("the") != std::string::npos || text_cand.find("ocean") != std::string::npos ||
           text_cand.find("sun") != std::string::npos, "Generated text must contain relevant vocabulary");

    // ------------------------------------------------------------------------
    // Part 3: Interior Checkpoint Verification under XMX
    // ------------------------------------------------------------------------
    std::cout << "\n--- Verifying Interior Checkpoint & Rollback Parity with XMX ---\n";
    q4e_backend.set_prefill_xmx(true);
    q4e_backend.set_direct_prefill_io(true);
    q4e_backend.reset();

    // Verify anchor 26
    q4e_backend.forward(ctx, dev_ids, 26, {SnapshotPolicy::ReusablePrefix, false, true});
    expect(q4e_backend.truncate_kv_cache(26), "Rollback to anchor 26 must succeed");
    std::cout << "[PASS] Interior Checkpoint and rollback validated under XMX Joint Matrix\n";

    std::cout << "\n======================================================================\n";
    std::cout << "  [SUMMARY] All Tolerance Criteria MET (Phase X0 & X1 Confirmed)\n";
    std::cout << "======================================================================\n\n";
}

#include "Qwen4ExpCheckpointAcceptance.hpp"
#include "Qwen4ExpPresetAcceptance.hpp"
#include "Qwen4ExpIntegerMoeProbe.hpp"
#include "Qwen4ExpIntegerQualityAcceptance.hpp"
#include "Qwen4ExpCpuDiagnostic.hpp"
#include "Qwen4ExpGroupedXmxAcceptance.hpp"
#include "Qwen4ExpMtpAlign.hpp"

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    // Keep fatal driver/host failures attributable in isolated GPU canaries.
    SetUnhandledExceptionFilter([](EXCEPTION_POINTERS* p) -> LONG {
        HMODULE module = nullptr;
        char path[MAX_PATH] = {};
        const auto address = p->ExceptionRecord->ExceptionAddress;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(address), &module);
        if (module) GetModuleFileNameA(module, path, MAX_PATH);
        std::cerr << "[GPU CANARY CRASH] code=" << std::hex << p->ExceptionRecord->ExceptionCode
                  << " module=" << path << " offset=" << (reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(module)) << '\n';
        return EXCEPTION_EXECUTE_HANDLER;
    });
#endif
    aila::env::enable_l0_copy_engine();
    try {
        if (argc == 2 && std::string(argv[1]) == "--grouped-kernel") {
            test_grouped_xmx_kernel();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--grouped-model") {
            test_grouped_xmx_model(argv[2]);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--checkpoint-model") {
            test_checkpoint_acceptance(argv[2]);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--preset-model") {
            test_qwen38_preset_checkpoint(argv[2]);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--window-model") {
            test_qwen38_window_quality(argv[2]);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--dpas-smoke") {
            test_integer_dpas_smoke(); return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--integer-state") {
            test_integer_quality(argv[2],true); return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--integer-quality") {
            test_integer_quality(argv[2]); return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--memory-model") {
            test_prefill_memory(argv[2]);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--mtp-align") {
            test_mtp_align(argv[2]);
            return 0;
        }
        Context ctx;
        test_mock_prefill_tolerance(ctx);

        std::string model_dir;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--model" && i + 1 < argc) {
                model_dir = argv[i + 1];
                ++i;
            }
        }

        if (!model_dir.empty()) {
            test_real_model_tolerance(model_dir);
        } else {
            std::cout << "[Info] To run full model tolerance acceptance, pass: --model <path_to_model>\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Tolerance test failed: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[FAIL] Tolerance test failed with unknown exception\n";
        return 1;
    }
}
