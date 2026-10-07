#pragma once

// Included inside the tolerance test's anonymous namespace. This probe loads
// only the real BF16 language-model head, never a backend or an expert worker.
uint16_t head_probe_bits(bf16 value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct HeadProbeScore {
    double nrmse = 0.0, kl = 0.0, max_abs = 0.0;
    size_t equal_bits = 0;
    int argmax_ref = 0, argmax_candidate = 0;
};

HeadProbeScore head_probe_score(const std::vector<bf16>& reference, const std::vector<bf16>& candidate) {
    expect(reference.size() == candidate.size() && !reference.empty(), "Head probe logits shape mismatch");
    HeadProbeScore result;
    double reference_sq = 0.0, error_sq = 0.0;
    double reference_max = -1e300, candidate_max = -1e300;
    for (size_t i = 0; i < reference.size(); ++i) {
        const uint16_t rb = head_probe_bits(reference[i]), cb = head_probe_bits(candidate[i]);
        expect((rb & 0x7f80u) != 0x7f80u && (cb & 0x7f80u) != 0x7f80u,
            "Head probe nonfinite logits (raw BF16 exponent guard)");
        const double r = float(reference[i]), c = float(candidate[i]), difference = c - r;
        reference_sq += r * r;
        error_sq += difference * difference;
        result.max_abs = std::max(result.max_abs, std::abs(difference));
        result.equal_bits += rb == cb;
        if (r > reference_max) { reference_max = r; result.argmax_ref = static_cast<int>(i); }
        if (c > candidate_max) { candidate_max = c; result.argmax_candidate = static_cast<int>(i); }
    }
    result.nrmse = std::sqrt(error_sq / std::max(reference_sq, 1e-30));
    double reference_sum = 0.0, candidate_sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        reference_sum += std::exp(double(float(reference[i])) - reference_max);
        candidate_sum += std::exp(double(float(candidate[i])) - candidate_max);
    }
    const double reference_log_z = reference_max + std::log(reference_sum);
    const double candidate_log_z = candidate_max + std::log(candidate_sum);
    for (size_t i = 0; i < reference.size(); ++i) {
        const double r = float(reference[i]), c = float(candidate[i]);
        const double probability = std::exp(r - reference_max) / reference_sum;
        result.kl += probability * ((r - reference_log_z) - (c - candidate_log_z));
    }
    // KL is nonnegative analytically; erase only double accumulation roundoff.
    result.kl = std::max(0.0, result.kl);
    return result;
}

double head_probe_quantile(const std::vector<double>& sorted, double fraction) {
    const double position = double(sorted.size() - 1) * fraction;
    const size_t lo = static_cast<size_t>(position), hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - double(lo));
}

inline int8_t head_probe_quantize_q8_value(float value, float maximum) {
    if (!(maximum > 0.0f)) return 0;
    if (maximum > 0x1p-110f && maximum <= 0x1p120f) {
        int magnitude = static_cast<int>(sycl::rint(sycl::clamp(sycl::fabs(value) * (127.0f / maximum), 0.0f, 127.0f)));
        const float numerator = sycl::fma(sycl::fabs(value), 254.0f / 256.0f, 0.0f);
        const float lower = sycl::fma(maximum, static_cast<float>(2 * magnitude - 1) / 256.0f, 0.0f);
        const float upper = sycl::fma(maximum, static_cast<float>(2 * magnitude + 1) / 256.0f, 0.0f);
        if (numerator > upper || (numerator == upper && (magnitude & 1))) ++magnitude;
        else if (numerator < lower || (numerator == lower && (magnitude & 1))) --magnitude;
        return static_cast<int8_t>(value < 0.0f ? -magnitude : magnitude);
    }
    const float ratio = maximum > 0x1p120f
        ? sycl::ext::intel::math::fdiv_rn(sycl::fma(value, 127.0f / 256.0f, 0.0f), maximum / 256.0f)
        : sycl::ext::intel::math::fdiv_rn(value * 127.0f, maximum);
    return static_cast<int8_t>(sycl::rint(sycl::clamp(ratio, -127.0f, 127.0f)));
}

// One complete subgroup owns a contiguous BF16 block64. This quantizer is
// used once for real raw weights and on every timed Q8 activation pipeline.
sycl::event head_probe_quantize_q8_64(Context& ctx, const bf16* input,
    int8_t* quantized, float* scales, int rows, int hidden) {
    constexpr int wg = 128, sg_size = 16, blocks_per_group = wg / sg_size;
    const size_t blocks = size_t(rows) * (hidden / 64);
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>((blocks + blocks_per_group - 1) / blocks_per_group * wg, wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t block = item.get_global_linear_id() / sg_size;
                const int lane = static_cast<int>(item.get_local_linear_id() % sg_size);
                if (block >= blocks) return; // Uniform subgroup exit, no WG barrier.
                sycl::vec<float, 4> values;
                float maximum = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint16_t bits = reinterpret_cast<const uint16_t*>(input)[block * 64 + lane * 4 + j];
                    if ((bits & 0x7f80u) == 0x7f80u) bits = 0;
                    values[j] = static_cast<float>(sycl::bit_cast<bf16>(bits));
                    maximum = sycl::max(maximum, sycl::fabs(values[j]));
                }
                maximum = sycl::reduce_over_group(item.get_sub_group(), maximum, sycl::maximum<float>());
                if (lane == 0) scales[block] = maximum / 127.0f;
                sycl::vec<int8_t, 4> codes;
#pragma unroll
                for (int j = 0; j < 4; ++j) codes[j] = head_probe_quantize_q8_value(values[j], maximum);
                *reinterpret_cast<sycl::vec<int8_t, 4>*>(quantized + block * 64 + lane * 4) = codes;
            });
    });
}

inline int32_t head_probe_signed_dot4(int32_t packed_a, int32_t packed_b) {
    // Local compiler idiom avoids dot_product.hpp's non-inline definitions.
    const int8_t* a = reinterpret_cast<const int8_t*>(&packed_a);
    const int8_t* b = reinterpret_cast<const int8_t*>(&packed_b);
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

sycl::event head_probe_q8_gemv(Context& ctx, const int8_t* weights,
    const float* weight_scales, const int8_t* xq, const float* xs,
    bf16* output, int hidden, int vocab) {
    constexpr int wg = 256, sg_size = 16, rows_per_group = wg / sg_size;
    const int blocks_per_row = hidden / 64;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(size_t(vocab / rows_per_group) * wg, wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int lane = static_cast<int>(item.get_local_linear_id() % sg_size);
                const int row = static_cast<int>(item.get_group(0)) * rows_per_group
                    + static_cast<int>(item.get_local_linear_id() / sg_size);
                const int8_t* w = weights + size_t(row) * hidden;
                const float* ws = weight_scales + size_t(row) * blocks_per_row;
                float acc = 0.0f;
                for (int k = lane * 8; k < hidden; k += sg_size * 8) {
                    const int32_t w0 = *reinterpret_cast<const int32_t*>(w + k);
                    const int32_t w1 = *reinterpret_cast<const int32_t*>(w + k + 4);
                    const int32_t x0 = *reinterpret_cast<const int32_t*>(xq + k);
                    const int32_t x1 = *reinterpret_cast<const int32_t*>(xq + k + 4);
                    const int32_t dot = head_probe_signed_dot4(w0, x0) + head_probe_signed_dot4(w1, x1);
                    acc = sycl::fma(static_cast<float>(dot), ws[k / 64] * xs[k / 64], acc);
                }
                const float sum = sycl::reduce_over_group(item.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) output[row] = bf16(sum);
            });
    });
}

void test_head_kernel(const std::string& directory) {
    std::cout << std::unitbuf << std::setprecision(10);
    constexpr int Hidden = 2560, Vocab = 248320, Modes = 4, Rounds = 3, Samples = 15, Warmup = 5, Guard = 8;
    constexpr double NrmseBound = 0.005, KlBound = 0.001;
    constexpr double Q8NrmseBound = 0.03, Q8KlBound = 0.001;
    constexpr std::array<unsigned, Rounds> Seeds = {20261009, 20261010, 20261011};
    constexpr std::array<const char*, Modes> Names = {"canonical_onednn", "raw_onednn", "native_sg16", "q8_dp4a_block64"};
    Context ctx;
    expect(ctx.queue().get_device().is_gpu(), "Head kernel probe requires a GPU");
    auto weights = LoadModelWeightsLazy(directory, ctx);
    expect(weights.size() == 0, "Head lazy loader unexpectedly uploaded other model tensors");
    const auto host = weights.host_view("lm_head.weight");
    expect(host.valid() && host.dtype == DT_BF16 && host.shape.size() == 2 &&
        host.shape[0] == Vocab && host.shape[1] == Hidden, "Head probe requires BF16 lm_head [248320,2560]");
    Tensor& raw = weights.upload("lm_head.weight", ctx);
    expect(weights.size() == 1 && raw.dtype() == dnnl::memory::data_type::bf16,
        "Head probe must upload only the raw BF16 head");
    Tensor canonical = Tensor::allocate(ctx, {Hidden, Vocab}, dnnl::memory::data_type::bf16);
    Tensor input = Tensor::allocate(ctx, {1, Hidden}, dnnl::memory::data_type::bf16);
    constexpr int64_t Q8WeightElements = int64_t(Vocab) * Hidden;
    constexpr int64_t Q8WeightBlocks = int64_t(Vocab) * (Hidden / 64);
    Tensor q8_weight_storage = Tensor::allocate(ctx, {Q8WeightElements + 2 * Guard}, dnnl::memory::data_type::u8);
    Tensor q8_weight_scale_storage = Tensor::allocate(ctx, {Q8WeightBlocks + 2 * Guard}, dnnl::memory::data_type::f32);
    Tensor q8_input_storage = Tensor::allocate(ctx, {Hidden + 2 * Guard}, dnnl::memory::data_type::u8);
    Tensor q8_input_scale_storage = Tensor::allocate(ctx, {Hidden / 64 + 2 * Guard}, dnnl::memory::data_type::f32);
    Tensor q8_output_storage = Tensor::allocate(ctx, {Vocab + 2 * Guard}, dnnl::memory::data_type::bf16);
    int8_t* const q8_weights = q8_weight_storage.data_as<int8_t>() + Guard;
    float* const q8_weight_scales = q8_weight_scale_storage.data_as<float>() + Guard;
    int8_t* const q8_input = q8_input_storage.data_as<int8_t>() + Guard;
    float* const q8_input_scales = q8_input_scale_storage.data_as<float>() + Guard;
    Tensor q8_weight_view = Tensor::view(ctx, q8_weights, {Vocab, Hidden}, dnnl::memory::data_type::u8);
    Tensor q8_weight_scale_view = Tensor::view(ctx, q8_weight_scales, {Vocab, Hidden / 64}, dnnl::memory::data_type::f32);
    Tensor q8_input_view = Tensor::view(ctx, q8_input, {Hidden}, dnnl::memory::data_type::u8);
    Tensor q8_input_scale_view = Tensor::view(ctx, q8_input_scales, {Hidden / 64}, dnnl::memory::data_type::f32);
    std::array<Tensor, Modes> output;
    for (int mode = 0; mode < Modes - 1; ++mode)
        output[mode] = Tensor::allocate(ctx, {1, Vocab}, dnnl::memory::data_type::bf16);
    output[3] = Tensor::view(ctx, q8_output_storage.data_as<bf16>() + Guard, {1, Vocab}, dnnl::memory::data_type::bf16);
    Linear canonical_linear, raw_linear;
    std::array<std::vector<bf16>, Rounds> inputs;
    std::array<std::vector<bf16>, Modes> logits;
    for (auto& values : logits) values.resize(Vocab);
    for (int seed = 0; seed < Rounds; ++seed) {
        std::mt19937 rng(Seeds[seed]);
        inputs[seed].resize(Hidden);
        for (auto& value : inputs[seed]) value = bf16(float(int(rng() % 2049) - 1024) / 1024.0f);
    }
    // Last resource owner: every exit drains submitted work before Linear,
    // output arenas, transpose, or the raw model weight can be destroyed.
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    ops::transpose(ctx, raw, canonical);
    auto* const q8_output = output[3].data_as<bf16>();
    ctx.queue().parallel_for(sycl::range<1>(Guard), [=](sycl::id<1> index) {
        const int i = static_cast<int>(index[0]);
        q8_weights[i - Guard] = q8_weights[Q8WeightElements + i] = int8_t(0x5a);
        q8_weight_scales[i - Guard] = q8_weight_scales[Q8WeightBlocks + i] = 123.25f;
        q8_input[i - Guard] = q8_input[Hidden + i] = int8_t(0x5a);
        q8_input_scales[i - Guard] = q8_input_scales[Hidden / 64 + i] = 123.25f;
        q8_output[i - Guard] = q8_output[Vocab + i] = bf16(111.0f);
    });
    // Convert the real raw [Vocab,Hidden] head only once, outside all timing.
    ops::quantize_bf16_weights_q8_64(ctx, raw, q8_weight_view, q8_weight_scale_view);
    ctx.synchronize();
    canonical_linear.init(ctx, canonical, Hidden, Vocab, /*preprocessed=*/true, /*prepare_gemv=*/false);
    raw_linear.init(ctx, raw, Hidden, Vocab, /*preprocessed=*/false, /*prepare_gemv=*/false);

    // Describe the same oneDNN primitives used by Linear, without changing its
    // attributes or introducing another weight copy. Inspection is untimed.
    using Md = dnnl::memory::desc;
    const auto bf = dnnl::memory::data_type::bf16;
    const Md src_md({1, Hidden}, bf, dnnl::memory::format_tag::ab);
    const Md dst_md({1, Vocab}, bf, dnnl::memory::format_tag::ab);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    const dnnl::matmul::primitive_desc canonical_pd(ctx.engine(), src_md,
        Md({Hidden, Vocab}, bf, dnnl::memory::format_tag::ab), dst_md, attr);
    const dnnl::matmul::primitive_desc raw_pd(ctx.engine(), src_md,
        Md({Hidden, Vocab}, bf, dnnl::memory::dims{1, Hidden}), dst_md, attr);
    const auto device = ctx.queue().get_device();
    std::cout << "[HEAD CONFIG] scope=isolated_head hidden=" << Hidden << " vocab=" << Vocab
        << " tokens=1 gpu_model_tensors=" << weights.size() << " raw_weight_bytes=" << raw.size_bytes()
        << " canonical_weight_bytes=" << canonical.size_bytes() << " device_bytes=" << ctx.current_allocated_bytes()
        << " q8_weight_bytes=" << Q8WeightElements << " q8_weight_scale_bytes=" << Q8WeightBlocks * sizeof(float)
        << " q8_weight_conversion_once=1 q8_weight_format=signed_int8_block64 q8_input_quantization_timed=1"
        << " device=" << device.get_info<sycl::info::device::name>()
        << " driver=" << device.get_info<sycl::info::device::driver_version>()
        << " warmup_per_mode_per_round=" << Warmup << " rounds=" << Rounds << " samples_per_mode_per_round=" << Samples
        << " h2d_transpose_init_d2h_excluded=1 no_full_model_claim=1\n";
    std::cout << "[HEAD LIBRARY] canonical=" << canonical_pd.impl_info_str() << " raw=" << raw_pd.impl_info_str()
        << " src=bf16 weight=bf16 dst=bf16 fpmath=default scratchpad=user canonical_strides=248320,1 raw_strides=1,2560\n";
    auto run = [&](int mode) {
        if (mode == 0) canonical_linear.forward(ctx, input, output[0], 1);
        else if (mode == 1) raw_linear.forward(ctx, input, output[1], 1);
        else if (mode == 2) ops::bf16_gemv_bf16(ctx, raw, input.data_as<bf16>(), output[2].data_as<bf16>(), Vocab, Hidden);
        else {
            ops::bf16_q8_head_forward(ctx, q8_weight_view, q8_weight_scale_view,
                input, output[3], q8_input_view, q8_input_scale_view, 1, Hidden, Vocab);
        }
    };
    auto check_guards = [&]() {
        std::array<int8_t, Guard> bytes;
        for (const auto span : {std::make_pair(q8_weights, Q8WeightElements), std::make_pair(q8_input, int64_t(Hidden))}) {
            ctx.memcpy_d2h(bytes.data(), span.first - Guard, sizeof(bytes));
            expect(std::all_of(bytes.begin(), bytes.end(), [](int8_t v) { return v == int8_t(0x5a); }), "Head Q8 byte prefix canary");
            ctx.memcpy_d2h(bytes.data(), span.first + span.second, sizeof(bytes));
            expect(std::all_of(bytes.begin(), bytes.end(), [](int8_t v) { return v == int8_t(0x5a); }), "Head Q8 byte suffix canary");
        }
        std::array<float, Guard> scales;
        for (const auto span : {std::make_pair(q8_weight_scales, Q8WeightBlocks), std::make_pair(q8_input_scales, int64_t(Hidden / 64))}) {
            ctx.memcpy_d2h(scales.data(), span.first - Guard, sizeof(scales));
            expect(std::all_of(scales.begin(), scales.end(), [](float v) { return v == 123.25f; }), "Head Q8 scale prefix canary");
            ctx.memcpy_d2h(scales.data(), span.first + span.second, sizeof(scales));
            expect(std::all_of(scales.begin(), scales.end(), [](float v) { return v == 123.25f; }), "Head Q8 scale suffix canary");
        }
        std::array<bf16, Guard> out;
        for (const auto* side : {q8_output - Guard, q8_output + Vocab}) {
            ctx.memcpy_d2h(out.data(), side, sizeof(out));
            expect(std::all_of(out.begin(), out.end(), [](bf16 v) { return head_probe_bits(v) == head_probe_bits(bf16(111.0f)); }),
                "Head Q8 output side canary");
        }
    };
    std::array<bool, Modes> valid = {true, true, true, true};
    for (int seed = 0; seed < Rounds; ++seed) {
        ctx.memcpy_h2d(input.data(), inputs[seed].data(), Hidden * sizeof(bf16));
        for (int mode = 0; mode < Modes; ++mode) {
            run(mode);
            ctx.synchronize();
            ctx.memcpy_d2h(logits[mode].data(), output[mode].data(), Vocab * sizeof(bf16));
        }
        for (int mode = 1; mode < Modes; ++mode) {
            const auto score = head_probe_score(logits[0], logits[mode]);
            const double nrmse_bound = mode == 3 ? Q8NrmseBound : NrmseBound;
            const double kl_bound = mode == 3 ? Q8KlBound : KlBound;
            const bool passed = score.nrmse <= nrmse_bound && score.kl <= kl_bound;
            valid[mode] = valid[mode] && passed;
            std::cout << "[HEAD CHECK] seed=" << Seeds[seed] << " mode=" << Names[mode]
                << " reference=canonical_onednn nrmse=" << score.nrmse << " kl_ref_candidate=" << score.kl
                << " max_abs=" << score.max_abs << " equal_bf16_logits=" << score.equal_bits << '/' << Vocab
                << " argmax_ref=" << score.argmax_ref << " argmax_candidate=" << score.argmax_candidate
                << " argmax_match=" << int(score.argmax_ref == score.argmax_candidate)
                << " bound_nrmse=" << nrmse_bound << " bound_kl=" << kl_bound << " finite_rawbits=1 passed=" << int(passed) << '\n';
        }
        check_guards();
    }
    std::array<std::vector<double>, Modes> all_times;
    for (auto& times : all_times) times.reserve(Rounds * Samples);
    const auto allocation_count = ctx.device_allocation_count();
    for (int round = 0; round < Rounds; ++round) {
        ctx.memcpy_h2d(input.data(), inputs[round].data(), Hidden * sizeof(bf16));
        for (int warmup = 0; warmup < Warmup; ++warmup)
            for (int mode = 0; mode < Modes; ++mode) { run(mode); ctx.synchronize(); }
        std::array<std::vector<double>, Modes> times;
        for (auto& values : times) values.reserve(Samples);
        for (int sample = 0; sample < Samples; ++sample) for (int order = 0; order < Modes; ++order) {
            const int mode = (round + sample + order) % Modes;
            const auto start = std::chrono::steady_clock::now();
            run(mode);
            ctx.synchronize();
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            times[mode].push_back(elapsed);
            all_times[mode].push_back(elapsed);
        }
        for (int mode = 0; mode < Modes; ++mode) {
            std::sort(times[mode].begin(), times[mode].end());
            std::cout << "[HEAD TIME] round=" << round << " seed=" << Seeds[round] << " mode=" << Names[mode]
                << " samples=" << Samples << " median_wall_ms=" << head_probe_quantile(times[mode], 0.5)
                << " p10_wall_ms=" << head_probe_quantile(times[mode], 0.1)
                << " p90_wall_ms=" << head_probe_quantile(times[mode], 0.9)
                << " submit_and_sync_included=1 input_quantization_included=" << int(mode == 3)
                << " comparable_for_speedup=" << int(valid[mode]) << '\n';
        }
    }
    for (int mode = 0; mode < Modes; ++mode) {
        std::sort(all_times[mode].begin(), all_times[mode].end());
        std::cout << "[HEAD SUMMARY] mode=" << Names[mode] << " samples=" << all_times[mode].size()
            << " median_wall_ms=" << head_probe_quantile(all_times[mode], 0.5)
            << " p10_wall_ms=" << head_probe_quantile(all_times[mode], 0.1)
            << " p90_wall_ms=" << head_probe_quantile(all_times[mode], 0.9)
            << " accuracy_class=" << (valid[mode] ? "matched" : "invalid")
            << " comparable_for_speedup=" << int(valid[mode]) << " input_quantization_included=" << int(mode == 3)
            << " scope=isolated_head\n";
    }
    check_guards();
    expect(ctx.device_allocation_count() == allocation_count, "Head probe hot-path USM allocation growth");
    expect(valid[1] && valid[2], "Head probe candidate exceeds NRMSE/KL contract");
    expect(valid[3], "Head Q8 probe candidate exceeds separate NRMSE=.03/KL=.001 contract");
    std::cout << "[HEAD PASS] seeds=3 no_hot_usm_growth=1 q8_side_canaries=1 q8_input_quantization_timed=1"
        << " original_canonical_onednn_reference=1 no_full_model_claim=1\n";
}
