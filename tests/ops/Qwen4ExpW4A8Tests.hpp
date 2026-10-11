// B2: adaptive-pipeline NF4 weight x signed-Q8 (W4A8) decode activations.
//
// Kernel-scope evidence only (no scheduler, no production sequence):
//   * the Q8 decode pair vs the BF16 pair it replaces (max_abs / NRMSE / cosine)
//   * both variants vs an independent f64 reference of the intended W4A8 math,
//     fed from the device's own readback Q8 codes and per-64 scales
//   * interleaved same-shape timing of the two pairs
//
// The Q8 arithmetic is intentionally not bit-exact with BF16: activations are
// rounded to signed 8-bit with per-64 block scales and the NF4 codebook is
// folded into the same 8-bit grids (exactly like the retired resident q8 path).

[[maybe_unused]] static void q4e_w4a8_stats(const std::vector<float>& actual,
                                            const std::vector<float>& expected,
                                            float& max_abs, float& nrmse, float& cosine) {
    if (actual.size() != expected.size() || actual.empty())
        throw std::runtime_error("w4a8 stats size mismatch");
    double sum_dd = 0.0, sum_ee = 0.0, sum_aa = 0.0, sum_ae = 0.0;
    max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) throw std::runtime_error("w4a8 stats nonfinite");
        const double d = a - e;
        max_abs = std::max(max_abs, static_cast<float>(std::abs(d)));
        sum_dd += d * d; sum_ee += e * e; sum_aa += a * a; sum_ae += a * e;
    }
    nrmse = sum_ee > 0.0 ? static_cast<float>(std::sqrt(sum_dd / sum_ee)) : 0.0f;
    const double denom = std::sqrt(sum_aa) * std::sqrt(sum_ee);
    cosine = denom > 0.0 ? static_cast<float>(sum_ae / denom) : 1.0f;
}

[[maybe_unused]] static std::vector<float> q4e_w4a8_to_f32(const std::vector<bf16>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

// BF16 outputs carry up to half a BF16 ULP of rounding, so the gate/up oracle
// uses a relative allowance on top of the absolute one.
[[maybe_unused]] static void q4e_w4a8_check_close_rel(const std::vector<bf16>& actual,
                                                      const std::vector<float>& expected,
                                                      const std::string& name,
                                                      float abs_tol, float rel_tol) {
    if (actual.size() != expected.size())
        throw std::runtime_error(name + " size mismatch");
    float worst_excess = -1.0f, max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float a = static_cast<float>(actual[i]);
        const float e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) throw std::runtime_error(name + " nonfinite");
        const float diff = std::abs(a - e);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / (std::abs(e) + 1e-30f));
        worst_excess = std::max(worst_excess, diff - (abs_tol + rel_tol * std::abs(e)));
    }
    if (worst_excess > 0.0f)
        throw std::runtime_error(name + " failed: max_abs=" + std::to_string(max_abs) +
                                 " max_rel=" + std::to_string(max_rel));
    std::cout << "[PASS] " << name << " (max_abs=" << max_abs << " max_rel=" << max_rel << ")\n";
}

// Production expert geometry H=2560, I=640, decode top_k=10. models the
// executor: <=16 active experts use the pinned host descriptor array, wider
// batches use the device copy.
[[maybe_unused]] static void run_q4e_w4a8_case(Context& ctx, int tokens, bool nested, int timing_iters) {
    constexpr int H = 2560;
    constexpr int I = 640;
    constexpr int Block = 64;
    constexpr int TopK = 10;
    const int tasks = tokens * TopK;
    const int unique_experts = std::min(8, tasks);
    const bool device_slots = tasks > 16;

    const size_t gu_bytes = static_cast<size_t>(2 * I) * (H / 2);
    const size_t gu_scales = static_cast<size_t>(2 * I) * (H / Block);
    const size_t dn_bytes = static_cast<size_t>(H) * (I / 2);
    const size_t dn_scales = static_cast<size_t>(H) * (I / Block);

    std::mt19937 rng(0x51B2u + static_cast<unsigned>(tokens) * 131u + static_cast<unsigned>(nested));
    const bool stage_sync = aila::env::read_flag("AILA_W4A8_STAGE_SYNC", false);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> nibble_dist(0, 15);
    std::uniform_real_distribution<float> absmax_dist(0.01f, 0.5f);
    std::uniform_real_distribution<float> input_dist(-0.3f, 0.3f);

    // Sparse codes keep the reconstructed nested scale inside the tested range.
    std::vector<float> nested_map_host(256);
    for (int i = 0; i < 256; ++i) nested_map_host[static_cast<size_t>(i)] =
        static_cast<float>(i + 1) / 257.0f;

    std::vector<Tensor> keep; // device owners; reserved so references stay valid
    keep.reserve(128);
    std::vector<std::vector<uint8_t>> gu_host(unique_experts), dn_host(unique_experts);
    std::vector<std::vector<float>> gu_abs_host(unique_experts), dn_abs_host(unique_experts);
    std::vector<std::vector<uint8_t>> gu_codes_host(unique_experts), dn_codes_host(unique_experts);
    std::vector<std::vector<float>> gu_nested_host(unique_experts), dn_nested_host(unique_experts);
    std::vector<ops::MoeExpertSlotDev> experts(unique_experts);

    Tensor nested_map_dev = Tensor::allocate(ctx, {256}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(nested_map_dev.data(), nested_map_host.data(), 256 * sizeof(float));

    for (int e = 0; e < unique_experts; ++e) {
        gu_host[static_cast<size_t>(e)].resize(gu_bytes);
        dn_host[static_cast<size_t>(e)].resize(dn_bytes);
        gu_abs_host[static_cast<size_t>(e)].resize(gu_scales);
        dn_abs_host[static_cast<size_t>(e)].resize(dn_scales);
        for (auto& v : gu_host[static_cast<size_t>(e)]) v = static_cast<uint8_t>(byte_dist(rng));
        for (auto& v : dn_host[static_cast<size_t>(e)]) v = static_cast<uint8_t>(byte_dist(rng));
        for (auto& v : gu_abs_host[static_cast<size_t>(e)]) v = absmax_dist(rng);
        for (auto& v : dn_abs_host[static_cast<size_t>(e)]) v = absmax_dist(rng);

        keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(gu_bytes)}, dnnl::memory::data_type::u8));
        ctx.memcpy_h2d(keep.back().data(), gu_host[static_cast<size_t>(e)].data(), gu_bytes);
        keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(gu_scales)}, dnnl::memory::data_type::f32));
        ctx.memcpy_h2d(keep.back().data(), gu_abs_host[static_cast<size_t>(e)].data(), gu_scales * sizeof(float));
        keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(dn_bytes)}, dnnl::memory::data_type::u8));
        ctx.memcpy_h2d(keep.back().data(), dn_host[static_cast<size_t>(e)].data(), dn_bytes);
        keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(dn_scales)}, dnnl::memory::data_type::f32));
        ctx.memcpy_h2d(keep.back().data(), dn_abs_host[static_cast<size_t>(e)].data(), dn_scales * sizeof(float));

        ops::MoeExpertSlotDev& slot = experts[static_cast<size_t>(e)];
        slot.gate_up_packed = static_cast<const uint8_t*>(keep[keep.size() - 4].data());
        slot.gate_up_absmax = keep[keep.size() - 3].data_as<float>();
        slot.down_packed = static_cast<const uint8_t*>(keep[keep.size() - 2].data());
        slot.down_absmax = keep[keep.size() - 1].data_as<float>();

        if (nested) {
            const size_t gu_nested_count = (gu_scales + 255) / 256;
            const size_t dn_nested_count = (dn_scales + 255) / 256;
            gu_codes_host[static_cast<size_t>(e)].resize(gu_scales);
            dn_codes_host[static_cast<size_t>(e)].resize(dn_scales);
            gu_nested_host[static_cast<size_t>(e)].resize(gu_nested_count);
            dn_nested_host[static_cast<size_t>(e)].resize(dn_nested_count);
            for (auto& v : gu_codes_host[static_cast<size_t>(e)]) v = static_cast<uint8_t>(nibble_dist(rng));
            for (auto& v : dn_codes_host[static_cast<size_t>(e)]) v = static_cast<uint8_t>(nibble_dist(rng));
            for (auto& v : gu_nested_host[static_cast<size_t>(e)]) v = absmax_dist(rng);
            for (auto& v : dn_nested_host[static_cast<size_t>(e)]) v = absmax_dist(rng);

            keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(gu_scales)}, dnnl::memory::data_type::u8));
            ctx.memcpy_h2d(keep.back().data(), gu_codes_host[static_cast<size_t>(e)].data(), gu_scales);
            keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(gu_nested_count)}, dnnl::memory::data_type::f32));
            ctx.memcpy_h2d(keep.back().data(), gu_nested_host[static_cast<size_t>(e)].data(), gu_nested_count * sizeof(float));
            keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(dn_scales)}, dnnl::memory::data_type::u8));
            ctx.memcpy_h2d(keep.back().data(), dn_codes_host[static_cast<size_t>(e)].data(), dn_scales);
            keep.push_back(Tensor::allocate(ctx, {static_cast<int64_t>(dn_nested_count)}, dnnl::memory::data_type::f32));
            ctx.memcpy_h2d(keep.back().data(), dn_nested_host[static_cast<size_t>(e)].data(), dn_nested_count * sizeof(float));

            slot.gate_up_nested.codes = static_cast<const uint8_t*>(keep[keep.size() - 4].data());
            slot.gate_up_nested.nested_absmax = keep[keep.size() - 3].data_as<float>();
            slot.gate_up_nested.nested_quant_map = nested_map_dev.data_as<float>();
            slot.gate_up_nested.offset = 0.0f;
            slot.gate_up_nested.blocksize = 256;
            slot.down_nested.codes = static_cast<const uint8_t*>(keep[keep.size() - 2].data());
            slot.down_nested.nested_absmax = keep[keep.size() - 1].data_as<float>();
            slot.down_nested.nested_quant_map = nested_map_dev.data_as<float>();
            slot.down_nested.offset = 0.0f;
            slot.down_nested.blocksize = 256;
        }
    }

    // Per-task expert / token / rank maps: the executor's flattened
    // (token * top_k + rank) grid.
    std::vector<int> expert_of(tasks), token_ids(tasks), rank_of(tasks);
    std::vector<ops::MoeExpertSlotDev> slots_host(tasks);
    for (int t = 0; t < tasks; ++t) {
        const int e = (t * 7 + 3) % unique_experts;
        expert_of[static_cast<size_t>(t)] = e;
        token_ids[static_cast<size_t>(t)] = t / TopK;
        rank_of[static_cast<size_t>(t)] = t % TopK;
        slots_host[static_cast<size_t>(t)] = experts[static_cast<size_t>(e)];
    }

    std::vector<bf16> x_host(static_cast<size_t>(tokens) * H);
    for (auto& v : x_host) v = bf16(input_dist(rng));

    Tensor x_dev = Tensor::allocate(ctx, {tokens, H}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));
    Tensor slots_dev_t = Tensor::allocate(ctx, {static_cast<int64_t>(tasks * sizeof(ops::MoeExpertSlotDev))}, dnnl::memory::data_type::u8);
    ctx.memcpy_h2d(slots_dev_t.data(), slots_host.data(), tasks * sizeof(ops::MoeExpertSlotDev));
    Tensor token_dev = Tensor::allocate(ctx, {tasks}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(token_dev.data(), token_ids.data(), tasks * sizeof(int));
    Tensor rank_dev = Tensor::allocate(ctx, {tasks}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(rank_dev.data(), rank_of.data(), tasks * sizeof(int));

    const ops::MoeExpertSlotDev* slots_ptr = device_slots
        ? static_cast<const ops::MoeExpertSlotDev*>(slots_dev_t.data())
        : slots_host.data();
    const int* token_ptr = device_slots ? token_dev.data_as<int>() : token_ids.data();
    const int* rank_ptr = device_slots ? rank_dev.data_as<int>() : rank_of.data();

    // Activation and intermediate arenas (identical to the executor's).
    Tensor xq_dev = Tensor::allocate(ctx, {tokens, H}, dnnl::memory::data_type::u8);
    Tensor xs_dev = Tensor::allocate(ctx, {tokens, H / Block}, dnnl::memory::data_type::f32);
    Tensor h_bf16_dev = Tensor::allocate(ctx, {tasks, I}, dnnl::memory::data_type::bf16);
    Tensor down_bf16_dev = Tensor::allocate(ctx, {tokens * TopK, H}, dnnl::memory::data_type::f32);
    Tensor h_q8_dev = Tensor::allocate(ctx, {tasks, I}, dnnl::memory::data_type::bf16);
    Tensor hq_dev = Tensor::allocate(ctx, {tasks, I}, dnnl::memory::data_type::u8);
    Tensor hs_dev = Tensor::allocate(ctx, {tasks, I / Block}, dnnl::memory::data_type::f32);
    Tensor down_q8_dev = Tensor::allocate(ctx, {tokens * TopK, H}, dnnl::memory::data_type::f32);

    // 1. BF16 reference pair (exactly what the adaptive executor runs today).
    auto bf16_gate_up = ops::moe_decode_gate_up_swiglu_grouped(
        ctx, slots_ptr, nullptr, x_dev.data_as<bf16>(), h_bf16_dev.data_as<bf16>(), tasks,
        H, I, Block, nullptr, token_ptr, nested);
    if (stage_sync) { ctx.synchronize(); std::cout << "[stage] bf16 gate_up ok\n"; }
    ops::moe_decode_down_to_rank_f32(
        ctx, slots_ptr, rank_ptr, h_bf16_dev.data_as<bf16>(), down_bf16_dev.data_as<float>(), tasks,
        TopK, I, H, Block, token_ptr, nullptr, &bf16_gate_up, nested);
    if (stage_sync) { ctx.synchronize(); std::cout << "[stage] bf16 down ok\n"; }

    // 2. Q8 pair: one quantization pass per decode row, then fused gate/up and
    //    the Q8 down projection.
    ops::quantize_moe_input_q8_64_rows(ctx, x_dev.data_as<bf16>(), reinterpret_cast<int8_t*>(xq_dev.data()),
        xs_dev.data_as<float>(), tokens, H);
    if (stage_sync) { ctx.synchronize(); std::cout << "[stage] quantize ok\n"; }
    auto q8_gate_up = ops::moe_decode_gate_up_swiglu_grouped_q8(
        ctx, slots_ptr, reinterpret_cast<const int8_t*>(xq_dev.data()), xs_dev.data_as<float>(),
        h_q8_dev.data_as<bf16>(), reinterpret_cast<int8_t*>(hq_dev.data()), hs_dev.data_as<float>(),
        tasks, H, I, token_ptr, nullptr, nested);
    if (stage_sync) { ctx.synchronize(); std::cout << "[stage] q8 gate_up ok\n"; }
    ops::moe_decode_down_to_rank_f32_q8(
        ctx, slots_ptr, rank_ptr, reinterpret_cast<const int8_t*>(hq_dev.data()),
        hs_dev.data_as<float>(), down_q8_dev.data_as<float>(), tasks, TopK, I, H, token_ptr,
        &q8_gate_up, nested);
    if (stage_sync) { ctx.synchronize(); std::cout << "[stage] q8 down ok\n"; }
    ctx.synchronize();

    std::vector<bf16> h_bf16(tasks * I), h_q8(tasks * I);
    std::vector<float> down_bf16(tasks * H), down_q8(tasks * H);
    std::vector<int8_t> xq(static_cast<size_t>(tokens) * H);
    std::vector<float> xs(static_cast<size_t>(tokens) * (H / Block));
    std::vector<int8_t> hq(static_cast<size_t>(tasks) * I);
    std::vector<float> hs(static_cast<size_t>(tasks) * (I / Block));
    ctx.memcpy_d2h(h_bf16.data(), h_bf16_dev.data(), h_bf16.size() * sizeof(bf16));
    ctx.memcpy_d2h(h_q8.data(), h_q8_dev.data(), h_q8.size() * sizeof(bf16));
    ctx.memcpy_d2h(down_bf16.data(), down_bf16_dev.data(), down_bf16.size() * sizeof(float));
    ctx.memcpy_d2h(down_q8.data(), down_q8_dev.data(), down_q8.size() * sizeof(float));
    ctx.memcpy_d2h(xq.data(), xq_dev.data(), xq.size() * sizeof(int8_t));
    ctx.memcpy_d2h(xs.data(), xs_dev.data(), xs.size() * sizeof(float));
    ctx.memcpy_d2h(hq.data(), hq_dev.data(), hq.size() * sizeof(int8_t));
    ctx.memcpy_d2h(hs.data(), hs_dev.data(), hs.size() * sizeof(float));

    // 3. Independent f64 reference of the intended W4A8 arithmetic, driven by
    //    the device's own readback codes so it validates the kernels' dot,
    //    scale and rank wiring rather than the quantizer's rounding.
    //    Row sampling keeps the scalar oracle affordable while still covering
    //    every task (token/rank) and both 64-element boundaries inside a row.
    constexpr int kRefRows = 64;
    int q8_codes[16];
    for (int i = 0; i < 16; ++i) q8_codes[i] = static_cast<int>(std::lround(kTestQuantMap[i] * 127.0f));
    const float lut_scale = 1.0f / 127.0f;
    auto scale_of = [&](bool gate, int e, size_t index) -> double {
        if (!nested) return gate ? gu_abs_host[static_cast<size_t>(e)][index] : dn_abs_host[static_cast<size_t>(e)][index];
        const auto& codes = gate ? gu_codes_host[static_cast<size_t>(e)] : dn_codes_host[static_cast<size_t>(e)];
        const auto& nested_abs = gate ? gu_nested_host[static_cast<size_t>(e)] : dn_nested_host[static_cast<size_t>(e)];
        return static_cast<double>(nested_map_host[codes[index]]) * nested_abs[index / 256];
    };
    auto nibble_of = [](const std::vector<uint8_t>& packed, size_t row, size_t row_stride, int d) {
        const uint8_t byte = packed[row * row_stride + static_cast<size_t>(d) / 2];
        return (d % 2 == 0) ? int(byte >> 4) : int(byte & 0xF);
    };

    std::vector<bf16> actual_h;
    std::vector<float> ref_h;
    actual_h.reserve(static_cast<size_t>(tasks) * kRefRows);
    ref_h.reserve(static_cast<size_t>(tasks) * kRefRows);
    for (int t = 0; t < tasks; ++t) {
        const int tok = token_ids[static_cast<size_t>(t)];
        const int e = expert_of[static_cast<size_t>(t)];
        const int8_t* xq_t = &xq[static_cast<size_t>(tok) * H];
        const float* xs_t = &xs[static_cast<size_t>(tok) * (H / Block)];
        for (int r = 0; r < kRefRows; ++r) {
            double gate = 0.0, up = 0.0;
            for (int d = 0; d < H; ++d) {
                const double xv = static_cast<double>(xq_t[d]) * static_cast<double>(xs_t[d / Block]);
                const int gate_nib = nibble_of(gu_host[static_cast<size_t>(e)], static_cast<size_t>(r), H / 2, d);
                const int up_nib = nibble_of(gu_host[static_cast<size_t>(e)], static_cast<size_t>(r + I), H / 2, d);
                gate += xv * static_cast<double>(q8_codes[gate_nib]) * lut_scale *
                        scale_of(true, e, static_cast<size_t>(r) * (H / Block) + d / Block);
                up += xv * static_cast<double>(q8_codes[up_nib]) * lut_scale *
                      scale_of(true, e, static_cast<size_t>(r + I) * (H / Block) + d / Block);
            }
            const double silu = gate / (1.0 + std::exp(-gate));
            ref_h.push_back(static_cast<float>(silu * up));
            actual_h.push_back(h_q8[static_cast<size_t>(t) * I + r]);
        }
    }

    std::vector<float> actual_down, ref_down;
    actual_down.reserve(static_cast<size_t>(tasks) * kRefRows);
    ref_down.reserve(static_cast<size_t>(tasks) * kRefRows);
    for (int t = 0; t < tasks; ++t) {
        const int tok = token_ids[static_cast<size_t>(t)];
        const int rank = rank_of[static_cast<size_t>(t)];
        const int e = expert_of[static_cast<size_t>(t)];
        const int8_t* hq_t = &hq[static_cast<size_t>(t) * I];
        const float* hs_t = &hs[static_cast<size_t>(t) * (I / Block)];
        for (int row = 0; row < kRefRows; ++row) {
            double acc = 0.0;
            for (int d = 0; d < I; ++d) {
                const double hv = static_cast<double>(hq_t[d]) * static_cast<double>(hs_t[d / Block]);
                const int nib = nibble_of(dn_host[static_cast<size_t>(e)], static_cast<size_t>(row), I / 2, d);
                acc += hv * static_cast<double>(q8_codes[nib]) * lut_scale *
                       scale_of(false, e, static_cast<size_t>(row) * (I / Block) + d / Block);
            }
            ref_down.push_back(static_cast<float>(acc));
            actual_down.push_back(down_q8[(static_cast<size_t>(tok) * TopK + rank) * H + row]);
        }
    }

    // 4. Report and verify.
    const std::string tag = "W4A8 tokens=" + std::to_string(tokens) + (nested ? " nested" : " plain");
    q4e_w4a8_check_close_rel(actual_h, ref_h, tag + " gate/up vs f64 W4A8 reference", 2e-2f, 2e-2f);
    check_close_f32(actual_down, ref_down, tag + " down vs f64 W4A8 reference", 1e-3f);

    float h_max = 0.0f, h_nrmse = 0.0f, h_cos = 0.0f;
    q4e_w4a8_stats(q4e_w4a8_to_f32(h_q8), q4e_w4a8_to_f32(h_bf16), h_max, h_nrmse, h_cos);
    float d_max = 0.0f, d_nrmse = 0.0f, d_cos = 0.0f;
    q4e_w4a8_stats(down_q8, down_bf16, d_max, d_nrmse, d_cos);
    std::cout << "[W4A8] " << tag << " gate/up: max_abs=" << h_max << " NRMSE=" << h_nrmse
              << " cosine=" << h_cos << " | down: max_abs=" << d_max << " NRMSE=" << d_nrmse
              << " cosine=" << d_cos << "\n";
    if (!(h_cos >= 0.995f) || !(d_cos >= 0.995f))
        throw std::runtime_error(tag + " Q8/BF16 cosine below the 0.995 tolerance contract");
    if (!(d_nrmse <= 0.05f))
        throw std::runtime_error(tag + " Q8/BF16 down NRMSE above 0.05");

    if (timing_iters <= 0) return;

    // 5. Same-shape timing: each kernel alone, then the two pipelines that the
    //    executor would submit (the Q8 pair includes its activation pass).
    auto bf16_gate_up_once = [&] {
        ops::moe_decode_gate_up_swiglu_grouped(
            ctx, slots_ptr, nullptr, x_dev.data_as<bf16>(), h_bf16_dev.data_as<bf16>(), tasks,
            H, I, Block, nullptr, token_ptr, nested);
    };
    auto bf16_down_once = [&] {
        ops::moe_decode_down_to_rank_f32(
            ctx, slots_ptr, rank_ptr, h_bf16_dev.data_as<bf16>(), down_bf16_dev.data_as<float>(),
            tasks, TopK, I, H, Block, token_ptr, nullptr, nullptr, nested);
    };
    auto q8_quantize_once = [&] {
        ops::quantize_moe_input_q8_64_rows(ctx, x_dev.data_as<bf16>(),
            reinterpret_cast<int8_t*>(xq_dev.data()), xs_dev.data_as<float>(), tokens, H);
    };
    auto q8_gate_up_once = [&] {
        ops::moe_decode_gate_up_swiglu_grouped_q8(
            ctx, slots_ptr, reinterpret_cast<const int8_t*>(xq_dev.data()), xs_dev.data_as<float>(),
            h_q8_dev.data_as<bf16>(), reinterpret_cast<int8_t*>(hq_dev.data()), hs_dev.data_as<float>(),
            tasks, H, I, token_ptr, nullptr, nested);
    };
    auto q8_down_once = [&] {
        ops::moe_decode_down_to_rank_f32_q8(
            ctx, slots_ptr, rank_ptr, reinterpret_cast<const int8_t*>(hq_dev.data()),
            hs_dev.data_as<float>(), down_q8_dev.data_as<float>(), tasks, TopK, I, H, token_ptr,
            nullptr, nested);
    };
    auto bf16_pair_once = [&] { bf16_gate_up_once(); bf16_down_once(); };
    auto q8_pair_once = [&] { q8_quantize_once(); q8_gate_up_once(); q8_down_once(); };

    auto measure = [&](auto&& submit) -> double {
        submit();
        ctx.synchronize();
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            ctx.synchronize();
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < timing_iters; ++i) submit();
            ctx.synchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count() /
                                      timing_iters);
        }
        return best;
    };
    const double t_bf16_gu = measure(bf16_gate_up_once);
    const double t_bf16_dn = measure(bf16_down_once);
    const double t_q8_quant = measure(q8_quantize_once);
    const double t_q8_gu = measure(q8_gate_up_once);
    const double t_q8_dn = measure(q8_down_once);
    const double t_bf16_pair = measure(bf16_pair_once);
    const double t_q8_pair = measure(q8_pair_once);
    std::cout << "[W4A8 timing] " << tag << " iters=" << timing_iters
              << " | pair bf16=" << t_bf16_pair << "us q8=" << t_q8_pair
              << "us ratio=" << (t_bf16_pair / t_q8_pair)
              << " | gate_up bf16=" << t_bf16_gu << "us q8=" << t_q8_gu << "us"
              << " | down bf16=" << t_bf16_dn << "us q8=" << t_q8_dn << "us"
              << " | q8_quantize=" << t_q8_quant << "us"
              << " | kernel_sum bf16=" << (t_bf16_gu + t_bf16_dn)
              << "us q8=" << (t_q8_quant + t_q8_gu + t_q8_dn) << "us\n";
}

[[maybe_unused]] static void test_q4e_w4a8_adaptive_decode(Context& ctx, bool timing) {
    std::cout << "\n=== B2 adaptive NF4 x Q8 (W4A8) decode kernels ===\n";
    run_q4e_w4a8_case(ctx, 1, false, timing ? 200 : 0);
    run_q4e_w4a8_case(ctx, 4, false, timing ? 100 : 0);
    run_q4e_w4a8_case(ctx, 1, true, 0);
}
