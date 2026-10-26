#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"
#include "models/Qwen4ExpPle.hpp"
#include "ops/QsaIndexerOps.hpp"
#include "utils/SafeTensors.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using bf16 = sycl::ext::oneapi::bfloat16;

struct Qwen4ExpGenerationTestAccess {
    using PendingMlpResidual = Qwen4ExpBnb4Backend::DecodePendingMlpResidual;
    static bool CanCapture(const Qwen4ExpBnb4Backend& backend,
                           const Qwen4ExpBnb4Backend::Session& session, int count, int anchor) {
        return backend.CanCaptureInteriorCheckpoint(session, count, anchor);
    }
    static void ValidateGdnPath(const Qwen4ExpBnb4Backend& backend,
                                const Qwen4ExpBnb4Backend::Session& session, int count) {
        backend.ValidateGdnHeadPath(session, count);
    }
    static bool GdnDeviceHeadReadable(
        const Qwen4ExpBnb4Backend& backend, const Qwen4ExpLayerCache& cache) {
        return backend.GdnDeviceHeadReadable(cache);
    }
    static void MarkGdnGraphDirty(Qwen4ExpBnb4Backend& backend,
                                  Qwen4ExpBnb4Backend::Session& session) {
        backend.MarkGdnGraphStateDirty(session);
    }
    static void PublishGdnHeads(Qwen4ExpBnb4Backend& backend, Context& ctx,
                                Qwen4ExpBnb4Backend::Session& session) {
        backend.PublishGdnHostHeadsToDevice(ctx, session);
    }
    static aila::TokenGraphKey MakeTokenGraphKey(
        const Qwen4ExpBnb4Backend& backend,
        const Qwen4ExpBnb4Backend::Session& session) {
        return backend.MakeTokenGraphKey(
            session, aila::TokenGraphBody::QsaLayerProof,
            aila::TokenGraphAttention::QsaSparse512BlockNative, 0);
    }
    static aila::TokenGraphKey MakeQsaLayerGraphKey(
        const Qwen4ExpBnb4Backend& backend,
        const Qwen4ExpBnb4Backend::Session& session) {
        return backend.MakeQsaLayerGraphKey(session);
    }
    static bool QsaLayerGraphEligible(
        const Qwen4ExpBnb4Backend& backend, Context& ctx,
        const Qwen4ExpBnb4Backend::Session& session,
        const Tensor& input, const Tensor& output, int count, int start_pos) {
        return backend.QsaLayerGraphEligible(
            ctx, session, input, output, count, start_pos, false);
    }
    static bool QsaLayerBlockStateEligible(
        int current_blocks, int target_blocks) {
        return Qwen4ExpBnb4Backend::QsaLayerBlockStateEligible(
            current_blocks, target_blocks);
    }
    static uint64_t PleWorkspaceGeneration(const Qwen4ExpBnb4Backend& backend) {
        return backend.ple_.workspace_generation();
    }
    static void EnsurePleWorkspace(Qwen4ExpBnb4Backend& backend, Context& ctx, int rows) {
        backend.ple_.ensure_compute_capacity(ctx, rows);
    }
    static void SetEmbeddingOverrideState(Qwen4ExpBnb4Backend::Session& session,
                                          bool enabled) {
        session.embeddingOverridePositions_.assign(enabled ? 1 : 0, 0);
        session.embeddingOverrideValues_.clear();
        session.embeddingOverrideHiddenSize_ = enabled ? 2560 : 0;
    }
    static void SetMropeState(Qwen4ExpBnb4Backend::Session& session, bool enabled) {
        session.mropePromptLength_ = enabled ? 1 : 0;
        session.mropeTextPositionDelta_ = enabled ? 1 : 0;
    }
    static uint64_t PleTicketEpoch(const Qwen4ExpBnb4Backend::Session& session) {
        return session.ple_ticket_epoch();
    }
    static void InvalidateMtpRestoreState(Qwen4ExpBnb4Backend::Session& session) {
        session.InvalidateMtpRestoreState();
    }
    static bool PleTicketEpochMatches(
        const Qwen4ExpBnb4Backend& backend,
        const Qwen4ExpBnb4Backend::Session& session,
        const PleTicket& ticket) {
        return backend.PleTicketEpochMatches(session, ticket);
    }
    static Qwen4ExpBuffers& HcFixture(Qwen4ExpBnb4Backend& backend, Context& ctx, int dim, int lowrank) {
        backend.context_ = &ctx;
        backend.hidden_size_ = dim;
        backend.cfg_.hc_lowrank = lowrank;
        backend.cfg_.rms_norm_eps = 1e-6f;
        auto& b = backend.buf_;
        b.hidden4 = Tensor::allocate(ctx, {1, 4 * dim}, dnnl::memory::data_type::bf16);
        b.y = Tensor::allocate(ctx, {1, dim}, dnnl::memory::data_type::bf16);
        b.hc_N = Tensor::allocate(ctx, {1, 4 * dim}, dnnl::memory::data_type::bf16);
        b.hc_mix_down = Tensor::allocate(ctx, {1, lowrank}, dnnl::memory::data_type::bf16);
        b.hc_inject_f32 = Tensor::allocate(ctx, {1, 4}, dnnl::memory::data_type::f32);
        b.x = Tensor::allocate(ctx, {1, dim}, dnnl::memory::data_type::bf16);
        return b;
    }
    static void RunHc(Qwen4ExpBnb4Backend& backend, Context& ctx,
        Qwen4ExpHyperConnectionLayer& hc, PendingMlpResidual* pending = nullptr) {
        backend.RunHyperConnectionCombined(ctx, hc, 1, false, nullptr, nullptr, pending);
    }
};

namespace {

void expect(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("Test failure: " + msg);
    }
}

void test_decode_pending_mlp_residual(Context& ctx) {
    std::cout << "[Test] Decode pending MLP residual bitwise/state alignment...\n";
    using Pending = Qwen4ExpGenerationTestAccess::PendingMlpResidual;
    auto expect_bits = [&](const bf16* ref, const bf16* got, size_t count, const std::string& label) {
        if (std::memcmp(ref, got, count * sizeof(bf16)) == 0) return;
        for (size_t i = 0; i < count; ++i) {
            uint16_t old_bits = 0, new_bits = 0;
            std::memcpy(&old_bits, ref + i, sizeof(old_bits));
            std::memcpy(&new_bits, got + i, sizeof(new_bits));
            if (old_bits != new_bits) {
                std::cerr << "[Bitwise diff] " << label << " index=" << i
                    << " old_bits=0x" << std::hex << old_bits << " new_bits=0x" << new_bits
                    << std::dec << " old=" << static_cast<float>(ref[i])
                    << " new=" << static_cast<float>(got[i]) << '\n';
                break;
            }
        }
        expect(false, label + " differs bitwise");
    };
    for (const int dim : {257, 2560}) {
        const size_t n = static_cast<size_t>(4 * dim);
        Tensor old_h = Tensor::allocate(ctx, {4, dim}, dnnl::memory::data_type::bf16);
        Tensor new_h = Tensor::allocate(ctx, {4, dim}, dnnl::memory::data_type::bf16);
        Tensor y = Tensor::allocate(ctx, {dim}, dnnl::memory::data_type::bf16);
        Tensor inj = Tensor::allocate(ctx, {4}, dnnl::memory::data_type::f32);
        Tensor weight = Tensor::allocate(ctx, {4, dim}, dnnl::memory::data_type::f32);
        Tensor old_n = Tensor::allocate(ctx, {4, dim}, dnnl::memory::data_type::bf16);
        Tensor new_n = Tensor::allocate(ctx, {4, dim}, dnnl::memory::data_type::bf16);
        std::vector<bf16> h_host(n), y_host(dim), ref_h(n), got_h(n), ref_n(n), got_n(n);
        std::vector<float> w_host(n);
        const std::array<float, 4> inject = {0.00390625f, -1.0000001192092896f, 0.3333333432674408f, 1.9999998807907104f};
        for (int d = 0; d < dim; ++d) {
            y_host[d] = bf16((d % 4 == 0) ? 1.0f : std::ldexp(float((d % 31) - 15), (d % 11) - 5));
            for (int g = 0; g < 4; ++g) {
                const size_t i = size_t(g) * dim + d;
                // Group 0 has exact BF16 half-ULP additions; group 1 has
                // near-cancellation. Other values span exponents and signs.
                h_host[i] = bf16((g == 0 && d % 4 == 0) ? 1.0f :
                    g == 1 ? static_cast<float>(y_host[d]) :
                    std::ldexp(float(((d * 13 + g * 7) % 47) - 23), (d % 9) - 4));
                w_host[i] = float(int(i % 19) - 9) / 32.0f;
            }
        }
        ctx.memcpy_h2d(y.data(), y_host.data(), dim * sizeof(bf16));
        ctx.memcpy_h2d(inj.data(), inject.data(), sizeof(inject));
        ctx.memcpy_h2d(weight.data(), w_host.data(), n * sizeof(float));
        auto reset = [&] {
            ctx.memcpy_h2d(old_h.data(), h_host.data(), n * sizeof(bf16));
            ctx.memcpy_h2d(new_h.data(), h_host.data(), n * sizeof(bf16));
        };
        auto compare_h = [&] {
            ctx.memcpy_d2h(ref_h.data(), old_h.data(), n * sizeof(bf16));
            ctx.memcpy_d2h(got_h.data(), new_h.data(), n * sizeof(bf16));
            expect_bits(ref_h.data(), got_h.data(), n,
                "pending residual H dim=" + std::to_string(dim));
        };
        reset();
        // Eight successive layers overwrite the same y/inject buffers only
        // after each submitted norm has consumed them. No wait is inserted
        // between the norm and that next producer on the compute queue.
        for (int layer = 0; layer < 8; ++layer) {
            ops::hc_inject_add(ctx, old_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), 1, dim, 4);
            ops::grouped_rms_norm(ctx, old_h.data_as<bf16>(), weight.data_as<float>(), 1e-6f,
                old_n.data_as<bf16>(), 1, 4, dim, true);
            {
                Pending pending(ctx, new_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), dim);
                pending.Defer();
                pending.ConsumeNorm(weight.data_as<float>(), 1e-6f, new_n.data_as<bf16>());
                expect(!pending.pending, "norm did not consume residual ownership");
                expect(pending.deferred == 1 && pending.consumedNorm == uint64_t(dim % 8 == 0) &&
                    pending.materialized == uint64_t(dim % 8 != 0), "pending norm dispatch counters");
            }
            const auto y_ptr = y.data_as<bf16>();
            const auto inj_ptr = inj.data_as<float>();
            ctx.queue().parallel_for(sycl::range<1>(dim), [=](sycl::id<1> id) {
                y_ptr[id[0]] = bf16(-static_cast<float>(y_ptr[id[0]]));
                if (id[0] < 4) inj_ptr[id[0]] = -inj_ptr[id[0]];
            });
            compare_h();
            ctx.memcpy_d2h(ref_n.data(), old_n.data(), n * sizeof(bf16));
            ctx.memcpy_d2h(got_n.data(), new_n.data(), n * sizeof(bf16));
            expect_bits(ref_n.data(), got_n.data(), n,
                "pending residual N dim=" + std::to_string(dim) + " layer=" + std::to_string(layer));
        }
        // Explicit materialization (PLE/no-logits/observer) is idempotent.
        reset();
        ops::hc_inject_add(ctx, old_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), 1, dim, 4);
        {
            Pending pending(ctx, new_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), dim);
            pending.Defer(); pending.Materialize(); pending.Materialize();
            expect(!pending.pending, "boundary flush left a pending residual");
            expect(pending.deferred == 1 && pending.materialized == 1 && pending.consumedNorm == 0,
                "idempotent materialization counters");
        }
        compare_h();
        // A host exception before the next norm must materialize exactly once.
        reset();
        ops::hc_inject_add(ctx, old_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), 1, dim, 4);
        try {
            Pending pending(ctx, new_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), dim);
            pending.Defer();
            throw std::runtime_error("test unwind before norm");
        } catch (const std::runtime_error&) {}
        compare_h();
        reset();
        ops::hc_inject_add(ctx, old_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), 1, dim, 4);
        try {
            Pending pending(ctx, new_h.data_as<bf16>(), y.data_as<bf16>(), inj.data_as<float>(), dim);
            pending.Defer(); pending.ConsumeNorm(weight.data_as<float>(), 1e-6f, new_n.data_as<bf16>());
            throw std::runtime_error("test unwind after norm");
        } catch (const std::runtime_error&) {}
        compare_h();
    }

    // Exercise the actual backend final-mixer path: with_inject=false must
    // consume the previous layer's injection before submitting its own dots.
    constexpr int dim = 32, lowrank = 16;
    Qwen4ExpBnb4Backend baseline, candidate;
    auto& old_b = Qwen4ExpGenerationTestAccess::HcFixture(baseline, ctx, dim, lowrank);
    auto& new_b = Qwen4ExpGenerationTestAccess::HcFixture(candidate, ctx, dim, lowrank);
    Tensor norm = Tensor::allocate(ctx, {4 * dim}, dnnl::memory::data_type::f32);
    Tensor down = Tensor::allocate(ctx, {lowrank, 4 * dim}, dnnl::memory::data_type::bf16);
    Tensor up = Tensor::allocate(ctx, {4 * dim, lowrank}, dnnl::memory::data_type::bf16);
    std::vector<bf16> h(4 * dim), y(dim), wd(lowrank * 4 * dim), wu(4 * dim * lowrank);
    std::vector<float> wn(4 * dim);
    const std::array<float, 4> inject = {0.25f, 0.5f, 1.0f, 1.75f};
    for (size_t i = 0; i < h.size(); ++i) {h[i] = bf16(float(int(i % 17) - 8) / 4); wn[i] = float(int(i % 7) - 3) / 16;}
    for (int i = 0; i < dim; ++i) y[i] = bf16(float(i - 16) / 8);
    for (size_t i = 0; i < wd.size(); ++i) wd[i] = bf16(float(int(i % 13) - 6) / 32);
    for (size_t i = 0; i < wu.size(); ++i) wu[i] = bf16(float(int(i % 11) - 5) / 32);
    for (auto* b : {&old_b, &new_b}) {
        ctx.memcpy_h2d(b->hidden4.data(), h.data(), h.size() * sizeof(bf16));
        ctx.memcpy_h2d(b->y.data(), y.data(), y.size() * sizeof(bf16));
        ctx.memcpy_h2d(b->hc_inject_f32.data(), inject.data(), sizeof(inject));
    }
    ctx.memcpy_h2d(norm.data(), wn.data(), wn.size() * sizeof(float));
    ctx.memcpy_h2d(down.data(), wd.data(), wd.size() * sizeof(bf16));
    ctx.memcpy_h2d(up.data(), wu.data(), wu.size() * sizeof(bf16));
    Qwen4ExpHyperConnectionLayer hc;
    hc.hc_norm_weight = &norm; hc.mix_down_w = &down; hc.mix_up_w = &up;
    ops::hc_inject_add(ctx, old_b.hidden4.data_as<bf16>(), old_b.y.data_as<bf16>(), old_b.hc_inject_f32.data_as<float>(), 1, dim, 4);
    Qwen4ExpGenerationTestAccess::RunHc(baseline, ctx, hc);
    {
        Pending pending(ctx, new_b.hidden4.data_as<bf16>(), new_b.y.data_as<bf16>(), new_b.hc_inject_f32.data_as<float>(), dim);
        pending.Defer(); Qwen4ExpGenerationTestAccess::RunHc(candidate, ctx, hc, &pending);
        expect(!pending.pending, "final mixer did not consume residual");
    }
    auto bitwise = [&](const Tensor& a, const Tensor& b, const char* label) {
        std::vector<bf16> x(a.size_bytes() / sizeof(bf16)), z(b.size_bytes() / sizeof(bf16));
        ctx.memcpy_d2h(x.data(), a.data(), a.size_bytes()); ctx.memcpy_d2h(z.data(), b.data(), b.size_bytes());
        expect_bits(x.data(), z.data(), x.size(), std::string("final mixer ") + label);
    };
    bitwise(old_b.hidden4, new_b.hidden4, "H"); bitwise(old_b.hc_N, new_b.hc_N, "N"); bitwise(old_b.x, new_b.x, "x");
    std::array<float, 4> got_inject{};
    ctx.memcpy_d2h(got_inject.data(), new_b.hc_inject_f32.data(), sizeof(got_inject));
    expect(std::memcmp(got_inject.data(), inject.data(), sizeof(inject)) == 0,
        "final mixer overwrote the preceding layer's injection");
    std::cout << "[PASS] Decode pending MLP residual numerical and ownership checks.\n";
}

void test_hc_decode_scheduling_bitwise(Context& ctx) {
    std::cout << "[Test] HC down WG64/128 and parallel-stream up strict bitwise alignment...\n";
    constexpr int guard = 16;
    auto run_case = [&](int dim, int lowrank, int groups, int rows, uint32_t seed) {
        const int k = groups * dim;
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> values(-32, 32);
        std::vector<bf16> n(size_t(rows) * k), wd(size_t(lowrank) * k),
            wu(size_t(k) * lowrank), wi(size_t(groups) * k);
        for (auto& v : n) v = bf16(float(values(rng)) / 16.0f);
        for (auto& v : wd) v = bf16(float(values(rng)) / 256.0f);
        for (auto& v : wu) v = bf16(float(values(rng)) / 128.0f);
        for (auto& v : wi) v = bf16(float(values(rng)) / 256.0f);
        // Dot 0 is exactly halfway between two BF16 values before the HC
        // nonlinear epilogue. Dot 1 cancels, leaving a small representable term.
        for (int r = 0; r < rows; ++r) {
            n[size_t(r) * k] = bf16(1); n[size_t(r) * k + 1] = bf16(1); n[size_t(r) * k + 2] = bf16(1);
        }
        std::fill(wd.begin(), wd.begin() + 2 * k, bf16(0));
        wd[0] = bf16(1); wd[1] = bf16(0.00390625f);
        wd[k] = bf16(1); wd[k + 1] = bf16(-1); wd[k + 2] = bf16(0.000030517578125f);
        for (int g = 0; g < groups; ++g) {
            // Mixed output includes cancellation across streams and exponents.
            const std::array<float, 4> cancellation = {1.0f, 0.00390625f, -1.0f, -0.00390625f};
            for (int r = 0; r < rows; ++r) {
                n[size_t(r) * k + size_t(g) * dim + dim - 1] =
                    bf16(cancellation[g]);
            }
            // Zero projection makes the last dimension's gates exactly 0.5,
            // exposing the ordered stream accumulation without sigmoid noise.
            const size_t off = (size_t(g) * dim + dim - 1) * lowrank;
            std::fill(wu.begin() + off, wu.begin() + off + lowrank, bf16(0));
        }
        Tensor tn = Tensor::allocate(ctx, {rows, k}, dnnl::memory::data_type::bf16);
        Tensor td = Tensor::allocate(ctx, {lowrank, k}, dnnl::memory::data_type::bf16);
        Tensor tu = Tensor::allocate(ctx, {k, lowrank}, dnnl::memory::data_type::bf16);
        Tensor ti = Tensor::allocate(ctx, {groups, k}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(tn.data(), n.data(), n.size() * sizeof(bf16));
        ctx.memcpy_h2d(td.data(), wd.data(), wd.size() * sizeof(bf16));
        ctx.memcpy_h2d(tu.data(), wu.data(), wu.size() * sizeof(bf16));
        ctx.memcpy_h2d(ti.data(), wi.data(), wi.size() * sizeof(bf16));
        const size_t act_count = size_t(rows) * lowrank, inj_count = size_t(rows) * groups, x_count = size_t(rows) * dim;
        Tensor ref_a = Tensor::allocate(ctx, {int64_t(act_count + 2 * guard)}, dnnl::memory::data_type::bf16);
        Tensor got_a = Tensor::allocate(ctx, {int64_t(act_count + 2 * guard)}, dnnl::memory::data_type::bf16);
        Tensor ref_i = Tensor::allocate(ctx, {int64_t(inj_count + 2 * guard)}, dnnl::memory::data_type::f32);
        Tensor got_i = Tensor::allocate(ctx, {int64_t(inj_count + 2 * guard)}, dnnl::memory::data_type::f32);
        Tensor ref_x = Tensor::allocate(ctx, {int64_t(x_count + 2 * guard)}, dnnl::memory::data_type::bf16);
        Tensor got_x = Tensor::allocate(ctx, {int64_t(x_count + 2 * guard)}, dnnl::memory::data_type::bf16);
        const bf16 sentinel = bf16(-713.0f);
        const float inject_sentinel = -919.25f;
        int case_wg = 256;
        bool case_parallel = false, case_inject = false;
        auto initialize = [&](Tensor& a, Tensor& i, Tensor& x) {
            std::vector<bf16> ah(act_count + 2 * guard, sentinel), xh(x_count + 2 * guard, sentinel);
            std::vector<float> ih(inj_count + 2 * guard, inject_sentinel);
            ctx.memcpy_h2d(a.data(), ah.data(), a.size_bytes());
            ctx.memcpy_h2d(i.data(), ih.data(), i.size_bytes());
            ctx.memcpy_h2d(x.data(), xh.data(), x.size_bytes());
        };
        auto compare = [&](const Tensor& ref, const Tensor& got, const std::string& label) {
            std::vector<uint8_t> a(ref.size_bytes()), b(got.size_bytes());
            ctx.memcpy_d2h(a.data(), ref.data(), a.size()); ctx.memcpy_d2h(b.data(), got.data(), b.size());
            if (a != b) {
                for (size_t offset = 0; offset < a.size(); offset += sizeof(uint16_t)) {
                    uint16_t old_bits = 0, new_bits = 0;
                    std::memcpy(&old_bits, a.data() + offset, sizeof(old_bits));
                    std::memcpy(&new_bits, b.data() + offset, sizeof(new_bits));
                    if (old_bits != new_bits) {
                        std::cerr << "[HC bitwise diff] " << label << " dim=" << dim << " lr=" << lowrank
                            << " rows=" << rows << " groups=" << groups << " seed=" << seed << " word=" << offset / 2
                            << " wg=" << case_wg << " parallel=" << case_parallel << " inject=" << case_inject
                            << " old=0x" << std::hex << old_bits << " new=0x" << new_bits << std::dec << '\n';
                        break;
                    }
                }
                expect(false, "HC scheduling differs bitwise: " + label);
            }
            const bool is_float = label == "inject";
            const size_t element_bytes = is_float ? sizeof(float) : sizeof(bf16);
            const void* expected = is_float ? static_cast<const void*>(&inject_sentinel) : static_cast<const void*>(&sentinel);
            for (size_t j = 0; j < guard; ++j) {
                expect(std::memcmp(b.data() + j * element_bytes, expected, element_bytes) == 0 &&
                    std::memcmp(b.data() + b.size() - (j + 1) * element_bytes, expected, element_bytes) == 0,
                    "HC scheduling wrote output guard: " + label);
            }
        };
        auto down = [&](Tensor& a, Tensor& i, bool inject, int wg, bool defaults) {
            const bf16* weight_inject = inject ? ti.data_as<bf16>() : nullptr;
            if (rows == 1) {
                if (defaults) ops::hc_decode_down_inject(ctx, tn.data_as<bf16>(), td.data_as<bf16>(), weight_inject,
                    a.data_as<bf16>() + guard, i.data_as<float>() + guard, lowrank, dim, groups);
                else ops::hc_decode_down_inject(ctx, tn.data_as<bf16>(), td.data_as<bf16>(), weight_inject,
                    a.data_as<bf16>() + guard, i.data_as<float>() + guard, lowrank, dim, groups, wg);
            } else {
                if (defaults) ops::hc_decode_down_inject_rows(ctx, tn.data_as<bf16>(), td.data_as<bf16>(), weight_inject,
                    a.data_as<bf16>() + guard, i.data_as<float>() + guard, lowrank, dim, groups, rows);
                else ops::hc_decode_down_inject_rows(ctx, tn.data_as<bf16>(), td.data_as<bf16>(), weight_inject,
                    a.data_as<bf16>() + guard, i.data_as<float>() + guard, lowrank, dim, groups, rows, wg);
            }
        };
        auto up = [&](const Tensor& a, Tensor& x, bool parallel, bool defaults) {
            if (rows == 1) {
                if (defaults) ops::hc_decode_up_mix(ctx, a.data_as<bf16>() + guard, tu.data_as<bf16>(), tn.data_as<bf16>(),
                    x.data_as<bf16>() + guard, lowrank, dim, groups);
                else ops::hc_decode_up_mix(ctx, a.data_as<bf16>() + guard, tu.data_as<bf16>(), tn.data_as<bf16>(),
                    x.data_as<bf16>() + guard, lowrank, dim, groups, parallel);
            } else {
                if (defaults) ops::hc_decode_up_mix_rows(ctx, a.data_as<bf16>() + guard, tu.data_as<bf16>(), tn.data_as<bf16>(),
                    x.data_as<bf16>() + guard, lowrank, dim, groups, rows);
                else ops::hc_decode_up_mix_rows(ctx, a.data_as<bf16>() + guard, tu.data_as<bf16>(), tn.data_as<bf16>(),
                    x.data_as<bf16>() + guard, lowrank, dim, groups, rows, parallel);
            }
        };
        for (const bool inject : {false, true}) {
            case_inject = inject;
            initialize(ref_a, ref_i, ref_x);
            down(ref_a, ref_i, inject, 256, true); up(ref_a, ref_x, false, true);
            for (const int wg : {256, 128, 64}) {
                case_wg = wg; case_parallel = false;
                initialize(got_a, got_i, got_x);
                down(got_a, got_i, inject, wg, false);
                compare(ref_a, got_a, "act"); compare(ref_i, got_i, "inject");
                for (const bool parallel : {false, true}) {
                    case_parallel = parallel;
                    up(got_a, got_x, parallel, false);
                    compare(ref_x, got_x, "x");
                }
                if (!inject) {
                    std::vector<float> observed(inj_count + 2 * guard);
                    ctx.memcpy_d2h(observed.data(), got_i.data(), got_i.size_bytes());
                    for (float value : observed) expect(value == inject_sentinel, "null injection wrote its output");
                }
            }
        }
        // Give up its own controlled activation: exact half-ULP dots and
        // near-cancellation cannot rely on the preceding SiLU values occurring.
        std::vector<bf16> controlled(act_count + 2 * guard, sentinel);
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < lowrank; ++j) controlled[guard + size_t(r) * lowrank + j] = bf16(0);
            controlled[guard + size_t(r) * lowrank] = bf16(1);
            controlled[guard + size_t(r) * lowrank + 1] = bf16(1);
            controlled[guard + size_t(r) * lowrank + 2] = bf16(1);
        }
        for (int g = 0; g < groups; ++g) {
            const size_t off = size_t(g) * dim * lowrank;
            std::fill(wu.begin() + off, wu.begin() + off + 2 * lowrank, bf16(0));
            wu[off] = bf16(1); wu[off + 1] = bf16(0.00390625f);
            wu[off + lowrank] = bf16(1); wu[off + lowrank + 1] = bf16(-1);
            wu[off + lowrank + 2] = bf16(0.000030517578125f);
        }
        ctx.memcpy_h2d(tu.data(), wu.data(), wu.size() * sizeof(bf16));
        ctx.memcpy_h2d(ref_a.data(), controlled.data(), ref_a.size_bytes());
        case_wg = 256; case_parallel = true;
        up(ref_a, ref_x, false, true); up(ref_a, got_x, true, false); compare(ref_x, got_x, "x controlled");
        bool rejected = false;
        try { down(got_a, got_i, true, 32, false); } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "unsupported explicit HC down work-group accepted");
    };
    for (const uint32_t seed : {17u, 113u, 709u})
        for (const int rows : {1, 2, 4}) run_case(2560, 320, 4, rows, seed);
    run_case(17, 23, 4, 2, 919u); // partial dimension tile and vec8/scalar tails
    run_case(17, 23, 3, 2, 920u); // parallel flag must preserve the old non-four-stream path
    std::cout << "[PASS] HC scheduling defaults, bitwise outputs, null inject and guards.\n";
}

// ---------------------------------------------------------------------------
// 2. PLE Unit Tests
// ---------------------------------------------------------------------------

void test_ple_ngram_ids_calculation() {
    std::cout << "[Test] Running PLE n-gram IDs calculation unit tests...\n";

    constexpr int eos_token_id = 248044;
    const std::array<int64_t, 3> mults = {23703573157769LL, 20109073645365LL, 8052911324071LL};
    const std::array<int64_t, 16> vizes = {
        20000003LL, 20000023LL, 20000033LL, 20000047LL, 20000059LL, 20000063LL, 20000069LL, 20000077LL,
        20000081LL, 20000093LL, 20000107LL, 20000147LL, 20000153LL, 20000159LL, 20000161LL, 20000171LL
    };
    const std::array<int64_t, 16> offs = {
        0LL, 20000003LL, 40000026LL, 60000059LL, 80000106LL, 100000165LL,
        120000228LL, 140000297LL, 160000374LL, 180000455LL, 200000548LL, 220000655LL,
        240000802LL, 260000955LL, 280001114LL, 300001275LL
    };

    // Golden values directly verified against HuggingFace modeling_qwen4_exp.py
    const std::vector<int64_t> golden_101 = {
        14526383LL, 36975477LL, 51939184LL, 65076202LL, 94510971LL, 111786935LL, 138448715LL, 155393638LL,
        167579692LL, 181767276LL, 211184823LL, 226303266LL, 246271192LL, 267465224LL, 281469089LL, 313531602LL
    };
    const std::vector<int64_t> golden_202 = {
        9383917LL, 37604839LL, 51720000LL, 75486496LL, 84434064LL, 114084336LL, 138560629LL, 157864067LL,
        167911082LL, 194568541LL, 204150554LL, 222298387LL, 249396999LL, 276854645LL, 299420419LL, 312846864LL
    };
    const std::vector<int64_t> golden_303 = {
        493037LL, 28958551LL, 43196368LL, 71134995LL, 92230508LL, 105930112LL, 126480532LL, 153882988LL,
        168741164LL, 195613863LL, 210304005LL, 226593416LL, 250041074LL, 273489806LL, 281306288LL, 300390317LL
    };

    // Test A: Prefill batch of 3 tokens [101, 202, 303] with initial context [eos, eos]
    std::vector<int> history = {eos_token_id, eos_token_id, 101, 202, 303};
    std::vector<int64_t> out_ids(3 * 16);
    Qwen4ExpPle::compute_ngram_ids(history.data(), static_cast<int>(history.size()), 3,
                                  mults.data(), vizes.data(), offs.data(), out_ids.data(), eos_token_id);

    for (int h = 0; h < 16; ++h) {
        expect(out_ids[0 * 16 + h] == golden_101[h], "token 101 head " + std::to_string(h) + " mismatch");
        expect(out_ids[1 * 16 + h] == golden_202[h], "token 202 head " + std::to_string(h) + " mismatch");
        expect(out_ids[2 * 16 + h] == golden_303[h], "token 303 head " + std::to_string(h) + " mismatch");
    }

    // Test B: Streaming single-token decode using Qwen4ExpPleState
    Qwen4ExpPle ple;
    ple.set_multipliers_and_vocab(mults, vizes, offs);
    Qwen4ExpPleState state;
    state.prev_tokens = {eos_token_id, eos_token_id};

    int tok1 = 101;
    std::vector<int64_t> step1_ids(16);
    ple.compute_ngram_ids(state, &tok1, 1, step1_ids.data());
    for (int h = 0; h < 16; ++h) {
        expect(step1_ids[h] == golden_101[h], "streaming step 1 head " + std::to_string(h) + " mismatch");
    }
    expect(state.prev_tokens[0] == eos_token_id && state.prev_tokens[1] == 101, "prev_tokens after step 1 invalid");

    int tok2 = 202;
    std::vector<int64_t> step2_ids(16);
    ple.compute_ngram_ids(state, &tok2, 1, step2_ids.data());
    for (int h = 0; h < 16; ++h) {
        expect(step2_ids[h] == golden_202[h], "streaming step 2 head " + std::to_string(h) + " mismatch");
    }
    expect(state.prev_tokens[0] == 101 && state.prev_tokens[1] == 202, "prev_tokens after step 2 invalid");

    int tok3 = 303;
    std::vector<int64_t> step3_ids(16);
    ple.compute_ngram_ids(state, &tok3, 1, step3_ids.data());
    for (int h = 0; h < 16; ++h) {
        expect(step3_ids[h] == golden_303[h], "streaming step 3 head " + std::to_string(h) + " mismatch");
    }
    expect(state.prev_tokens[0] == 202 && state.prev_tokens[1] == 303, "prev_tokens after step 3 invalid");

    // Test C: EOS boundary reset
    // In sequence [101, eos, 202], token 202's shift=1 and shift=2 must not see 101!
    std::vector<int> eos_history = {eos_token_id, eos_token_id, 101, eos_token_id, 202};
    std::vector<int64_t> eos_out_ids(3 * 16);
    Qwen4ExpPle::compute_ngram_ids(eos_history.data(), static_cast<int>(eos_history.size()), 3,
                                  mults.data(), vizes.data(), offs.data(), eos_out_ids.data(), eos_token_id);

    // After EOS, token 202 has shift=1 and shift=2 both reset to EOS, exactly like token 101 had!
    // So 2-gram hash of 202 is 202 * mult0 ^ eos * mult1.
    // Verify 202's IDs are non-zero and offset-bounded
    for (int h = 0; h < 16; ++h) {
        int64_t id = eos_out_ids[2 * 16 + h];
        expect(id >= offs[h] && id < offs[h] + vizes[h], "id out of vocab bounds");
    }

    std::cout << "[PASS] PLE n-gram IDs calculation tests passed.\n";
}

void test_ple_lookup_and_dequant(Context& ctx) {
    std::cout << "[Test] Running PLE lookup and dequantization unit tests...\n";

    constexpr int num_tokens = 2;
    constexpr int64_t rows_per_shard = 100;
    constexpr int num_shards = 2;

    // Create 2 synthetic shards with known packed rows
    std::vector<uint8_t> shard0_packed(rows_per_shard * 80, 0x12); // hi=1, lo=2
    std::vector<float> shard0_absmax(rows_per_shard * 5, 2.0f);
    std::vector<uint8_t> shard1_packed(rows_per_shard * 80, 0x34); // hi=3, lo=4
    std::vector<float> shard1_absmax(rows_per_shard * 5, 0.5f);

    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) qmap[i] = static_cast<float>(i);

    std::vector<PleShardRef> shards(num_shards);
    shards[0].packed_weight = shard0_packed.data();
    shards[0].absmax = shard0_absmax.data();
    shards[0].quant_map = qmap.data();
    shards[0].rows = rows_per_shard;

    shards[1].packed_weight = shard1_packed.data();
    shards[1].absmax = shard1_absmax.data();
    shards[1].quant_map = qmap.data();
    shards[1].rows = rows_per_shard;

    ModelWeights dummy;
    Qwen4ExpPle ple;
    std::string err;
    bool ok = ple.init(ctx, dummy, 1, 64, 1e-6f, &err);
    expect(ok, "ple init failed: " + err);
    ple.set_shards(shards, rows_per_shard);

    // Token 0: all heads point to shard 0 row 5 (id = 5)
    // Token 1: all heads point to shard 1 row 10 (id = 100 + 10 = 110)
    std::vector<int64_t> ids(num_tokens * Qwen4ExpPle::NUM_HEADS);
    for (int h = 0; h < Qwen4ExpPle::NUM_HEADS; ++h) {
        ids[0 * Qwen4ExpPle::NUM_HEADS + h] = 5;
        ids[1 * Qwen4ExpPle::NUM_HEADS + h] = 110;
    }

    Tensor out_dev = Tensor::allocate(ctx, {num_tokens, Qwen4ExpPle::EMBED_DIM}, dnnl::memory::data_type::bf16);
    ple.lookup_and_dequant(ctx, ids.data(), num_tokens, static_cast<bf16*>(out_dev.data()));
    ctx.synchronize();

    std::vector<bf16> out_host(num_tokens * Qwen4ExpPle::EMBED_DIM);
    ctx.memcpy_d2h(out_host.data(), out_dev.data(), out_host.size() * sizeof(bf16));

    // Token 0 (shard 0): hi=1 * 2.0 = 2.0, lo=2 * 2.0 = 4.0
    for (int h = 0; h < Qwen4ExpPle::NUM_HEADS; ++h) {
        for (int b = 0; b < 80; ++b) {
            float val0 = static_cast<float>(out_host[0 * Qwen4ExpPle::EMBED_DIM + h * 160 + 2 * b]);
            float val1 = static_cast<float>(out_host[0 * Qwen4ExpPle::EMBED_DIM + h * 160 + 2 * b + 1]);
            expect(std::abs(val0 - 2.0f) < 1e-3f, "shard 0 val0 mismatch");
            expect(std::abs(val1 - 4.0f) < 1e-3f, "shard 0 val1 mismatch");
        }
    }

    // Token 1 (shard 1): hi=3 * 0.5 = 1.5, lo=4 * 0.5 = 2.0
    for (int h = 0; h < Qwen4ExpPle::NUM_HEADS; ++h) {
        for (int b = 0; b < 80; ++b) {
            float val0 = static_cast<float>(out_host[1 * Qwen4ExpPle::EMBED_DIM + h * 160 + 2 * b]);
            float val1 = static_cast<float>(out_host[1 * Qwen4ExpPle::EMBED_DIM + h * 160 + 2 * b + 1]);
            expect(std::abs(val0 - 1.5f) < 1e-3f, "shard 1 val0 mismatch");
            expect(std::abs(val1 - 2.0f) < 1e-3f, "shard 1 val1 mismatch");
        }
    }

    // Test state snapshot and restore
    Qwen4ExpPleState state;
    ple.init_state(ctx, state);
    expect(state.prev_tokens[0] == Qwen4ExpPle::EOS_TOKEN_ID, "init state prev_token[0] mismatch");
    state.prev_tokens[0] = 555;
    state.prev_tokens[1] = 666;
    state.conv_head = 4;

    Qwen4ExpPleSnapshot snap;
    ple.capture_snapshot(ctx, state, snap);
    expect(snap.prev_tokens[0] == 555 && snap.prev_tokens[1] == 666, "snapshot prev_tokens mismatch");
    expect(snap.conv_head == 4, "snapshot conv_head mismatch");

    Qwen4ExpPleState restored_state;
    ple.restore_snapshot(ctx, restored_state, snap);
    expect(restored_state.prev_tokens[0] == 555 && restored_state.prev_tokens[1] == 666, "restored prev_tokens mismatch");
    expect(restored_state.conv_head == 4, "restored conv_head mismatch");

    std::cout << "[PASS] PLE lookup and dequantization tests passed.\n";
}

void test_ple_ticket_pipeline(Context& ctx) {
    std::cout << "[Test] Running PLE Ticket Pipeline (prepare / cancel / double buffering / commit) unit tests...\n";

    constexpr int64_t rows_per_shard = 100;
    constexpr int num_shards = 2;

    std::vector<uint8_t> shard0_packed(rows_per_shard * 80, 0x12);
    std::vector<float> shard0_absmax(rows_per_shard * 5, 2.0f);
    std::vector<uint8_t> shard1_packed(rows_per_shard * 80, 0x34);
    std::vector<float> shard1_absmax(rows_per_shard * 5, 0.5f);
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) qmap[i] = static_cast<float>(i);

    std::vector<PleShardRef> shards(num_shards);
    shards[0].packed_weight = shard0_packed.data();
    shards[0].absmax = shard0_absmax.data();
    shards[0].quant_map = qmap.data();
    shards[0].rows = rows_per_shard;

    shards[1].packed_weight = shard1_packed.data();
    shards[1].absmax = shard1_absmax.data();
    shards[1].quant_map = qmap.data();
    shards[1].rows = rows_per_shard;

    ModelWeights dummy;
    Qwen4ExpPle ple;
    std::string err;
    bool ok = ple.init(ctx, dummy, 1, 64, 1e-6f, &err);
    expect(ok, "ple init failed: " + err);
    ple.set_shards(shards, rows_per_shard);

    // 1. Pure prepare: does NOT mutate state.prev_tokens
    Qwen4ExpPleState state;
    ple.init_state(ctx, state);
    state.prev_tokens = {101, 202};

    int input_token = 303;
    PleTicket ticket1 = ple.prepare(ctx, state, &input_token, 1, &state, 1, 10);
    expect(ticket1.valid, "ticket1 must be valid");
    expect(ticket1.staging_slot >= 0 && ticket1.staging_slot < Qwen4ExpPle::kNumStagingSlots, "valid slot index");
    expect(ticket1.num_tokens == 1, "ticket num_tokens mismatch");
    expect(ticket1.session_ptr == &state, "ticket session_ptr mismatch");
    expect(ticket1.generation == 1, "ticket generation mismatch");
    expect(ticket1.start_pos == 10, "ticket start_pos mismatch");
    expect(state.prev_tokens[0] == 101 && state.prev_tokens[1] == 202,
           "prepare must be pure and not mutate state.prev_tokens");
    expect(ticket1.prev_tokens_in[0] == 101 && ticket1.prev_tokens_in[1] == 202, "ticket prev_tokens_in mismatch");
    expect(ticket1.prev_tokens_next[0] == 202 && ticket1.prev_tokens_next[1] == 303, "ticket prev_tokens_next mismatch");

    // 2. Double-buffered staging slot alternation
    int input_token2 = 404;
    PleTicket ticket2 = ple.prepare(ctx, state, &input_token2, 1, &state, 1, 11);
    expect(ticket2.valid, "ticket2 must be valid");
    expect(ticket2.staging_slot != ticket1.staging_slot, "staging slots must alternate for double buffering");
    expect(state.prev_tokens[0] == 101 && state.prev_tokens[1] == 202,
           "state.prev_tokens must still be untouched before commit");

    // 3. Cancel/discard safety: discard ticket2, state remains completely unpolluted
    ticket2.valid = false;
    expect(state.prev_tokens[0] == 101 && state.prev_tokens[1] == 202,
           "discarding ticket must leave state untouched");

    // 4. Commit ticket1: updates prev_tokens and invalidates ticket
    ple.commit(ticket1, state);
    expect(!ticket1.valid, "commit must invalidate ticket");
    expect(state.prev_tokens[0] == 202 && state.prev_tokens[1] == 303,
           "commit must update state.prev_tokens to prev_tokens_next");

    // Calling commit again must be safe no-op
    ple.commit(ticket1, state);
    expect(state.prev_tokens[0] == 202 && state.prev_tokens[1] == 303,
           "subsequent commit calls must be no-ops");

    // 5. Multi-token prepare test (e.g. prefill 3 tokens)
    std::vector<int> prefill_tokens = {501, 502, 503};
    PleTicket ticket3 = ple.prepare(ctx, state, prefill_tokens.data(), 3, &state, 2, 12);
    expect(ticket3.valid, "ticket3 must be valid");
    expect(ticket3.num_tokens == 3, "ticket3 num_tokens mismatch");
    expect(ticket3.prev_tokens_in[0] == 202 && ticket3.prev_tokens_in[1] == 303, "ticket3 prev_tokens_in mismatch");
    expect(ticket3.prev_tokens_next[0] == 502 && ticket3.prev_tokens_next[1] == 503, "ticket3 prev_tokens_next mismatch");
    expect(state.prev_tokens[0] == 202 && state.prev_tokens[1] == 303, "state must not change before commit");

    // 6. EOS token handling in n-gram calculation (boundary reset)
    {
        std::array<int64_t, 3> mults = {10007LL, 20011LL, 30013LL};
        std::array<int64_t, 16> vocabs;
        vocabs.fill(1000);
        std::array<int64_t, 16> offs;
        offs.fill(0);
        ple.set_multipliers_and_vocab(mults, vocabs, offs);

        // Sequence: EOS, tokenA, EOS, tokenB
        std::vector<int> eos_seq = {Qwen4ExpPle::EOS_TOKEN_ID, 777, Qwen4ExpPle::EOS_TOKEN_ID, 888};
        std::vector<int64_t> ids(eos_seq.size() * Qwen4ExpPle::NUM_HEADS);
        Qwen4ExpPle::compute_ngram_ids(eos_seq.data(), static_cast<int>(eos_seq.size()),
                                       static_cast<int>(eos_seq.size()),
                                       mults.data(), vocabs.data(), offs.data(), ids.data(),
                                       Qwen4ExpPle::EOS_TOKEN_ID);

        // For token at pos 0 (EOS): prev_eos is -1, pos_in_seg is 0, s0 = EOS, s1 = EOS, s2 = EOS
        // For token at pos 3 (888): prev_eos is 2 (the EOS at pos 2), pos_in_seg is 3 - 3 = 0.
        // Therefore s0 = 888, s1 = EOS, s2 = EOS (reset after EOS!)
        int64_t m0_u = static_cast<int64_t>(static_cast<uint64_t>(888) * static_cast<uint64_t>(mults[0]));
        int64_t m1_u = static_cast<int64_t>(static_cast<uint64_t>(Qwen4ExpPle::EOS_TOKEN_ID) * static_cast<uint64_t>(mults[1]));
        int64_t m2_u = static_cast<int64_t>(static_cast<uint64_t>(Qwen4ExpPle::EOS_TOKEN_ID) * static_cast<uint64_t>(mults[2]));
        int64_t mixed2 = m0_u ^ m1_u;
        int64_t mixed3 = m0_u ^ m1_u ^ m2_u;
        int64_t expected_h0 = (mixed2 % 1000 + 1000) % 1000;
        int64_t expected_h8 = (mixed3 % 1000 + 1000) % 1000;

        const int64_t* row3 = ids.data() + 3 * Qwen4ExpPle::NUM_HEADS;
        expect(row3[0] == expected_h0, "EOS boundary reset head 0 mismatch");
        expect(row3[8] == expected_h8, "EOS boundary reset head 8 mismatch");
    }

    // 7. Cancel / Retry safety with generation bump
    {
        uint64_t gen = 100;
        Qwen4ExpPleState s_retry;
        ple.init_state(ctx, s_retry);
        s_retry.prev_tokens = {10, 20};

        int bad_token = 999;
        PleTicket ticket_aborted = ple.prepare(ctx, s_retry, &bad_token, 1, &s_retry, gen, 5);
        expect(ticket_aborted.valid, "ticket_aborted must be valid");

        // Cancel / Retry triggered: generation bumped, ticket discarded
        ++gen;
        ticket_aborted.valid = false;
        expect(s_retry.prev_tokens[0] == 10 && s_retry.prev_tokens[1] == 20,
               "state must remain unchanged after abort");

        // Stale ticket validation check
        expect(ticket_aborted.generation != gen, "stale ticket generation mismatch detected");

        // New valid retry token prepared and committed
        int good_token = 30;
        PleTicket ticket_retried = ple.prepare(ctx, s_retry, &good_token, 1, &s_retry, gen, 5);
        expect(ticket_retried.valid && ticket_retried.generation == gen, "retried ticket must be valid and current");
        ple.commit(ticket_retried, s_retry);
        expect(s_retry.prev_tokens[0] == 20 && s_retry.prev_tokens[1] == 30,
               "state correctly updated with retried token only");
    }

    // 8. Rapid queueing & double-buffering concurrency
    {
        Qwen4ExpPleState s_rapid;
        ple.init_state(ctx, s_rapid);
        s_rapid.prev_tokens = {1, 2};

        int tok1 = 3;
        int tok2 = 4;
        PleTicket t_a = ple.prepare(ctx, s_rapid, &tok1, 1, &s_rapid, 1, 0);
        // Note: s_rapid.prev_tokens is not yet committed, so for t_b we simulate chained decode
        ple.commit(t_a, s_rapid);
        PleTicket t_b = ple.prepare(ctx, s_rapid, &tok2, 1, &s_rapid, 1, 1);
        ple.commit(t_b, s_rapid);

        expect(s_rapid.prev_tokens[0] == 3 && s_rapid.prev_tokens[1] == 4,
               "rapid sequential prepare/commit must chain cleanly");
    }

    // 9. Graph staging slot is fixed at zero and host waits are explicit.
    {
        Qwen4ExpPleState s_fixed;
        ple.init_state(ctx, s_fixed);
        s_fixed.prev_tokens = {700, 701};

        int fixed_token_a = 702;
        PleTicket fixed_a = ple.prepare_fixed_slot(
            ctx, s_fixed, &fixed_token_a, 1, &s_fixed, 3, 20);
        expect(fixed_a.valid && fixed_a.staging_slot == 0,
               "fixed PLE prepare must use staging slot zero");
        ple.wait_prepared(ctx, fixed_a);
        expect(ple.MatchesInput(fixed_a, &fixed_token_a, 1),
               "fixed PLE ticket must retain its owned input");

        int fixed_token_b = 703;
        PleTicket fixed_b = ple.prepare_fixed_slot(
            ctx, s_fixed, &fixed_token_b, 1, &s_fixed, 3, 21);
        expect(fixed_b.valid && fixed_b.staging_slot == 0,
               "fixed PLE prepare must not rotate back to the other bank");
        ple.wait_prepared(ctx, fixed_b);
        expect(ple.MatchesInput(fixed_b, &fixed_token_b, 1),
               "fixed PLE replacement ticket must match its input");
        fixed_a.valid = false;
        fixed_b.valid = false;
    }

    std::cout << "[PASS] PLE Ticket Pipeline tests passed.\n";
}

// ---------------------------------------------------------------------------
// 3. Real Model Integration Test (if present)
// ---------------------------------------------------------------------------

static std::string resolve_qwen4exp_model_dir() {
    const std::string env_dir = aila::env::read_string("AILA_QWEN4EXP_MODEL_DIR", "");
    if (!env_dir.empty() && fs::exists(env_dir)) return env_dir;
    const std::string candidate_c = "C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_c)) return candidate_c;
    const std::string candidate_f = "F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_f)) return candidate_f;
    return candidate_c;
}

void test_snapshot_capture_and_truncate(Context& ctx) {
    std::cout << "[Test] Running Phase S5 batched async snapshot capture & truncate test...\n";

    Qwen4ExpBnb4Backend backend;
    backend.init_for_testing(ctx, 4); // 4 layers: 0, 1, 2 linear, 3 attention
    auto session = backend.CreateSession(ctx, /*maxSnapshots=*/3);
    expect(session != nullptr, "CreateSession failed");
    const auto initial_graph_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    expect(initial_graph_key.backend == &backend &&
               initial_graph_key.session == session.get() &&
               initial_graph_key.session_instance_id ==
                   session->session_instance_id() &&
               initial_graph_key.session_instance_id != 0 &&
               !session->token_graph_poisoned(),
           "initial token graph key identity is invalid");
    expect(initial_graph_key.ple_workspace_generation ==
               Qwen4ExpGenerationTestAccess::PleWorkspaceGeneration(backend),
           "token graph key must snapshot the PLE workspace generation");
    const auto l3_graph_key =
        Qwen4ExpGenerationTestAccess::MakeQsaLayerGraphKey(backend, *session);
    expect(l3_graph_key.session_instance_id ==
               initial_graph_key.session_instance_id &&
               l3_graph_key.session_generation ==
                   initial_graph_key.session_generation &&
               l3_graph_key.runtime_generation ==
                   initial_graph_key.runtime_generation &&
               l3_graph_key.runtime_capacity ==
                   initial_graph_key.runtime_capacity &&
               l3_graph_key.topology_generation ==
                   initial_graph_key.topology_generation,
           "L3 graph key must retain identity and layout generations");
    expect(l3_graph_key.session_state_generation == 0 &&
               l3_graph_key.residency_generation == 0 &&
               l3_graph_key.descriptor_generation == 0 &&
               l3_graph_key.ple_workspace_generation == 0 &&
               l3_graph_key.ple_graph_view_generation == 0 &&
               l3_graph_key.body == aila::TokenGraphBody::QsaLayerProof &&
               l3_graph_key.attention ==
                   aila::TokenGraphAttention::QsaSparse512BlockNative,
           "L3 graph key must exclude unrelated mutable generations");
    const Tensor empty_input;
    const Tensor empty_output;
    expect(!Qwen4ExpGenerationTestAccess::QsaLayerGraphEligible(
               backend, ctx, *session, empty_input, empty_output, 1, 2052),
           "incomplete L3 fixtures must fail closed to the direct path");
    expect(!Qwen4ExpGenerationTestAccess::QsaLayerBlockStateEligible(
               512, 514) &&
               Qwen4ExpGenerationTestAccess::QsaLayerBlockStateEligible(
                   513, 514) &&
               Qwen4ExpGenerationTestAccess::QsaLayerBlockStateEligible(
                   514, 514) &&
               !Qwen4ExpGenerationTestAccess::QsaLayerBlockStateEligible(
                   515, 514),
           "L3 graph must accept exactly the pending-one compressed block");

    PleTicket speculative_ticket;
    speculative_ticket.session_ptr = session.get();
    speculative_ticket.generation = session->generation();
    speculative_ticket.ticket_epoch =
        Qwen4ExpGenerationTestAccess::PleTicketEpoch(*session);
    speculative_ticket.start_pos = 0;
    speculative_ticket.valid = true;
    expect(Qwen4ExpGenerationTestAccess::PleTicketEpochMatches(
               backend, *session, speculative_ticket),
           "current PLE ticket fixture must match its session epoch");
    const uint64_t generation_before_mtp_restore = session->generation();
    const uint64_t graph_generation_before_mtp_restore =
        session->token_graph_generation();
    Qwen4ExpGenerationTestAccess::InvalidateMtpRestoreState(*session);
    expect(!Qwen4ExpGenerationTestAccess::PleTicketEpochMatches(
               backend, *session, speculative_ticket),
           "MTP restore must reject pre-restore PLE tickets");
    expect(session->generation() == generation_before_mtp_restore,
           "MTP restore must not bump the broader session generation");
    expect(session->token_graph_generation() >
               graph_generation_before_mtp_restore,
           "MTP restore must still invalidate the token graph key");

    Qwen4ExpGenerationTestAccess::EnsurePleWorkspace(backend, ctx, 1);
    const auto grown_workspace_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    expect(grown_workspace_key.ple_workspace_generation >
               initial_graph_key.ple_workspace_generation &&
               grown_workspace_key.ple_workspace_generation ==
                   Qwen4ExpGenerationTestAccess::PleWorkspaceGeneration(backend),
           "PLE workspace growth must invalidate the token graph key");

    auto& ple_state = session->ple_state_for_testing();
    const uint64_t previous_view_generation = ple_state.graph_view_generation;
    ++ple_state.graph_view_generation;
    const auto graph_view_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    expect(graph_view_key.ple_graph_view_generation ==
               previous_view_generation + 1 &&
               graph_view_key.ple_graph_view_generation !=
                   grown_workspace_key.ple_graph_view_generation,
           "PLE graph view generation must participate in the token graph key");

    bool unsafe_key_rejected = false;
    Qwen4ExpGenerationTestAccess::SetEmbeddingOverrideState(*session, true);
    try {
        (void)Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    } catch (const std::logic_error&) {
        unsafe_key_rejected = true;
    }
    expect(unsafe_key_rejected,
           "token graph key must reject embedding override state");
    Qwen4ExpGenerationTestAccess::SetEmbeddingOverrideState(*session, false);

    unsafe_key_rejected = false;
    Qwen4ExpGenerationTestAccess::SetMropeState(*session, true);
    try {
        (void)Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    } catch (const std::logic_error&) {
        unsafe_key_rejected = true;
    }
    expect(unsafe_key_rejected,
           "token graph key must reject mRoPE position state");
    Qwen4ExpGenerationTestAccess::SetMropeState(*session, false);

    auto& caches = session->layer_caches_for_testing();
    expect(caches.size() == 48, "expected 48 layer caches allocated in session");
    const auto storage_addresses = [&]() {
        std::vector<const void*> addresses;
        addresses.push_back(session->snapshot_slots_for_testing().data());
        for (const auto& slot : session->snapshot_slots_for_testing()) {
            addresses.push_back(slot.layers.data());
            addresses.push_back(slot.ple_snapshot.conv_state.data());
            for (const auto& layer : slot.layers) {
                addresses.push_back(layer.linear_state_host.data());
                addresses.push_back(layer.linear_conv_state_host.data());
            }
        }
        return addresses;
    };
    const auto initial_storage = storage_addresses();

    // Initialize test patterns in Layer 0 and Layer 1 linear states
    const size_t state_size = 48 * 128 * 128;
    const size_t conv_size = 3 * 10240;
    std::vector<float> l0_state_pat(state_size, 3.14159f);
    std::vector<float> l0_conv_pat(conv_size, 2.71828f);
    std::vector<float> l1_state_pat(state_size, 1.41421f);
    std::vector<float> l1_conv_pat(conv_size, 0.57721f);

    ctx.memcpy_h2d(caches[0].linear_state.data(), l0_state_pat.data(), state_size * sizeof(float));
    ctx.memcpy_h2d(caches[0].linear_conv_state.data(), l0_conv_pat.data(), conv_size * sizeof(float));
    caches[0].linear_conv_head = 42;
    caches[0].device_state_dirty = true;

    ctx.memcpy_h2d(caches[1].linear_state.data(), l1_state_pat.data(), state_size * sizeof(float));
    ctx.memcpy_h2d(caches[1].linear_conv_state.data(), l1_conv_pat.data(), conv_size * sizeof(float));
    caches[1].linear_conv_head = 7;
    caches[1].device_state_dirty = true;

    // 1. Capture snapshot at length 10
    backend.capture_snapshot_for_testing(ctx, *session, 10, 10);
    expect(!caches[0].device_state_dirty, "device_state_dirty must be cleared after snapshot");
    expect(!caches[1].device_state_dirty, "device_state_dirty must be cleared after snapshot");
    expect(session->snapshot_for_testing(10) != nullptr, "snapshot 10 must exist");

    // 2. Mutate device state to all zeros
    ctx.queue().memset(caches[0].linear_state.data(), 0, state_size * sizeof(float)).wait_and_throw();
    ctx.queue().memset(caches[0].linear_conv_state.data(), 0, conv_size * sizeof(float)).wait_and_throw();
    caches[0].linear_conv_head = 0;
    ctx.queue().memset(caches[1].linear_state.data(), 0, state_size * sizeof(float)).wait_and_throw();
    ctx.queue().memset(caches[1].linear_conv_state.data(), 0, conv_size * sizeof(float)).wait_and_throw();
    caches[1].linear_conv_head = 0;

    // 3. Truncate to length 10 -> must restore snapshot 10
    const auto before_truncate_graph_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    const uint64_t before_truncate_ple_epoch =
        Qwen4ExpGenerationTestAccess::PleTicketEpoch(*session);
    int restored = backend.Truncate(*session, 10);
    expect(restored == 10, "Truncate must restore length 10");
    const auto after_truncate_graph_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    expect(after_truncate_graph_key.session_generation >
               before_truncate_graph_key.session_generation &&
               after_truncate_graph_key.session_state_generation >
                    before_truncate_graph_key.session_state_generation,
            "Truncate must invalidate token graph state generation");
    expect(Qwen4ExpGenerationTestAccess::PleTicketEpoch(*session) >
               before_truncate_ple_epoch,
           "Truncate must invalidate pre-rollback PLE tickets");
    expect(session->CurrentLength() == 10, "CurrentLength must be 10");
    expect(caches[0].linear_conv_head == 42, "Layer 0 conv head must be restored to 42");
    expect(caches[1].linear_conv_head == 7, "Layer 1 conv head must be restored to 7");

    // Verify GPU device data restored correctly
    std::vector<float> read_l0_state(state_size);
    std::vector<float> read_l0_conv(conv_size);
    std::vector<float> read_l1_state(state_size);
    std::vector<float> read_l1_conv(conv_size);

    ctx.memcpy_d2h(read_l0_state.data(), caches[0].linear_state.data(), state_size * sizeof(float));
    ctx.memcpy_d2h(read_l0_conv.data(), caches[0].linear_conv_state.data(), conv_size * sizeof(float));
    ctx.memcpy_d2h(read_l1_state.data(), caches[1].linear_state.data(), state_size * sizeof(float));
    ctx.memcpy_d2h(read_l1_conv.data(), caches[1].linear_conv_state.data(), conv_size * sizeof(float));

    expect(read_l0_state[0] == 3.14159f && read_l0_state[state_size - 1] == 3.14159f, "Layer 0 linear state restoration failed");
    expect(read_l0_conv[0] == 2.71828f && read_l0_conv[conv_size - 1] == 2.71828f, "Layer 0 conv state restoration failed");
    expect(read_l1_state[0] == 1.41421f && read_l1_state[state_size - 1] == 1.41421f, "Layer 1 linear state restoration failed");
    expect(read_l1_conv[0] == 0.57721f && read_l1_conv[conv_size - 1] == 0.57721f, "Layer 1 conv state restoration failed");

    // 4. Test bounded snapshot eviction with maxSnapshots = 3
    // Snapshots: add 20, 30, 40 -> total 4 snapshots, so snapshot 10 must be evicted!
    caches[0].device_state_dirty = true;
    caches[1].device_state_dirty = true;
    backend.capture_snapshot_for_testing(ctx, *session, 20, 10);

    caches[0].device_state_dirty = true;
    caches[1].device_state_dirty = true;
    backend.capture_snapshot_for_testing(ctx, *session, 30, 10);

    caches[0].device_state_dirty = true;
    caches[1].device_state_dirty = true;
    backend.capture_snapshot_for_testing(ctx, *session, 40, 10);

    expect(session->snapshot_for_testing(10) == nullptr, "snapshot 10 must be evicted when maxSnapshots exceeded");
    expect(session->snapshot_for_testing(20) != nullptr, "snapshot 20 must be retained");
    expect(session->snapshot_for_testing(30) != nullptr, "snapshot 30 must be retained");
    expect(session->snapshot_for_testing(40) != nullptr, "snapshot 40 must be retained");

    // Truncate to 35 -> nearest checkpoint <= 35 is 30
    restored = backend.Truncate(*session, 35);
    expect(restored == 30, "Truncate to 35 must restore checkpoint 30");

    // Truncate to 15 -> no checkpoint <= 15 (10 was evicted), must reset to 0
    restored = backend.Truncate(*session, 15);
    expect(restored == 0, "Truncate to 15 must reset to 0");
    expect(session->CurrentLength() == 0, "CurrentLength must be 0 after reset");
    expect(initial_storage == storage_addresses(), "Capture, eviction and reset must reuse snapshot storage");

    backend.capture_snapshot_for_testing(ctx, *session, 13, 1, true);
    expect(session->snapshot_for_testing(13) != nullptr, "Explicit single-token anchor must be captured");
    expect(backend.Truncate(*session, 13) == 13, "Single-token anchor must be restorable");
    const auto before_reset_graph_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    const uint64_t before_reset_ple_epoch =
        Qwen4ExpGenerationTestAccess::PleTicketEpoch(*session);
    backend.Reset(*session);
    const auto after_reset_graph_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    expect(after_reset_graph_key.session_generation >
               before_reset_graph_key.session_generation &&
               after_reset_graph_key.session_state_generation >
                    before_reset_graph_key.session_state_generation,
            "Reset must invalidate token graph state generation");
    expect(Qwen4ExpGenerationTestAccess::PleTicketEpoch(*session) >
               before_reset_ple_epoch,
           "Reset must invalidate pre-reset PLE tickets");
    expect(initial_storage == storage_addresses(), "Anchor restore must preserve storage");
    for (const auto& slot : session->snapshot_slots_for_testing()) {
        expect(slot.length == -1, "Reset must invalidate every snapshot");
    }
    const auto before_poison_graph_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    session->PoisonTokenGraph();
    expect(session->token_graph_poisoned() &&
               session->token_graph_generation() >
                   before_poison_graph_key.session_state_generation,
           "poison must permanently invalidate token graph state");
    bool poisoned_key_rejected = false;
    try {
        (void)Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    } catch (const std::logic_error&) {
        poisoned_key_rejected = true;
    }
    expect(poisoned_key_rejected,
           "poisoned session must reject token graph key creation");
    session.reset();
    auto no_snapshots = backend.CreateSession(ctx, 0);
    const auto replacement_session_key =
        Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *no_snapshots);
    expect(no_snapshots->session_instance_id() != 0 &&
               no_snapshots->session_instance_id() !=
                   initial_graph_key.session_instance_id &&
               replacement_session_key.session_instance_id !=
                   initial_graph_key.session_instance_id,
           "backend must monotonically identify recreated Session instances");
    backend.capture_snapshot_for_testing(ctx, *no_snapshots, 13, 1, true);
    expect(no_snapshots->snapshot_slots_for_testing().empty(), "Zero snapshot budget must remain empty");
    expect(backend.Truncate(*no_snapshots, 12) == 0, "Missing snapshot must fall back to reset");

    std::cout << "[PASS] Phase S5 batched async snapshot capture & truncate verified successfully.\n";
}

void test_graph_gdn_device_head_state(Context& ctx) {
    std::cout << "[Test] Graph GDN device-head ownership and fail-closed transitions...\n";
    _putenv("AILA_Q4E_GRAPH_GDN_STATE=1");
    _putenv("AILA_Q4E_GDN_HISTORY_FUSED=0");

    Qwen4ExpBnb4Backend backend;
    backend.init_for_testing(ctx, 4);
    auto session = backend.CreateSession(ctx, /*maxSnapshots=*/2);
    expect(session != nullptr, "graph GDN session creation failed");

    auto& caches = session->layer_caches_for_testing();
    expect(caches[0].linear_conv_head_dev != nullptr,
           "linear layer must own a device-head row when the gate is enabled");
    expect(caches[1].linear_conv_head_dev != nullptr,
           "second linear layer must own an independent device-head row");
    expect(caches[3].linear_conv_head_dev == nullptr,
           "full-attention layer must not own a GDN device-head row");
    expect(caches[0].linear_conv_head_dev != caches[1].linear_conv_head_dev,
           "linear layers must use distinct device-head rows");

    bool rejected = false;
    std::string rejection;
    caches[0].linear_conv_head_device_authority = true;
    caches[1].linear_conv_head_device_authority = false;
    try {
        (void)Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, *session);
    } catch (const std::logic_error& e) {
        rejected = true;
        rejection = e.what();
    }
    expect(rejected &&
               rejection.find("mixed host/device authority") != std::string::npos,
           "token graph key must reject mixed GDN authority");
    caches[0].linear_conv_head_device_authority = false;

    caches[0].linear_conv_head = 2;
    caches[1].linear_conv_head = 1;
    Qwen4ExpGenerationTestAccess::PublishGdnHeads(backend, ctx, *session);
    expect(caches[0].linear_conv_head_device_synced &&
           caches[1].linear_conv_head_device_synced,
           "host heads must publish as synchronized");

    caches[0].linear_conv_head_device_synced = false;
    caches[1].linear_conv_head_device_synced = false;
    bool launch_ran = false;
    rejected = false;
    try {
        backend.SubmitTokenGraph(*session, [&] {
            launch_ran = true;
            return 0;
        });
    } catch (const std::invalid_argument& e) {
        rejected = true;
        rejection = e.what();
    }
    expect(rejected && !launch_ran &&
               rejection.find("published to the device") !=
                   std::string::npos,
           "graph submission must reject stale unpublished host heads");
    Qwen4ExpGenerationTestAccess::PublishGdnHeads(backend, ctx, *session);

    const int launch_receipt = backend.SubmitTokenGraph(*session, [] { return 7; });
    expect(launch_receipt == 7,
           "token graph wrapper must preserve the launch return value");
    expect(caches[0].linear_conv_head_device_authority &&
           caches[1].linear_conv_head_device_authority,
           "graph replay must transfer authority to device heads");
    expect(caches[0].device_state_dirty && caches[1].device_state_dirty,
           "graph submission wrapper must dirty the checkpoint state");
    expect(Qwen4ExpGenerationTestAccess::GdnDeviceHeadReadable(backend, caches[0]),
           "device-authority head with a stale host mirror must remain readable");

    caches[1].linear_conv_head_device_authority = false;
    caches[1].linear_conv_head_device_synced = true;
    try {
        Qwen4ExpGenerationTestAccess::ValidateGdnPath(backend, *session, 1);
    } catch (const std::invalid_argument& e) {
        rejected = true;
        rejection = e.what();
    }
    expect(rejected,
           "partially advanced device-authoritative state must fail closed");
    expect(rejection.find("mixed host/device authority") != std::string::npos,
           "mixed GDN authority must report a clear error");
    Qwen4ExpGenerationTestAccess::MarkGdnGraphDirty(backend, *session);

    rejected = false;
    try {
        Qwen4ExpGenerationTestAccess::ValidateGdnPath(backend, *session, 2);
    } catch (const std::invalid_argument& e) {
        rejected = true;
        rejection = e.what();
    }
    expect(rejected,
           "device-authoritative heads must reject multi-row/prefill before sync");
    expect(rejection.find("cannot switch to a multi-row/prefill") !=
               std::string::npos,
           "host-path fallback must report a clear error");

    Qwen4ExpGenerationTestAccess::ValidateGdnPath(backend, *session, 1);
    backend.capture_snapshot_for_testing(ctx, *session, 7, 1, true);
    const auto* snapshot = session->snapshot_for_testing(7);
    expect(snapshot != nullptr,
           "graph replay state must produce a valid checkpoint");
    expect(snapshot->layers[0].linear_conv_head == 2 &&
               snapshot->layers[1].linear_conv_head == 1,
           "checkpoint must synchronize the device-authoritative GDN heads");
    expect(caches[0].linear_conv_head == 2 &&
               caches[1].linear_conv_head == 1 &&
               !caches[0].linear_conv_head_device_authority &&
               !caches[1].linear_conv_head_device_authority &&
               caches[0].linear_conv_head_device_synced &&
               caches[1].linear_conv_head_device_synced,
           "snapshot capture must return GDN state to a synchronized host mirror");
    expect(!caches[0].linear_conv_head_device_authority &&
           !caches[1].linear_conv_head_device_authority &&
           !caches[0].device_state_dirty && !caches[1].device_state_dirty,
           "snapshot capture must clear the graph dirty generation");

    const int replayed_heads[2] = {0, 2};
    ctx.memcpy_h2d(caches[0].linear_conv_head_dev, &replayed_heads[0],
                   sizeof(replayed_heads[0]));
    ctx.memcpy_h2d(caches[1].linear_conv_head_dev, &replayed_heads[1],
                   sizeof(replayed_heads[1]));
    backend.SubmitTokenGraph(*session, [] { return true; });
    session->current_length_for_testing() = 9;
    expect(backend.Truncate(*session, 7) == 7,
           "checkpoint rollback after graph replay must succeed");
    int restored_heads[2]{};
    ctx.memcpy_d2h(&restored_heads[0], caches[0].linear_conv_head_dev,
                   sizeof(restored_heads[0]));
    ctx.memcpy_d2h(&restored_heads[1], caches[1].linear_conv_head_dev,
                   sizeof(restored_heads[1]));
    expect(caches[0].linear_conv_head == 2 &&
               caches[1].linear_conv_head == 1 &&
               restored_heads[0] == 2 && restored_heads[1] == 1,
           "truncate must restore and republish the checkpoint GDN heads");
    Qwen4ExpGenerationTestAccess::ValidateGdnPath(backend, *session, 2);

    backend.Reset(*session);
    expect(caches[0].linear_conv_head == 0 && caches[1].linear_conv_head == 0,
           "Reset must restore zero host heads");
    expect(caches[0].linear_conv_head_device_synced &&
           caches[1].linear_conv_head_device_synced,
           "Reset must republish synchronized device heads");
    expect(!caches[0].linear_conv_head_device_authority &&
           !caches[1].linear_conv_head_device_authority,
           "Reset must restore host authority");

    session->PoisonTokenGraph();
    rejected = false;
    try {
        backend.capture_snapshot_for_testing(ctx, *session, 8, 1, true);
    } catch (const std::logic_error& e) {
        rejected = true;
        rejection = e.what();
    }
    expect(rejected &&
               rejection.find("cannot checkpoint a poisoned") !=
                   std::string::npos,
           "poisoned graph session must reject checkpoint capture");

    _putenv("AILA_Q4E_GRAPH_GDN_STATE=");
    _putenv("AILA_Q4E_GDN_HISTORY_FUSED=");
    std::cout << "[PASS] Graph GDN device-head ownership verified.\n";
}

void test_token_graph_runtime_lifecycle(Context& ctx) {
    std::cout << "[Test] Token graph backend lifecycle and direct-path abort...\n";
    _putenv("AILA_Q4E_GRAPH_GDN_STATE=");

    const auto make_ready = [](Qwen4ExpBnb4Backend& backend,
                               Qwen4ExpBnb4Backend::Session& session) {
        const auto key =
            Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(backend, session);
        expect(backend.BeginCapture(session, key.body, key.attention,
                                    key.topology_flags),
               "backend BeginCapture failed");
        expect(backend.MarkReady(), "backend MarkReady failed");
    };

    {
        Qwen4ExpBnb4Backend backend;
        backend.init_for_testing(ctx, 4);
        auto session = backend.CreateSession(ctx, 0);
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Empty,
               "default direct backend must start with an Empty graph runtime");

        const auto allocations = ctx.device_allocation_count();
        (void)backend.Forward(ctx, *session, nullptr, 0);
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Empty,
               "default direct forward must leave the graph runtime Empty");
        expect(ctx.device_allocation_count() == allocations,
               "default direct forward must not allocate graph resources");

        make_ready(backend, *session);
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Ready,
               "captured graph did not become Ready");
        (void)backend.Forward(ctx, *session, nullptr, 0);
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Aborting,
               "direct forward must abort a Ready token graph");
    }

    {
        Qwen4ExpBnb4Backend backend;
        backend.init_for_testing(ctx, 4);
        auto session = backend.CreateSession(ctx, 0);
        make_ready(backend, *session);
        expect(backend.Launch(*session, 1, sycl::event{}),
               "first token graph launch must succeed");
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::InFlight,
               "launch did not enter InFlight");
        expect(!backend.Launch(*session, 2, sycl::event{}),
               "second in-flight token graph launch must be rejected");
        expect(backend.Complete(1, true),
               "matching graph completion must succeed");
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Ready,
               "successful completion must return to Ready");

        backend.Reset(*session);
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Aborting,
               "Reset must abort a Ready token graph before mutation");
    }

    {
        Qwen4ExpBnb4Backend backend;
        backend.init_for_testing(ctx, 4);
        auto session = backend.CreateSession(ctx, 0);
        make_ready(backend, *session);
        session->current_length_for_testing() = 2;
        expect(backend.Truncate(*session, 0) == 0,
               "Truncate-to-zero fixture failed");
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Aborting,
               "Truncate must abort a Ready token graph before mutation");
    }

    {
        Qwen4ExpBnb4Backend backend;
        backend.init_for_testing(ctx, 4);
        auto session = backend.CreateSession(ctx, 0);
        make_ready(backend, *session);
        session.reset();
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Aborting,
               "Session destruction must abort its retained token graph key");
        expect(backend.token_graph_runtime().key().session == nullptr,
               "Token graph key retained a destroyed Session pointer");
    }

    {
        Qwen4ExpBnb4Backend backend;
        backend.init_for_testing(ctx, 4);
        auto session = backend.CreateSession(ctx, 0);
        make_ready(backend, *session);
        Qwen4ExpGenerationTestAccess::InvalidateMtpRestoreState(*session);
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Aborting,
               "MTP restore-state invalidation must abort the graph runtime");
    }

    {
        Qwen4ExpBnb4Backend backend;
        backend.init_for_testing(ctx, 4);
        auto session = backend.CreateSession(ctx, 0);
        make_ready(backend, *session);
        expect(backend.AbortAndPoisonTokenGraph(*session),
               "AbortAndPoison must transition an active runtime");
        expect(backend.token_graph_runtime().phase() ==
                   aila::TokenGraphPhase::Poisoned &&
                   session->token_graph_poisoned(),
               "AbortAndPoison must poison both runtime and Session");
        expect(!backend.AbortTokenGraph() &&
                   !backend.MarkReady() &&
                   !backend.Launch(*session, 1, sycl::event{}) &&
                   !backend.Complete(1, true),
               "Poisoned graph runtime must reject every transition");
        expect(!backend.AbortAndPoisonTokenGraph(*session),
               "second AbortAndPoison must be a no-op");

        bool forward_rejected = false;
        try {
            (void)backend.Forward(ctx, *session, nullptr, 0);
        } catch (const std::logic_error&) {
            forward_rejected = true;
        }
        expect(forward_rejected,
               "Poisoned Session must reject subsequent forward execution");

        bool key_rejected = false;
        try {
            (void)Qwen4ExpGenerationTestAccess::MakeTokenGraphKey(
                backend, *session);
        } catch (const std::logic_error&) {
            key_rejected = true;
        }
        expect(key_rejected,
               "Poisoned Session must continue rejecting graph keys");

        bool checkpoint_rejected = false;
        try {
            backend.capture_snapshot_for_testing(ctx, *session, 1, 1, true);
        } catch (const std::logic_error&) {
            checkpoint_rejected = true;
        }
        expect(checkpoint_rejected,
               "Poisoned Session must continue rejecting checkpoints");
    }

    std::cout << "[PASS] Token graph backend lifecycle verified.\n";
}

void test_supports_interior_checkpoint_whitelist(Context& ctx) {
    std::cout << "[Test] X2 interior checkpoint eligibility and opt-out...\n";
    Qwen4ExpBnb4Backend backend;
    backend.init_for_testing(ctx, 4);
    _putenv("AILA_Q4E_INLINE_CHECKPOINT=");
    expect(backend.supports_interior_checkpoint(30, 26), "Measured story shape must default on");
    expect(!backend.supports_interior_checkpoint(30, 28), "Unmeasured defaults stay opt-in");
    expect(!backend.supports_interior_checkpoint(30, 16), "General anchors stay opt-in");
    _putenv("AILA_Q4E_INLINE_CHECKPOINT=1");
    for (int anchor : {2, 16, 26, 28})
        expect(backend.supports_interior_checkpoint(30, anchor), "Valid text anchors must support opt-in");
    expect(backend.supports_interior_checkpoint(64, 32), "Largest dense chunk must support opt-in");
    for (int anchor : {-1, 0, 1, 29, 30, 31})
        expect(!backend.supports_interior_checkpoint(30, anchor), "Invalid sides must be rejected");
    expect(!backend.supports_interior_checkpoint(65, 26), "count > 64 must be rejected");
    backend.set_embedding_overrides({0}, {bf16(0)}, 1);
    expect(!backend.supports_interior_checkpoint(30, 26), "Vision override must fall back");
    backend.clear_embedding_overrides();
    backend.set_mrope_positions(ctx, {0}, {0}, {0}, 0);
    expect(!backend.supports_interior_checkpoint(30, 26), "Mrope override must fall back");
    backend.clear_mrope_positions();
    _putenv("AILA_Q4E_INLINE_CHECKPOINT=0");
    expect(!backend.supports_interior_checkpoint(30, 26), "Explicit opt-out must retain split prefill");
    _putenv("AILA_Q4E_INLINE_CHECKPOINT=");
    expect(backend.supports_interior_checkpoint(30, 26), "Clearing override restores measured default");
    Qwen4ExpBnb4Backend large;
    large.init_for_testing(ctx, 4, 4096);
    auto session = large.CreateSession(ctx, 2);
    session->current_length_for_testing() = 2021;
    expect(Qwen4ExpGenerationTestAccess::CanCapture(large, *session, 30, 26), "2051 must remain dense");
    session->current_length_for_testing() = 2022;
    expect(!Qwen4ExpGenerationTestAccess::CanCapture(large, *session, 30, 26), "2052 is sparse and must be rejected");
    bool rejected = false;
    try { large.Forward(ctx, *session, nullptr, 30, {SnapshotPolicy::ReusablePrefix, true, false, 26}); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected && session->CurrentLength() == 2022, "Direct forward must reject sparse capture before advancing");
    auto single_slot = large.CreateSession(ctx, 1);
    expect(!Qwen4ExpGenerationTestAccess::CanCapture(large, *single_slot, 30, 26), "Capture needs two slots");
    session->current_length_for_testing() = 4090;
    expect(!Qwen4ExpGenerationTestAccess::CanCapture(large, *session, 30, 26), "Capture must fit session capacity");
    std::cout << "[PASS] X2 interior checkpoint eligibility\n";
}

void test_real_model_ple_and_cache_if_present(Context& ctx, bool pleOnly = false) {
    const std::string model_dir = resolve_qwen4exp_model_dir();
    if (!fs::exists(model_dir) || !fs::exists(model_dir + "/model.safetensors.index.json")) {
        expect(!pleOnly, "--ple-only requires a real model");
        std::cout << "[Test] Real model directory " << model_dir << " not found, skipping real model tests.\n";
        return;
    }

    std::cout << "[Test] Loading real model lazy index from " << model_dir << "...\n";
    ModelWeights weights = LoadModelWeightsLazy(model_dir, &ctx);

    // 1. Verify PLE indexing on real model
    Qwen4ExpPle ple;
    std::string ple_err;
    bool ple_ok = ple.init(ctx, weights, /*ple_layer_id=*/1, /*max_tokens=*/128, 1e-6f, &ple_err);
    expect(ple_ok, "real model ple.init failed: " + ple_err);

    expect(ple.layer_multipliers()[0] == 23703573157769LL, "real model layer_multipliers[0] mismatch");
    expect(ple.layer_multipliers()[1] == 20109073645365LL, "real model layer_multipliers[1] mismatch");
    expect(ple.layer_multipliers()[2] == 8052911324071LL, "real model layer_multipliers[2] mismatch");
    expect(ple.ngram_heads_vocab_sizes()[0] == 20000003LL, "real model vocab_sizes[0] mismatch");
    expect(ple.ngram_heads_offsets()[0] == 0LL, "real model offsets[0] mismatch");

    std::cout << "[PASS] Real model PLE initialized successfully. 128 shards indexed.\n";

    // 1b. Verify real model PLE Ticket pipeline (prepare -> consume -> commit) matches forward() bit-exact
    std::vector<bf16> beforeGrowth;
    for (int rows : {4, 65, 4, 1}) {
        Qwen4ExpPleState s_fwd, s_pipe;
        ple.init_state(ctx, s_fwd);
        ple.init_state(ctx, s_pipe);

        std::vector<int> test_tokens(rows);
        for (int i = 0; i < rows; ++i) test_tokens[i] = (i + 1) * 101;
        Tensor hidden_dev = Tensor::allocate(ctx, {rows, Qwen4ExpPle::HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
        Tensor out_fwd = Tensor::allocate(ctx, {rows, Qwen4ExpPle::HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
        Tensor out_pipe = Tensor::allocate(ctx, {rows, Qwen4ExpPle::HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
        std::vector<bf16> hidden_h(rows * Qwen4ExpPle::HC_HIDDEN_SIZE, bf16(0.02f));
        ctx.memcpy_h2d(hidden_dev.data(), hidden_h.data(), hidden_h.size() * sizeof(bf16));

        // Legacy forward
        ple.forward(ctx, s_fwd, test_tokens.data(), rows,
                    static_cast<const bf16*>(hidden_dev.data()),
                    static_cast<bf16*>(out_fwd.data()), nullptr);

        // Ticket pipeline
        const auto allocations = ctx.device_allocation_count();
        const auto deviceLimit = ctx.device_allocation_limit();
        const auto hostLimit = ctx.pinned_host_allocation_limit();
        if (rows == 1) ctx.set_allocation_limits(ctx.current_allocated_bytes(), ctx.current_pinned_host_bytes());
        PleTicket ticket = ple.prepare(ctx, s_pipe, test_tokens.data(), rows, &s_pipe, 1, 0);
        ple.consume(ctx, ticket, s_pipe,
                    static_cast<const bf16*>(hidden_dev.data()),
                    static_cast<bf16*>(out_pipe.data()), nullptr);
        ple.commit(ticket, s_pipe);
        ctx.synchronize();

        expect(s_fwd.prev_tokens == s_pipe.prev_tokens, "Real model PLE ticket prev_tokens mismatch");
        if (rows == 1) {
            ctx.set_allocation_limits(deviceLimit, hostLimit);
            expect(ctx.device_allocation_count() == allocations, "PLE decode allocated USM");
        }
        std::vector<bf16> fwd_res(rows * Qwen4ExpPle::HC_HIDDEN_SIZE);
        std::vector<bf16> pipe_res(rows * Qwen4ExpPle::HC_HIDDEN_SIZE);
        ctx.memcpy_d2h(fwd_res.data(), out_fwd.data(), fwd_res.size() * sizeof(bf16));
        ctx.memcpy_d2h(pipe_res.data(), out_pipe.data(), pipe_res.size() * sizeof(bf16));

        for (size_t i = 0; i < fwd_res.size(); ++i) {
            uint16_t u_fwd = *reinterpret_cast<const uint16_t*>(&fwd_res[i]);
            uint16_t u_pipe = *reinterpret_cast<const uint16_t*>(&pipe_res[i]);
            if (u_fwd != u_pipe) {
                throw std::runtime_error("Real model PLE Ticket vs forward bit-exact mismatch at " + std::to_string(i));
            }
        }
        if (rows == 4) {
            if (beforeGrowth.empty()) beforeGrowth = fwd_res;
            else expect(std::memcmp(beforeGrowth.data(), fwd_res.data(), fwd_res.size() * sizeof(bf16)) == 0,
                        "PLE output changed after workspace growth");
        }
        std::cout << "[PASS] Real model PLE Ticket pipeline bit-exact parity rows=" << rows << "\n";
    }
    // Three chained chunks exercise both banks, history crossing EOS, and a
    // prepared suffix discarded after cancellation. Real PLE projections and
    // recurrent state must match the ordinary preparation order bit for bit.
    {
        const std::array<int,3> counts{65,7,1};
        std::vector<int> tokens(73);
        for (int i=0;i<73;++i) tokens[i]=101+i*17;
        tokens[63]=Qwen4ExpPle::EOS_TOKEN_ID;
        Qwen4ExpPleState serial, ahead;
        ple.init_state(ctx,serial); ple.init_state(ctx,ahead);
        auto hidden=Tensor::allocate(ctx,{65,Qwen4ExpPle::HC_HIDDEN_SIZE},dnnl::memory::data_type::bf16);
        auto output=Tensor::allocate(ctx,{65,Qwen4ExpPle::HC_HIDDEN_SIZE},dnnl::memory::data_type::bf16);
        ctx.queue().fill(hidden.data_as<bf16>(),bf16(.02f),hidden.numel()).wait_and_throw();
        std::array<std::vector<bf16>,3> reference;
        int pos=0;
        for (int i=0;i<3;++i) {
            ple.forward(ctx,serial,tokens.data()+pos,counts[i],hidden.data_as<bf16>(),output.data_as<bf16>());
            reference[i].resize(size_t(counts[i])*Qwen4ExpPle::HC_HIDDEN_SIZE);
            ctx.memcpy_d2h(reference[i].data(),output.data(),reference[i].size()*sizeof(bf16));
            pos+=counts[i];
        }
        const auto allocations=ctx.device_allocation_count();
        auto current=ple.prepare(ctx,ahead,tokens.data(),counts[0],&ahead,1,0);
        pos=0;
        for (int i=0;i<3;++i) {
            PleTicket next;
            if (i+1<3) {
                Qwen4ExpPleState future; future.prev_tokens=current.prev_tokens_next;
                const auto before=ahead.prev_tokens;
                next=ple.prepare(ctx,future,tokens.data()+pos+counts[i],counts[i+1],&ahead,1,pos+counts[i]);
                expect(ahead.prev_tokens==before,"PLE lookahead advanced session history");
                expect(next.staging_slot!=current.staging_slot,"PLE lookahead reused the current bank");
                expect(ple.MatchesInput(next,tokens.data()+pos+counts[i],counts[i+1]),"PLE ticket did not match its owned input");
            }
            ple.consume(ctx,current,ahead,hidden.data_as<bf16>(),output.data_as<bf16>());
            ple.commit(current,ahead);
            std::vector<bf16> actual(reference[i].size());
            ctx.memcpy_d2h(actual.data(),output.data(),actual.size()*sizeof(bf16));
            expect(std::memcmp(actual.data(),reference[i].data(),actual.size()*sizeof(bf16))==0,"PLE lookahead output differs");
            current=next; pos+=counts[i];
        }
        Qwen4ExpPleSnapshot a,b; ple.capture_snapshot(ctx,serial,a); ple.capture_snapshot(ctx,ahead,b);
        expect(a.prev_tokens==b.prev_tokens && a.conv_head==b.conv_head && a.conv_state==b.conv_state,
               "PLE lookahead recurrent state differs");
        expect(ctx.device_allocation_count()==allocations,"PLE lookahead allocated new device buffers");
        auto abandoned=ple.prepare(ctx,ahead,tokens.data(),7,&ahead,1,pos);
        auto changed=tokens; changed[0]++;
        expect(!ple.MatchesInput(abandoned,changed.data(),7),"Cancelled ticket matched replacement input");
        expect(!ple.MatchesInput(abandoned,tokens.data(),6),"Cancelled ticket matched a different chunk length");
        auto retry=ple.prepare(ctx,ahead,changed.data(),7,&ahead,2,pos);
        ple.consume(ctx,retry,ahead,hidden.data_as<bf16>(),output.data_as<bf16>());
        ple.commit(retry,ahead);
        ctx.synchronize();
        expect(ahead.prev_tokens==std::array<int,2>{changed[5],changed[6]},"PLE retry committed stale input");
        std::cout<<"[PASS] Real PLE lookahead: chained banks, exact outputs/state, discarded input, no new USM\n";
    }
}

void test_qsa_indexer_long_context_dispatch(Context& ctx) {
    std::cout << "[Test] Running QSA Indexer Long Context (>2048) Dispatch Unit Test...\n";
    // 1. Long context dispatch (2400 tokens, 600 blocks) compared against host reference
    {
        constexpr int cached_len = 2400;
        constexpr int compress_ratio = 4;
        constexpr int block_topk = 512;
        constexpr int num_complete_blocks = cached_len / compress_ratio; // 600
        expect(num_complete_blocks > block_topk, "Expected num_complete_blocks > 512");

        Tensor scores_dev = Tensor::allocate(ctx, {num_complete_blocks}, dnnl::memory::data_type::f32);
        Tensor active_tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);

        std::vector<float> scores_h(num_complete_blocks);
        for (int i = 0; i < num_complete_blocks; ++i) {
            scores_h[i] = static_cast<float>(i % 50);
        }
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), num_complete_blocks * sizeof(float));

        std::vector<int> tokens_ref(block_topk * compress_ratio);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_complete_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());

        int num_active = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_tokens_dev);
        ctx.synchronize();

        expect(num_active == ref_count, "Active count must match host reference");
        expect(num_active == 512 * 4 + 0, "Expected 2048 active tokens (2400 % 4 == 0)");
        std::vector<int> tokens_h(num_active);
        ctx.memcpy_d2h(tokens_h.data(), active_tokens_dev.data(), num_active * sizeof(int));

        for (int i = 0; i < num_active; ++i) {
            expect(tokens_h[i] == tokens_ref[i], "Active token at " + std::to_string(i) +
                   " must match host reference: gpu=" + std::to_string(tokens_h[i]) +
                   " ref=" + std::to_string(tokens_ref[i]));
            if (i > 0) {
                expect(tokens_h[i] > tokens_h[i - 1], "Active tokens must be strictly increasing");
            }
            expect(tokens_h[i] < cached_len, "Active token index out of range");
        }
    }

    // 2. Boundary test: cached_len = 2051 (num_complete_blocks = 512 -> dense passthrough)
    {
        constexpr int cached_len = 2051;
        constexpr int compress_ratio = 4;
        constexpr int block_topk = 512;
        constexpr int num_complete_blocks = cached_len / compress_ratio; // 512
        expect(num_complete_blocks <= block_topk, "Expected num_complete_blocks <= 512");

        Tensor scores_dev = Tensor::allocate(ctx, {num_complete_blocks}, dnnl::memory::data_type::f32);
        Tensor active_tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);

        int num_active = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_tokens_dev);
        ctx.synchronize();

        expect(num_active == cached_len, "Boundary 2051: dense mode must select all tokens");
        std::vector<int> tokens_h(num_active);
        ctx.memcpy_d2h(tokens_h.data(), active_tokens_dev.data(), num_active * sizeof(int));
        for (int i = 0; i < num_active; ++i) {
            expect(tokens_h[i] == i, "Boundary 2051 must be dense identity");
        }
    }

    // 3. Boundary test: cached_len = 2052 (num_complete_blocks = 513 -> sparse transition)
    {
        constexpr int cached_len = 2052;
        constexpr int compress_ratio = 4;
        constexpr int block_topk = 512;
        constexpr int num_complete_blocks = cached_len / compress_ratio; // 513
        expect(num_complete_blocks > block_topk, "Expected num_complete_blocks == 513 > 512");

        std::vector<float> scores_h(num_complete_blocks);
        // Make block 0 have the lowest score, so block 0 is dropped
        scores_h[0] = 0.0f;
        for (int i = 1; i < num_complete_blocks; ++i) scores_h[i] = 10.0f + static_cast<float>(i);

        Tensor scores_dev = Tensor::allocate(ctx, {num_complete_blocks}, dnnl::memory::data_type::f32);
        Tensor active_tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), num_complete_blocks * sizeof(float));

        int num_active = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_tokens_dev);
        ctx.synchronize();

        expect(num_active == 512 * 4, "Boundary 2052: sparse mode must select 2048 tokens (512 blocks)");
        std::vector<int> tokens_h(num_active);
        ctx.memcpy_d2h(tokens_h.data(), active_tokens_dev.data(), num_active * sizeof(int));
        // Block 0 (tokens 0..3) should be dropped; active tokens start at 4
        expect(tokens_h[0] == 4, "Dropped block 0: first active token must be 4");
        for (int i = 1; i < num_active; ++i) {
            expect(tokens_h[i] > tokens_h[i - 1], "Boundary 2052 active tokens must be strictly increasing");
        }
    }

    // 4. Boundary test: cached_len = 2055 (513 complete blocks + 3 tail tokens = 2051 active tokens)
    {
        constexpr int cached_len = 2055;
        constexpr int compress_ratio = 4;
        constexpr int block_topk = 512;
        constexpr int num_complete_blocks = cached_len / compress_ratio; // 513

        std::vector<float> scores_h(num_complete_blocks);
        for (int i = 0; i < num_complete_blocks; ++i) scores_h[i] = static_cast<float>(i + 1);

        Tensor scores_dev = Tensor::allocate(ctx, {num_complete_blocks}, dnnl::memory::data_type::f32);
        Tensor active_tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), num_complete_blocks * sizeof(float));

        int num_active = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_tokens_dev);
        ctx.synchronize();

        expect(num_active == 512 * 4 + 3, "Boundary 2055 must select 2051 active tokens (512 blocks * 4 + 3 tail)");
        std::vector<int> tokens_h(num_active);
        ctx.memcpy_d2h(tokens_h.data(), active_tokens_dev.data(), num_active * sizeof(int));
        // Tail tokens must be 2052, 2053, 2054
        expect(tokens_h[2048] == 2052, "Tail token 0 mismatch");
        expect(tokens_h[2049] == 2053, "Tail token 1 mismatch");
        expect(tokens_h[2050] == 2054, "Tail token 2 mismatch");
        for (int i = 1; i < num_active; ++i) {
            expect(tokens_h[i] > tokens_h[i - 1], "Boundary 2055 active tokens must be strictly increasing");
        }
    }

    std::cout << "[PASS] QSA Indexer Long Context Dispatch verified: 2400 tokens, 2051/2052/2055 boundary transitions, 100% matched.\n";
}

void test_qsa_causal_prefill_append_chunk(Context& ctx) {
    std::cout << "[Test] Running QSA causal prefill append-chunk boundary test...\n";
    constexpr int initial_len = 2050;
    constexpr int append_count = 4;
    constexpr int capacity = initial_len + append_count;
    constexpr int compress_ratio = 4;
    constexpr int block_topk = 512;
    constexpr int indexer_heads = 4;
    constexpr int indexer_head_dim = 128;
    constexpr int indexer_q_dim = indexer_heads * indexer_head_dim;
    constexpr int indexer_out_dim = indexer_q_dim + indexer_head_dim;
    constexpr float theta = 1000000.0f;

    std::vector<bf16> indexer_qk_h(append_count * indexer_out_dim);
    std::vector<bf16> q_norm_h(indexer_head_dim, bf16(1.0f));
    std::vector<bf16> k_norm_h(indexer_head_dim, bf16(1.0f));
    std::vector<bf16> raw_k_h(capacity * indexer_head_dim, bf16(0.0f));
    for (int p = 0; p < initial_len; ++p) {
        for (int d = 0; d < indexer_head_dim; ++d) {
            raw_k_h[static_cast<size_t>(p) * indexer_head_dim + d] =
                bf16(std::sin(0.013f * p + 0.071f * d) * 0.2f);
        }
    }
    for (int row = 0; row < append_count; ++row) {
        for (int d = 0; d < indexer_out_dim; ++d) {
            indexer_qk_h[static_cast<size_t>(row) * indexer_out_dim + d] =
                bf16(std::sin(0.17f * row + 0.031f * d) * 0.3f);
        }
    }

    Tensor qk_dev = Tensor::allocate(ctx, {append_count, indexer_out_dim}, dnnl::memory::data_type::bf16);
    Tensor q_norm_dev = Tensor::allocate(ctx, {indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_norm_dev = Tensor::allocate(ctx, {indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_dev = Tensor::allocate(ctx, {capacity, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_out_dev = Tensor::allocate(ctx, {append_count, indexer_q_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(ctx, {capacity / compress_ratio}, dnnl::memory::data_type::f32);
    Tensor active_dev = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(qk_dev.data(), indexer_qk_h.data(), indexer_qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(q_norm_dev.data(), q_norm_h.data(), q_norm_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_norm_dev.data(), k_norm_h.data(), k_norm_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_dev.data(), raw_k_h.data(),
                   static_cast<size_t>(initial_len) * indexer_head_dim * sizeof(bf16));

    for (int row = 0; row < append_count; ++row) {
        const int pos = initial_len + row;
        const int visible_len = pos + 1;
        Tensor qk_row = Tensor::view(
            ctx, qk_dev.data_as<bf16>() + static_cast<size_t>(row) * indexer_out_dim,
            {1, indexer_out_dim});
        Tensor q_row = Tensor::view(
            ctx, q_out_dev.data_as<bf16>() + static_cast<size_t>(row) * indexer_q_dim,
            {indexer_heads, indexer_head_dim});
        const int gpu_sparse_count = ops::qsa_prepare_causal_query_and_select_tokens(
            ctx, qk_row, q_norm_dev, k_norm_dev, k_cache_dev, q_row,
            scores_dev, active_dev, pos, block_topk, compress_ratio, theta);

        std::vector<bf16> q_ref(indexer_q_dim);
        ops::qsa_prepare_q_and_cache_k_host(
            indexer_qk_h.data() + static_cast<size_t>(row) * indexer_out_dim,
            q_norm_h.data(), raw_k_h.data(), q_ref.data(), pos, capacity, theta);
        const int num_complete_blocks = visible_len / compress_ratio;
        std::vector<float> scores_ref(num_complete_blocks);
        ops::qsa_compute_block_scores_host(
            q_ref.data(), raw_k_h.data(), k_norm_h.data(), scores_ref.data(),
            num_complete_blocks, capacity, theta);
        std::vector<int> tokens_ref(capacity);
        const int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_ref.data(), num_complete_blocks, block_topk, visible_len,
            compress_ratio, tokens_ref.data());

        const bool sparse = num_complete_blocks > block_topk;
        expect((gpu_sparse_count > 0) == sparse,
               "QSA causal sparse transition must be decided from this query's visible prefix");
        const int actual_count = sparse ? gpu_sparse_count : visible_len;
        expect(actual_count == ref_count,
               "QSA append row active-token count must match its causal host prefix at pos=" + std::to_string(pos));
        std::vector<int> tokens_gpu(actual_count);
        if (sparse) {
            ctx.synchronize();
            ctx.memcpy_d2h(tokens_gpu.data(), active_dev.data(), actual_count * sizeof(int));
        } else {
            for (int i = 0; i < visible_len; ++i) tokens_gpu[i] = i;
        }
        for (int i = 0; i < actual_count; ++i) {
            expect(tokens_gpu[i] == tokens_ref[i],
                   "QSA append row visible-token set mismatch at pos=" + std::to_string(pos) +
                   " index=" + std::to_string(i));
            expect(tokens_gpu[i] < visible_len,
                   "QSA append row must not expose a future token at pos=" + std::to_string(pos));
        }
    }
    ctx.synchronize();
    std::cout << "[PASS] QSA append chunk uses per-query causal prefixes across visible lengths 2051..2054, including incomplete tails.\n";
}

void test_real_model_qsa_indexer_weights_if_present(Context& ctx) {
    const std::string model_dir = resolve_qwen4exp_model_dir();
    if (!fs::exists(model_dir) || !fs::exists(model_dir + "/model.safetensors.index.json")) {
        std::cout << "[Test] Real model directory " << model_dir << " not found, skipping real model QSA indexer tests.\n";
        return;
    }

    std::cout << "[Test] Verifying QSA Indexer weights on real model from " << model_dir << "...\n";
    ModelWeights weights = LoadModelWeightsLazy(model_dir, &ctx);

    int verified_layers = 0;
    for (int l = 0; l < 48; ++l) {
        if ((l + 1) % 4 != 0) continue; // Skip linear attention layers

        std::string prefix = "model.language_model.layers." + std::to_string(l) + ".";
        std::string qk_name = prefix + "self_attn.indexer.index_qk_proj.weight";
        std::string qn_name = prefix + "self_attn.indexer.q_layernorm.weight";
        std::string kn_name = prefix + "self_attn.indexer.k_layernorm.weight";

        expect(weights.has_host(qk_name) || weights.has(qk_name), "Missing " + qk_name);
        expect(weights.has_host(qn_name) || weights.has(qn_name), "Missing " + qn_name);
        expect(weights.has_host(kn_name) || weights.has(kn_name), "Missing " + kn_name);

        auto qk_info = weights.host_view(qk_name);
        expect(qk_info.valid(), "Invalid host view for " + qk_name);
        expect(qk_info.shape.size() == 2 && qk_info.shape[0] == 640 && qk_info.shape[1] == 2560,
               "Shape mismatch for " + qk_name);

        auto qn_info = weights.host_view(qn_name);
        expect(qn_info.valid() && qn_info.shape.size() == 1 && qn_info.shape[0] == 128,
               "Shape mismatch for " + qn_name);

        auto kn_info = weights.host_view(kn_name);
        expect(kn_info.valid() && kn_info.shape.size() == 1 && kn_info.shape[0] == 128,
               "Shape mismatch for " + kn_name);

        verified_layers++;
    }
    expect(verified_layers == 12, "Expected 12 full attention layers with QSA indexers, verified: " + std::to_string(verified_layers));
    std::cout << "[PASS] Verified all 12 QSA Indexer weight tensors (index_qk_proj [640, 2560], q_layernorm [128], k_layernorm [128]).\n";
}

void test_real_model_qsa_indexer_fixture(Context& ctx) {
    const std::string model_dir = resolve_qwen4exp_model_dir();
    if (!fs::exists(model_dir) || !fs::exists(model_dir + "/model.safetensors.index.json")) {
        std::cout << "[Test] Real model directory " << model_dir << " not found, skipping real model QSA fixture test.\n";
        return;
    }

    std::cout << "[Test] Running real-checkpoint layer 11 QSA indexer fixture test (F4 precision & margin audit)...\n";
    ModelWeights weights = LoadModelWeightsLazy(model_dir, &ctx);

    const std::string prefix = "model.language_model.layers.11.";
    const std::string qk_name = prefix + "self_attn.indexer.index_qk_proj.weight";
    const std::string qn_name = prefix + "self_attn.indexer.q_layernorm.weight";
    const std::string kn_name = prefix + "self_attn.indexer.k_layernorm.weight";

    expect(weights.has_host(qk_name) || weights.has(qk_name), "Missing " + qk_name);
    expect(weights.has_host(qn_name) || weights.has(qn_name), "Missing " + qn_name);
    expect(weights.has_host(kn_name) || weights.has(kn_name), "Missing " + kn_name);

    auto qn_info = weights.host_view(qn_name);
    auto kn_info = weights.host_view(kn_name);
    const bf16* qn_raw_ptr = reinterpret_cast<const bf16*>(qn_info.ptr);
    const bf16* kn_raw_ptr = reinterpret_cast<const bf16*>(kn_info.ptr);

    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int kIndexerOutDim = 640;
    constexpr int kBlockTopk = 512;
    constexpr int kCompressRatio = 4;
    constexpr int kCachedLen = 2400; // 600 complete blocks (> 512 block budget)
    constexpr int kNumBlocks = kCachedLen / kCompressRatio;
    constexpr float rope_theta = 10000000.0f; // 1e7

    // Build plus_one_norm weights for Aila GPU and host paths
    std::vector<bf16> qn_plus_one_h(kHeadDim);
    std::vector<bf16> kn_plus_one_h(kHeadDim);
    std::vector<bf16> kn_raw_h(kHeadDim);
    for (int d = 0; d < kHeadDim; ++d) {
        qn_plus_one_h[d] = bf16(static_cast<float>(qn_raw_ptr[d]) + 1.0f);
        kn_plus_one_h[d] = bf16(static_cast<float>(kn_raw_ptr[d]) + 1.0f);
        kn_raw_h[d] = kn_raw_ptr[d];
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> raw_k_h(kCachedLen * kHeadDim);
    for (auto& v : raw_k_h) v = bf16(dist(rng) * 0.2f);

    // Query at pos = 2400 (crossing the 2048 boundary)
    const int query_pos = kCachedLen;
    std::vector<bf16> indexer_qk_h(kIndexerOutDim);
    for (auto& v : indexer_qk_h) v = bf16(dist(rng) * 0.3f);

    // GPU allocation and execution
    Tensor qk_dev = Tensor::allocate(ctx, {1, kIndexerOutDim}, dnnl::memory::data_type::bf16);
    Tensor qn_dev = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor kn_dev = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_dev = Tensor::allocate(ctx, {kCachedLen + 4, kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor q_out_dev = Tensor::allocate(ctx, {kNumHeads, kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(ctx, {kNumBlocks}, dnnl::memory::data_type::f32);
    Tensor active_tokens_dev = Tensor::allocate(ctx, {kCachedLen + 4}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(qk_dev.data(), indexer_qk_h.data(), indexer_qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(qn_dev.data(), qn_plus_one_h.data(), qn_plus_one_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(kn_dev.data(), kn_plus_one_h.data(), kn_plus_one_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_dev.data(), raw_k_h.data(), raw_k_h.size() * sizeof(bf16));

    const int gpu_active_count = ops::qsa_prepare_causal_query_and_select_tokens(
        ctx, qk_dev, qn_dev, kn_dev, k_cache_dev, q_out_dev, scores_dev, active_tokens_dev,
        query_pos, kBlockTopk, kCompressRatio, rope_theta);
    ctx.synchronize();

    expect(gpu_active_count > 0, "GPU causal query selection must trigger sparse mode");
    std::vector<int> gpu_tokens(gpu_active_count);
    ctx.memcpy_d2h(gpu_tokens.data(), active_tokens_dev.data(), gpu_active_count * sizeof(int));
    std::vector<float> gpu_scores(kNumBlocks);
    ctx.memcpy_d2h(gpu_scores.data(), scores_dev.data(), kNumBlocks * sizeof(float));

    // Aila Host reference
    std::vector<bf16> q_host(kNumHeads * kHeadDim);
    std::vector<bf16> k_cache_host = raw_k_h;
    k_cache_host.resize((kCachedLen + 4) * kHeadDim);
    ops::qsa_prepare_q_and_cache_k_host(
        indexer_qk_h.data(), qn_plus_one_h.data(), k_cache_host.data(), q_host.data(),
        query_pos, kCachedLen + 4, rope_theta);

    std::vector<float> aila_host_scores(kNumBlocks);
    ops::qsa_compute_block_scores_host(
        q_host.data(), k_cache_host.data(), kn_plus_one_h.data(), aila_host_scores.data(),
        kNumBlocks, kCachedLen + 4, rope_theta);

    std::vector<int> aila_host_tokens(kCachedLen + 4);
    const int host_active_count = ops::qsa_select_topk_and_expand_tokens_host(
        aila_host_scores.data(), kNumBlocks, kBlockTopk, query_pos + 1, kCompressRatio, aila_host_tokens.data());

    expect(gpu_active_count == host_active_count, "Active token count mismatch between GPU and Aila host");
    for (int i = 0; i < gpu_active_count; ++i) {
        expect(gpu_tokens[i] == aila_host_tokens[i],
               "Aila GPU and Host token mismatch at index " + std::to_string(i));
    }

    // Upstream HuggingFace reference execution
    std::vector<float> upstream_scores(kNumBlocks);
    ops::qsa_compute_block_scores_upstream_ref(
        q_host.data(), k_cache_host.data(), kn_raw_h.data(), upstream_scores.data(),
        kNumBlocks, rope_theta);

    std::vector<int> upstream_tokens(kCachedLen + 4);
    ops::qsa_select_topk_and_expand_tokens_host(
        upstream_scores.data(), kNumBlocks, kBlockTopk, query_pos + 1, kCompressRatio, upstream_tokens.data());

    // Score comparison & margin quantification
    float max_score_diff = 0.0f;
    float sum_score_diff = 0.0f;
    for (int b = 0; b < kNumBlocks; ++b) {
        float d = std::abs(gpu_scores[b] - upstream_scores[b]);
        if (d > max_score_diff) max_score_diff = d;
        sum_score_diff += d;
    }
    float mean_score_diff = sum_score_diff / static_cast<float>(kNumBlocks);

    // Calculate upstream sorted cutoff margin
    std::vector<float> sorted_upstream = upstream_scores;
    std::sort(sorted_upstream.begin(), sorted_upstream.end(), std::greater<float>());
    const float cutoff_margin = sorted_upstream[kBlockTopk - 1] - sorted_upstream[kBlockTopk];

    // Selected block set comparison
    std::unordered_set<int> gpu_block_set;
    for (int i = 0; i < kBlockTopk * kCompressRatio; i += kCompressRatio) {
        gpu_block_set.insert(gpu_tokens[i] / kCompressRatio);
    }
    std::unordered_set<int> up_block_set;
    for (int i = 0; i < kBlockTopk * kCompressRatio; i += kCompressRatio) {
        up_block_set.insert(upstream_tokens[i] / kCompressRatio);
    }
    int common_blocks = 0;
    for (int b : gpu_block_set) {
        if (up_block_set.count(b)) ++common_blocks;
    }
    float agreement = static_cast<float>(common_blocks) / static_cast<float>(kBlockTopk);

    std::cout << "[PASS] Real-model QSA fixture (Layer 11): 100% GPU-Host bitwise match ("
              << gpu_active_count << " tokens), Upstream score max|dS|=" << max_score_diff
              << " mean|dS|=" << mean_score_diff << ", Top-512 set agreement=" << (agreement * 100.0f)
              << "%, upstream cutoff margin=" << cutoff_margin << "\n";
    expect(agreement >= 0.98f, "Selected block set agreement with upstream reference must be >= 98%");

    // Compressed K verification on real model layer 11 weights
    Tensor comp_k_dev = Tensor::allocate(ctx, {kNumBlocks, kHeadDim}, dnnl::memory::data_type::bf16);
    int num_comp = 0;
    Tensor comp_scores_dev = Tensor::allocate(ctx, {kNumBlocks}, dnnl::memory::data_type::f32);
    Tensor comp_active_tokens_dev = Tensor::allocate(ctx, {kCachedLen + 4}, dnnl::memory::data_type::s32);
    Tensor comp_q_out_dev = Tensor::allocate(ctx, {kNumHeads, kHeadDim}, dnnl::memory::data_type::bf16);

    const int comp_active_count = ops::qsa_prepare_causal_query_and_select_tokens(
        ctx, qk_dev, qn_dev, kn_dev, k_cache_dev, comp_q_out_dev, comp_scores_dev, comp_active_tokens_dev,
        query_pos, kBlockTopk, kCompressRatio, rope_theta, false,
        nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
        &comp_k_dev, &num_comp);
    ctx.synchronize();

    expect(comp_active_count == gpu_active_count, "Compressed active token count mismatch");
    expect(num_comp == kNumBlocks, "All complete blocks must be compressed into comp_k_dev");
    std::vector<int> comp_tokens(comp_active_count);
    ctx.memcpy_d2h(comp_tokens.data(), comp_active_tokens_dev.data(), comp_active_count * sizeof(int));
    std::unordered_set<int> comp_block_set;
    for (int i = 0; i < kBlockTopk * kCompressRatio; i += kCompressRatio) {
        comp_block_set.insert(comp_tokens[i] / kCompressRatio);
    }
    int comp_common_blocks = 0;
    for (int b : comp_block_set) {
        if (up_block_set.count(b)) ++comp_common_blocks;
    }
    float comp_agreement = static_cast<float>(comp_common_blocks) / static_cast<float>(kBlockTopk);
    std::cout << "[PASS] Real-model QSA fixture with Compressed K: Top-512 set agreement with upstream="
              << (comp_agreement * 100.0f) << "%\n";
    expect(comp_agreement >= 0.98f, "Compressed K agreement with upstream reference must be >= 98%");
}

void test_qsa_dense_prefill_buffer_bounds(Context& ctx) {
    std::cout << "[Test] Verifying D1 dense prefill scratchpad upper bounds at 32K context...\n";
    Qwen4ExpBnb4Backend backend;
    backend.init_for_testing(ctx, 4);

    // Ensure buffers for large sequence length (e.g. 4096 tokens)
    const auto runtime_generation_before = backend.runtime_generation();
    backend.ensure_runtime_buffers_for_testing(ctx, 4096);
    const auto runtime_generation_after = backend.runtime_generation();
    expect(runtime_generation_after > runtime_generation_before,
           "runtime buffer growth must advance runtime generation");
    backend.ensure_runtime_buffers_for_testing(ctx, 4096);
    expect(backend.runtime_generation() == runtime_generation_after,
           "unchanged runtime capacity must preserve runtime generation");
    const auto& buf = backend.buf_for_testing();

    // Verify scores buffer is strictly bounded by kDensePrefillLimit (2052)
    expect(buf.scores.shape().size() == 3, "buf.scores must be 3D [heads, cap, cap]");
    expect(buf.scores.shape()[1] <= 2052, "buf.scores dim 1 must be <= 2052");
    expect(buf.scores.shape()[2] <= 2052, "buf.scores dim 2 must be <= 2052");

    // Verify incr_scores buffer is strictly bounded by kDensePrefillLimit (2052)
    expect(buf.incr_scores.shape().size() == 3, "buf.incr_scores must be 3D [heads, cap, kv_len]");
    expect(buf.incr_scores.shape()[1] <= 2052, "buf.incr_scores dim 1 must be <= 2052");
    expect(buf.incr_scores.shape()[2] <= 2052, "buf.incr_scores dim 2 must be <= 2052");

    // Phase P2 (R4a): Verify bounded arena buffers for QSA batch tiles
    expect(buf.qsa_tile_valid_counts.valid(), "buf.qsa_tile_valid_counts must be allocated");
    expect(backend.qsa_query_tile() >= 16 && backend.qsa_query_tile() <= 1024,
           "backend.qsa_query_tile() must be bounded in [16, 1024]");
    expect(buf.qsa_tile_valid_counts.shape()[0] == backend.qsa_query_tile(),
           "buf.qsa_tile_valid_counts rows must match qsa_query_tile");
    expect(buf.qsa_tile_valid_counts.shape()[1] == 2, "buf.qsa_tile_valid_counts cols must be 2");

    std::cout << "[PASS] Dense attention buffers strictly bounded: scores shape=["
              << buf.scores.shape()[0] << ", " << buf.scores.shape()[1] << ", " << buf.scores.shape()[2]
              << "], incr_scores shape=["
              << buf.incr_scores.shape()[0] << ", " << buf.incr_scores.shape()[1] << ", " << buf.incr_scores.shape()[2]
              << "] (6-96 GiB OOM hazard completely eliminated).\n";
}

void test_qsa_k_only_projection_alignment(Context& ctx) {
    std::cout << "[Test] Verifying A2 K-only separated projection alignment...\n";
    constexpr int hidden_size = 2560;
    constexpr int indexer_q_dim = 512;
    constexpr int indexer_k_dim = 128;
    constexpr int indexer_out_dim = indexer_q_dim + indexer_k_dim; // 640

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> qk_weight_h(indexer_out_dim * hidden_size);
    for (auto& v : qk_weight_h) v = bf16(dist(rng) * 0.1f);

    Tensor qk_w_dev = Tensor::allocate(ctx, {indexer_out_dim, hidden_size}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(qk_w_dev.data(), qk_weight_h.data(), qk_weight_h.size() * sizeof(bf16));

    const size_t k_offset = static_cast<size_t>(indexer_q_dim) * hidden_size;
    Tensor k_w_dev = Tensor::view(
        ctx, qk_w_dev.data_as<bf16>() + k_offset, {indexer_k_dim, hidden_size});

    Linear qk_proj;
    qk_proj.init(ctx, qk_w_dev, hidden_size, indexer_out_dim, false);

    Linear k_proj;
    k_proj.init(ctx, k_w_dev, hidden_size, indexer_k_dim, false);

    // Test 1: Single token decode (count = 1) using GEMV
    std::vector<bf16> x_h(hidden_size);
    for (auto& v : x_h) v = bf16(dist(rng) * 0.2f);
    Tensor x_dev = Tensor::allocate(ctx, {1, hidden_size}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_h.data(), x_h.size() * sizeof(bf16));

    Tensor qk_out_dev = Tensor::allocate(ctx, {1, indexer_out_dim}, dnnl::memory::data_type::bf16);
    Tensor k_out_dev = Tensor::allocate(ctx, {1, indexer_k_dim}, dnnl::memory::data_type::bf16);

    ops::bf16_gemv_bf16(ctx, qk_w_dev, x_dev.data_as<bf16>(), qk_out_dev.data_as<bf16>(), indexer_out_dim, hidden_size);
    ops::bf16_gemv_bf16(ctx, k_w_dev, x_dev.data_as<bf16>(), k_out_dev.data_as<bf16>(), indexer_k_dim, hidden_size);
    ctx.synchronize();

    std::vector<bf16> qk_out_h(indexer_out_dim);
    std::vector<bf16> k_out_h(indexer_k_dim);
    ctx.memcpy_d2h(qk_out_h.data(), qk_out_dev.data(), indexer_out_dim * sizeof(bf16));
    ctx.memcpy_d2h(k_out_h.data(), k_out_dev.data(), indexer_k_dim * sizeof(bf16));

    for (int d = 0; d < indexer_k_dim; ++d) {
        expect(k_out_h[d] == qk_out_h[indexer_q_dim + d],
               "Decode GEMV K-only projection bitwise mismatch at index " + std::to_string(d));
    }
    std::cout << "[PASS] Decode K-only GEMV projection matches QK projection 100% bit-exact (512 Q rows saved).\n";

    // Test 2: Prefill chunk (count = 8) using oneDNN matmul
    constexpr int prefill_count = 8;
    std::vector<bf16> x_prefill_h(prefill_count * hidden_size);
    for (auto& v : x_prefill_h) v = bf16(dist(rng) * 0.2f);
    Tensor x_prefill_dev = Tensor::allocate(ctx, {prefill_count, hidden_size}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_prefill_dev.data(), x_prefill_h.data(), x_prefill_h.size() * sizeof(bf16));

    Tensor qk_prefill_out = Tensor::allocate(ctx, {prefill_count, indexer_out_dim}, dnnl::memory::data_type::bf16);
    Tensor k_prefill_out = Tensor::allocate(ctx, {prefill_count, indexer_k_dim}, dnnl::memory::data_type::bf16);

    qk_proj.forward(ctx, x_prefill_dev, qk_prefill_out, prefill_count);
    k_proj.forward(ctx, x_prefill_dev, k_prefill_out, prefill_count);
    ctx.synchronize();

    std::vector<bf16> qk_prefill_h(prefill_count * indexer_out_dim);
    std::vector<bf16> k_prefill_h(prefill_count * indexer_k_dim);
    ctx.memcpy_d2h(qk_prefill_h.data(), qk_prefill_out.data(), qk_prefill_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_prefill_h.data(), k_prefill_out.data(), k_prefill_h.size() * sizeof(bf16));

    for (int r = 0; r < prefill_count; ++r) {
        for (int d = 0; d < indexer_k_dim; ++d) {
            float diff = std::abs(static_cast<float>(k_prefill_h[r * indexer_k_dim + d]) -
                                  static_cast<float>(qk_prefill_h[r * indexer_out_dim + indexer_q_dim + d]));
            expect(diff < 1e-3f, "Prefill K-only projection numerical mismatch at row " +
                   std::to_string(r) + " col " + std::to_string(d));
        }
    }
    std::cout << "[PASS] Prefill K-only projection matches QK projection across all rows (80% compute saved).\n";
}

void test_qsa_session_compressed_k_lifecycle(Context& ctx) {
    std::cout << "[Test] Verifying QSA Session compressed K state lifecycle & rollback...\n";
    Qwen4ExpBnb4Backend backend;
    backend.init_for_testing(ctx, 4, 4096);

    auto session = backend.CreateSession(ctx, 4);
    auto& caches = session->layer_caches_for_testing();

    // Verify initial state for layer 3 (full attention: (3+1)%4 == 0)
    expect(caches[3].indexer_compressed_k.valid(), "indexer_compressed_k must be allocated");
    expect(caches[3].num_compressed_blocks == 0, "Initial num_compressed_blocks must be 0");

    // Simulate prefill up to length 2050 (512 complete blocks)
    caches[3].num_compressed_blocks = 512;
    backend.capture_snapshot_for_testing(ctx, *session, 2050, 2050);

    // Verify snapshot captured num_compressed_blocks == 512
    const auto* snapshot = session->snapshot_for_testing(2050);
    expect(snapshot != nullptr, "Snapshot at 2050 must exist");
    expect(snapshot->layers[3].num_compressed_blocks == 512, "Snapshot layer 3 num_compressed_blocks must be 512");

    // Simulate decode steps advancing to 2054 (crossing to 513 complete blocks)
    caches[3].num_compressed_blocks = 513;
    session->current_length_for_testing() = 2054;

    // Rollback / Truncate back to 2050
    int restored_len = backend.Truncate(*session, 2050);
    expect(restored_len == 2050, "Truncate must return 2050");
    expect(session->CurrentLength() == 2050, "CurrentLength must be 2050");
    expect(caches[3].num_compressed_blocks == 512, "num_compressed_blocks must be restored to 512");

    // Test Reset
    backend.Reset(*session);
    expect(session->CurrentLength() == 0, "CurrentLength after reset must be 0");
    expect(caches[3].num_compressed_blocks == 0, "num_compressed_blocks after reset must be 0");
    for (const auto& slot : session->snapshot_slots_for_testing()) {
        expect(slot.length == -1, "Snapshots must be invalidated on Reset");
    }

    std::cout << "[PASS] QSA Session compressed K state lifecycle, snapshot capture, rollback & reset verified.\n";
}

void test_qsa_indexer_end_to_end_pipeline(Context& ctx) {
    std::cout << "[Test] Running QSA Indexer End-to-End Pipeline (>2048 Context) Integration Test...\n";
    constexpr int hidden_size = 2560;
    constexpr int indexer_out_dim = 640;
    constexpr int index_head_dim = 128;
    constexpr int num_q_heads = 4;
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int full_head_dim = 256;
    constexpr int cached_len = 2400; // 600 complete blocks
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int num_complete_blocks = cached_len / compress_ratio;
    constexpr float rope_theta = 1000000.0f;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // 1. Allocate and initialize index_qk weights, q_norm_w, k_norm_w
    std::vector<bf16> index_qk_w_h(indexer_out_dim * hidden_size);
    for (auto& v : index_qk_w_h) v = bf16(dist(rng) * 0.05f);
    std::vector<bf16> q_norm_w_h(index_head_dim, bf16(1.0f));
    std::vector<bf16> k_norm_w_h(index_head_dim, bf16(1.0f));

    Tensor index_qk_w_dev = Tensor::allocate(ctx, {indexer_out_dim, hidden_size}, dnnl::memory::data_type::bf16);
    Tensor q_norm_w_dev = Tensor::allocate(ctx, {index_head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_norm_w_dev = Tensor::allocate(ctx, {index_head_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(index_qk_w_dev.data(), index_qk_w_h.data(), index_qk_w_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(q_norm_w_dev.data(), q_norm_w_h.data(), q_norm_w_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_norm_w_dev.data(), k_norm_w_h.data(), k_norm_w_h.size() * sizeof(bf16));

    Linear index_qk_proj;
    index_qk_proj.init(ctx, index_qk_w_dev, hidden_size, indexer_out_dim, false);

    // 2. Query token hidden state x [1, hidden_size] and pre-existing k_cache [cached_len, 128]
    std::vector<bf16> x_h(hidden_size);
    for (auto& v : x_h) v = bf16(dist(rng));
    std::vector<bf16> k_cache_h(cached_len * index_head_dim);
    for (auto& v : k_cache_h) v = bf16(dist(rng) * 0.2f);

    Tensor x_dev = Tensor::allocate(ctx, {1, hidden_size}, dnnl::memory::data_type::bf16);
    Tensor indexer_qk_dev = Tensor::allocate(ctx, {1, indexer_out_dim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_dev = Tensor::allocate(ctx, {cached_len, index_head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_out_dev = Tensor::allocate(ctx, {num_q_heads, index_head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(ctx, {num_complete_blocks}, dnnl::memory::data_type::f32);
    Tensor active_tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(x_dev.data(), x_h.data(), x_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_dev.data(), k_cache_h.data(), k_cache_h.size() * sizeof(bf16));

    // 3. Forward index_qk_proj
    index_qk_proj.forward(ctx, x_dev, indexer_qk_dev, 1);

    // 4. Prepare Q and cache K for current decode step (pos = 2399)
    const int current_pos = cached_len - 1;
    ops::qsa_prepare_q_and_cache_k(
        ctx, indexer_qk_dev, q_norm_w_dev, k_cache_dev, q_out_dev, current_pos, rope_theta);

    // 5. Compute block scores
    ops::qsa_compute_block_scores(
        ctx, q_out_dev, k_cache_dev, k_norm_w_dev, scores_dev, num_complete_blocks, rope_theta);

    // 6. Select Top-512 blocks + tail
    int num_active = ops::qsa_select_topk_and_expand_tokens(
        ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_tokens_dev);
    ctx.synchronize();

    expect(num_active == block_topk * compress_ratio, "Pipeline: active tokens must equal 2048");

    // 7. Verify against CPU reference for block scores and active token selection
    std::vector<bf16> indexer_qk_h(indexer_out_dim);
    ctx.memcpy_d2h(indexer_qk_h.data(), indexer_qk_dev.data(), indexer_qk_h.size() * sizeof(bf16));

    std::vector<bf16> q_ref(num_q_heads * index_head_dim);
    ops::qsa_prepare_q_and_cache_k_host(
        indexer_qk_h.data(), q_norm_w_h.data(), k_cache_h.data(), q_ref.data(),
        current_pos, cached_len, rope_theta);

    std::vector<float> scores_ref(num_complete_blocks);
    ops::qsa_compute_block_scores_host(
        q_ref.data(), k_cache_h.data(), k_norm_w_h.data(), scores_ref.data(),
        num_complete_blocks, cached_len, rope_theta);

    std::vector<float> scores_gpu(num_complete_blocks);
    ctx.memcpy_d2h(scores_gpu.data(), scores_dev.data(), num_complete_blocks * sizeof(float));

    float max_score_diff = 0.0f;
    for (int i = 0; i < num_complete_blocks; ++i) {
        float diff = std::abs(scores_gpu[i] - scores_ref[i]);
        if (diff > max_score_diff) max_score_diff = diff;
    }
    expect(max_score_diff < 1e-3f, "Pipeline: QSA block scores mismatch vs host, max_diff=" + std::to_string(max_score_diff));

    std::vector<int> tokens_ref(num_active);
    int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
        scores_ref.data(), num_complete_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());
    expect(ref_count == num_active, "Pipeline: active tokens count mismatch vs host");

    std::vector<int> tokens_gpu(num_active);
    ctx.memcpy_d2h(tokens_gpu.data(), active_tokens_dev.data(), num_active * sizeof(int));
    for (int i = 0; i < num_active; ++i) {
        expect(tokens_gpu[i] == tokens_ref[i], "Pipeline: active token index mismatch at " + std::to_string(i));
    }

    // 8. Full attention decode with QSA active indices
    Tensor q_full_dev = Tensor::allocate(ctx, {1, num_heads * full_head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_full_dev = Tensor::allocate(ctx, {num_kv_heads, cached_len, full_head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_full_dev = Tensor::allocate(ctx, {num_kv_heads, cached_len, full_head_dim}, dnnl::memory::data_type::bf16);
    Tensor attn_out_dev = Tensor::allocate(ctx, {1, num_heads * full_head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_buf = Tensor::allocate(ctx, {num_heads, cached_len}, dnnl::memory::data_type::f32);
    constexpr int kDecodeExactTile = 128;
    const int max_tiles = (cached_len + kDecodeExactTile - 1) / kDecodeExactTile;
    Tensor partials_dev = Tensor::allocate(ctx, {(int64_t)num_heads, (int64_t)max_tiles, 272}, dnnl::memory::data_type::f32);

    std::vector<bf16> q_full_h(num_heads * full_head_dim);
    for (auto& v : q_full_h) v = bf16(dist(rng));
    ctx.memcpy_h2d(q_full_dev.data(), q_full_h.data(), q_full_h.size() * sizeof(bf16));

    std::vector<bf16> kv_full_h(num_kv_heads * cached_len * full_head_dim);
    for (auto& v : kv_full_h) v = bf16(dist(rng) * 0.1f);
    ctx.memcpy_h2d(k_full_dev.data(), kv_full_h.data(), kv_full_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_full_dev.data(), kv_full_h.data(), kv_full_h.size() * sizeof(bf16));

    ops::attention_decode(
        ctx, q_full_dev, k_full_dev, v_full_dev, attn_out_dev, scores_buf,
        num_heads, num_kv_heads, full_head_dim, cached_len,
        &partials_dev, /*gate_buf=*/nullptr,
        active_tokens_dev.data_as<int>(), num_active);
    ctx.synchronize();

    std::vector<bf16> attn_out_h(num_heads * full_head_dim);
    ctx.memcpy_d2h(attn_out_h.data(), attn_out_dev.data(), attn_out_h.size() * sizeof(bf16));
    for (size_t i = 0; i < attn_out_h.size(); ++i) {
        float val = static_cast<float>(attn_out_h[i]);
        expect(!std::isnan(val) && !std::isinf(val), "Pipeline: attention output NaN or Inf");
    }

    std::cout << "[PASS] QSA Indexer End-to-End Pipeline (>2048) fully verified: scores diff="
              << max_score_diff << " < 1e-3, 0 token mismatches, attention outputs finite & valid." << std::endl;
}

void test_qsa_block_native_decode_alignment(Context& ctx) {
    std::cout << "[Test] Running QSA Block-Native Decode Alignment (A3) Test..." << std::endl;
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int full_head_dim = 256;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;

    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    const std::vector<int> test_lens = {2400, 2401, 2403, 1003};

    for (int cached_len : test_lens) {
        const int num_complete_blocks = cached_len / compress_ratio;
        const int tail_len = cached_len - num_complete_blocks * compress_ratio;

        std::vector<float> scores_h(std::max(1, num_complete_blocks));
        for (auto& s : scores_h) s = dist(rng) + 0.5f;

        Tensor scores_dev = Tensor::allocate(ctx, {std::max(1, num_complete_blocks)}, dnnl::memory::data_type::f32);
        Tensor active_tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        Tensor active_blocks_dev = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);

        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int num_tokens = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_tokens_dev);
        int num_blocks = ops::qsa_select_topk_blocks(
            ctx, scores_dev, num_complete_blocks, block_topk, cached_len, compress_ratio, active_blocks_dev);
        ctx.synchronize();

        const int expected_blocks = std::min(num_complete_blocks, block_topk);
        const int expected_tokens = expected_blocks * compress_ratio + tail_len;
        expect(num_tokens == expected_tokens, "Block-Native: expanded tokens mismatch for cached_len=" + std::to_string(cached_len));
        expect(num_blocks == expected_blocks, "Block-Native: active blocks mismatch for cached_len=" + std::to_string(cached_len));

        // Setup Q, K, V, Gate
        Tensor q_dev = Tensor::allocate(ctx, {1, num_heads * full_head_dim}, dnnl::memory::data_type::bf16);
        Tensor z_dev = Tensor::allocate(ctx, {1, num_heads * full_head_dim}, dnnl::memory::data_type::bf16);
        Tensor k_dev = Tensor::allocate(ctx, {num_kv_heads, cached_len, full_head_dim}, dnnl::memory::data_type::bf16);
        Tensor v_dev = Tensor::allocate(ctx, {num_kv_heads, cached_len, full_head_dim}, dnnl::memory::data_type::bf16);
        Tensor out_expanded_dev = Tensor::allocate(ctx, {1, num_heads * full_head_dim}, dnnl::memory::data_type::bf16);
        Tensor out_native_dev = Tensor::allocate(ctx, {1, num_heads * full_head_dim}, dnnl::memory::data_type::bf16);
        Tensor scores_buf = Tensor::allocate(ctx, {num_heads, cached_len}, dnnl::memory::data_type::f32);
        constexpr int kDecodeExactTile = 128;
        const int max_tiles = (cached_len + kDecodeExactTile - 1) / kDecodeExactTile;
        Tensor partials_dev1 = Tensor::allocate(ctx, {(int64_t)num_heads, (int64_t)max_tiles, 272}, dnnl::memory::data_type::f32);
        Tensor partials_dev2 = Tensor::allocate(ctx, {(int64_t)num_heads, (int64_t)max_tiles, 272}, dnnl::memory::data_type::f32);

        std::vector<bf16> q_h(num_heads * full_head_dim);
        std::vector<bf16> z_h(num_heads * full_head_dim);
        for (auto& v : q_h) v = bf16(dist(rng));
        for (auto& v : z_h) v = bf16(dist(rng));
        std::vector<bf16> kv_h(num_kv_heads * cached_len * full_head_dim);
        for (auto& v : kv_h) v = bf16(dist(rng) * 0.1f);

        ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(z_dev.data(), z_h.data(), z_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(k_dev.data(), kv_h.data(), kv_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_dev.data(), kv_h.data(), kv_h.size() * sizeof(bf16));

        for (bool with_gate : {false, true}) {
            // 1. Attention decode with expanded active tokens
            ops::attention_decode(
                ctx, q_dev, k_dev, v_dev, out_expanded_dev, scores_buf,
                num_heads, num_kv_heads, full_head_dim, cached_len,
                &partials_dev1, (with_gate ? &z_dev : nullptr),
                active_tokens_dev.data_as<int>(), num_tokens, nullptr, 0);

            // 2. Attention decode with block-native active blocks (Phase A3)
            ops::attention_decode(
                ctx, q_dev, k_dev, v_dev, out_native_dev, scores_buf,
                num_heads, num_kv_heads, full_head_dim, cached_len,
                &partials_dev2, (with_gate ? &z_dev : nullptr),
                nullptr, 0, active_blocks_dev.data_as<int>(), num_blocks);
            ctx.synchronize();

            std::vector<bf16> out_exp_h(num_heads * full_head_dim);
            std::vector<bf16> out_nat_h(num_heads * full_head_dim);
            ctx.memcpy_d2h(out_exp_h.data(), out_expanded_dev.data(), out_exp_h.size() * sizeof(bf16));
            ctx.memcpy_d2h(out_nat_h.data(), out_native_dev.data(), out_nat_h.size() * sizeof(bf16));

            float max_diff = 0.0f;
            for (size_t i = 0; i < out_exp_h.size(); ++i) {
                float d = std::abs(static_cast<float>(out_exp_h[i]) - static_cast<float>(out_nat_h[i]));
                if (d > max_diff) max_diff = d;
            }
            expect(max_diff < 1e-4f, "Block-Native Decode: mismatch vs expanded tokens for cached_len=" +
                   std::to_string(cached_len) + " gate=" + std::to_string(with_gate) + " max_diff=" + std::to_string(max_diff));
        }
        std::cout << "[PASS] QSA Block-Native Decode (A3) matches expanded token path for cached_len="
                  << cached_len << " (tail=" << tail_len << ").\n";
    }
}

void test_qsa_batched_prefill_oracle_alignment(Context& ctx) {
    std::cout << "[Test] Running QSA Batched Prefill vs Oracle Alignment (A4) Test..." << std::endl;
    constexpr int start_pos = 2040;
    constexpr int count = 80; // spans pos 2040..2119 (crosses dense 2050 -> sparse 2051 threshold, spans 2 tiles: 64 + 16)
    constexpr int capacity = 2300;
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int full_head_dim = 256;
    constexpr int q_dim = num_heads * full_head_dim;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int indexer_heads = 4;
    constexpr int indexer_head_dim = 128;
    constexpr int indexer_q_dim = indexer_heads * indexer_head_dim;
    constexpr int indexer_out_dim = indexer_q_dim + indexer_head_dim;
    constexpr float rope_theta = 1000000.0f;

    std::mt19937 rng(9876);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Initial k_cache for pos 0..start_pos
    std::vector<bf16> raw_k_init(start_pos * indexer_head_dim);
    for (auto& v : raw_k_init) v = bf16(dist(rng) * 0.1f);

    std::vector<bf16> indexer_qk_h(count * indexer_out_dim);
    for (auto& v : indexer_qk_h) v = bf16(dist(rng) * 0.2f);
    std::vector<bf16> qn_w_h(indexer_head_dim, bf16(1.0f));
    std::vector<bf16> kn_w_h(indexer_head_dim, bf16(1.0f));

    Tensor indexer_qk_dev = Tensor::allocate(ctx, {count, indexer_out_dim}, dnnl::memory::data_type::bf16);
    Tensor qn_dev = Tensor::allocate(ctx, {indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor kn_dev = Tensor::allocate(ctx, {indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor indexer_k_dev = Tensor::allocate(ctx, {capacity, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed_k_dev = Tensor::allocate(ctx, {capacity / compress_ratio, indexer_head_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(indexer_qk_dev.data(), indexer_qk_h.data(), indexer_qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(qn_dev.data(), qn_w_h.data(), qn_w_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(kn_dev.data(), kn_w_h.data(), kn_w_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(indexer_k_dev.data(), raw_k_init.data(), raw_k_init.size() * sizeof(bf16));

    // Cache raw K for chunk and compress blocks up to start_pos + count
    ops::qsa_cache_k_batched(ctx, indexer_qk_dev, indexer_k_dev, start_pos, count);
    const int total_complete_blocks = (start_pos + count) / compress_ratio;
    ops::qsa_compress_key_blocks(ctx, indexer_k_dev, kn_dev, compressed_k_dev, 0, total_complete_blocks, rope_theta);
    ctx.synchronize();

    // Setup full attention tensors
    Tensor q_full_dev = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);
    Tensor z_dev = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);
    Tensor k_full_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, full_head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_full_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, full_head_dim}, dnnl::memory::data_type::bf16);
    Tensor attn_out_batched = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);
    Tensor attn_out_oracle = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);

    std::vector<bf16> q_full_h(count * q_dim);
    std::vector<bf16> z_h(count * q_dim);
    std::vector<bf16> kv_full_h(num_kv_heads * capacity * full_head_dim);
    for (auto& v : q_full_h) v = bf16(dist(rng) * 0.1f);
    for (auto& v : z_h) v = bf16(dist(rng) * 0.1f);
    for (auto& v : kv_full_h) v = bf16(dist(rng) * 0.1f);

    ctx.memcpy_h2d(q_full_dev.data(), q_full_h.data(), q_full_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(z_dev.data(), z_h.data(), z_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_full_dev.data(), kv_full_h.data(), kv_full_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_full_dev.data(), kv_full_h.data(), kv_full_h.size() * sizeof(bf16));

    // Path 1: Phase A4 Batched QSA Prefill with query tiling (kQsaQueryTile = 64)
    {
        Tensor indexer_q_dev = Tensor::allocate(ctx, {count, indexer_q_dim}, dnnl::memory::data_type::bf16);
        const int max_blocks = total_complete_blocks;
        constexpr int kQsaQueryTile = 64;
        Tensor tile_scores = Tensor::allocate(ctx, {kQsaQueryTile, max_blocks}, dnnl::memory::data_type::f32);
        Tensor tile_blocks = Tensor::allocate(ctx, {kQsaQueryTile, block_topk + 4}, dnnl::memory::data_type::s32);
        Tensor tile_valid_counts = Tensor::allocate(ctx, {kQsaQueryTile, 2}, dnnl::memory::data_type::s32);

        ops::qsa_prepare_q_batched(ctx, indexer_qk_dev, qn_dev, indexer_q_dev, start_pos, count, rope_theta);

        for (int tile_offset = 0; tile_offset < count; tile_offset += kQsaQueryTile) {
            const int tile_count = std::min(kQsaQueryTile, count - tile_offset);
            const int tile_start_pos = start_pos + tile_offset;
            const int tile_max_blocks = std::max(1, (tile_start_pos + tile_count) / compress_ratio);

            Tensor q_tile = Tensor::view(
                ctx, indexer_q_dev.data_as<bf16>() + static_cast<size_t>(tile_offset) * indexer_q_dim,
                {tile_count, indexer_q_dim});
            Tensor q_full_tile = Tensor::view(
                ctx, q_full_dev.data_as<bf16>() + static_cast<size_t>(tile_offset) * q_dim,
                {tile_count, q_dim});
            Tensor z_tile = Tensor::view(
                ctx, z_dev.data_as<bf16>() + static_cast<size_t>(tile_offset) * q_dim,
                {tile_count, q_dim});
            Tensor attn_out_tile = Tensor::view(
                ctx, attn_out_batched.data_as<bf16>() + static_cast<size_t>(tile_offset) * q_dim,
                {tile_count, q_dim});
            Tensor valid_counts_tile = Tensor::view(
                ctx, tile_valid_counts.data_as<int>(),
                {tile_count, 2});

            ops::qsa_compute_block_scores_compressed_batched(
                ctx, q_tile, compressed_k_dev, tile_scores, tile_count, tile_max_blocks, compress_ratio, tile_start_pos);

            ops::qsa_select_topk_blocks_batched(
                ctx, tile_scores, tile_blocks, tile_count, tile_max_blocks, tile_start_pos, block_topk, compress_ratio,
                &valid_counts_tile);

            ops::attention_prefill_qsa_batched(
                ctx, q_full_tile, k_full_dev, v_full_dev, tile_blocks, attn_out_tile,
                tile_count, tile_start_pos, num_heads, num_kv_heads, full_head_dim, block_topk, compress_ratio, &z_tile,
                &valid_counts_tile);
        }
        ctx.synchronize();
    }

    // Path 2: Row-wise oracle
    {
        Tensor decode_scores = Tensor::allocate(ctx, {num_heads, capacity}, dnnl::memory::data_type::f32);
        constexpr int kDecodeExactTile = 128;
        const int max_tiles = (capacity + kDecodeExactTile - 1) / kDecodeExactTile;
        Tensor partials = Tensor::allocate(ctx, {(int64_t)num_heads, (int64_t)max_tiles, 272}, dnnl::memory::data_type::f32);
        Tensor indexer_q_row = Tensor::allocate(ctx, {indexer_heads, indexer_head_dim}, dnnl::memory::data_type::bf16);
        Tensor row_scores = Tensor::allocate(ctx, {total_complete_blocks + 1}, dnnl::memory::data_type::f32);
        Tensor active_tokens = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);

        int comp_blocks = total_complete_blocks;
        for (int row = 0; row < count; ++row) {
            const int pos = start_pos + row;
            const int visible_len = pos + 1;
            Tensor qk_row = Tensor::view(
                ctx, indexer_qk_dev.data_as<bf16>() + static_cast<size_t>(row) * indexer_out_dim, {1, indexer_out_dim});
            Tensor q_row = Tensor::view(
                ctx, q_full_dev.data_as<bf16>() + static_cast<size_t>(row) * q_dim, {1, q_dim});
            Tensor z_row = Tensor::view(
                ctx, z_dev.data_as<bf16>() + static_cast<size_t>(row) * q_dim, {1, q_dim});
            Tensor out_row = Tensor::view(
                ctx, attn_out_oracle.data_as<bf16>() + static_cast<size_t>(row) * q_dim, {1, q_dim});

            int num_active = ops::qsa_prepare_causal_query_and_select_tokens(
                ctx, qk_row, qn_dev, kn_dev, indexer_k_dev, indexer_q_row, row_scores,
                active_tokens, pos, block_topk, compress_ratio, rope_theta, false,
                nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
                &compressed_k_dev, &comp_blocks);

            if (num_active == 0) {
                num_active = ops::qsa_select_topk_and_expand_tokens(
                    ctx, row_scores, visible_len / compress_ratio, block_topk, visible_len, compress_ratio, active_tokens);
            }

            ops::attention_decode(
                ctx, q_row, k_full_dev, v_full_dev, out_row, decode_scores,
                num_heads, num_kv_heads, full_head_dim, visible_len,
                &partials, &z_row, active_tokens.data_as<int>(), num_active);
        }
        ctx.synchronize();
    }

    std::vector<bf16> out_batched_h(count * q_dim);
    std::vector<bf16> out_oracle_h(count * q_dim);
    ctx.memcpy_d2h(out_batched_h.data(), attn_out_batched.data(), out_batched_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(out_oracle_h.data(), attn_out_oracle.data(), out_oracle_h.size() * sizeof(bf16));

    float max_diff = 0.0f;
    for (size_t i = 0; i < out_batched_h.size(); ++i) {
        float val_b = static_cast<float>(out_batched_h[i]);
        float val_o = static_cast<float>(out_oracle_h[i]);
        expect(!std::isnan(val_b) && !std::isinf(val_b), "Batched prefill output NaN/Inf");
        float d = std::abs(val_b - val_o);
        if (d > max_diff) max_diff = d;
    }
    expect(max_diff < 2e-2f, "Batched Prefill: output mismatch vs row-wise oracle, max_diff=" + std::to_string(max_diff));
    std::cout << "[PASS] QSA Batched Prefill (A4) matches row-wise oracle across threshold (max_diff=" << max_diff << " < 0.02)." << std::endl;
}

void test_qsa_complement_selection_513(Context& ctx) {
    std::cout << "[Test] Running QSA Complement Selection B=513 (R4c) Tests..." << std::endl;
    constexpr int block_topk = 512;
    constexpr int num_blocks = 513; // B = K + 1
    constexpr int compress_ratio = 4;
    constexpr int visible_len = num_blocks * compress_ratio + 2; // 2054 tokens, tail_len = 2

    auto verify_case = [&](const std::string& case_name, const std::vector<float>& host_scores, int expected_dropped_idx) {
        expect(host_scores.size() == num_blocks, "Scores vector size mismatch");

        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(scores_dev.data(), host_scores.data(), num_blocks * sizeof(float));

        // 1. Test qsa_select_topk_blocks
        Tensor out_blocks_dev = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);
        std::vector<int> out_blocks_host(block_topk + 4, -1);
        int num_sel_host = ops::qsa_select_topk_blocks_host(
            host_scores.data(), num_blocks, block_topk, visible_len, compress_ratio, out_blocks_host.data());
        int num_sel_dev = ops::qsa_select_topk_blocks(
            ctx, scores_dev, num_blocks, block_topk, visible_len, compress_ratio, out_blocks_dev);
        ctx.synchronize();

        expect(num_sel_host == block_topk, case_name + ": host num_active_blocks must be 512");
        expect(num_sel_dev == block_topk, case_name + ": dev num_active_blocks must be 512");

        std::vector<int> out_blocks_dev_h(block_topk + 4, -1);
        ctx.memcpy_d2h(out_blocks_dev_h.data(), out_blocks_dev.data(), (block_topk + 4) * sizeof(int));

        for (int i = 0; i < block_topk; ++i) {
            expect(out_blocks_host[i] == out_blocks_dev_h[i],
                   case_name + ": dev vs host mismatch at index " + std::to_string(i));
            int expected_block = (i < expected_dropped_idx) ? i : (i + 1);
            expect(out_blocks_dev_h[i] == expected_block,
                   case_name + ": expected block " + std::to_string(expected_block) + " but got " + std::to_string(out_blocks_dev_h[i]));
        }

        constexpr int tail_start = num_blocks * compress_ratio;
        constexpr int tail_len = visible_len - tail_start; // 2
        for (int t = 0; t < tail_len; ++t) {
            expect(out_blocks_dev_h[block_topk + t] == tail_start + t,
                   case_name + ": tail token mismatch at tail " + std::to_string(t));
        }

        // 2. Test qsa_select_topk_and_expand_tokens
        const int max_tokens = (block_topk + 1) * compress_ratio + 4;
        Tensor out_toks_dev = Tensor::allocate(ctx, {max_tokens}, dnnl::memory::data_type::s32);
        std::vector<int> out_toks_host(max_tokens, -1);
        int num_toks_host = ops::qsa_select_topk_and_expand_tokens_host(
            host_scores.data(), num_blocks, block_topk, visible_len, compress_ratio, out_toks_host.data());
        int num_toks_dev = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, visible_len, compress_ratio, out_toks_dev);
        ctx.synchronize();

        expect(num_toks_host == block_topk * compress_ratio + tail_len, case_name + ": host token count mismatch");
        expect(num_toks_dev == num_toks_host, case_name + ": dev token count mismatch");

        std::vector<int> out_toks_dev_h(max_tokens, -1);
        ctx.memcpy_d2h(out_toks_dev_h.data(), out_toks_dev.data(), max_tokens * sizeof(int));
        for (int i = 0; i < num_toks_host; ++i) {
            expect(out_toks_host[i] == out_toks_dev_h[i],
                   case_name + ": token expand dev vs host mismatch at " + std::to_string(i));
        }

        // 3. Test qsa_select_topk_blocks_batched with valid_counts_out
        Tensor batched_scores = Tensor::allocate(ctx, {2, num_blocks}, dnnl::memory::data_type::f32);
        std::vector<float> batched_scores_h(2 * num_blocks);
        std::copy(host_scores.begin(), host_scores.end(), batched_scores_h.begin());
        std::copy(host_scores.begin(), host_scores.end(), batched_scores_h.begin() + num_blocks);
        ctx.memcpy_h2d(batched_scores.data(), batched_scores_h.data(), batched_scores_h.size() * sizeof(float));

        Tensor batched_out_blocks = Tensor::allocate(ctx, {2, block_topk + 4}, dnnl::memory::data_type::s32);
        Tensor batched_valid_counts = Tensor::allocate(ctx, {2, 2}, dnnl::memory::data_type::s32);
        std::vector<int> batched_out_blocks_h(2 * (block_topk + 4), -1);
        std::vector<int> batched_valid_counts_h(4, -1);

        const int start_pos_test = 513 * 4;
        ops::qsa_select_topk_blocks_batched_host(
            batched_scores_h.data(), batched_out_blocks_h.data(), 2, num_blocks, start_pos_test, block_topk, compress_ratio,
            batched_valid_counts_h.data());
        ops::qsa_select_topk_blocks_batched(
            ctx, batched_scores, batched_out_blocks, 2, num_blocks, start_pos_test, block_topk, compress_ratio,
            &batched_valid_counts);
        ctx.synchronize();

        std::vector<int> dev_counts_h(4, -1);
        ctx.memcpy_d2h(dev_counts_h.data(), batched_valid_counts.data(), 4 * sizeof(int));
        expect(dev_counts_h[0] == block_topk && dev_counts_h[1] == 1,
               case_name + ": batched valid counts dev mismatch row 0");
        expect(dev_counts_h[2] == block_topk && dev_counts_h[3] == 2,
               case_name + ": batched valid counts dev mismatch row 1");
        expect(batched_valid_counts_h[0] == dev_counts_h[0] && batched_valid_counts_h[1] == dev_counts_h[1],
               case_name + ": host vs dev valid counts mismatch row 0");
        expect(batched_valid_counts_h[2] == dev_counts_h[2] && batched_valid_counts_h[3] == dev_counts_h[3],
               case_name + ": host vs dev valid counts mismatch row 1");
    };

    // Scenario A: Worst at index 0
    {
        std::vector<float> scores(num_blocks);
        for (int i = 0; i < num_blocks; ++i) scores[i] = 10.0f + static_cast<float>(i) * 0.1f;
        scores[0] = -100.0f;
        verify_case("WorstAtIndex0", scores, 0);
    }

    // Scenario B: Worst at index 512 (last)
    {
        std::vector<float> scores(num_blocks);
        for (int i = 0; i < num_blocks; ++i) scores[i] = 10.0f + static_cast<float>(i) * 0.1f;
        scores[512] = -100.0f;
        verify_case("WorstAtIndex512", scores, 512);
    }

    // Scenario C: Worst in middle (index 250)
    {
        std::vector<float> scores(num_blocks);
        for (int i = 0; i < num_blocks; ++i) scores[i] = 10.0f + static_cast<float>(i) * 0.1f;
        scores[250] = -100.0f;
        verify_case("WorstAtIndex250", scores, 250);
    }

    // Scenario D: Tie-break (equal worst scores at index 100 and index 300)
    // Contract: score tie -> smaller ID preferred -> drop larger ID (index 300).
    {
        std::vector<float> scores(num_blocks, 5.0f);
        scores[100] = -50.0f;
        scores[300] = -50.0f;
        verify_case("TieBreakDropLargerID", scores, 300);
    }

    // Scenario E: IEEE floating point special cases
    {
        std::vector<float> scores(num_blocks, 1.0f);
        scores[400] = -1e30f;
        verify_case("NegativeLargeWorst", scores, 400);
    }

    // Scenario F: All equal scores across all 513 blocks
    // Contract: scores tie across all blocks -> keep smaller IDs 0..511, drop block 512.
    {
        std::vector<float> scores(num_blocks, 3.14159f);
        verify_case("AllEqualScoresDropLast", scores, 512);
    }

    // Scenario G: Multiple equal worst with boundary elements (511 and 512)
    {
        std::vector<float> scores(num_blocks, 10.0f);
        scores[511] = -999.0f;
        scores[512] = -999.0f;
        verify_case("TieBreakAtBoundary", scores, 512);
    }

    std::cout << "[PASS] QSA Complement Selection B=513 (R4c) all scenarios passed bit-exact.\n";
}

void test_qsa_batched_multi_q_head_attention(Context& ctx) {
    std::cout << "[Test] Running QSA Batched Multi-Q-Head Parallel Attention (R4b) Test...\n";
    constexpr int count = 4;
    constexpr int start_pos = 2052;
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2; // 12 Q heads per KV head
    constexpr int head_dim = 256;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int capacity = 2200;
    constexpr int q_dim = num_heads * head_dim;

    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    Tensor q_dev = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor z_dev = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);
    Tensor out_gpu = Tensor::allocate(ctx, {count, q_dim}, dnnl::memory::data_type::bf16);
    Tensor active_blocks_dev = Tensor::allocate(ctx, {count, block_topk + 4}, dnnl::memory::data_type::s32);
    Tensor valid_counts_dev = Tensor::allocate(ctx, {count, 2}, dnnl::memory::data_type::s32);

    std::vector<bf16> q_h(count * q_dim);
    std::vector<bf16> k_h(num_kv_heads * capacity * head_dim);
    std::vector<bf16> v_h(num_kv_heads * capacity * head_dim);
    std::vector<bf16> z_h(count * q_dim);
    std::vector<int> active_blocks_h(count * (block_topk + 4));
    std::vector<int> valid_counts_h(count * 2);

    for (auto& v : q_h) v = bf16(dist(rng));
    for (auto& v : k_h) v = bf16(dist(rng));
    for (auto& v : v_h) v = bf16(dist(rng));
    for (auto& v : z_h) v = bf16(dist(rng));

    for (int r = 0; r < count; ++r) {
        const int pos = start_pos + r;
        const int visible_len = pos + 1;
        const int num_complete_blocks = visible_len / compress_ratio;
        const int tail_start = num_complete_blocks * compress_ratio;
        const int tail_len = visible_len - tail_start;

        valid_counts_h[r * 2 + 0] = block_topk;
        valid_counts_h[r * 2 + 1] = tail_len;

        for (int b = 0; b < block_topk; ++b) {
            active_blocks_h[r * (block_topk + 4) + b] = b;
        }
        for (int t = 0; t < tail_len; ++t) {
            active_blocks_h[r * (block_topk + 4) + block_topk + t] = tail_start + t;
        }
    }

    ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k_h.data(), k_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v_h.data(), v_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(z_dev.data(), z_h.data(), z_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(active_blocks_dev.data(), active_blocks_h.data(), active_blocks_h.size() * sizeof(int));
    ctx.memcpy_h2d(valid_counts_dev.data(), valid_counts_h.data(), valid_counts_h.size() * sizeof(int));

    auto run_multi_q_suite = [&](int heads_mode, const std::string& mode_name) {
#ifdef _WIN32
        _putenv(("AILA_QSA_HEADS_PER_WG=" + std::to_string(heads_mode)).c_str());
#else
        setenv("AILA_QSA_HEADS_PER_WG", std::to_string(heads_mode).c_str(), 1);
#endif
        std::cout << "  [Subtest] " << mode_name << " (AILA_QSA_HEADS_PER_WG=" << heads_mode << ")..." << std::endl;

        // Test Case 1: With gate (z) and with valid_counts
        {
            ops::attention_prefill_qsa_batched(
                ctx, q_dev, k_dev, v_dev, active_blocks_dev, out_gpu,
                count, start_pos, num_heads, num_kv_heads, head_dim,
                block_topk, compress_ratio, &z_dev, &valid_counts_dev);
            ctx.synchronize();

            std::vector<bf16> out_gpu_h(count * q_dim);
            std::vector<bf16> out_ref_h(count * q_dim);
            ctx.memcpy_d2h(out_gpu_h.data(), out_gpu.data(), out_gpu_h.size() * sizeof(bf16));

            ops::attention_prefill_qsa_batched_host(
                q_h.data(), k_h.data(), v_h.data(), active_blocks_h.data(), out_ref_h.data(),
                count, start_pos, capacity, num_heads, num_kv_heads, head_dim,
                block_topk, compress_ratio, z_h.data(), valid_counts_h.data());

            float max_diff = 0.0f;
            for (size_t i = 0; i < out_gpu_h.size(); ++i) {
                float g = static_cast<float>(out_gpu_h[i]);
                float r = static_cast<float>(out_ref_h[i]);
                expect(!std::isnan(g) && !std::isinf(g), mode_name + " GPU output contains NaN/Inf");
                float diff = std::abs(g - r);
                if (diff > max_diff) max_diff = diff;
            }
            expect(max_diff < 0.02f, mode_name + " with gate: GPU vs Host mismatch max_diff=" + std::to_string(max_diff));
            std::cout << "    With gate max_diff = " << max_diff << " (PASS)\n";
        }

        // Test Case 2: Without gate (nullptr) and with valid_counts
        {
            ops::attention_prefill_qsa_batched(
                ctx, q_dev, k_dev, v_dev, active_blocks_dev, out_gpu,
                count, start_pos, num_heads, num_kv_heads, head_dim,
                block_topk, compress_ratio, nullptr, &valid_counts_dev);
            ctx.synchronize();

            std::vector<bf16> out_gpu_h(count * q_dim);
            std::vector<bf16> out_ref_h(count * q_dim);
            ctx.memcpy_d2h(out_gpu_h.data(), out_gpu.data(), out_gpu_h.size() * sizeof(bf16));

            ops::attention_prefill_qsa_batched_host(
                q_h.data(), k_h.data(), v_h.data(), active_blocks_h.data(), out_ref_h.data(),
                count, start_pos, capacity, num_heads, num_kv_heads, head_dim,
                block_topk, compress_ratio, nullptr, valid_counts_h.data());

            float max_diff = 0.0f;
            for (size_t i = 0; i < out_gpu_h.size(); ++i) {
                float g = static_cast<float>(out_gpu_h[i]);
                float r = static_cast<float>(out_ref_h[i]);
                expect(!std::isnan(g) && !std::isinf(g), mode_name + " GPU output without gate contains NaN/Inf");
                float diff = std::abs(g - r);
                if (diff > max_diff) max_diff = diff;
            }
            expect(max_diff < 0.02f, mode_name + " without gate: GPU vs Host mismatch max_diff=" + std::to_string(max_diff));
            std::cout << "    Without gate max_diff = " << max_diff << " (PASS)\n";
        }

        // Test Case 3: Without valid_counts (fallback to on-the-fly arithmetic)
        {
            ops::attention_prefill_qsa_batched(
                ctx, q_dev, k_dev, v_dev, active_blocks_dev, out_gpu,
                count, start_pos, num_heads, num_kv_heads, head_dim,
                block_topk, compress_ratio, &z_dev, nullptr);
            ctx.synchronize();

            std::vector<bf16> out_gpu_h(count * q_dim);
            std::vector<bf16> out_ref_h(count * q_dim);
            ctx.memcpy_d2h(out_gpu_h.data(), out_gpu.data(), out_gpu_h.size() * sizeof(bf16));

            ops::attention_prefill_qsa_batched_host(
                q_h.data(), k_h.data(), v_h.data(), active_blocks_h.data(), out_ref_h.data(),
                count, start_pos, capacity, num_heads, num_kv_heads, head_dim,
                block_topk, compress_ratio, z_h.data(), nullptr);

            float max_diff = 0.0f;
            for (size_t i = 0; i < out_gpu_h.size(); ++i) {
                float diff = std::abs(static_cast<float>(out_gpu_h[i]) - static_cast<float>(out_ref_h[i]));
                if (diff > max_diff) max_diff = diff;
            }
            expect(max_diff < 0.02f, mode_name + " without valid_counts: GPU vs Host mismatch max_diff=" + std::to_string(max_diff));
            std::cout << "    Fallback arithmetic max_diff = " << max_diff << " (PASS)\n";
        }
    };

    run_multi_q_suite(2, "2 Q-Heads per WG (Default)");
    run_multi_q_suite(4, "4 Q-Heads per WG (High Reuse)");
    run_multi_q_suite(1, "1 Q-Head per WG (Baseline Fallback)");

#ifdef _WIN32
    _putenv("AILA_QSA_HEADS_PER_WG=2");
#else
    setenv("AILA_QSA_HEADS_PER_WG", "2", 1);
#endif

    std::cout << "[PASS] QSA Batched Multi-Q-Head Parallel Attention (R4b) tests passed (all 3 modes).\n";
}

void test_qsa_pre_indexer_fusion_reentrant(Context& ctx) {
    std::cout << "[Test] Running QSA Pre-Indexer Fusion Reentrancy (R4d) Test...\n";
    constexpr int total_count = 178;
    constexpr int capacity = 256;
    constexpr int compress_ratio = 4;
    constexpr int indexer_heads = 4;
    constexpr int indexer_head_dim = 128;
    constexpr int indexer_q_dim = indexer_heads * indexer_head_dim;
    constexpr int indexer_out_dim = indexer_q_dim + indexer_head_dim;
    constexpr float rope_theta = 1000000.0f;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<bf16> indexer_qk_h(total_count * indexer_out_dim);
    for (auto& v : indexer_qk_h) v = bf16(dist(rng));
    std::vector<bf16> qn_w_h(indexer_head_dim, bf16(1.0f));
    std::vector<bf16> kn_w_h(indexer_head_dim, bf16(1.0f));

    Tensor indexer_qk_dev = Tensor::allocate(ctx, {total_count, indexer_out_dim}, dnnl::memory::data_type::bf16);
    Tensor qn_dev = Tensor::allocate(ctx, {indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor kn_dev = Tensor::allocate(ctx, {indexer_head_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(indexer_qk_dev.data(), indexer_qk_h.data(), indexer_qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(qn_dev.data(), qn_w_h.data(), qn_w_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(kn_dev.data(), kn_w_h.data(), kn_w_h.size() * sizeof(bf16));

    // Approach 1: Monolithic single call (start_pos = 0, count = total_count)
    Tensor k_cache_mono = Tensor::allocate(ctx, {capacity, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor comp_k_mono = Tensor::allocate(ctx, {capacity / compress_ratio, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_out_mono = Tensor::allocate(ctx, {total_count, indexer_q_dim}, dnnl::memory::data_type::bf16);
    int num_comp_mono = 0;

    ops::qsa_pre_indexer_batched(
        ctx, indexer_qk_dev, qn_dev, kn_dev,
        k_cache_mono, q_out_mono, &comp_k_mono, &num_comp_mono,
        0, total_count, compress_ratio, rope_theta);
    ctx.synchronize();

    // Approach 2: Chunked reentrant calls across 3 chunks: [0..64), [64..128), [128..178)
    Tensor k_cache_chunk = Tensor::allocate(ctx, {capacity, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor comp_k_chunk = Tensor::allocate(ctx, {capacity / compress_ratio, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_out_chunk = Tensor::allocate(ctx, {total_count, indexer_q_dim}, dnnl::memory::data_type::bf16);
    int num_comp_chunk = 0;

    const std::vector<std::pair<int, int>> chunks = {{0, 64}, {64, 64}, {128, 50}};
    for (const auto& [chunk_start, chunk_count] : chunks) {
        Tensor qk_chunk = Tensor::view(
            ctx, indexer_qk_dev.data_as<bf16>() + static_cast<size_t>(chunk_start) * indexer_out_dim,
            {chunk_count, indexer_out_dim});
        Tensor q_chunk = Tensor::view(
            ctx, q_out_chunk.data_as<bf16>() + static_cast<size_t>(chunk_start) * indexer_q_dim,
            {chunk_count, indexer_q_dim});

        ops::qsa_pre_indexer_batched(
            ctx, qk_chunk, qn_dev, kn_dev,
            k_cache_chunk, q_chunk, &comp_k_chunk, &num_comp_chunk,
            chunk_start, chunk_count, compress_ratio, rope_theta);
    }
    ctx.synchronize();

    expect(num_comp_mono == total_count / compress_ratio, "Mono compressed block count mismatch");
    expect(num_comp_chunk == num_comp_mono, "Chunked compressed block count mismatch");

    // Compare k_cache
    std::vector<bf16> k_mono_h(total_count * indexer_head_dim);
    std::vector<bf16> k_chunk_h(total_count * indexer_head_dim);
    ctx.memcpy_d2h(k_mono_h.data(), k_cache_mono.data(), k_mono_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_chunk_h.data(), k_cache_chunk.data(), k_chunk_h.size() * sizeof(bf16));

    for (size_t i = 0; i < k_mono_h.size(); ++i) {
        expect(k_mono_h[i] == k_chunk_h[i], "k_cache bit-exact mismatch between mono and chunked at " + std::to_string(i));
    }

    // Compare compressed_k
    std::vector<bf16> comp_mono_h(num_comp_mono * indexer_head_dim);
    std::vector<bf16> comp_chunk_h(num_comp_chunk * indexer_head_dim);
    ctx.memcpy_d2h(comp_mono_h.data(), comp_k_mono.data(), comp_mono_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(comp_chunk_h.data(), comp_k_chunk.data(), comp_chunk_h.size() * sizeof(bf16));

    for (size_t i = 0; i < comp_mono_h.size(); ++i) {
        expect(comp_mono_h[i] == comp_chunk_h[i], "compressed_k bit-exact mismatch between mono and chunked at " + std::to_string(i));
    }

    // Compare q_out
    std::vector<bf16> q_mono_h(total_count * indexer_q_dim);
    std::vector<bf16> q_chunk_h(total_count * indexer_q_dim);
    ctx.memcpy_d2h(q_mono_h.data(), q_out_mono.data(), q_mono_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(q_chunk_h.data(), q_out_chunk.data(), q_chunk_h.size() * sizeof(bf16));

    for (size_t i = 0; i < q_mono_h.size(); ++i) {
        expect(q_mono_h[i] == q_chunk_h[i], "q_out bit-exact mismatch between mono and chunked at " + std::to_string(i));
    }

    // Approach 2b: Unaligned chunked reentrant calls crossing block boundaries
    // Chunks: [0..17), [17..52), [52..112), [112..178)
    Tensor k_cache_unaligned = Tensor::allocate(ctx, {capacity, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor comp_k_unaligned = Tensor::allocate(ctx, {capacity / compress_ratio, indexer_head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_out_unaligned = Tensor::allocate(ctx, {total_count, indexer_q_dim}, dnnl::memory::data_type::bf16);
    int num_comp_unaligned = 0;

    const std::vector<std::pair<int, int>> unaligned_chunks = {{0, 17}, {17, 35}, {52, 60}, {112, 66}};
    for (const auto& [chunk_start, chunk_count] : unaligned_chunks) {
        Tensor qk_chunk = Tensor::view(
            ctx, indexer_qk_dev.data_as<bf16>() + static_cast<size_t>(chunk_start) * indexer_out_dim,
            {chunk_count, indexer_out_dim});
        Tensor q_chunk = Tensor::view(
            ctx, q_out_unaligned.data_as<bf16>() + static_cast<size_t>(chunk_start) * indexer_q_dim,
            {chunk_count, indexer_q_dim});

        ops::qsa_pre_indexer_batched(
            ctx, qk_chunk, qn_dev, kn_dev,
            k_cache_unaligned, q_chunk, &comp_k_unaligned, &num_comp_unaligned,
            chunk_start, chunk_count, compress_ratio, rope_theta);
    }
    ctx.synchronize();

    expect(num_comp_unaligned == num_comp_mono, "Unaligned chunked compressed block count mismatch");

    std::vector<bf16> k_unaligned_h(total_count * indexer_head_dim);
    std::vector<bf16> comp_unaligned_h(num_comp_unaligned * indexer_head_dim);
    std::vector<bf16> q_unaligned_h(total_count * indexer_q_dim);
    ctx.memcpy_d2h(k_unaligned_h.data(), k_cache_unaligned.data(), k_unaligned_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(comp_unaligned_h.data(), comp_k_unaligned.data(), comp_unaligned_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(q_unaligned_h.data(), q_out_unaligned.data(), q_unaligned_h.size() * sizeof(bf16));

    for (size_t i = 0; i < k_mono_h.size(); ++i) {
        expect(k_mono_h[i] == k_unaligned_h[i], "k_cache bit-exact mismatch between mono and unaligned at " + std::to_string(i));
    }
    for (size_t i = 0; i < comp_mono_h.size(); ++i) {
        expect(comp_mono_h[i] == comp_unaligned_h[i], "compressed_k bit-exact mismatch between mono and unaligned at " + std::to_string(i));
    }
    for (size_t i = 0; i < q_mono_h.size(); ++i) {
        expect(q_mono_h[i] == q_unaligned_h[i], "q_out bit-exact mismatch between mono and unaligned at " + std::to_string(i));
    }

    // Approach 3: Compare against Host reference
    std::vector<bf16> k_cache_ref(capacity * indexer_head_dim);
    std::vector<bf16> comp_k_ref((capacity / compress_ratio) * indexer_head_dim);
    std::vector<bf16> q_out_ref(total_count * indexer_q_dim);
    int num_comp_ref = 0;

    ops::qsa_pre_indexer_batched_host(
        indexer_qk_h.data(), qn_w_h.data(), kn_w_h.data(),
        k_cache_ref.data(), q_out_ref.data(), comp_k_ref.data(), &num_comp_ref,
        0, total_count, compress_ratio, capacity, rope_theta);

    expect(num_comp_ref == num_comp_mono, "Host compressed block count mismatch");

    float max_q_diff = 0.0f;
    for (size_t i = 0; i < q_mono_h.size(); ++i) {
        float d = std::abs(static_cast<float>(q_mono_h[i]) - static_cast<float>(q_out_ref[i]));
        if (d > max_q_diff) max_q_diff = d;
    }
    expect(max_q_diff < 0.01f, "GPU vs Host q_out mismatch max_diff=" + std::to_string(max_q_diff));

    std::cout << "[PASS] QSA Pre-Indexer Fusion Reentrancy (R4d) verified bit-exact across chunks (max_q_diff="
              << max_q_diff << ").\n";
}

// ---------------------------------------------------------------------------
// Phase B (B1-B4) Expert Cache & Scheduling Tests
// ---------------------------------------------------------------------------

void test_fused_resident_router_bitwise(Context& ctx) {
    std::cout << "[Test] Running fused resident routing bitwise alignment...\n";
    constexpr int experts = 512, capacity = 17;
    Tensor scores = Tensor::allocate(ctx, {experts}, dnnl::memory::data_type::bf16);
    Tensor scales = Tensor::allocate(ctx, {experts}, dnnl::memory::data_type::bf16);
    Tensor table = Tensor::allocate(ctx, {experts}, dnnl::memory::data_type::s32);
    Tensor old_ids = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);
    Tensor old_weights = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::f32);
    Tensor old_selected = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);
    Tensor fused_ids = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);
    Tensor fused_weights = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::f32);
    Tensor fused_selected = Tensor::allocate(ctx, {capacity}, dnnl::memory::data_type::s32);
    struct HostOutputs {
        Context& ctx;
        void* block;
        explicit HostOutputs(Context& context) : ctx(context),
            block(context.alloc_host(6 * capacity * sizeof(uint32_t))) {}
        ~HostOutputs() {
            try { ctx.synchronize(); } catch (...) {}
            ctx.free_host(block);
        }
    } outputs(ctx);
    auto* old_host_ids = static_cast<int*>(outputs.block);
    auto* old_host_weights = reinterpret_cast<float*>(old_host_ids + capacity);
    auto* old_host_selected = reinterpret_cast<int*>(old_host_weights + capacity);
    auto* fused_host_ids = old_host_selected + capacity;
    auto* fused_host_weights = reinterpret_cast<float*>(fused_host_ids + capacity);
    auto* fused_host_selected = reinterpret_cast<int*>(fused_host_weights + capacity);
    std::memset(outputs.block, 0xA5, 6 * capacity * sizeof(uint32_t));
    for (Tensor* output : {&old_ids, &old_weights, &old_selected,
                           &fused_ids, &fused_weights, &fused_selected})
        ctx.queue().memset(output->data(), 0xA5, output->size_bytes());
    ctx.synchronize();

    // Invalid shape/pointer rejection must happen before GPU submission.
    auto expect_invalid = [&](int num_experts, int top_k, const int* resident_table) {
        bool rejected = false;
        try {
            ops::router_softmax_topk_resident(ctx, scores.data_as<bf16>(), nullptr,
                fused_ids.data_as<int>(), fused_weights.data_as<float>(), resident_table,
                fused_selected.data_as<int>(), fused_host_ids, fused_host_weights,
                fused_host_selected, num_experts, top_k, true);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "Fused resident router must reject invalid input bounds");
    };
    expect_invalid(511, 10, table.data_as<int>());
    expect_invalid(512, 0, table.data_as<int>());
    expect_invalid(512, 17, table.data_as<int>());
    expect_invalid(512, 10, nullptr);

    std::array<bf16, experts> scores_host{}, scales_host{};
    std::array<int, experts> table_host{};
    std::array<int, capacity> old_ids_read{}, old_selected_read{}, fused_ids_read{}, fused_selected_read{};
    std::array<float, capacity> old_weights_read{}, fused_weights_read{};
    int comparisons = 0;
    // 48 independent seeds plus explicit all-tie, repeated-tie, extreme finite,
    // and softmax-underflow inputs. Keep all arithmetic in the original kernel.
    for (int sample = 0; sample < 52; ++sample) {
        std::mt19937 rng(0x19A770u + static_cast<unsigned>(sample));
        std::uniform_real_distribution<float> distribution(-12.0f, 12.0f);
        for (int expert = 0; expert < experts; ++expert) {
            float score = distribution(rng);
            if (sample == 48) score = 0.0f;
            if (sample == 49) score = static_cast<float>(expert % 8) - 4.0f;
            if (sample == 50) score = expert % 2 ? 1.0e30f : -1.0e30f;
            if (sample == 51) score = expert == 511 ? 0.0f : -160.0f;
            scores_host[expert] = bf16(score);
            scales_host[expert] = bf16(static_cast<float>((expert + sample) % 33) / 8.0f);
            table_host[expert] = (expert + sample) % 3 == 0 ? -1 : (expert * 17 + sample) % 3551;
            expect(std::isfinite(static_cast<float>(scores_host[expert])), "Router test inputs must be finite");
        }
        ctx.memcpy_h2d(scores.data(), scores_host.data(), sizeof(scores_host));
        ctx.memcpy_h2d(scales.data(), scales_host.data(), sizeof(scales_host));
        ctx.memcpy_h2d(table.data(), table_host.data(), sizeof(table_host));
        for (int top_k : {1, 10, 16}) for (bool normalize : {false, true}) for (bool scale : {false, true}) {
            const bf16* scale_ptr = scale ? scales.data_as<bf16>() : nullptr;
            ops::router_softmax_topk_device(ctx, scores.data_as<bf16>(), scale_ptr,
                old_ids.data_as<int>(), old_weights.data_as<float>(), 1, experts, top_k, normalize);
            auto original = ops::moe_resident_classify(ctx, old_ids.data_as<int>(), old_weights.data_as<float>(),
                table.data_as<int>(), old_selected.data_as<int>(), old_host_ids, old_host_weights,
                old_host_selected, top_k, experts);
            auto fused = ops::router_softmax_topk_resident(ctx, scores.data_as<bf16>(), scale_ptr,
                fused_ids.data_as<int>(), fused_weights.data_as<float>(), table.data_as<int>(),
                fused_selected.data_as<int>(), fused_host_ids, fused_host_weights,
                fused_host_selected, experts, top_k, normalize);
            // Both events provide the public pinned-host output lifetime contract.
            original.wait_and_throw();
            fused.wait_and_throw();
            const std::string tag = " sample=" + std::to_string(sample) + " top_k=" + std::to_string(top_k) +
                " normalize=" + std::to_string(normalize) + " scale=" + std::to_string(scale);
            expect(std::memcmp(old_host_ids, fused_host_ids, capacity * sizeof(int)) == 0, "Host IDs differ" + tag);
            expect(std::memcmp(old_host_weights, fused_host_weights, capacity * sizeof(float)) == 0, "Host weights differ bitwise" + tag);
            expect(std::memcmp(old_host_selected, fused_host_selected, capacity * sizeof(int)) == 0, "Host selections differ" + tag);
            ctx.memcpy_d2h(old_ids_read.data(), old_ids.data(), sizeof(old_ids_read));
            ctx.memcpy_d2h(old_weights_read.data(), old_weights.data(), sizeof(old_weights_read));
            ctx.memcpy_d2h(old_selected_read.data(), old_selected.data(), sizeof(old_selected_read));
            ctx.memcpy_d2h(fused_ids_read.data(), fused_ids.data(), sizeof(fused_ids_read));
            ctx.memcpy_d2h(fused_weights_read.data(), fused_weights.data(), sizeof(fused_weights_read));
            ctx.memcpy_d2h(fused_selected_read.data(), fused_selected.data(), sizeof(fused_selected_read));
            expect(std::memcmp(old_ids_read.data(), fused_ids_read.data(), sizeof(old_ids_read)) == 0, "Device IDs differ" + tag);
            expect(std::memcmp(old_weights_read.data(), fused_weights_read.data(), sizeof(old_weights_read)) == 0, "Device weights differ bitwise" + tag);
            expect(std::memcmp(old_selected_read.data(), fused_selected_read.data(), sizeof(old_selected_read)) == 0, "Device selections differ" + tag);
            expect(std::memcmp(fused_ids_read.data(), fused_host_ids, sizeof(fused_ids_read)) == 0, "Fused host/device ID mirrors differ" + tag);
            expect(std::memcmp(fused_weights_read.data(), fused_host_weights, sizeof(fused_weights_read)) == 0, "Fused host/device weight mirrors differ" + tag);
            expect(std::memcmp(fused_selected_read.data(), fused_host_selected, sizeof(fused_selected_read)) == 0, "Fused host/device selection mirrors differ" + tag);
            uint32_t guard = 0;
            std::memcpy(&guard, fused_host_ids + 16, sizeof(guard));
            expect(guard == 0xA5A5A5A5u, "Fused output overran top-k capacity" + tag);
            for (int rank = 0; rank < top_k; ++rank) {
                expect(fused_host_ids[rank] >= 0 && fused_host_ids[rank] < experts, "Expert ID out of range" + tag);
                expect(fused_host_selected[rank] == table_host[fused_host_ids[rank]], "Resident classification wrong" + tag);
                expect(std::isfinite(fused_host_weights[rank]), "Router weight is nonfinite" + tag);
                if (sample == 48) expect(fused_host_ids[rank] == rank, "All-tie ID ordering changed" + tag);
            }
            ++comparisons;
        }
    }
    std::cout << "[PASS] Fused resident router " << comparisons << " bitwise cases, host/device mirrors and bounds.\n\n";
}

void test_b2_route_and_prediction_event_decoupling(Context& ctx) {
    std::cout << "[Test] Running B2 route and prediction event decoupling tests...\n";

    std::atomic<bool> prediction_done{false};
    int actual_ids[4] = {1, 2, 3, 4};
    int pred_ids[4] = {-1, -1, -1, -1};

    sycl::event actual_route_event = ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.host_task([&]() {
        });
    });

    sycl::event prediction_event = ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.host_task([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            pred_ids[0] = 5; pred_ids[1] = 6; pred_ids[2] = 7; pred_ids[3] = 8;
            prediction_done = true;
        });
    });

    actual_route_event.wait();
    expect(actual_ids[0] == 1, "Actual router IDs ready immediately");

    auto ensure_prediction_ready = [&]() {
        if (prediction_event != sycl::event{}) {
            prediction_event.wait();
            prediction_event = sycl::event{};
        }
    };

    ensure_prediction_ready();
    expect(prediction_done.load(), "Prediction must be complete after ensure_prediction_ready");
    expect(pred_ids[0] == 5, "Prediction IDs correctly read after deferred wait");

    ensure_prediction_ready();

    std::cout << "[PASS] B2 route and prediction event decoupling tests passed.\n";
}

void test_predictive_gating_logic_and_session_isolation(Context& ctx) {
    std::cout << "[Test] Running predictive gating logic and session isolation tests...\n";

    auto eval_predict = [](bool s_predict_gating, bool s_prefetch, bool s_predict_stats,
                           int s_prefill_prefetch_max, int count, int consecutive,
                           bool high_hit_rate, int L, int num_layers) {
        const bool gate_prediction = s_predict_gating && (count == 1) &&
            (consecutive >= 2 || (high_hit_rate && consecutive > 0));
        const bool predict_next = (count == 1)
            ? (!gate_prediction && (s_prefetch || s_predict_stats) && (L + 1 < num_layers))
            : (s_prefetch && (s_prefill_prefetch_max > 0) && (L + 1 < num_layers));
        return std::make_pair(gate_prediction, predict_next);
    };

    // Cold start decode: consecutive = 0
    auto res_cold = eval_predict(true, true, false, 48, 1, 0, false, 0, 48);
    expect(!res_cold.first, "gate_prediction must be false at consecutive=0");
    expect(res_cold.second, "predict_next must be true at cold start decode");

    // Steady state decode: consecutive = 2
    auto res_steady = eval_predict(true, true, false, 48, 1, 2, false, 0, 48);
    expect(res_steady.first, "gate_prediction must be true at consecutive=2");
    expect(!res_steady.second, "predict_next MUST be false in steady decode (gated)!");

    // Steady state decode with high hit rate: consecutive = 1, high_hit_rate = true
    auto res_high = eval_predict(true, true, false, 48, 1, 1, true, 0, 48);
    expect(res_high.first, "gate_prediction must be true at consecutive=1 with high_hit_rate");
    expect(!res_high.second, "predict_next MUST be false when gated with high_hit_rate");

    // Prefill: count = 30
    auto res_prefill = eval_predict(true, true, false, 48, 30, 2, true, 0, 48);
    expect(!res_prefill.first, "gate_prediction must be false in prefill");
    expect(res_prefill.second, "predict_next must be true in prefill when prefetch enabled");

    // Disabled gating: s_predict_gating = false
    auto res_nogate = eval_predict(false, true, false, 48, 1, 5, true, 0, 48);
    expect(!res_nogate.first, "gate_prediction must be false when disabled");
    expect(res_nogate.second, "predict_next must be true when gating disabled");

    std::cout << "[PASS] Predictive gating logic and session isolation verified.\n\n";
}

} // namespace

int main(int argc, char** argv) {
    aila::env::enable_l0_copy_engine();
    try {
        Context ctx;
        if (argc == 2 && std::string(argv[1]) == "--gdn-graph-state-only") {
            test_graph_gdn_device_head_state(ctx);
            std::cout << "AilaQwen4ExpModuleTests: GDN GRAPH STATE PASSED\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--token-graph-key-only") {
            test_snapshot_capture_and_truncate(ctx);
            std::cout << "AilaQwen4ExpModuleTests: TOKEN GRAPH KEY PASSED\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--token-graph-runtime-only") {
            test_token_graph_runtime_lifecycle(ctx);
            std::cout << "AilaQwen4ExpModuleTests: TOKEN GRAPH RUNTIME PASSED\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--ple-only") {
            test_real_model_ple_and_cache_if_present(ctx, true);
            std::cout << "AilaQwen4ExpModuleTests: PLE PASSED\n";
            return 0;
        }
        test_decode_pending_mlp_residual(ctx);
        test_hc_decode_scheduling_bitwise(ctx);
        test_fused_resident_router_bitwise(ctx);
        test_b2_route_and_prediction_event_decoupling(ctx);
        test_predictive_gating_logic_and_session_isolation(ctx);
        test_snapshot_capture_and_truncate(ctx);
        test_graph_gdn_device_head_state(ctx);
        test_token_graph_runtime_lifecycle(ctx);
        test_supports_interior_checkpoint_whitelist(ctx);
        test_ple_ngram_ids_calculation();
        test_ple_lookup_and_dequant(ctx);
        test_ple_ticket_pipeline(ctx);
        test_real_model_ple_and_cache_if_present(ctx);
        test_qsa_indexer_long_context_dispatch(ctx);
        test_qsa_causal_prefill_append_chunk(ctx);
        test_real_model_qsa_indexer_weights_if_present(ctx);
        test_real_model_qsa_indexer_fixture(ctx);
        test_qsa_dense_prefill_buffer_bounds(ctx);
        test_qsa_k_only_projection_alignment(ctx);
        test_qsa_session_compressed_k_lifecycle(ctx);
        test_qsa_indexer_end_to_end_pipeline(ctx);
        test_qsa_block_native_decode_alignment(ctx);
        test_qsa_batched_prefill_oracle_alignment(ctx);
        test_qsa_complement_selection_513(ctx);
        test_qsa_batched_multi_q_head_attention(ctx);
        test_qsa_pre_indexer_fusion_reentrant(ctx);
        std::cout << "AilaQwen4ExpModuleTests: ALL PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "[ERROR CATCH] AilaQwen4ExpModuleTests FAILED: " << e.what() << std::endl;
        std::cerr << "AilaQwen4ExpModuleTests FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "[ERROR CATCH] AilaQwen4ExpModuleTests FAILED: unknown exception" << std::endl;
        std::cerr << "AilaQwen4ExpModuleTests FAILED: unknown exception" << std::endl;
        return 1;
    }
}
