// B4-① regression: the adaptive native-arena multi-row CPU verifier must
// reproduce the per-row T1 CPU reference bit for bit.
//
// The legacy frozen-cache verifier (Qwen4ExpExpertExecutor::ExecuteVerifyRows)
// is unreachable once the adaptive pipeline is active, so this test pins the
// arena verifier against the independent per-row T1 CPU reference built from
// the original synthetic weights, and against a single multi-row batch of the
// same tasks. rows=2/3/4 plus partial/empty tail rows are covered.
#include "core/Context.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include "models/Qwen4ExpHostArena.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

struct Weight {
    std::vector<uint8_t> packed, codes;
    std::vector<float> scales, map, qmap;
    int rows, cols, experts;
    Weight(int n, int r, int c) : rows(r), cols(c), experts(n) {
        packed.resize(size_t(n) * r * c / 2);
        codes.resize(size_t(n) * r * c / 64);
        scales.resize(codes.size() / 256);
        map.resize(256);
        qmap.resize(16);
        for (size_t i = 0; i < packed.size(); ++i) packed[i] = uint8_t(i * 13 + i / 17 + 3);
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(i * 7 + i / 29);
        for (size_t i = 0; i < scales.size(); ++i)
            scales[i] = 0.003f + 0.00001f * float(i % 97) + 0.01f * float(i / (size_t(r) * c / 64 / 256));
        for (int i = 0; i < 256; ++i) map[i] = 0.5f + i / 256.0f;
    }
    Bnb4BitHostWeightRef Ref() const {
        Bnb4BitHostWeightRef r;
        r.name = "synthetic";
        r.packed_weight = {packed.data(), {experts, rows, cols / 2}, DT_U8, packed.size()};
        r.absmax = {codes.data(), {int64_t(codes.size())}, DT_U8, codes.size()};
        r.nested_absmax = {reinterpret_cast<const uint8_t*>(scales.data()), {int64_t(scales.size())}, DT_F32,
                           scales.size() * 4};
        r.nested_quant_map = {reinterpret_cast<const uint8_t*>(map.data()), {256}, DT_F32, map.size() * 4};
        r.quant_map = {reinterpret_cast<const uint8_t*>(qmap.data()), {16}, DT_F32, qmap.size() * 4};
        r.quant_state.quant_type = "nf4";
        r.quant_state.shape = {experts, rows, cols};
        r.quant_state.blocksize = 64;
        r.quant_state.nested = true;
        r.quant_state.nested_blocksize = 256;
        r.quant_state.nested_offset = 0.002f;
        return r;
    }
};

// Original synthetic source for expert e, independent of the arena copy.
static ops::cpu::CpuExpertTask original_task(const Weight& gu, const Weight& dn,
    const aila::ExpertSlotGeometry& g, int e, int token) {
    ops::cpu::CpuExpertTask task{};
    auto& s = task.slot;
    s.gate_up_packed = gu.packed.data() + size_t(e) * g.gu_packed_bytes;
    s.down_packed = dn.packed.data() + size_t(e) * g.dn_packed_bytes;
    s.gate_up_nested = {gu.codes.data() + size_t(e) * g.gu_absmax_count,
                        gu.scales.data() + size_t(e) * g.gu_absmax_count / 256, gu.map.data(), 0.002f, 256};
    s.down_nested = {dn.codes.data() + size_t(e) * g.dn_absmax_count,
                     dn.scales.data() + size_t(e) * g.dn_absmax_count / 256, dn.map.data(), 0.002f, 256};
    task.token = token;
    return task;
}

int main() {
    try {
        _putenv_s("AILA_Q4E_ARENA_VERIFY_ROWS", "1");
        _putenv_s("AILA_Q4E_CPU_RESULT_HOST_READ", "0");
        _putenv_s("AILA_Q4E_COLD_MODE", "2");
        _putenv_s("AILA_Q4E_CPU_MAX_EXPERTS", "16");
        _putenv_s("AILA_Q4E_CPU_THREADS", "18");
        _putenv_s("AILA_Q4E_CPU_TILE_POOL", "0");
        _putenv_s("AILA_Q4E_CPU_NF4_Q8", "1");
        _putenv_s("AILA_Q4E_CPU_Q8_ACTIVATION64", "1");
        _putenv_s("AILA_Q4E_CPU_SCALE_SHIFT", "1");

        constexpr int H = 2560, I = 640, experts = 8, top_k = 10;
        Context ctx;
        Weight gu(experts, 2 * I, H), dn(experts, H, I);

        aila::HostArenaConfig hc;
        hc.num_layers = 1;
        hc.num_experts_per_layer = experts;
        hc.hidden_size = H;
        hc.moe_intermediate_size = I;
        // Experts 2 and 5 are "GPU resident": the arena holds only the RAM
        // complement, so they must be excluded from every CPU task.
        // LoadLayerDirect copies a full slot into each GPU owner, so these
        // stand-ins must be sized for the real compact geometry.
        const size_t slot_bytes = size_t(2 * I) * H / 2 + size_t(2 * I) * H / 64 +
                                  size_t(H) * I / 2 + size_t(H) * I / 64;
        auto* fake_gpu = static_cast<uint8_t*>(ctx.alloc_device(2 * slot_bytes));
        std::vector<uint8_t*> owners(experts, nullptr);
        owners[2] = fake_gpu;
        owners[5] = fake_gpu + slot_bytes;
        aila::Qwen4ExpHostArena arena;
        std::string error;
        require(arena.Allocate(ctx, hc, &error, owners), error.c_str());
        arena.LoadLayerDirect(0, gu.Ref(), dn.Ref());
        ctx.synchronize();
        require(arena.ResidentExperts() == size_t(experts - 2), "arena RAM complement size");
        require(!arena.GetExpertPointer(0, 2) && !arena.GetExpertPointer(0, 5), "GPU-owned experts leave the arena");
        const auto& geom = arena.Geometry();

        // The executor only needs a bounded CPU worker here; the arena path must
        // never consult the (retired) cache for sources or eligibility.
        aila::Qwen4ExpExpertExecutor cpu;
        cpu.init(ctx, 1, top_k, H, I);
        require(cpu.cpu_overlap_enabled(), "arena verifier requires the bounded CPU worker");

        struct Case { int rows; const char* name; std::vector<int> ids; };
        // ids are [rows*top_k], token-major. Experts 2/5 are GPU-resident.
        // The first `cold[t]` ranks of each token pick rotating CPU experts;
        // the rest are the GPU-resident experts.
        std::vector<int> cpu_experts;
        for (int e = 0; e < experts; ++e)
            if (e != 2 && e != 5) cpu_experts.push_back(e);
        auto build = [&](int rows, const std::vector<int>& cold) {
            std::vector<int> ids;
            int rot = 0;
            for (int t = 0; t < rows; ++t) {
                for (int k = 0; k < top_k; ++k) {
                    if (k < cold[t]) ids.push_back(cpu_experts[size_t((rot + k) % cpu_experts.size())]);
                    else ids.push_back((k & 1) ? 5 : 2);
                }
                rot += cold[t];
            }
            return ids;
        };
        const std::array<Case, 4> cases = {{{2, "rows=2", build(2, {10, 6})},
                                            {3, "rows=3 (partial tail)", build(3, {10, 10, 1})},
                                            {4, "rows=4 (empty tail)", build(4, {10, 10, 10, 0})},
                                            {4, "rows=4 (full capacity)", build(4, {10, 10, 10, 10})}}};

        const size_t rank_rows = size_t(4) * top_k;  // backend capability ceiling
        auto* down = static_cast<float*>(ctx.alloc_device(rank_rows * H * sizeof(float)));
        std::vector<bf16> x(4 * H);
        for (int t = 0; t < 4; ++t)
            for (int j = 0; j < H; ++j) x[size_t(t) * H + j] = bf16((int((j + 7 * t) % 31) - 15) * 0.01f);
        auto* xd = static_cast<bf16*>(ctx.alloc_device(x.size() * sizeof(bf16)));
        ctx.memcpy_h2d(xd, x.data(), size_t(4) * H * sizeof(bf16));

        for (const auto& tc : cases) {
            const int rows = tc.rows;
            std::vector<int> cold_ranks;  // ascending global ranks on the CPU
            for (int rank = 0; rank < rows * top_k; ++rank) {
                const int e = tc.ids[size_t(rank)];
                if (arena.GetExpertPointer(0, e)) cold_ranks.push_back(rank);
            }
            require(!cold_ranks.empty(), "case must exercise CPU cold experts");

            const auto before = cpu.stats().cpu_tasks_completed;
            ctx.queue().fill(down, -7.0f, rank_rows * H);
            cpu.StageResidentInput(xd, rows);
            const int predicted = cpu.StartArenaVerifyRows(arena, 0, rows, tc.ids.data(), sycl::event{});
            require(predicted == int(cold_ranks.size()), "coordinator must predict the CPU task count");
            cpu.JoinArenaVerifyRows(down).wait_and_throw();
            ctx.synchronize();
            require(cpu.stats().cpu_tasks_completed - before == uint64_t(cold_ranks.size()),
                    "worker must complete exactly the predicted CPU tasks");

            // Reference 1: independent per-row T1 CPU forward on the original source.
            std::vector<bf16> per_h(I);
            std::vector<float> per_out(H);
            // Reference 2: one multi-row batch over the same tasks.
            std::vector<ops::cpu::CpuExpertTask> batch;
            for (size_t c = 0; c < cold_ranks.size(); ++c) {
                const int rank = cold_ranks[c];
                batch.push_back(original_task(gu, dn, geom, tc.ids[size_t(rank)], rank / top_k));
            }
            std::vector<bf16> batch_h(batch.size() * I);
            std::vector<float> batch_out(batch.size() * H);
            ops::cpu::cpu_expert_batch_forward(batch.data(), int(batch.size()), x.data(), rows,
                batch_h.data(), batch_out.data(), nullptr, ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2,
                nullptr, true);

            std::vector<float> actual(rank_rows * H);
            ctx.memcpy_d2h(actual.data(), down, rank_rows * H * sizeof(float));

            for (size_t c = 0; c < cold_ranks.size(); ++c) {
                const int rank = cold_ranks[c];
                const int token = rank / top_k;
                // Single-row reference: x points at this row, so token is 0.
                const auto task = original_task(gu, dn, geom, tc.ids[size_t(rank)], 0);
                ops::cpu::cpu_expert_batch_forward(&task, 1, x.data() + size_t(token) * H, 1, per_h.data(),
                    per_out.data(), nullptr, ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2, nullptr, true);
                require(std::memcmp(per_out.data(), batch_out.data() + c * H, H * sizeof(float)) == 0,
                        "multi-row batch must equal the per-row T1 reference");
                require(std::memcmp(per_out.data(), actual.data() + size_t(rank) * H, H * sizeof(float)) == 0,
                        "arena verifier rank row must equal the per-row T1 reference bit for bit");
            }
            // Every GPU rank must stay untouched (its slot keeps the sentinel).
            size_t cpu_cursor = 0;
            for (int rank = 0; rank < rows * top_k; ++rank) {
                const bool is_cpu = cpu_cursor < cold_ranks.size() && cold_ranks[cpu_cursor] == rank;
                if (is_cpu) { ++cpu_cursor; continue; }
                for (int j = 0; j < H; ++j)
                    require(actual[size_t(rank) * H + j] == -7.0f, "GPU rank must not be written by the CPU verifier");
            }
            for (int rank = rows * top_k; rank < int(rank_rows); ++rank)
                for (int j = 0; j < H; ++j)
                    require(actual[size_t(rank) * H + j] == -7.0f, "ranks beyond the window stay untouched");

            std::cout << "[arena verify] " << tc.name << " cpu_tasks=" << cold_ranks.size()
                      << " bitwise=T1-per-row\n";
        }

        // Gate must reject the arena verifier when the arena is gone, proving
        // the new path never silently falls back onto the retired cache.
        arena.Release();
        require(!cpu.CanExecuteArenaVerifyRows(arena, 2, top_k), "released arena must fail the gate");
        ctx.free_device(down);
        ctx.free_device(xd);
        ctx.free_device(fake_gpu);
        std::cout << "PASS: adaptive arena multi-row verifier == per-row T1 CPU reference (rows 2/3/4)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
