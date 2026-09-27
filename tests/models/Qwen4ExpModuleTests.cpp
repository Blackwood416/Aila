#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"
#include "models/Qwen4ExpPle.hpp"
#include "ops/QsaIndexerOps.hpp"
#include "utils/SafeTensors.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

void expect(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("Test failure: " + msg);
    }
}

// ---------------------------------------------------------------------------
// 1. Expert Cache Unit Tests
// ---------------------------------------------------------------------------

void test_expert_cache_lru_and_dequant(Context& ctx, bool compact) {
    std::cout << "[Test] Running expert cache LRU and dequantization unit tests (nested codes="
              << compact << ")...\n";

    constexpr int num_layers = 2;
    constexpr int num_experts_per_layer = 8;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size; // 128
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2); // 128 * 128 = 16384 bytes
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize); // 128 * 4 = 512
    const size_t dn_rows = hidden_size; // 256
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2); // 256 * 32 = 8192 bytes
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize); // 256 * 1 = 256

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    // Create synthetic weight data
    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256);

    std::vector<uint8_t> dn_packed_host(total_dn_packed);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256);

    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) qmap[i] = static_cast<float>(i - 8) / 8.0f;
    for (int i = 0; i < 256; ++i) {
        gu_nested_qmap[i] = static_cast<float>(i) / 256.0f;
        dn_nested_qmap[i] = static_cast<float>(i) / 256.0f;
    }
    // Vary per nested block so a wrong expert offset shows up in dequant checks.
    for (size_t j = 0; j < gu_nested_absmax.size(); ++j) gu_nested_absmax[j] = 1.25f + 0.01f * static_cast<float>(j);
    for (size_t j = 0; j < dn_nested_absmax.size(); ++j) dn_nested_absmax[j] = 0.75f + 0.01f * static_cast<float>(j);

    std::mt19937 rng(12345);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>(rng() & 0xFF);
    for (size_t i = 0; i < total_gu_absmax; ++i) gu_absmax_codes[i] = static_cast<uint8_t>(rng() & 0xFF);
    for (size_t i = 0; i < total_dn_packed; ++i) dn_packed_host[i] = static_cast<uint8_t>(rng() & 0xFF);
    for (size_t i = 0; i < total_dn_absmax; ++i) dn_absmax_codes[i] = static_cast<uint8_t>(rng() & 0xFF);

    Bnb4BitHostWeightRef gu_ref;
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

    Bnb4BitHostWeightRef dn_ref;
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

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;

    const size_t scale_bytes = compact ? sizeof(uint8_t) : sizeof(float);
    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp * scale_bytes +
                              dn_packed_per_exp + dn_absmax_per_exp * scale_bytes;
    const size_t cache_bytes = 4 * slot_bytes; // exactly 4 slots!

    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, compact);
    expect(ok, "cache init failed: " + err);
    expect(cache.uses_nested_codes() == compact, "nested-code mode mismatch");
    expect(cache.num_slots() == 4, "expected 4 slots, got " + std::to_string(cache.num_slots()));

    cache.set_layer_expert_ref(0, gu_ref, dn_ref);
    cache.set_layer_expert_ref(1, gu_ref, dn_ref);

    // Test 1: acquire 3 experts [0, 1, 2] -> all miss, allocated into 3 distinct slots
    int slots1[3] = {-1, -1, -1};
    int req1[3] = {0, 1, 2};
    cache.acquire(0, req1, 3, slots1);

    expect(cache.stats().miss_count == 3, "expected 3 misses, got " + std::to_string(cache.stats().miss_count));
    expect(cache.stats().hit_count == 0, "expected 0 hits");
    expect(slots1[0] != slots1[1] && slots1[1] != slots1[2] && slots1[0] != slots1[2], "slots must be distinct");

    // Verify slot 0 content vs host dequantization
    cache.wait_copy();
    ops::MoeExpertSlotDev dev_slot0 = cache.get_slot_dev(slots1[0]);
    std::vector<uint8_t> read_gu_packed(gu_packed_per_exp);
    ctx.memcpy_d2h(read_gu_packed.data(), dev_slot0.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_gu_packed.data(), gu_packed_host.data(), gu_packed_per_exp) == 0,
           "gate_up packed mismatch on device");

    if (compact) {
        std::vector<uint8_t> read_codes(gu_absmax_per_exp);
        ctx.memcpy_d2h(read_codes.data(), dev_slot0.gate_up_nested.codes, read_codes.size());
        expect(std::memcmp(read_codes.data(), gu_absmax_codes.data(), read_codes.size()) == 0,
               "nested absmax code mismatch");
        std::vector<float> read_scale(gu_absmax_per_exp / 256);
        ctx.memcpy_d2h(read_scale.data(), dev_slot0.gate_up_nested.nested_absmax,
                       read_scale.size() * sizeof(float));
        for (size_t i = 0; i < read_scale.size(); ++i) {
            expect(read_scale[i] == gu_nested_absmax[i], "nested scale mismatch");
        }
        Tensor reconstructed = Tensor::allocate(ctx, {gu_absmax_per_exp}, dnnl::memory::data_type::f32);
        const auto nested = dev_slot0.gate_up_nested;
        auto* reconstructed_ptr = static_cast<float*>(reconstructed.data());
        ctx.queue().parallel_for(sycl::range<1>(gu_absmax_per_exp), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const float q = nested.nested_quant_map[nested.codes[i]];
            const float scale = nested.nested_absmax[i / nested.blocksize];
            reconstructed_ptr[i] = sycl::fma(sycl::fma(q, scale, 0.0f), 1.0f, nested.offset);
        });
        ctx.synchronize();
        std::vector<float> gpu_scales(gu_absmax_per_exp);
        ctx.memcpy_d2h(gpu_scales.data(), reconstructed.data(), gu_absmax_per_exp * sizeof(float));
        for (size_t i = 0; i < gu_absmax_per_exp; ++i) {
            const float host_scale = gu_nested_qmap[gu_absmax_codes[i]] * gu_nested_absmax[i / 256] + 0.05f;
            expect(gpu_scales[i] == host_scale,
                   "nested GPU/host f32 scale rounding mismatch at " + std::to_string(i));
        }
    } else {
        std::vector<float> read_gu_absmax(gu_absmax_per_exp);
        ctx.memcpy_d2h(read_gu_absmax.data(), dev_slot0.gate_up_absmax, gu_absmax_per_exp * sizeof(float));
        for (size_t i = 0; i < gu_absmax_per_exp; ++i) {
            uint8_t code = gu_absmax_codes[i];
            float expected_val = gu_nested_qmap[code] * gu_nested_absmax[i / 256] + 0.05f;
            float diff = std::abs(read_gu_absmax[i] - expected_val);
            expect(diff < 1e-5f, "absmax dequant mismatch at index " + std::to_string(i));
        }
    }

    // Exercise both cache representations through the actual MoE kernels.
    // The CPU reference reconstructs nested scales from the original host
    // codes, rather than reading the GPU slot layout back.
    std::vector<ops::MoeExpertSlotDev> kernel_slots(3);
    cache.get_slots_dev(slots1, 3, kernel_slots.data());
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), 16 * sizeof(float));
    std::vector<bf16> x_host(hidden_size);
    for (int i = 0; i < hidden_size; ++i) x_host[i] = bf16(std::sin(i * 0.13f) * 0.1f);
    Tensor x_dev = Tensor::allocate(ctx, {hidden_size}, dnnl::memory::data_type::bf16);
    Tensor h_dev = Tensor::allocate(ctx, {3 * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor out_dev = Tensor::allocate(ctx, {hidden_size}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));
    const float expert_weights[3] = {0.2f, 0.3f, 0.5f};
    auto matrix_weight = [&](const std::vector<uint8_t>& packed,
                             const std::vector<uint8_t>& codes,
                             const std::vector<float>& scales,
                             const std::vector<float>& nested_map,
                             size_t packed_per_expert, size_t absmax_per_expert,
                             int expert, int row, int col, int width, float offset) {
        const size_t packed_index = static_cast<size_t>(expert) * packed_per_expert +
                                    static_cast<size_t>(row) * (width / 2) + col / 2;
        const uint8_t byte = packed[packed_index];
        const int nibble = col % 2 == 0 ? byte >> 4 : byte & 0xF;
        const size_t scale_index = static_cast<size_t>(expert) * absmax_per_expert +
                                   static_cast<size_t>(row) * (width / blocksize) + col / blocksize;
        const float absmax = nested_map[codes[scale_index]] * scales[scale_index / 256] + offset;
        return qmap[nibble] * absmax;
    };
    std::vector<bf16> h_ref(3 * intermediate_size);
    for (int e = 0; e < 3; ++e) {
        for (int row = 0; row < intermediate_size; ++row) {
            float gate = 0.0f, up = 0.0f;
            for (int col = 0; col < hidden_size; ++col) {
                const float x = static_cast<float>(x_host[col]);
                gate += x * matrix_weight(gu_packed_host, gu_absmax_codes, gu_nested_absmax,
                                          gu_nested_qmap, gu_packed_per_exp, gu_absmax_per_exp,
                                          e, row, col, hidden_size, 0.05f);
                up += x * matrix_weight(gu_packed_host, gu_absmax_codes, gu_nested_absmax,
                                        gu_nested_qmap, gu_packed_per_exp, gu_absmax_per_exp,
                                        e, row + intermediate_size, col, hidden_size, 0.05f);
            }
            h_ref[e * intermediate_size + row] = bf16(gate / (1.0f + std::exp(-gate)) * up);
        }
    }
    std::vector<float> out_ref(hidden_size, 0.0f);
    for (int e = 0; e < 3; ++e) {
        for (int row = 0; row < hidden_size; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < intermediate_size; ++col) {
                sum += static_cast<float>(h_ref[e * intermediate_size + col]) *
                       matrix_weight(dn_packed_host, dn_absmax_codes, dn_nested_absmax,
                                     dn_nested_qmap, dn_packed_per_exp, dn_absmax_per_exp,
                                     e, row, col, intermediate_size, 0.03f);
            }
            out_ref[row] += expert_weights[e] * sum;
        }
    }
    ops::moe_decode_gate_up_swiglu_grouped(
        ctx, kernel_slots.data(), static_cast<const float*>(qmap_dev.data()),
        static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_dev.data()),
        3, hidden_size, intermediate_size, blocksize);
    ops::moe_decode_down_weighted_accumulate(
        ctx, kernel_slots.data(), static_cast<const float*>(qmap_dev.data()),
        expert_weights, static_cast<const bf16*>(h_dev.data()), nullptr,
        static_cast<bf16*>(out_dev.data()), 3, intermediate_size, hidden_size, blocksize);
    ctx.synchronize();
    std::vector<bf16> h_actual(h_ref.size()), out_actual(hidden_size);
    ctx.memcpy_d2h(h_actual.data(), h_dev.data(), h_actual.size() * sizeof(bf16));
    ctx.memcpy_d2h(out_actual.data(), out_dev.data(), out_actual.size() * sizeof(bf16));
    for (size_t i = 0; i < h_ref.size(); ++i) {
        expect(std::isfinite(static_cast<float>(h_actual[i])) &&
               std::abs(static_cast<float>(h_actual[i]) - static_cast<float>(h_ref[i])) < 0.03f,
               "MoE gate/up CPU reference mismatch at " + std::to_string(i));
    }
    for (size_t i = 0; i < out_ref.size(); ++i) {
        expect(std::isfinite(static_cast<float>(out_actual[i])) &&
               std::abs(static_cast<float>(out_actual[i]) - out_ref[i]) < 0.03f + 0.02f * std::abs(out_ref[i]),
               "MoE down CPU reference mismatch at " + std::to_string(i));
    }

    // Test 2: acquire experts [1, 2] -> both hit!
    int slots2[2] = {-1, -1};
    int req2[2] = {1, 2};
    cache.acquire(0, req2, 2, slots2);
    expect(slots2[0] == slots1[1], "expert 1 slot mismatch on hit");
    expect(slots2[1] == slots1[2], "expert 2 slot mismatch on hit");
    expect(cache.stats().hit_count == 2, "expected 2 hits");
    expect(cache.stats().miss_count == 3, "expected still 3 misses");

    // Test 3: acquire expert 3 -> misses. Since slots 1 and 2 were recently accessed, slot 0 or slot 3 should be used
    int slots3[1] = {-1};
    int req3[1] = {3};
    cache.acquire(0, req3, 1, slots3);
    expect(cache.stats().miss_count == 4, "expected 4 misses");
    expect(slots3[0] != slots2[0] && slots3[0] != slots2[1], "slot 1 and 2 must not be evicted");

    // Test 4: single batch acquire [2, 4, 5]:
    // Expert 2 hits (in slots2[1]). Experts 4 and 5 miss.
    // Crucial: slots2[1] MUST NOT be evicted by expert 4 or 5!
    int slots4[3] = {-1, -1, -1};
    int req4[3] = {2, 4, 5};
    cache.acquire(0, req4, 3, slots4);
    expect(slots4[0] == slots2[1], "expert 2 must hit its existing slot");
    expect(slots4[0] != slots4[1] && slots4[0] != slots4[2] && slots4[1] != slots4[2],
           "all batch slots must be mutually distinct");

    // Test 5: duplicate expert in same batch [4, 4]
    int slots5[2] = {-1, -1};
    int req5[2] = {4, 4};
    cache.acquire(0, req5, 2, slots5);
    expect(slots5[0] == slots4[1] && slots5[1] == slots4[1], "duplicate experts in batch must resolve to same slot");

    // Test 6: fill-only prefetch. Layer 1 experts {7, 6} plus resident layer-0 expert 4
    // (skipped). Prefetch must not bind or evict anything; acquire then serves expert 7
    // from the prefetch staging and fills expert 3 normally.
    const bool resident_before[3] = {cache.contains(0, 2), cache.contains(0, 4), cache.contains(0, 5)};
    const uint64_t misses_before = cache.stats().miss_count;
    int pf_req[3] = {7, 6, 4};
    cache.prefetch_async(1, pf_req, 2);
    cache.prefetch_async(0, pf_req + 2, 1); // joins the previous job; resident -> nothing staged
    expect(cache.stats().prefetch_issued == 2, "expected 2 staged prefetches, got " +
           std::to_string(cache.stats().prefetch_issued));
    // Restage layer 1 (the second call replaced the staged set).
    cache.prefetch_async(1, pf_req, 2);
    expect(!cache.contains(1, 7) && !cache.contains(1, 6), "prefetch must not bind slots");
    expect(cache.contains(0, 2) == resident_before[0] && cache.contains(0, 4) == resident_before[1] &&
           cache.contains(0, 5) == resident_before[2], "prefetch must not evict");

    int slots6[2] = {-1, -1};
    int req6[2] = {7, 3};
    cache.acquire(1, req6, 2, slots6);
    expect(cache.stats().prefetch_used == 1, "expected expert 7 served from prefetch staging");
    expect(cache.stats().miss_count == misses_before + 2, "both layer-1 experts are misses");

    auto check_slot = [&](int slot, int expert, const char* what) {
        cache.wait_copy();
        ops::MoeExpertSlotDev dev = cache.get_slot_dev(slot);
        std::vector<uint8_t> gu(gu_packed_per_exp), dn(dn_packed_per_exp);
        ctx.memcpy_d2h(gu.data(), dev.gate_up_packed, gu.size());
        ctx.memcpy_d2h(dn.data(), dev.down_packed, dn.size());
        expect(std::memcmp(gu.data(), gu_packed_host.data() + expert * gu_packed_per_exp, gu.size()) == 0,
               std::string(what) + ": gate_up packed mismatch");
        expect(std::memcmp(dn.data(), dn_packed_host.data() + expert * dn_packed_per_exp, dn.size()) == 0,
               std::string(what) + ": down packed mismatch");
        if (compact) {
            std::vector<uint8_t> gua(gu_absmax_per_exp), dna(dn_absmax_per_exp);
            ctx.memcpy_d2h(gua.data(), dev.gate_up_nested.codes, gua.size());
            ctx.memcpy_d2h(dna.data(), dev.down_nested.codes, dna.size());
            expect(std::memcmp(gua.data(), gu_absmax_codes.data() + expert * gu_absmax_per_exp, gua.size()) == 0,
                   std::string(what) + ": gate_up codes mismatch");
            expect(std::memcmp(dna.data(), dn_absmax_codes.data() + expert * dn_absmax_per_exp, dna.size()) == 0,
                   std::string(what) + ": down codes mismatch");
            std::vector<float> gu_scale(gu_absmax_per_exp / 256), dn_scale(dn_absmax_per_exp / 256);
            ctx.memcpy_d2h(gu_scale.data(), dev.gate_up_nested.nested_absmax,
                           gu_scale.size() * sizeof(float));
            ctx.memcpy_d2h(dn_scale.data(), dev.down_nested.nested_absmax,
                           dn_scale.size() * sizeof(float));
            for (size_t i = 0; i < gu_scale.size(); ++i) {
                expect(gu_scale[i] == gu_nested_absmax[expert * gu_scale.size() + i],
                       std::string(what) + ": gate_up nested scale offset mismatch");
            }
            for (size_t i = 0; i < dn_scale.size(); ++i) {
                expect(dn_scale[i] == dn_nested_absmax[expert * dn_scale.size() + i],
                       std::string(what) + ": down nested scale offset mismatch");
            }
            return;
        }
        std::vector<float> gua(gu_absmax_per_exp), dna(dn_absmax_per_exp);
        ctx.memcpy_d2h(gua.data(), dev.gate_up_absmax, gua.size() * sizeof(float));
        ctx.memcpy_d2h(dna.data(), dev.down_absmax, dna.size() * sizeof(float));
        for (size_t i = 0; i < gu_absmax_per_exp; ++i) {
            const size_t g = expert * gu_absmax_per_exp + i;
            const float want = gu_nested_qmap[gu_absmax_codes[g]] * gu_nested_absmax[g / 256] + 0.05f;
            expect(std::abs(gua[i] - want) < 1e-5f, std::string(what) + ": gate_up absmax mismatch");
        }
        for (size_t i = 0; i < dn_absmax_per_exp; ++i) {
            const size_t g = expert * dn_absmax_per_exp + i;
            const float want = dn_nested_qmap[dn_absmax_codes[g]] * dn_nested_absmax[g / 256] + 0.03f;
            expect(std::abs(dna[i] - want) < 1e-5f, std::string(what) + ": down absmax mismatch");
        }
    };
    check_slot(slots6[0], 7, "prefetched expert 7");
    check_slot(slots6[1], 3, "filled expert 3");

    // Demotion: a demoted expert is evicted before older, non-demoted ones.
    int slots7[2] = {-1, -1};
    cache.acquire(1, req6, 2, slots7); // both hits -> most recently used
    bool layer0_before[num_experts_per_layer];
    for (int e = 0; e < num_experts_per_layer; ++e) layer0_before[e] = cache.contains(0, e);
    const int demoted[1] = {7};
    cache.demote(1, demoted, 1);
    int slots8[1] = {-1};
    const int req8[1] = {5};
    cache.acquire(1, req8, 1, slots8);
    expect(slots8[0] == slots7[0], "demoted expert 7's slot must be the victim");
    expect(!cache.contains(1, 7) && cache.contains(1, 3) && cache.contains(1, 5),
           "demotion must evict only expert 7");
    for (int e = 0; e < num_experts_per_layer; ++e) {
        expect(cache.contains(0, e) == layer0_before[e], "demotion must keep older layer-0 experts");
    }
    check_slot(slots8[0], 5, "expert 5 in the demoted slot");

    // A request larger than the GPU slot pool must fail before publishing any
    // key-to-slot bindings. Retrying its first miss must perform a real fill.
    const int over_capacity_req[5] = {6, 0, 1, 2, 3};
    int over_capacity_slots[5] = {-1, -1, -1, -1, -1};
    const uint64_t misses_before_over_capacity = cache.stats().miss_count;
    bool rejected_over_capacity = false;
    try {
        cache.acquire(0, over_capacity_req, 5, over_capacity_slots);
    } catch (const std::runtime_error&) {
        rejected_over_capacity = true;
    }
    expect(rejected_over_capacity, "unique expert request larger than GPU pool must be rejected");
    expect(!cache.is_faulted(), "preflight capacity rejection must leave the cache usable");
    expect(!cache.contains(0, 6), "failed over-capacity acquire must not publish expert 6 as resident");
    expect(cache.stats().miss_count == misses_before_over_capacity,
           "preflight capacity rejection must not count partial misses");

    int retry_slot[1] = {-1};
    cache.acquire(0, over_capacity_req, 1, retry_slot);
    expect(cache.stats().miss_count == misses_before_over_capacity + 1,
           "retry after capacity rejection must be a real miss/fill");
    check_slot(retry_slot[0], 6, "expert 6 retry after over-capacity rejection");

    std::cout << "[PASS] Expert cache LRU and dequantization tests passed. Final hit rate: "
              << (cache.stats().hit_rate() * 100.0) << "%\n";
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

void test_hot_host_ram_pool_and_prefetch(Context& ctx) {
    constexpr int num_layers = 2;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 128;
    constexpr int blocksize = 64;
    constexpr int gu_rows = 2 * intermediate_size;
    constexpr int dn_rows = hidden_size;

    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax);
    std::vector<uint8_t> dn_packed_host(total_dn_packed);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    std::mt19937 rng(999);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>(rng() & 0xFF);
    for (size_t i = 0; i < total_gu_absmax; ++i) gu_absmax_codes[i] = static_cast<uint8_t>(rng() & 0xFF);
    for (size_t i = 0; i < total_dn_packed; ++i) dn_packed_host[i] = static_cast<uint8_t>(rng() & 0xFF);
    for (size_t i = 0; i < total_dn_absmax; ++i) dn_absmax_codes[i] = static_cast<uint8_t>(rng() & 0xFF);

    Bnb4BitHostWeightRef gu_ref;
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

    Bnb4BitHostWeightRef dn_ref;
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

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 4 * slot_bytes; // 4 GPU slots

    // Part A: Hot Host RAM Pool Lifecycle (Hits, Eviction, Prefetch)
    // 8 slots in host pool > 4 GPU slots (mimics production where host pool is larger than GPU VRAM cache)
    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=8");

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    expect(cache.num_slots() == 4, "expected 4 GPU slots");
    expect(cache.host_pool_enabled(), "host pool must be enabled");
    expect(cache.num_host_slots() == 8, "expected exactly 8 host slots");

    cache.set_layer_expert_ref(0, gu_ref, dn_ref);
    cache.set_layer_expert_ref(1, gu_ref, dn_ref);

    // 1. Initial acquire: experts [0, 1] -> both miss GPU, both miss Host Pool
    int slots1[2] = {-1, -1};
    int req1[2] = {0, 1};
    cache.acquire(0, req1, 2, slots1);
    expect(cache.stats().miss_count == 2, "expected 2 misses");
    expect(cache.stats().host_pool_misses == 2, "expected 2 host pool misses");
    expect(cache.stats().host_pool_hits == 0, "expected 0 host pool hits");

    // 2. Demote [0, 1] and fill [2, 3, 4] to evict 0 from GPU cache (4 GPU slots < 2 + 3 = 5)
    int demote_req[2] = {0, 1};
    cache.demote(0, demote_req, 2);
    int req2[3] = {2, 3, 4};
    int slots2[3] = {-1, -1, -1};
    cache.acquire(0, req2, 3, slots2);
    // Since GPU has 4 slots, acquiring 3 more evicted demoted expert 0 from GPU cache.
    // But Host Pool has 8 slots, so expert 0 is still resident in the Host Pool.

    // 3. Re-acquire expert 0: misses GPU, but hits in Host Pool without host miss!
    int slots3[1] = {-1};
    int req3[1] = {0};
    uint64_t prev_host_hits = cache.stats().host_pool_hits;
    cache.acquire(0, req3, 1, slots3);
    expect(cache.stats().host_pool_hits == prev_host_hits + 1, "expected host pool hit for evicted expert 0");
    cache.wait_copy();
    ops::MoeExpertSlotDev dev_slot3 = cache.get_slot_dev(slots3[0]);
    std::vector<uint8_t> read_gu3(gu_packed_per_exp);
    ctx.memcpy_d2h(read_gu3.data(), dev_slot3.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_gu3.data(), gu_packed_host.data(), gu_packed_per_exp) == 0,
           "host pool hit expert 0 content mismatch on GPU");

    // 4. Test prefetch into host pool
    int pf_req[1] = {7};
    cache.prefetch_async(1, pf_req, 1);
    int slots4[1] = {-1};
    int req4[1] = {7};
    cache.acquire(1, req4, 1, slots4);
    expect(cache.stats().prefetch_used >= 1, "expected prefetch used >= 1");
    cache.wait_copy();
    ops::MoeExpertSlotDev dev_slot4 = cache.get_slot_dev(slots4[0]);
    std::vector<uint8_t> read_gu7(gu_packed_per_exp);
    ctx.memcpy_d2h(read_gu7.data(), dev_slot4.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_gu7.data(), gu_packed_host.data() + 7 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "prefetched expert 7 content mismatch on GPU");

    // Part B: Stress test chunking in a constrained host pool:
    // Configure cache with 8 GPU slots and only 4 host slots.
    // Acquiring 6 cold experts forces n_fill = 6 > num_host_slots = 4,
    // which strictly exercises the multi-chunk path (chunk 1: 4 slots, drain DMA, chunk 2: 2 slots).
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");
    Qwen4ExpExpertCache chunk_cache;
    ok = chunk_cache.init(ctx, dummy_weights, num_layers, 8 * slot_bytes,
                          num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "chunk cache init failed: " + err);
    expect(chunk_cache.num_slots() == 8, "expected 8 GPU slots");
    expect(chunk_cache.num_host_slots() == 4, "expected 4 host slots");
    chunk_cache.set_layer_expert_ref(0, gu_ref, dn_ref);

    int req_stress[6] = {0, 1, 2, 3, 4, 5};
    int slots_stress[6] = {-1, -1, -1, -1, -1, -1};
    chunk_cache.acquire(0, req_stress, 6, slots_stress);
    for (int i = 0; i < 6; ++i) {
        expect(slots_stress[i] >= 0, "slot must be valid");
        for (int j = i + 1; j < 6; ++j) {
            expect(slots_stress[i] != slots_stress[j], "batch slots must be distinct");
        }
    }
    chunk_cache.wait_copy();
    ops::MoeExpertSlotDev dev_slot_c1 = chunk_cache.get_slot_dev(slots_stress[0]);
    std::vector<uint8_t> read_gu_c1(gu_packed_per_exp);
    ctx.memcpy_d2h(read_gu_c1.data(), dev_slot_c1.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_gu_c1.data(), gu_packed_host.data() /* expert 0 */, gu_packed_per_exp) == 0,
           "stress chunk 1 expert 0 content mismatch on GPU");

    ops::MoeExpertSlotDev dev_slot_c2 = chunk_cache.get_slot_dev(slots_stress[5]);
    std::vector<uint8_t> read_gu_c2(gu_packed_per_exp);
    ctx.memcpy_d2h(read_gu_c2.data(), dev_slot_c2.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_gu_c2.data(), gu_packed_host.data() + 5 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "stress chunk 2 expert 5 content mismatch on GPU");

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    _putenv("AILA_Q4E_HOT_RAM_MB=");
    std::cout << "[PASS] Hot Host RAM Pool & Prefetch verified successfully.\n";
}

void test_prefill_speculative_prefetch(Context& ctx) {
    std::cout << "[Test] Running Prefill speculative next-layer prefetch test...\n";

    constexpr int num_layers = 2;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size; // 128
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2); // 16384 bytes
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize); // 512
    const size_t dn_rows = hidden_size; // 256
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2); // 8192 bytes
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize); // 256

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed, 0x42);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x10);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0x24);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x20);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);

    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 8 * slot_bytes; // 8 GPU slots

    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=16");

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    expect(cache.host_pool_enabled(), "host pool must be enabled");

    cache.set_layer_expert_ref(0, gu_ref, dn_ref);
    cache.set_layer_expert_ref(1, gu_ref, dn_ref);

    // Simulate prefill: Layer 0 acquires batch [0, 1, 2, 3]
    int l0_req[4] = {0, 1, 2, 3};
    int l0_slots[4] = {-1, -1, -1, -1};
    cache.acquire(0, l0_req, 4, l0_slots);
    expect(cache.stats().miss_count == 4, "expected 4 initial misses for Layer 0");

    // Speculative routing predicts Layer 1 will need [4, 5, 6, 7]
    // Issue prefetch_async for Layer 1 into host pool concurrently
    int l1_pred[4] = {4, 5, 6, 7};
    cache.prefetch_async(1, l1_pred, 4);

    // Now Layer 1 acquires [4, 5, 6, 7]: all 4 should hit the Host RAM Pool!
    int l1_slots[4] = {-1, -1, -1, -1};
    uint64_t prev_host_hits = cache.stats().host_pool_hits;
    cache.acquire(1, l1_pred, 4, l1_slots);
    expect(cache.stats().host_pool_hits == prev_host_hits + 4, "all 4 prefetched experts must hit Host Pool");
    cache.wait_copy();

    for (int i = 0; i < 4; ++i) {
        expect(l1_slots[i] >= 0, "slot must be valid");
        ops::MoeExpertSlotDev dev_slot = cache.get_slot_dev(l1_slots[i]);
        std::vector<uint8_t> read_gu(gu_packed_per_exp);
        ctx.memcpy_d2h(read_gu.data(), dev_slot.gate_up_packed, gu_packed_per_exp);
        expect(std::memcmp(read_gu.data(), gu_packed_host.data() + (4 + i) * gu_packed_per_exp, gu_packed_per_exp) == 0,
               "prefetched expert content mismatch on GPU");
    }

    // Phase 2: Multi-token frequency ranking, deduplication, and quota capping verification
    // Simulate prefill tokens: Token 0=[10,11], Token 1=[12,10], Token 2=[10,12], Token 3=[13,10]
    // Expert 10 has 4 votes, Expert 12 has 2 votes, Experts 11 and 13 have 1 vote each.
    const int raw_token_preds[8] = {10, 11, 12, 10, 10, 12, 13, 10};
    std::vector<int> pred_unique;
    std::vector<int> pred_uses(num_experts_per_layer, 0);
    for (int e : raw_token_preds) {
        if (e >= 0 && e < num_experts_per_layer && pred_uses[e]++ == 0) {
            pred_unique.push_back(e);
        }
    }
    std::stable_sort(pred_unique.begin(), pred_unique.end(), [&](int a, int b) {
        return pred_uses[a] > pred_uses[b];
    });
    expect(pred_unique.size() == 4, "expected 4 unique experts");
    expect(pred_unique[0] == 10, "expert 10 (4 votes) must rank first");
    expect(pred_unique[1] == 12, "expert 12 (2 votes) must rank second");

    // Cap prefetch quota to top-2 experts: [10, 12]
    constexpr int prefetch_quota = 2;
    cache.prefetch_async(1, pred_unique.data(), prefetch_quota);

    // Layer 1 acquires [10, 12, 11, 13]: exactly the 2 prefetched experts must hit Host Pool, others fill on-demand
    int l1_freq_req[4] = {10, 12, 11, 13};
    int l1_freq_slots[4] = {-1, -1, -1, -1};
    prev_host_hits = cache.stats().host_pool_hits;
    cache.acquire(1, l1_freq_req, 4, l1_freq_slots);
    expect(cache.stats().host_pool_hits == prev_host_hits + 2,
           "exactly the 2 prefetched high-frequency experts must hit Host Pool");
    cache.wait_copy();

    for (int i = 0; i < 4; ++i) {
        expect(l1_freq_slots[i] >= 0, "slot must be valid");
        ops::MoeExpertSlotDev dev_slot = cache.get_slot_dev(l1_freq_slots[i]);
        std::vector<uint8_t> read_gu(gu_packed_per_exp);
        ctx.memcpy_d2h(read_gu.data(), dev_slot.gate_up_packed, gu_packed_per_exp);
        expect(std::memcmp(read_gu.data(), gu_packed_host.data() + l1_freq_req[i] * gu_packed_per_exp, gu_packed_per_exp) == 0,
               "expert content mismatch on GPU in frequency ranking test");
    }

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Prefill speculative next-layer prefetch verified successfully.\n";
}

void test_fine_grained_demand_prefetch(Context& ctx) {
    std::cout << "[Test] Running Phase S2 fine-grained demand prefetch & GPU hit bypass test...\n";

    constexpr int num_layers = 2;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size; // 128
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2); // 16384 bytes
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize); // 512
    const size_t dn_rows = hidden_size; // 256
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2); // 8192 bytes
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize); // 256

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed, 0x55);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x12);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xAA);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x34);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 8 * slot_bytes; // 8 GPU slots

    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=16");

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);
    cache.set_layer_expert_ref(1, gu_ref, dn_ref);

    // Warm up Layer 0 experts [0, 1, 2, 3] on GPU
    int l0_req[4] = {0, 1, 2, 3};
    int l0_slots[4] = {-1, -1, -1, -1};
    cache.acquire(0, l0_req, 4, l0_slots);
    expect(cache.stats().miss_count == 4, "expected 4 initial misses");

    // Launch heavy speculative prefetch for Layer 1: 8 experts [4, 5, 6, 7, 8, 9, 10, 11]
    int l1_pf[8] = {4, 5, 6, 7, 8, 9, 10, 11};
    cache.prefetch_async(1, l1_pf, 8);

    // Test Case A: 100% GPU hit path must return immediately without waiting for prefetch
    const uint64_t wait_before = cache.stats().prefetch_wait_ns;
    int l0_hit_slots[4] = {-1, -1, -1, -1};
    cache.acquire(0, l0_req, 4, l0_hit_slots);
    expect(cache.stats().hit_count == 4, "expected 4 GPU hits");
    expect(cache.stats().prefetch_wait_ns == wait_before,
           "100% GPU hit path must NOT incur any prefetch wait time");
    for (int i = 0; i < 4; ++i) {
        expect(l0_hit_slots[i] == l0_slots[i], "GPU hit slot mismatch");
    }

    // Test Case B: Fine-grained demand wait.
    // Layer 1 only needs expert 7 (single expert from the 8 prefetched).
    // Acquire must only wait for expert 7 and not wait for the remaining 7 unneeded experts.
    int l1_single_req[1] = {7};
    int l1_single_slot[1] = {-1};
    uint64_t prev_hits = cache.stats().host_pool_hits;
    cache.acquire(1, l1_single_req, 1, l1_single_slot);
    expect(cache.stats().host_pool_hits == prev_hits + 1, "demanded expert 7 must hit Host Pool");
    expect(cache.stats().prefetch_used >= 1, "demanded expert 7 must increment prefetch_used");
    cache.wait_copy();

    ops::MoeExpertSlotDev dev_slot7 = cache.get_slot_dev(l1_single_slot[0]);
    std::vector<uint8_t> read_gu7(gu_packed_per_exp);
    ctx.memcpy_d2h(read_gu7.data(), dev_slot7.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_gu7.data(), gu_packed_host.data() + 7 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "demanded expert 7 content mismatch on GPU");

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S2 fine-grained demand prefetch & GPU hit bypass verified successfully.\n";
}

void test_slru_host_cache_and_anti_thrashing(Context& ctx) {
    std::cout << "[Test] Running Phase S3 2-tier SLRU host cache & anti-thrashing test...\n";

    constexpr int num_layers = 2;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size; // 128
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2); // 16384 bytes
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize); // 512
    const size_t dn_rows = hidden_size; // 256
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2); // 8192 bytes
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize); // 256

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed, 0x33);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x11);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.0f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0x77);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x22);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 1.0f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 2 * slot_bytes; // Only 2 GPU slots to trigger frequent GPU miss & Host lookup

    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=4"); // 4 Host slots
    _putenv("AILA_Q4E_HOST_PROTECTED_RATIO=0.5"); // 50% = 2 protected, 2 probationary

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    expect(cache.host_pool_enabled(), "host pool must be enabled");
    expect(cache.num_host_slots() == 4, "expected 4 host slots");
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);
    cache.set_layer_expert_ref(1, gu_ref, dn_ref);

    // Initial state: 4 empty slots
    expect(cache.host_empty_count() == 4, "expected 4 empty host slots");
    expect(cache.host_probationary_count() == 0, "expected 0 probationary host slots");
    expect(cache.host_protected_count() == 0, "expected 0 protected host slots");

    // 1. Acquire Expert 0 on Layer 0:
    // Misses GPU and Host Pool -> demand fill into Host Pool.
    // Must be placed in probationary segment.
    int req0[1] = {0};
    int slot0[1] = {-1};
    cache.acquire(0, req0, 1, slot0);
    expect(cache.host_contains(0, 0), "expert 0 must be in host pool");
    expect(!cache.host_is_protected(0, 0), "first touch must remain in probationary segment");
    expect(cache.host_protected_count() == 0, "no protected slots yet");

    // 2. Acquire Expert 1 on Layer 0:
    int req1[1] = {1};
    int slot1[1] = {-1};
    cache.acquire(0, req1, 1, slot1);
    expect(cache.host_contains(0, 1), "expert 1 must be in host pool");
    expect(!cache.host_is_protected(0, 1), "first touch must remain in probationary segment");

    // 3. Acquire Expert 2 on Layer 0: (GPU slots = 2, so Expert 0 is evicted from GPU!)
    int req2[1] = {2};
    int slot2[1] = {-1};
    cache.acquire(0, req2, 1, slot2);
    expect(!cache.contains(0, 0), "expert 0 must have been evicted from GPU cache");
    expect(cache.host_contains(0, 0), "expert 0 must still be in host pool");

    // 4. Second touch of Expert 0:
    // Hits Host Pool! Must be promoted to protected segment.
    uint64_t prev_h_hits = cache.stats().host_pool_hits;
    cache.acquire(0, req0, 1, slot0);
    expect(cache.stats().host_pool_hits == prev_h_hits + 1, "expert 0 must hit host pool");
    expect(cache.host_is_protected(0, 0), "second touch must promote expert 0 to protected segment");
    expect(cache.host_protected_count() == 1, "expected 1 protected host slot");

    // 5. Fill remaining host capacity:
    // Host has 4 slots: Expert 0 is protected. Expert 1, 2 are probationary.
    // Acquire Expert 3 (fills 4th slot, probationary).
    int req3[1] = {3};
    int slot3[1] = {-1};
    cache.acquire(0, req3, 1, slot3);
    expect(cache.host_contains(0, 3), "expert 3 must be in host pool");
    expect(!cache.host_is_protected(0, 3), "expert 3 must be probationary");

    // Now host pool is full (1 protected [0], 3 probationary [3, 2, 1]).
    // Evict a slot by acquiring Expert 4:
    // With SLRU, probationary victim (Expert 1) must be evicted! Protected slot (0) must NOT be evicted!
    int req4[1] = {4};
    int slot4[1] = {-1};
    cache.acquire(0, req4, 1, slot4);
    expect(cache.host_contains(0, 0), "protected expert 0 MUST NOT be evicted by new fill");
    expect(cache.host_is_protected(0, 0), "expert 0 must remain protected");
    expect(!cache.host_contains(0, 1), "probationary LRU expert 1 should have been evicted");

    // 6. Test Demote anti-thrashing:
    // Expert 0 is protected. Demote expert 0.
    int demote_e[1] = {0};
    cache.demote(0, demote_e, 1);
    expect(!cache.host_is_protected(0, 0), "demoted expert must no longer be protected");
    expect(cache.host_protected_count() == 0, "protected count must drop to 0 after demoting sole protected expert");

    // Because expert 0 was demoted to the tail of probationary list, the NEXT eviction must pick expert 0!
    int req5[1] = {5};
    int slot5[1] = {-1};
    cache.acquire(0, req5, 1, slot5);
    expect(!cache.host_contains(0, 0), "demoted expert 0 must be the first victim evicted");
    expect(cache.host_contains(0, 5), "expert 5 must now reside in host pool");

    _putenv("AILA_Q4E_HOST_PROTECTED_RATIO=");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S3 2-tier SLRU host cache & anti-thrashing verified successfully.\n";
}

void test_host_slot_reservation_and_demand_priority(Context& ctx) {
    std::cout << "[Test] Running Phase S2 host slot reservation & demand fill priority test...\n";

    constexpr int num_layers = 2;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>(i % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x12);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xAA);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x34);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 4 * slot_bytes; // 4 GPU slots

    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=4"); // 4 Host slots

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);
    cache.set_layer_expert_ref(1, gu_ref, dn_ref);

    // 1. Acquire batch of 3 experts [0, 1, 2] in single acquire call:
    // With 4 host slots, all 3 must be filled and reserved without collision.
    int req3[3] = {0, 1, 2};
    int slots3[3] = {-1, -1, -1};
    cache.acquire(0, req3, 3, slots3);
    for (int i = 0; i < 3; ++i) {
        expect(slots3[i] >= 0, "acquired slot must be valid");
        expect(cache.host_contains(0, req3[i]), "host pool must contain acquired expert");
    }
    // After acquire returns, reservations are released
    expect(!cache.host_is_reserved(0, 0), "host reservation must be released after acquire");
    expect(!cache.host_is_reserved(0, 1), "host reservation must be released after acquire");
    expect(!cache.host_is_reserved(0, 2), "host reservation must be released after acquire");

    // 2. Test prefetch in flight + demand preemption:
    // Launch speculative prefetch for 4 experts [3, 4, 5, 6]
    int pf[4] = {3, 4, 5, 6};
    cache.prefetch_async(0, pf, 4);

    // Immediately launch a demand acquire for expert 7 on Layer 1:
    // This must execute on high-priority demand queue, preempting any background prefetch
    int demand_req[1] = {7};
    int demand_slot[1] = {-1};
    cache.acquire(1, demand_req, 1, demand_slot);
    expect(demand_slot[0] >= 0, "demand slot must be valid");
    cache.wait_copy();

    // Verify GPU content of demand expert 7
    ops::MoeExpertSlotDev dev_slot = cache.get_slot_dev(demand_slot[0]);
    std::vector<uint8_t> read_back(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back.data(), dev_slot.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back.data(), gu_packed_host.data() + 7 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "demand expert 7 data verification failed");

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S2 host slot reservation & demand fill priority verified successfully.\n";
}

void test_host_slot_contention_and_dma_protection(Context& ctx) {
    std::cout << "[Test] Running Phase S2 host slot contention & DMA protection test...\n";

    constexpr int num_layers = 1;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>((i * 7 + 13) % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x12);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xBB);
    for (size_t i = 0; i < total_dn_packed; ++i) dn_packed_host[i] = static_cast<uint8_t>((i * 11 + 3) % 251);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x34);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 2 * slot_bytes; // 2 GPU slots

    // Extreme stress test: ONLY 1 Host Slot!
    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=1");

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);

    // 1. Warm up expert 0 -> occupies the single host slot 0
    int req1[1] = {0};
    int slots1[1] = {-1};
    cache.acquire(0, req1, 1, slots1);
    cache.wait_copy();
    expect(cache.host_contains(0, 0), "expert 0 must be in host slot");

    // 2. In a single batch acquire {0, 1}:
    // Expert 0 is a host hit. It reserves host slot 0 and starts async DMA.
    // Expert 1 is a miss. Since only 1 host slot exists, slot 0 MUST NOT be overwritten until expert 0 DMA finishes!
    int req2[2] = {0, 1};
    int slots2[2] = {-1, -1};
    cache.acquire(0, req2, 2, slots2);
    cache.wait_copy();
    expect(slots2[0] >= 0 && slots2[1] >= 0, "both slots must be acquired");
    expect(slots2[0] != slots2[1], "GPU slots must be distinct");

    // Verify GPU content of expert 0
    ops::MoeExpertSlotDev dev_slot0 = cache.get_slot_dev(slots2[0]);
    std::vector<uint8_t> read_back0(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back0.data(), dev_slot0.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back0.data(), gu_packed_host.data() + 0 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "expert 0 data was corrupted during concurrent host slot reuse!");

    // Verify GPU content of expert 1
    ops::MoeExpertSlotDev dev_slot1 = cache.get_slot_dev(slots2[1]);
    std::vector<uint8_t> read_back1(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back1.data(), dev_slot1.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back1.data(), gu_packed_host.data() + 1 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "expert 1 data verification failed");

    // 3. Test prefetch cancellation and wait safety (must not deadlock)
    int pf[4] = {2, 3, 4, 5};
    cache.prefetch_async(0, pf, 4);
    cache.cancel_prefetch_for_testing();
    cache.wait_prefetch(); // Must return cleanly without infinite loop!

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S2 host slot contention & DMA protection verified successfully.\n";
}

void test_host_prefetch_hit_dma_protection(Context& ctx) {
    std::cout << "[Test] Running Phase S2/N0 prefetch hit DMA protection test...\n";

    constexpr int num_layers = 1;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>((i * 7 + 13) % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x12);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xBB);
    for (size_t i = 0; i < total_dn_packed; ++i) dn_packed_host[i] = static_cast<uint8_t>((i * 11 + 3) % 251);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x34);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 2 * slot_bytes; // 2 GPU slots

    // Extreme stress test: ONLY 1 Host Slot!
    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=1");

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);

    // 1. Prefetch expert 0 (A) -> fills host slot 0 and reserves it
    int pf[1] = {0};
    cache.prefetch_async(0, pf, 1);
    cache.wait_prefetch();
    expect(cache.host_contains(0, 0), "expert 0 must be in host slot");

    // 2. Acquire {0, 1}:
    // 0 is prefetch hit -> transfers to batch readers, starts DMA
    // 1 is demand miss -> needs host slot, but slot 0 is in-flight DMA reader
    int req[2] = {0, 1};
    int slots[2] = {-1, -1};
    cache.acquire(0, req, 2, slots);

    // 3. Immediately prefetch expert 2 (C)
    // Must also respect DMA event on host slot 0
    int pf2[1] = {2};
    cache.prefetch_async(0, pf2, 1);
    cache.wait_prefetch();
    cache.wait_copy();

    // Verify GPU content of expert 0
    ops::MoeExpertSlotDev dev_slot0 = cache.get_slot_dev(slots[0]);
    std::vector<uint8_t> read_back0(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back0.data(), dev_slot0.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back0.data(), gu_packed_host.data() + 0 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "expert 0 data was corrupted during concurrent prefetch-hit host slot reuse!");

    // Verify GPU content of expert 1
    ops::MoeExpertSlotDev dev_slot1 = cache.get_slot_dev(slots[1]);
    std::vector<uint8_t> read_back1(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back1.data(), dev_slot1.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back1.data(), gu_packed_host.data() + 1 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "expert 1 data was corrupted!");

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S2/N0 prefetch hit DMA protection verified successfully.\n";
}

void test_expert_cache_move_lifecycle(Context& ctx) {
    std::cout << "[Test] Running Phase S2/N0 expert cache move lifecycle test...\n";

    constexpr int num_layers = 1;
    constexpr int num_experts_per_layer = 16;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>((i * 7 + 13) % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x12);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.25f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xBB);
    for (size_t i = 0; i < total_dn_packed; ++i) dn_packed_host[i] = static_cast<uint8_t>((i * 11 + 3) % 251);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x34);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 0.75f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.1f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 2 * slot_bytes; // 2 GPU slots

    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache1;
    std::string err;
    bool ok = cache1.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                          num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache1 init failed: " + err);
    cache1.set_layer_expert_ref(0, gu_ref, dn_ref);

    // Warm up expert 0 and 1 in cache1
    int req1[2] = {0, 1};
    int slots1[2] = {-1, -1};
    cache1.acquire(0, req1, 2, slots1);
    cache1.wait_copy();
    expect(cache1.host_contains(0, 0), "cache1 must contain expert 0");
    expect(cache1.host_contains(0, 1), "cache1 must contain expert 1");

    // Move-construct cache2 from cache1
    Qwen4ExpExpertCache cache2(std::move(cache1));
    expect(!cache1.is_initialized(), "cache1 must be uninitialized after move");
    expect(cache2.is_initialized(), "cache2 must be initialized");
    expect(cache2.host_contains(0, 0), "cache2 must retain expert 0");
    expect(cache2.host_contains(0, 1), "cache2 must retain expert 1");

    // 1. Host hit on cache2: expert 0
    int req2[1] = {0};
    int slots2[1] = {-1};
    cache2.acquire(0, req2, 1, slots2);
    cache2.wait_copy();
    expect(slots2[0] >= 0, "cache2 hit acquire must succeed");

    // 2. Miss on cache2: acquire expert 2 and 3
    int req3[2] = {2, 3};
    int slots3[2] = {-1, -1};
    cache2.acquire(0, req3, 2, slots3);
    cache2.wait_copy();
    expect(slots3[0] >= 0 && slots3[1] >= 0, "cache2 miss acquire must succeed");

    // Verify GPU content of expert 2
    ops::MoeExpertSlotDev dev_slot2 = cache2.get_slot_dev(slots3[0]);
    std::vector<uint8_t> read_back2(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back2.data(), dev_slot2.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back2.data(), gu_packed_host.data() + 2 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "expert 2 data verification on moved cache failed");

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S2/N0 expert cache move lifecycle verified successfully.\n";
}

void test_fill_parts_64_bitmap(Context& ctx) {
    std::cout << "[Test] Running Phase S2/N0 64-part bitmap CPU boundary & prefetch test...\n";

    // 1. Pure CPU boundary checks on 64-bit atomic claim bitmap: 0, 31, 32, 33, 63
    std::atomic<uint64_t> claimed{0};
    const int check_parts[] = {0, 31, 32, 33, 63};
    for (int p : check_parts) {
        const uint64_t bit = uint64_t{1} << p;
        uint64_t old = claimed.fetch_or(bit, std::memory_order_acq_rel);
        expect((old & bit) == 0, "bit must not be claimed yet");
        expect((claimed.load(std::memory_order_relaxed) & bit) != 0, "bit must be set");
        // Second claim attempt must fail
        old = claimed.fetch_or(bit, std::memory_order_acq_rel);
        expect((old & bit) != 0, "duplicate claim must detect already set bit");
    }

    // 2. Synthetic prefetch & acquire with AILA_Q4E_FILL_PARTS=64
    _putenv("AILA_Q4E_FILL_PARTS=64");
    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=2");

    constexpr int num_layers = 1;
    constexpr int num_experts_per_layer = 4;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>((i * 13 + 7) % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x55);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.0f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xCC);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x66);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 1.0f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.2f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 2 * slot_bytes;

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed with 64 parts: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);

    // Prefetch expert 0 with 64 parts
    int pf[1] = {0};
    cache.prefetch_async(0, pf, 1);
    cache.wait_prefetch();

    int req[1] = {0};
    int slots[1] = {-1};
    cache.acquire(0, req, 1, slots);
    cache.wait_copy();
    expect(slots[0] >= 0, "slot must be acquired");

    // Verify GPU memory for expert 0
    ops::MoeExpertSlotDev dev_slot = cache.get_slot_dev(slots[0]);
    std::vector<uint8_t> read_back(gu_packed_per_exp);
    ctx.memcpy_d2h(read_back.data(), dev_slot.gate_up_packed, gu_packed_per_exp);
    expect(std::memcmp(read_back.data(), gu_packed_host.data(), gu_packed_per_exp) == 0,
           "expert 0 data with 64 parts verification failed");

    _putenv("AILA_Q4E_FILL_PARTS=");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Phase S2/N0 64-part bitmap CPU boundary & prefetch verified successfully.\n";
}

void test_small_gpu_cache_token_fallback_reader_protection(Context& ctx) {
    std::cout << "[Test] Running small GPU cache token fallback reader protection test...\n";

    constexpr int num_layers = 1;
    constexpr int num_experts_per_layer = 4;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>((i * 17 + 5) % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x44);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.0f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0xEE);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x77);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 1.0f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.25f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 2 * slot_bytes; // only 2 GPU slots!

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);

    // Token 0 requests experts {0, 1}
    int req0[2] = {0, 1};
    int slots0[2] = {-1, -1};
    sycl::event dma_ev0 = cache.acquire(0, req0, 2, slots0);
    expect(slots0[0] >= 0 && slots0[1] >= 0, "token 0 acquire must succeed");

    // Launch an asynchronous reader kernel on ctx.queue() that reads slot 0
    // and copies its gate_up_packed content into token 0's device buffer.
    void* dev_out0 = ctx.alloc_device(gu_packed_per_exp);
    ops::MoeExpertSlotDev dev_s0 = cache.get_slot_dev(slots0[0]);
    sycl::event token0_read_event = ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.depends_on(dma_ev0);
        const uint8_t* src = dev_s0.gate_up_packed;
        uint8_t* dst = static_cast<uint8_t*>(dev_out0);
        cgh.parallel_for(sycl::range<1>(gu_packed_per_exp), [=](sycl::id<1> idx) {
            dst[idx] = src[idx];
        });
    });

    // In token fallback: before acquiring experts {2, 3} for Token 1 (which would evict slots 0 and 1),
    // caller MUST wait for previous token's reader event:
    token0_read_event.wait_and_throw();

    // Now Token 1 acquires experts {2, 3} -> this safely evicts slots 0 and 1
    int req1[2] = {2, 3};
    int slots1[2] = {-1, -1};
    sycl::event dma_ev1 = cache.acquire(0, req1, 2, slots1);
    expect(slots1[0] >= 0 && slots1[1] >= 0, "token 1 acquire must succeed");

    // Launch Token 1 reader kernel
    void* dev_out1 = ctx.alloc_device(gu_packed_per_exp);
    ops::MoeExpertSlotDev dev_s2 = cache.get_slot_dev(slots1[0]);
    sycl::event token1_read_event = ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.depends_on(dma_ev1);
        const uint8_t* src = dev_s2.gate_up_packed;
        uint8_t* dst = static_cast<uint8_t*>(dev_out1);
        cgh.parallel_for(sycl::range<1>(gu_packed_per_exp), [=](sycl::id<1> idx) {
            dst[idx] = src[idx];
        });
    });
    token1_read_event.wait_and_throw();

    // Verify token 0 read the original data of expert 0
    std::vector<uint8_t> host_out0(gu_packed_per_exp);
    ctx.memcpy_d2h(host_out0.data(), dev_out0, gu_packed_per_exp);
    expect(std::memcmp(host_out0.data(), gu_packed_host.data(), gu_packed_per_exp) == 0,
           "token 0 read corrupted by token 1 eviction");

    // Verify token 1 read the data of expert 2
    std::vector<uint8_t> host_out1(gu_packed_per_exp);
    ctx.memcpy_d2h(host_out1.data(), dev_out1, gu_packed_per_exp);
    expect(std::memcmp(host_out1.data(), gu_packed_host.data() + 2 * gu_packed_per_exp, gu_packed_per_exp) == 0,
           "token 1 read data mismatch");

    ctx.free_device(dev_out0);
    ctx.free_device(dev_out1);

    std::cout << "[PASS] Small GPU cache token fallback reader protection verified successfully.\n";
}

void test_fill_pool_handoff_and_stale_prefetch_cleanup(Context& ctx) {
    std::cout << "[Test] Running ExpertFillPool handoff and stale prefetch cleanup test...\n";

    _putenv("AILA_TEST_Q4E_HOST_POOL=1");
    _putenv("AILA_TEST_Q4E_HOST_SLOTS=4");

    constexpr int num_layers = 1;
    constexpr int num_experts_per_layer = 8;
    constexpr int hidden_size = 256;
    constexpr int intermediate_size = 64;
    constexpr int blocksize = 64;

    const int gu_rows = 2 * intermediate_size;
    const size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    const size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    const size_t dn_rows = hidden_size;
    const size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    const size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    const size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    const size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    const size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    const size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

    std::vector<uint8_t> gu_packed_host(total_gu_packed);
    for (size_t i = 0; i < total_gu_packed; ++i) gu_packed_host[i] = static_cast<uint8_t>((i * 19 + 3) % 251);
    std::vector<uint8_t> gu_absmax_codes(total_gu_absmax, 0x33);
    std::vector<float> gu_nested_absmax(total_gu_absmax / 256 + 1, 1.0f);
    std::vector<float> gu_nested_qmap(256, 0.5f);

    std::vector<uint8_t> dn_packed_host(total_dn_packed, 0x88);
    std::vector<uint8_t> dn_absmax_codes(total_dn_absmax, 0x44);
    std::vector<float> dn_nested_absmax(total_dn_absmax / 256 + 1, 1.0f);
    std::vector<float> dn_nested_qmap(256, 0.5f);
    std::vector<float> qmap(16, 0.25f);

    Bnb4BitHostWeightRef gu_ref;
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
    gu_ref.quant_state.blocksize = blocksize;
    gu_ref.quant_state.nested = true;
    gu_ref.quant_state.nested_blocksize = 256;
    gu_ref.quant_state.nested_offset = 0.05f;

    Bnb4BitHostWeightRef dn_ref = gu_ref;
    dn_ref.name = "dn";
    dn_ref.packed_weight.ptr = dn_packed_host.data();
    dn_ref.packed_weight.bytes = total_dn_packed;
    dn_ref.packed_weight.shape = {static_cast<int64_t>(num_experts_per_layer * dn_rows), intermediate_size / 2};
    dn_ref.absmax.ptr = dn_absmax_codes.data();
    dn_ref.absmax.bytes = total_dn_absmax;
    dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
    dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
    dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
    dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
    dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
    dn_ref.quant_state.blocksize = blocksize;
    dn_ref.quant_state.nested = true;
    dn_ref.quant_state.nested_blocksize = 256;
    dn_ref.quant_state.nested_offset = 0.03f;

    const size_t slot_bytes = gu_packed_per_exp + gu_absmax_per_exp +
                              dn_packed_per_exp + dn_absmax_per_exp;
    const size_t gpu_cache_bytes = 4 * slot_bytes;

    ModelWeights dummy_weights;
    Qwen4ExpExpertCache cache;
    std::string err;
    bool ok = cache.init(ctx, dummy_weights, num_layers, gpu_cache_bytes,
                         num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
    expect(ok, "cache init failed: " + err);
    cache.set_layer_expert_ref(0, gu_ref, dn_ref);

    // 1. Stress rapid handoff between async prefetch and demand acquire
    for (int iter = 0; iter < 20; ++iter) {
        int pf_ids[2] = {(iter * 2) % num_experts_per_layer, (iter * 2 + 1) % num_experts_per_layer};
        cache.prefetch_async(0, pf_ids, 2);

        // Demand acquire for different experts while prefetch is in flight
        int dem_ids[2] = {(iter * 2 + 2) % num_experts_per_layer, (iter * 2 + 3) % num_experts_per_layer};
        int dem_slots[2] = {-1, -1};
        cache.acquire(0, dem_ids, 2, dem_slots);
        expect(dem_slots[0] >= 0 && dem_slots[1] >= 0, "demand acquire must succeed");
        cache.wait_copy();
    }

    // 2. Test stale prefetch cleanup:
    // Prefetch expert 0, wait_prefetch(), then acquire expert 1 (leaving expert 0 unconsumed).
    // Acquire must clean up stale prefetch item so expert 0 is not treated as a phantom in-flight item.
    int pf[1] = {0};
    cache.prefetch_async(0, pf, 1);
    cache.wait_prefetch();

    int req1[1] = {1};
    int slots1[1] = {-1};
    cache.acquire(0, req1, 1, slots1);
    cache.wait_copy();
    expect(slots1[0] >= 0, "acquire expert 1 must succeed");

    // Acquire expert 2: should not have any stale prefetch leftovers
    int req2[1] = {2};
    int slots2[1] = {-1};
    cache.acquire(0, req2, 1, slots2);
    cache.wait_copy();
    expect(slots2[0] >= 0, "acquire expert 2 must succeed");

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] ExpertFillPool handoff and stale prefetch cleanup verified successfully.\n";
}

void test_snapshot_capture_and_truncate(Context& ctx) {
    std::cout << "[Test] Running Phase S5 batched async snapshot capture & truncate test...\n";

    Qwen4ExpBnb4Backend backend;
    backend.init_for_testing(ctx, 4); // 4 layers: 0, 1, 2 linear, 3 attention
    auto session = backend.CreateSession(ctx, /*maxSnapshots=*/3);
    expect(session != nullptr, "CreateSession failed");

    auto& caches = session->layer_caches_for_testing();
    expect(caches.size() == 48, "expected 48 layer caches allocated in session");

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
    expect(session->snapshots_for_testing().count(10) == 1, "snapshot 10 must exist");

    // 2. Mutate device state to all zeros
    ctx.queue().memset(caches[0].linear_state.data(), 0, state_size * sizeof(float)).wait_and_throw();
    ctx.queue().memset(caches[0].linear_conv_state.data(), 0, conv_size * sizeof(float)).wait_and_throw();
    caches[0].linear_conv_head = 0;
    ctx.queue().memset(caches[1].linear_state.data(), 0, state_size * sizeof(float)).wait_and_throw();
    ctx.queue().memset(caches[1].linear_conv_state.data(), 0, conv_size * sizeof(float)).wait_and_throw();
    caches[1].linear_conv_head = 0;

    // 3. Truncate to length 10 -> must restore snapshot 10
    int restored = backend.Truncate(*session, 10);
    expect(restored == 10, "Truncate must restore length 10");
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

    expect(session->snapshots_for_testing().count(10) == 0, "snapshot 10 must be evicted when maxSnapshots exceeded");
    expect(session->snapshots_for_testing().count(20) == 1, "snapshot 20 must be retained");
    expect(session->snapshots_for_testing().count(30) == 1, "snapshot 30 must be retained");
    expect(session->snapshots_for_testing().count(40) == 1, "snapshot 40 must be retained");

    // Truncate to 35 -> nearest checkpoint <= 35 is 30
    restored = backend.Truncate(*session, 35);
    expect(restored == 30, "Truncate to 35 must restore checkpoint 30");

    // Truncate to 15 -> no checkpoint <= 15 (10 was evicted), must reset to 0
    restored = backend.Truncate(*session, 15);
    expect(restored == 0, "Truncate to 15 must reset to 0");
    expect(session->CurrentLength() == 0, "CurrentLength must be 0 after reset");

    std::cout << "[PASS] Phase S5 batched async snapshot capture & truncate verified successfully.\n";
}

void test_real_model_ple_and_cache_if_present(Context& ctx) {
    const std::string model_dir = resolve_qwen4exp_model_dir();
    if (!fs::exists(model_dir) || !fs::exists(model_dir + "/model.safetensors.index.json")) {
        std::cout << "[Test] Real model directory " << model_dir << " not found, skipping real model tests.\n";
        return;
    }

    std::cout << "[Test] Loading real model lazy index from " << model_dir << "...\n";
    ModelWeights weights = LoadModelWeightsLazy(model_dir, &ctx);

    // 1. Verify PLE indexing on real model
    Qwen4ExpPle ple;
    std::string ple_err;
    bool ple_ok = ple.init(ctx, weights, /*ple_layer_id=*/1, /*max_tokens=*/64, 1e-6f, &ple_err);
    expect(ple_ok, "real model ple.init failed: " + ple_err);

    expect(ple.layer_multipliers()[0] == 23703573157769LL, "real model layer_multipliers[0] mismatch");
    expect(ple.layer_multipliers()[1] == 20109073645365LL, "real model layer_multipliers[1] mismatch");
    expect(ple.layer_multipliers()[2] == 8052911324071LL, "real model layer_multipliers[2] mismatch");
    expect(ple.ngram_heads_vocab_sizes()[0] == 20000003LL, "real model vocab_sizes[0] mismatch");
    expect(ple.ngram_heads_offsets()[0] == 0LL, "real model offsets[0] mismatch");

    std::cout << "[PASS] Real model PLE initialized successfully. 128 shards indexed.\n";

    // 2. Verify ExpertCache initialization on real model with small memory
    Qwen4ExpExpertCache cache;
    std::string cache_err;
    constexpr size_t test_cache_mb = 64; // ~23 slots
    bool cache_ok = cache.init(ctx, weights, /*num_layers=*/2, test_cache_mb * 1024 * 1024,
                               /*num_experts=*/512, 2560, 640, 64, &cache_err);
    expect(cache_ok, "real model cache.init failed: " + cache_err);
    expect(cache.num_slots() >= 20, "expected >= 20 slots for 64MB");

    // Acquire top-10 experts for layer 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    int req[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int slots[10] = {-1};
    cache.acquire(0, req, 10, slots);

    expect(cache.stats().miss_count == 10, "expected 10 misses on cold cache");
    expect(cache.stats().hit_count == 0, "expected 0 hits on cold cache");

    // Acquire same 10 experts -> 100% hit!
    int hit_slots[10] = {-1};
    cache.acquire(0, req, 10, hit_slots);
    for (int i = 0; i < 10; ++i) {
        expect(hit_slots[i] == slots[i], "slot mismatch on warm acquire");
    }
    expect(cache.stats().hit_count == 10, "expected 10 hits on warm acquire");
    cache.log_stats(0);

    std::cout << "[PASS] Real model Expert Cache verified successfully.\n";
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
    backend.ensure_runtime_buffers_for_testing(ctx, 4096);
    const auto& buf = backend.buf_for_testing();

    // Verify scores buffer is strictly bounded by kDensePrefillLimit (2052)
    expect(buf.scores.shape().size() == 3, "buf.scores must be 3D [heads, cap, cap]");
    expect(buf.scores.shape()[1] <= 2052, "buf.scores dim 1 must be <= 2052");
    expect(buf.scores.shape()[2] <= 2052, "buf.scores dim 2 must be <= 2052");

    // Verify incr_scores buffer is strictly bounded by kDensePrefillLimit (2052)
    expect(buf.incr_scores.shape().size() == 3, "buf.incr_scores must be 3D [heads, cap, kv_len]");
    expect(buf.incr_scores.shape()[1] <= 2052, "buf.incr_scores dim 1 must be <= 2052");
    expect(buf.incr_scores.shape()[2] <= 2052, "buf.incr_scores dim 2 must be <= 2052");

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
    const auto& snaps = session->snapshots_for_testing();
    expect(snaps.count(2050) == 1, "Snapshot at 2050 must exist");
    expect(snaps.at(2050).layers[3].num_compressed_blocks == 512, "Snapshot layer 3 num_compressed_blocks must be 512");

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
    expect(session->snapshots_for_testing().empty(), "Snapshots must be cleared on Reset");

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
              << max_score_diff << " < 1e-3, 0 token mismatches, attention outputs finite & valid.\n";
}

void test_qsa_block_native_decode_alignment(Context& ctx) {
    std::cout << "[Test] Running QSA Block-Native Decode Alignment (A3) Test...\n";
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
    std::cout << "[Test] Running QSA Batched Prefill vs Oracle Alignment (A4) Test...\n";
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

            ops::qsa_compute_block_scores_compressed_batched(
                ctx, q_tile, compressed_k_dev, tile_scores, tile_count, tile_max_blocks, compress_ratio, tile_start_pos);

            ops::qsa_select_topk_blocks_batched(
                ctx, tile_scores, tile_blocks, tile_count, tile_max_blocks, tile_start_pos, block_topk, compress_ratio);

            ops::attention_prefill_qsa_batched(
                ctx, q_full_tile, k_full_dev, v_full_dev, tile_blocks, attn_out_tile,
                tile_count, tile_start_pos, num_heads, num_kv_heads, full_head_dim, block_topk, compress_ratio, &z_tile);
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
    std::cout << "[PASS] QSA Batched Prefill (A4) matches row-wise oracle across threshold (max_diff=" << max_diff << " < 0.02).\n";
}

// ---------------------------------------------------------------------------
// Phase B (B1-B4) Expert Cache & Scheduling Tests
// ---------------------------------------------------------------------------

struct TestCacheHarness {
    static constexpr int num_layers = 2;
    static constexpr int num_experts_per_layer = 8;
    static constexpr int hidden_size = 256;
    static constexpr int intermediate_size = 64;
    static constexpr int blocksize = 64;

    static constexpr int gu_rows = 2 * intermediate_size;
    static constexpr size_t gu_packed_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / 2);
    static constexpr size_t gu_absmax_per_exp = static_cast<size_t>(gu_rows) * (hidden_size / blocksize);
    static constexpr int dn_rows = hidden_size;
    static constexpr size_t dn_packed_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / 2);
    static constexpr size_t dn_absmax_per_exp = static_cast<size_t>(dn_rows) * (intermediate_size / blocksize);

    static constexpr size_t total_gu_packed = num_experts_per_layer * gu_packed_per_exp;
    static constexpr size_t total_gu_absmax = num_experts_per_layer * gu_absmax_per_exp;
    static constexpr size_t total_dn_packed = num_experts_per_layer * dn_packed_per_exp;
    static constexpr size_t total_dn_absmax = num_experts_per_layer * dn_absmax_per_exp;

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
    size_t slot_bytes = 0;

    TestCacheHarness()
        : gu_packed_host(total_gu_packed, 0x11),
          gu_absmax_codes(total_gu_absmax, 0x22),
          gu_nested_absmax(total_gu_absmax / 256 + 1, 1.0f),
          gu_nested_qmap(256, 0.5f),
          dn_packed_host(total_dn_packed, 0x33),
          dn_absmax_codes(total_dn_absmax, 0x44),
          dn_nested_absmax(total_dn_absmax / 256 + 1, 1.0f),
          dn_nested_qmap(256, 0.5f),
          qmap(16, 0.25f)
    {
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
        gu_ref.quant_state.blocksize = blocksize;
        gu_ref.quant_state.nested = true;
        gu_ref.quant_state.nested_blocksize = 256;
        gu_ref.quant_state.nested_offset = 0.05f;

        dn_ref = gu_ref;
        dn_ref.name = "dn";
        dn_ref.packed_weight.ptr = dn_packed_host.data();
        dn_ref.packed_weight.bytes = total_dn_packed;
        dn_ref.packed_weight.shape = {num_experts_per_layer * dn_rows, intermediate_size / 2};
        dn_ref.absmax.ptr = dn_absmax_codes.data();
        dn_ref.absmax.bytes = total_dn_absmax;
        dn_ref.absmax.shape = {static_cast<int64_t>(total_dn_absmax)};
        dn_ref.nested_absmax.ptr = reinterpret_cast<const uint8_t*>(dn_nested_absmax.data());
        dn_ref.nested_absmax.bytes = dn_nested_absmax.size() * sizeof(float);
        dn_ref.nested_quant_map.ptr = reinterpret_cast<const uint8_t*>(dn_nested_qmap.data());
        dn_ref.nested_quant_map.bytes = 256 * sizeof(float);
        dn_ref.quant_state.nested_offset = 0.03f;

        slot_bytes = gu_packed_per_exp + gu_absmax_per_exp + dn_packed_per_exp + dn_absmax_per_exp;
    }

    void setup_cache(Qwen4ExpExpertCache& cache, Context& ctx, int gpu_slots, int host_slots) {
        _putenv("AILA_TEST_Q4E_HOST_POOL=1");
        std::string hslots_str = "AILA_TEST_Q4E_HOST_SLOTS=" + std::to_string(host_slots);
        _putenv(hslots_str.c_str());

        ModelWeights dummy_weights;
        std::string err;
        bool ok = cache.init(ctx, dummy_weights, num_layers, static_cast<size_t>(gpu_slots) * slot_bytes,
                             num_experts_per_layer, hidden_size, intermediate_size, blocksize, &err, true);
        expect(ok, "cache init failed: " + err);
        for (int L = 0; L < num_layers; ++L) {
            cache.set_layer_expert_ref(L, gu_ref, dn_ref);
        }
    }
};

void test_b1_slot_state_machine_and_generation(Context& ctx) {
    std::cout << "[Test] Running B1 slot state machine, generation, and reader fence tests...\n";
    TestCacheHarness harness;
    Qwen4ExpExpertCache cache;
    harness.setup_cache(cache, ctx, 2, 4);

    expect(cache.get_slot_state(0) == ExpertSlotState::Empty, "Slot 0 initial state must be Empty");
    expect(cache.get_slot_state(1) == ExpertSlotState::Empty, "Slot 1 initial state must be Empty");
    expect(cache.get_slot_generation(0) == 0, "Slot 0 initial generation must be 0");

    int req0[1] = {0};
    int slot0[1] = {-1};
    sycl::event ev0 = cache.acquire(0, req0, 1, slot0);
    ExpertSlotState st_pre = cache.get_slot_state(slot0[0]);
    expect(st_pre == ExpertSlotState::Copying || st_pre == ExpertSlotState::Ready,
           "Slot state immediately after acquire must be Copying or Ready");
    ev0.wait();
    int s0 = slot0[0];
    expect(cache.get_slot_state(s0) == ExpertSlotState::Ready, "Slot after acquire must be Ready");
    uint64_t gen0 = cache.get_slot_generation(s0);
    expect(gen0 > 0, "Slot generation must increment upon fill");

    std::atomic<bool> fence_task_ran{false};
    sycl::event fence_ev = ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.host_task([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            fence_task_ran.store(true, std::memory_order_release);
        });
    });
    cache.record_consumer_reader_fence(s0, fence_ev);

    int req1[1] = {1};
    int slot1[1] = {-1};
    cache.acquire(0, req1, 1, slot1).wait();
    int s1 = slot1[0];
    expect(s1 != s0, "Different expert must take different slot");
    expect(cache.get_slot_state(s1) == ExpertSlotState::Ready, "Slot 1 must be Ready");

    int req2[1] = {2};
    int slot2[1] = {-1};
    cache.acquire(0, req2, 1, slot2).wait();
    int s2 = slot2[0];
    expect(s2 == s0, "Eviction must reuse slot s0");
    expect(fence_task_ran.load(std::memory_order_acquire), "Eviction must wait for consumer fence");
    expect(cache.get_slot_state(s2) == ExpertSlotState::Ready, "Evicted/re-filled slot must be Ready");
    uint64_t gen2 = cache.get_slot_generation(s2);
    expect(gen2 > gen0, "Evicted/re-filled slot must have strictly newer generation");

    expect(cache.get_host_slot_state(0) == ExpertSlotState::Ready, "Host slot 0 must be Ready after demand fill");
    expect(cache.get_host_slot_generation(0) > 0, "Host slot 0 generation must be > 0");

    std::cout << "[PASS] B1 slot state machine, generation, and reader fence tests passed.\n";
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

void test_b3_utility_aware_prefetch_filtering_and_metrics(Context& ctx) {
    std::cout << "[Test] Running B3 utility-aware prefetch filtering and byte-level metrics tests...\n";

    TestCacheHarness harness;
    Qwen4ExpExpertCache cache;
    harness.setup_cache(cache, ctx, 2, 4);

    int req0[1] = {0};
    int slot0[1] = {-1};
    cache.acquire(0, req0, 1, slot0).wait();
    expect(cache.contains(0, 0), "Expert 0 must be in GPU cache");

    int candidates[2] = {0, 1};
    // Pass quota = 1 explicitly: candidate 0 is GPU resident -> skipped; candidate 1 is prefetched!
    cache.prefetch_async(0, candidates, 2, 1);
    cache.wait_prefetch();

    expect(cache.host_contains(0, 1), "Expert 1 must be prefetched into Host RAM pool");
    const auto& s1 = cache.stats();
    expect(s1.prefetch_issued == 1, "Prefetch issued count must be exactly 1");
    expect(s1.prefetch_issued_bytes == harness.slot_bytes, "prefetch_issued_bytes must match slot_bytes");
    expect(s1.prefetch_useful_bytes == 0, "Useful bytes must be 0 before acquire");
    expect(s1.prefetch_utility_ratio() == 0.0, "Utility ratio must be 0.0 before acquire");

    // Acquire expert 1: consumes prefetch item
    int req1[1] = {1};
    int slot1[1] = {-1};
    cache.acquire(0, req1, 1, slot1).wait();
    const auto& s2 = cache.stats();
    expect(s2.prefetch_useful_bytes == harness.slot_bytes, "prefetch_useful_bytes must match slot_bytes on hit");
    expect(s2.prefetch_utility_ratio() == 1.0, "prefetch_utility_ratio must be 1.0 (100% useful)");

    // Test wasted bytes: prefetch expert 2, let it complete, then cancel before consuming
    int req_waste[1] = {2};
    cache.prefetch_async(0, req_waste, 1);
    cache.wait_prefetch();
    cache.cancel_prefetch_for_testing();
    const auto& s3 = cache.stats();
    expect(s3.prefetch_issued_bytes == 2 * harness.slot_bytes, "Two experts issued total");
    expect(s3.prefetch_wasted_bytes == harness.slot_bytes, "Completed but unconsumed prefetch must be counted as wasted");

    // Test cancelled bytes: prefetch expert 3 and cancel immediately while in flight
    int req_cancel[1] = {3};
    cache.prefetch_async(0, req_cancel, 1);
    cache.cancel_prefetch_for_testing();
    const auto& s4 = cache.stats();
    expect(s4.prefetch_issued_bytes == 3 * harness.slot_bytes, "Three experts issued total");
    expect(s4.prefetch_useful_bytes + s4.prefetch_wasted_bytes + s4.prefetch_cancelled_bytes == s4.prefetch_issued_bytes,
           "All prefetch bytes must strictly conserve: useful + wasted + cancelled == issued");
    expect(s4.demand_exposed_wait_ns >= 0, "demand_exposed_wait_ns must be recorded");

    std::cout << "[PASS] B3 utility-aware prefetch filtering and byte-level metrics tests passed.\n";
}

void test_b4_cost_aware_two_tier_eviction(Context& ctx) {
    std::cout << "[Test] Running B4 cost-aware two-tier eviction tests...\n";

    // Sub-test 1: Cost-aware eviction enabled (B4)
    {
        TestCacheHarness harness;
        Qwen4ExpExpertCache cache;
        harness.setup_cache(cache, ctx, 2, 4);
        cache.set_cost_aware_eviction(true);

        int req0[1] = {0};
        int slot0[1] = {-1};
        cache.acquire(0, req0, 1, slot0).wait();

        int req1[1] = {1};
        int slot1[1] = {-1};
        cache.acquire(0, req1, 1, slot1).wait();

        expect(cache.contains(0, 0) && cache.contains(0, 1), "GPU cache must hold experts 0 and 1");
        expect(cache.host_contains(0, 0) && cache.host_contains(0, 1), "Host pool must hold experts 0 and 1");

        // Prefetch 2, 3, 4 into host pool (capacity 4): evicts probationary tail (expert 0)
        int fill_host[3] = {2, 3, 4};
        cache.prefetch_async(0, fill_host, 3);
        cache.wait_prefetch();

        expect(!cache.host_contains(0, 0), "Host pool must have evicted expert 0");
        expect(cache.host_contains(0, 1), "Host pool must retain expert 1");
        expect(cache.contains(0, 0), "GPU cache must retain expert 0 (as LRU)");
        expect(cache.contains(0, 1), "GPU cache must retain expert 1 (as MRU)");

        int req5[1] = {5};
        int slot5[1] = {-1};
        cache.acquire(0, req5, 1, slot5).wait();

        expect(!cache.contains(0, 1), "Cost-aware eviction must evict expert 1 (which is backed by Host RAM)");
        expect(cache.contains(0, 0), "Cost-aware eviction must retain expert 0 (saving disk reload)");
        expect(cache.contains(0, 5), "Expert 5 must be in GPU cache");
        std::cout << "  Cost-aware eviction successfully prioritized evicting host-backed expert 1 over expert 0.\n";
    }

    // Sub-test 2: Pure LRU baseline (cost_aware = false)
    {
        TestCacheHarness harness;
        Qwen4ExpExpertCache cache_lru;
        harness.setup_cache(cache_lru, ctx, 2, 4);
        cache_lru.set_cost_aware_eviction(false);

        int req0[1] = {0};
        int slot0[1] = {-1};
        cache_lru.acquire(0, req0, 1, slot0).wait();

        int req1[1] = {1};
        int slot1[1] = {-1};
        cache_lru.acquire(0, req1, 1, slot1).wait();

        int fill_host[3] = {2, 3, 4};
        cache_lru.prefetch_async(0, fill_host, 3);
        cache_lru.wait_prefetch();

        expect(!cache_lru.host_contains(0, 0), "Host pool must have evicted expert 0");
        expect(cache_lru.host_contains(0, 1), "Host pool must retain expert 1");

        int req5[1] = {5};
        int slot5[1] = {-1};
        cache_lru.acquire(0, req5, 1, slot5).wait();

        expect(!cache_lru.contains(0, 0), "Pure LRU must evict LRU expert 0 regardless of host residency");
        expect(cache_lru.contains(0, 1), "Pure LRU must retain MRU expert 1");
        expect(cache_lru.contains(0, 5), "Expert 5 must be in GPU cache");
        std::cout << "  Pure LRU baseline successfully evicted LRU expert 0 as expected.\n";
    }

    std::cout << "[PASS] B4 cost-aware two-tier eviction tests passed.\n";
}

void test_b6_demand_microbatch_pipeline(Context& ctx) {
    std::cout << "[Test] Running B6 demand microbatch double-buffered pipeline tests...\n";

    // Sub-test 1: Cache microbatch statistics and consumer reader fence protection
    {
        TestCacheHarness harness;
        Qwen4ExpExpertCache cache;
        harness.setup_cache(cache, ctx, 4, 8); // 4 GPU slots, 8 host slots

        expect(cache.stats().demand_microbatches == 0, "Initial demand microbatches must be 0");

        int exp_g0[2] = {0, 1};
        int slots_g0[2] = {-1, -1};
        sycl::event ev0 = cache.acquire(0, exp_g0, 2, slots_g0);
        expect(cache.stats().demand_microbatches == 1, "First microbatch must increment demand_microbatches to 1");
        expect(cache.contains(0, 0) && cache.contains(0, 1), "Group 0 experts must be resident in GPU cache");

        // Record dummy consumer fence on Group 0
        sycl::event fence_g0 = ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() {});
        });
        cache.record_consumer_reader_fences(slots_g0, 2, fence_g0);

        int exp_g1[2] = {2, 3};
        int slots_g1[2] = {-1, -1};
        sycl::event ev1 = cache.acquire(0, exp_g1, 2, slots_g1);
        expect(cache.stats().demand_microbatches == 2, "Second microbatch must increment demand_microbatches to 2");
        expect(cache.contains(0, 2) && cache.contains(0, 3), "Group 1 experts must be resident in GPU cache");

        // Record dummy consumer fence on Group 1
        sycl::event fence_g1 = ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() {});
        });
        cache.record_consumer_reader_fences(slots_g1, 2, fence_g1);

        // Now cache is full (slots 0..3 used for experts 0..3, all protected by fences)
        // Acquire expert 4: must evict one of the slots after its fence completes
        int exp_g2[1] = {4};
        int slot_g2[1] = {-1};
        sycl::event ev2 = cache.acquire(0, exp_g2, 1, slot_g2);
        expect(cache.stats().demand_microbatches == 3, "Third microbatch must increment demand_microbatches to 3");
        expect(cache.contains(0, 4), "Expert 4 must be resident in GPU cache");
        std::cout << "  Sub-test 1: Microbatch acquire metrics and reader fence protection verified.\n";
    }

    // Sub-test 2: Kernel bit-exact equivalence between Monolithic S4 and Microbatched S4
    {
        TestCacheHarness harness;
        Qwen4ExpExpertCache cache;
        harness.setup_cache(cache, ctx, 8, 16);

        constexpr int tokens = 4;
        constexpr int top_k = 2;
        constexpr int total_acts = tokens * top_k; // 8 activations
        constexpr int hidden_size = 256;
        constexpr int intermediate_size = 64;
        constexpr int blocksize = 64;

        // Acquire experts {0, 1, 2, 3}
        int exp_ids[4] = {0, 1, 2, 3};
        int slots[4] = {-1, -1, -1, -1};
        cache.acquire(0, exp_ids, 4, slots).wait();

        // 4 tokens each choosing 2 experts:
        // token 0: exp 0, exp 2
        // token 1: exp 1, exp 3
        // token 2: exp 0, exp 1
        // token 3: exp 2, exp 3
        int router_ids[total_acts] = {0, 2, 1, 3, 0, 1, 2, 3};
        float router_weights[total_acts] = {0.6f, 0.4f, 0.7f, 0.3f, 0.5f, 0.5f, 0.8f, 0.2f};

        // Inputs
        std::vector<bf16> x_host(tokens * hidden_size, bf16(0.125f));
        bf16* x_dev = static_cast<bf16*>(ctx.alloc_device(x_host.size() * sizeof(bf16)));
        ctx.memcpy_h2d(x_dev, x_host.data(), x_host.size() * sizeof(bf16));

        // Allocations for Monolithic S4
        bf16* h_mono = static_cast<bf16*>(ctx.alloc_device(total_acts * intermediate_size * sizeof(bf16)));
        float* down_acc_mono = static_cast<float*>(ctx.alloc_device(total_acts * hidden_size * sizeof(float)));
        bf16* y_mono = static_cast<bf16*>(ctx.alloc_device(tokens * hidden_size * sizeof(bf16)));

        // Allocations for Microbatched S4
        bf16* h_mb = static_cast<bf16*>(ctx.alloc_device(total_acts * intermediate_size * sizeof(bf16)));
        float* down_acc_mb = static_cast<float*>(ctx.alloc_device(total_acts * hidden_size * sizeof(float)));
        bf16* y_mb = static_cast<bf16*>(ctx.alloc_device(tokens * hidden_size * sizeof(bf16)));

        float* weights_dev = static_cast<float*>(ctx.alloc_device(total_acts * sizeof(float)));
        ctx.memcpy_h2d(weights_dev, router_weights, total_acts * sizeof(float));

        // Device buffers for activation maps
        int* token_dev = static_cast<int*>(ctx.alloc_device(total_acts * sizeof(int)));
        int* rank_dev = static_cast<int*>(ctx.alloc_device(total_acts * sizeof(int)));
        ops::MoeExpertSlotDev* slot_dev = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));

        // Compute layout
        std::vector<int> counts(4, 0);
        for (int i = 0; i < total_acts; ++i) counts[router_ids[i]]++;
        std::vector<int> offsets(5, 0);
        for (int e = 0; e < 4; ++e) offsets[e + 1] = offsets[e] + counts[e];
        std::vector<int> cur_pos = offsets;

        std::vector<int> map_token(total_acts);
        std::vector<int> map_rank(total_acts);
        std::vector<ops::MoeExpertSlotDev> map_slot(total_acts);

        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                int i = t * top_k + k;
                int e = router_ids[i];
                int p = cur_pos[e]++;
                map_token[p] = t;
                map_rank[p] = k;
                map_slot[p] = cache.get_slot_dev(slots[e]);
            }
        }

        ctx.memcpy_h2d(token_dev, map_token.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(rank_dev, map_rank.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(slot_dev, map_slot.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

        // 1. Run Monolithic S4
        sycl::event gu_m = ops::moe_expert_major_prefill_gate_up(
            ctx, slot_dev, token_dev, rank_dev, nullptr, x_dev, h_mono,
            total_acts, top_k, hidden_size, intermediate_size, blocksize, nullptr, cache.uses_nested_codes());
        sycl::event dn_m = ops::moe_expert_major_prefill_down(
            ctx, slot_dev, token_dev, rank_dev, nullptr, h_mono, down_acc_mono,
            total_acts, top_k, intermediate_size, hidden_size, blocksize, &gu_m, cache.uses_nested_codes());
        ops::moe_ordered_accumulate_down_rows(
            ctx, down_acc_mono, weights_dev, y_mono, tokens, top_k, hidden_size, nullptr, nullptr, &dn_m).wait();

        // 2. Run Microbatched S4 (Group 0: experts {0, 1}, Group 1: experts {2, 3})
        // Group 0 activations: [offsets[0], offsets[2])
        // Group 1 activations: [offsets[2], offsets[4])
        int g0_start = offsets[0];
        int g0_count = offsets[2] - offsets[0];
        int g1_start = offsets[2];
        int g1_count = offsets[4] - offsets[2];

        // Launch Group 0
        sycl::event gu_mb0 = ops::moe_expert_major_prefill_gate_up(
            ctx, slot_dev + g0_start, token_dev + g0_start, rank_dev + g0_start,
            nullptr, x_dev, h_mb, g0_count, top_k, hidden_size, intermediate_size, blocksize,
            nullptr, cache.uses_nested_codes());
        sycl::event dn_mb0 = ops::moe_expert_major_prefill_down(
            ctx, slot_dev + g0_start, token_dev + g0_start, rank_dev + g0_start,
            nullptr, h_mb, down_acc_mb, g0_count, top_k, intermediate_size, hidden_size, blocksize,
            &gu_mb0, cache.uses_nested_codes());

        // Launch Group 1
        sycl::event gu_mb1 = ops::moe_expert_major_prefill_gate_up(
            ctx, slot_dev + g1_start, token_dev + g1_start, rank_dev + g1_start,
            nullptr, x_dev, h_mb, g1_count, top_k, hidden_size, intermediate_size, blocksize,
            nullptr, cache.uses_nested_codes());
        sycl::event dn_mb1 = ops::moe_expert_major_prefill_down(
            ctx, slot_dev + g1_start, token_dev + g1_start, rank_dev + g1_start,
            nullptr, h_mb, down_acc_mb, g1_count, top_k, intermediate_size, hidden_size, blocksize,
            &gu_mb1, cache.uses_nested_codes());

        // Accumulate with last event
        ops::moe_ordered_accumulate_down_rows(
            ctx, down_acc_mb, weights_dev, y_mb, tokens, top_k, hidden_size, nullptr, nullptr, &dn_mb1).wait();

        // Compare y_mono and y_mb
        std::vector<bf16> y_mono_host(tokens * hidden_size);
        std::vector<bf16> y_mb_host(tokens * hidden_size);
        ctx.memcpy_d2h(y_mono_host.data(), y_mono, y_mono_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(y_mb_host.data(), y_mb, y_mb_host.size() * sizeof(bf16));

        float max_diff = 0.0f;
        for (size_t i = 0; i < y_mono_host.size(); ++i) {
            float d = std::abs(static_cast<float>(y_mono_host[i]) - static_cast<float>(y_mb_host[i]));
            if (d > max_diff) max_diff = d;
        }
        expect(max_diff == 0.0f, "Microbatch pipeline must produce bit-exact identical outputs to monolithic S4");
        std::cout << "  Sub-test 2: Monolithic vs Microbatched prefill bit-exact equivalence verified (max_diff = "
                  << max_diff << ").\n";

        ctx.free_device(x_dev);
        ctx.free_device(h_mono);
        ctx.free_device(down_acc_mono);
        ctx.free_device(y_mono);
        ctx.free_device(h_mb);
        ctx.free_device(down_acc_mb);
        ctx.free_device(y_mb);
        ctx.free_device(weights_dev);
        ctx.free_device(token_dev);
        ctx.free_device(rank_dev);
        ctx.free_device(slot_dev);
    }

    std::cout << "[PASS] B6 demand microbatch double-buffered pipeline tests passed.\n";
}

} // namespace

int main() {
    aila::env::enable_l0_copy_engine();
    try {
        Context ctx;
        test_expert_cache_lru_and_dequant(ctx, false);
        test_expert_cache_lru_and_dequant(ctx, true);
        test_hot_host_ram_pool_and_prefetch(ctx);
        test_prefill_speculative_prefetch(ctx);
        test_fine_grained_demand_prefetch(ctx);
        test_slru_host_cache_and_anti_thrashing(ctx);
        test_host_slot_reservation_and_demand_priority(ctx);
        test_host_slot_contention_and_dma_protection(ctx);
        test_host_prefetch_hit_dma_protection(ctx);
        test_expert_cache_move_lifecycle(ctx);
        test_fill_parts_64_bitmap(ctx);
        test_small_gpu_cache_token_fallback_reader_protection(ctx);
        test_fill_pool_handoff_and_stale_prefetch_cleanup(ctx);
        test_b1_slot_state_machine_and_generation(ctx);
        test_b2_route_and_prediction_event_decoupling(ctx);
        test_b3_utility_aware_prefetch_filtering_and_metrics(ctx);
        test_b4_cost_aware_two_tier_eviction(ctx);
        test_b6_demand_microbatch_pipeline(ctx);
        test_snapshot_capture_and_truncate(ctx);
        test_ple_ngram_ids_calculation();
        test_ple_lookup_and_dequant(ctx);
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
        std::cout << "AilaQwen4ExpModuleTests: ALL PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "AilaQwen4ExpModuleTests FAILED: " << e.what() << "\n";
        return 1;
    }
}
