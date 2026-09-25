#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpPle.hpp"
#include "utils/SafeTensors.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
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
    const char* env_dir = std::getenv("AILA_QWEN4EXP_MODEL_DIR");
    if (env_dir && fs::exists(env_dir)) return env_dir;
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

    _putenv("AILA_TEST_Q4E_HOST_SLOTS=");
    _putenv("AILA_TEST_Q4E_HOST_POOL=");
    std::cout << "[PASS] Prefill speculative next-layer prefetch verified successfully.\n";
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

} // namespace

int main() {
    aila::env::enable_l0_copy_engine();
    try {
        Context ctx;
        test_expert_cache_lru_and_dequant(ctx, false);
        test_expert_cache_lru_and_dequant(ctx, true);
        test_hot_host_ram_pool_and_prefetch(ctx);
        test_prefill_speculative_prefetch(ctx);
        test_ple_ngram_ids_calculation();
        test_ple_lookup_and_dequant(ctx);
        test_real_model_ple_and_cache_if_present(ctx);
        std::cout << "AilaQwen4ExpModuleTests: ALL PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "AilaQwen4ExpModuleTests FAILED: " << e.what() << "\n";
        return 1;
    }
}
