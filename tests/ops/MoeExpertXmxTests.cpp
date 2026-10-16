// Unit test for ops::moe_expert_xmx_prefill_chunk (R1 prefill expert XMX).
// Small synthetic shapes (H=256, I=64, bs=64) keep the run trivial; the checks
// cover the two-bank dequant, the padded-band gather/scatter contract, the
// nullable H2D ready gate, nested and flat NF4 scales, and B=1 / B=2 batches.

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertXmx.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int H = 256;          // hidden
constexpr int I = 64;           // intermediate
constexpr int Block = 64;       // NF4 quantization blocksize
constexpr int NestedBlock = 4;  // NF4 blocks covered by one nested absmax value
constexpr int RowsPerBand = 16;
constexpr int TopK = 4;
constexpr int Tokens = 8;
constexpr int Members = 3;      // experts 0/1 nested, expert 2 flat absmax
constexpr float kSentinel = -1234.5f;

constexpr float kNf4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f};

void expect(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("MoeExpertXmxTests: " + message);
}

struct HostProjection {
    int rows = 0;
    int cols = 0;
    bool nested = true;
    float offset = 0.0f;
    std::vector<uint8_t> packed;
    std::vector<uint8_t> codes;
    std::vector<float> nested_absmax;
    std::vector<float> nested_qmap;
    std::vector<float> flat_absmax;

    void init(int row_count, int col_count, uint32_t seed, bool use_nested) {
        rows = row_count;
        cols = col_count;
        nested = use_nested;
        std::mt19937 rng(seed);
        packed.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols) / 2);
        for (uint8_t& byte : packed) byte = static_cast<uint8_t>(rng());
        const size_t blocks = packed.size() / (Block / 2);
        if (nested) {
            codes.resize(blocks);
            for (uint8_t& code : codes) code = static_cast<uint8_t>(rng());
            nested_absmax.resize((blocks + NestedBlock - 1) / NestedBlock);
            for (size_t j = 0; j < nested_absmax.size(); ++j) {
                nested_absmax[j] = 0.02f + 0.001f * static_cast<float>(j % 5);
            }
            nested_qmap.resize(256);
            for (int j = 0; j < 256; ++j) {
                nested_qmap[static_cast<size_t>(j)] = static_cast<float>(j + 1) / 257.0f;
            }
            offset = 0.0005f;
        } else {
            flat_absmax.resize(blocks);
            for (size_t j = 0; j < blocks; ++j) {
                flat_absmax[j] = 0.02f + 0.0005f * static_cast<float>(j % 97);
            }
        }
    }

    // Decoded scale of one NF4 block, mirroring expert_scale() in the op.
    float scale(size_t block) const {
        if (nested) {
            const float product = nested_qmap[codes[block]] * nested_absmax[block / NestedBlock];
            return product + offset;
        }
        return flat_absmax[block];
    }

    // Logical weight [rows][cols] = [N][K], K contiguous nibbles with the high
    // nibble first, decoded the way Bnb4BitLinear::dequantize_weight does.
    float weight(int row, int col) const {
        const size_t element = static_cast<size_t>(row) * static_cast<size_t>(cols) +
                               static_cast<size_t>(col);
        const uint8_t byte = packed[element / 2];
        const int code = element % 2 == 0 ? (byte >> 4) : (byte & 15);
        return kNf4[code] * scale(element / Block);
    }
};

struct ExpertFixture {
    HostProjection gate_up;
    HostProjection down;
};

template <typename T>
Tensor upload(Context& ctx, const T* values, size_t count) {
    Tensor tensor = Tensor::allocate(ctx, {static_cast<int64_t>(count * sizeof(T))},
                                     dnnl::memory::data_type::u8);
    ctx.memcpy_h2d(tensor.data(), values, count * sizeof(T));
    return tensor;
}

template <typename T>
Tensor upload(Context& ctx, const std::vector<T>& values) {
    return upload(ctx, values.data(), values.size());
}

// Double-precision row reference: gather -> gate_up GEMM -> SwiGLU (bf16 output)
// -> down GEMM.  Rows are indexed b * RowsPerBand + m; padded rows are zero.
std::vector<float> reference_rows(const std::array<ExpertFixture, Members>& fixtures,
                                  int first_member, const std::vector<int>& counts,
                                  const std::vector<int>& tokens, const std::vector<bf16>& x) {
    const int members = static_cast<int>(counts.size());
    std::vector<float> rows(static_cast<size_t>(members) * RowsPerBand * H, 0.0f);
    std::vector<double> a1(static_cast<size_t>(RowsPerBand) * H);
    std::vector<double> c1(static_cast<size_t>(RowsPerBand) * 2 * I);
    std::vector<double> a2(static_cast<size_t>(RowsPerBand) * I);
    int act = 0;
    for (int b = 0; b < members; ++b) {
        const HostProjection& gate_up = fixtures[static_cast<size_t>(first_member + b)].gate_up;
        const HostProjection& down = fixtures[static_cast<size_t>(first_member + b)].down;
        const int count = counts[static_cast<size_t>(b)];
        for (int m = 0; m < RowsPerBand; ++m) {
            for (int h = 0; h < H; ++h) {
                a1[static_cast<size_t>(m) * H + h] =
                    m < count
                        ? static_cast<double>(static_cast<float>(
                              x[static_cast<size_t>(tokens[static_cast<size_t>(act + m)]) * H + h]))
                        : 0.0;
            }
            for (int n = 0; n < 2 * I; ++n) {
                double sum = 0.0;
                for (int k = 0; k < H; ++k) {
                    sum += a1[static_cast<size_t>(m) * H + k] *
                           static_cast<double>(gate_up.weight(n, k));
                }
                c1[static_cast<size_t>(m) * 2 * I + n] = sum;
            }
            for (int i = 0; i < I; ++i) {
                const float gate = static_cast<float>(c1[static_cast<size_t>(m) * 2 * I + i]);
                const float up = static_cast<float>(c1[static_cast<size_t>(m) * 2 * I + I + i]);
                const float silu = gate / (1.0f + std::exp(-gate));
                a2[static_cast<size_t>(m) * I + i] =
                    static_cast<double>(static_cast<float>(bf16(silu * up)));
            }
            for (int h = 0; h < H; ++h) {
                double sum = 0.0;
                for (int k = 0; k < I; ++k) {
                    sum += a2[static_cast<size_t>(m) * I + k] *
                           static_cast<double>(down.weight(h, k));
                }
                rows[(static_cast<size_t>(b) * RowsPerBand + m) * H + h] =
                    static_cast<float>(sum);
            }
        }
        act += count;
    }
    return rows;
}

// Every real destination row must be written exactly once, every other row must
// still hold the sentinel, and the written rows must match the reference.
void check_chunk(const std::vector<float>& down_acc, const std::vector<int>& tokens,
                 const std::vector<int>& ranks, const std::vector<int>& counts,
                 const std::vector<float>& reference, const std::string& name) {
    std::vector<bool> written(static_cast<size_t>(Tokens) * TopK, false);
    int act = 0;
    for (size_t b = 0; b < counts.size(); ++b) {
        for (int m = 0; m < counts[b]; ++m) {
            written[static_cast<size_t>(tokens[act + m]) * TopK + ranks[act + m]] = true;
        }
        act += counts[b];
    }
    for (size_t row = 0; row < written.size(); ++row) {
        for (int h = 0; h < H; ++h) {
            const float value = down_acc[row * H + h];
            if (written[row]) {
                expect(value != kSentinel,
                       name + ": down_acc row " + std::to_string(row) + " was not written");
            } else {
                expect(value == kSentinel,
                       name + ": padding clobbered down_acc row " + std::to_string(row));
            }
        }
    }
    double sq_err = 0.0;
    double sq_ref = 0.0;
    double max_abs = 0.0;
    act = 0;
    for (size_t b = 0; b < counts.size(); ++b) {
        for (int m = 0; m < counts[b]; ++m) {
            const size_t row = static_cast<size_t>(tokens[act + m]) * TopK + ranks[act + m];
            for (int h = 0; h < H; ++h) {
                const double actual = static_cast<double>(down_acc[row * H + h]);
                const double expected =
                    static_cast<double>(reference[(b * RowsPerBand + m) * H + h]);
                sq_err += (actual - expected) * (actual - expected);
                sq_ref += expected * expected;
                max_abs = std::max(max_abs, std::abs(actual - expected));
            }
        }
        act += counts[b];
    }
    const double nrmse = std::sqrt(sq_err / std::max(sq_ref, 1e-30));
    std::cout << name << " nrmse=" << nrmse << " max_abs=" << max_abs << "\n";
    expect(nrmse <= 0.01, name + ": nrmse above 1% against the fp64 reference");
}

void run_pipeline_test() {
    Context ctx;
    std::array<ExpertFixture, Members> fixtures;
    fixtures[0].gate_up.init(2 * I, H, 101u, true);
    fixtures[0].down.init(H, I, 102u, true);
    fixtures[1].gate_up.init(2 * I, H, 201u, true);
    fixtures[1].down.init(H, I, 202u, true);
    fixtures[2].gate_up.init(2 * I, H, 301u, false);
    fixtures[2].down.init(H, I, 302u, false);

    std::vector<float> qmap(256);
    for (int j = 0; j < 256; ++j) {
        qmap[static_cast<size_t>(j)] = static_cast<float>(j + 1) / 257.0f;
    }
    Tensor qmap_dev = upload(ctx, qmap);

    struct ExpertDevices {
        Tensor gu_packed;
        Tensor gu_codes;
        Tensor gu_absmax;
        Tensor dn_packed;
        Tensor dn_codes;
        Tensor dn_absmax;
    };
    std::array<ExpertDevices, Members> dev;
    std::vector<ops::MoeExpertSlotDev> slots_host(Members);
    for (int e = 0; e < Members; ++e) {
        const size_t index = static_cast<size_t>(e);
        const HostProjection& gate_up = fixtures[index].gate_up;
        const HostProjection& down = fixtures[index].down;
        dev[index].gu_packed = upload(ctx, gate_up.packed);
        dev[index].dn_packed = upload(ctx, down.packed);
        ops::MoeExpertSlotDev& slot = slots_host[index];
        slot = ops::MoeExpertSlotDev{};
        slot.gate_up_packed = dev[index].gu_packed.data_as<uint8_t>();
        slot.down_packed = dev[index].dn_packed.data_as<uint8_t>();
        if (gate_up.nested) {
            dev[index].gu_codes = upload(ctx, gate_up.codes);
            dev[index].gu_absmax = upload(ctx, gate_up.nested_absmax);
            dev[index].dn_codes = upload(ctx, down.codes);
            dev[index].dn_absmax = upload(ctx, down.nested_absmax);
            slot.gate_up_nested = {dev[index].gu_codes.data_as<uint8_t>(),
                                   dev[index].gu_absmax.data_as<float>(),
                                   qmap_dev.data_as<float>(), gate_up.offset, NestedBlock};
            slot.down_nested = {dev[index].dn_codes.data_as<uint8_t>(),
                                dev[index].dn_absmax.data_as<float>(),
                                qmap_dev.data_as<float>(), down.offset, NestedBlock};
        } else {
            dev[index].gu_absmax = upload(ctx, gate_up.flat_absmax);
            dev[index].dn_absmax = upload(ctx, down.flat_absmax);
            slot.gate_up_absmax = dev[index].gu_absmax.data_as<float>();
            slot.down_absmax = dev[index].dn_absmax.data_as<float>();
        }
    }
    Tensor slots_dev = upload(ctx, slots_host);

    // Case A/B plan: member 0 = 16 rows, member 1 = 5 rows (11 padded rows).
    std::vector<int> bases_a{0, 16, 21};
    std::vector<int> tokens_a;
    std::vector<int> ranks_a;
    for (int t = 0; t < Tokens; ++t) {
        tokens_a.push_back(t);
        ranks_a.push_back(0);
        tokens_a.push_back(t);
        ranks_a.push_back(1);
    }
    const int extra_tokens[5] = {5, 6, 7, 3, 4};
    const int extra_ranks[5] = {2, 2, 2, 3, 3};
    for (int j = 0; j < 5; ++j) {
        tokens_a.push_back(extra_tokens[j]);
        ranks_a.push_back(extra_ranks[j]);
    }

    // Case C plan: expert 2 only, 7 real rows, flat absmax.
    std::vector<int> bases_c{0, 7};
    std::vector<int> tokens_c{0, 1, 2, 3, 4, 5, 6};
    std::vector<int> ranks_c{0, 1, 2, 3, 0, 1, 2};

    std::vector<bf16> x_host(static_cast<size_t>(Tokens) * H);
    for (size_t j = 0; j < x_host.size(); ++j) {
        x_host[j] = bf16(static_cast<float>(static_cast<int>(j * 37 % 512) - 256) / 512.0f);
    }
    Tensor x_dev = upload(ctx, x_host);

    Tensor bases_a_dev = upload(ctx, bases_a);
    Tensor bases_c_dev = upload(ctx, bases_c);
    Tensor tokens_a_dev = upload(ctx, tokens_a);
    Tensor tokens_c_dev = upload(ctx, tokens_c);
    Tensor ranks_a_dev = upload(ctx, ranks_a);
    Tensor ranks_c_dev = upload(ctx, ranks_c);

    const size_t acc_elements = static_cast<size_t>(Tokens) * TopK * H;
    std::vector<float> acc_host(acc_elements, kSentinel);
    Tensor acc_a_dev =
        Tensor::allocate(ctx, {static_cast<int64_t>(acc_elements)}, dnnl::memory::data_type::f32);
    Tensor acc_b_dev =
        Tensor::allocate(ctx, {static_cast<int64_t>(acc_elements)}, dnnl::memory::data_type::f32);
    Tensor acc_c_dev =
        Tensor::allocate(ctx, {static_cast<int64_t>(acc_elements)}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(acc_a_dev.data(), acc_host.data(), acc_elements * sizeof(float));
    ctx.memcpy_h2d(acc_b_dev.data(), acc_host.data(), acc_elements * sizeof(float));
    ctx.memcpy_h2d(acc_c_dev.data(), acc_host.data(), acc_elements * sizeof(float));

    ops::MoeExpertXmxScratch scratch;
    expect(ops::moe_expert_xmx_scratch_create(ctx, scratch, 32, H, I),
           "moe_expert_xmx_scratch_create failed");
    expect(scratch.max_chunk_rows == 32, "scratch row capacity is not 32");

    ops::MoeExpertXmxChunk chunk_a;
    chunk_a.num_experts = 2;
    chunk_a.top_k = TopK;
    chunk_a.rows_per_band = RowsPerBand;
    chunk_a.member_row_base = bases_a_dev.data_as<int>();
    chunk_a.member_token_ids = tokens_a_dev.data_as<int>();
    chunk_a.member_ranks = ranks_a_dev.data_as<int>();
    chunk_a.slots = reinterpret_cast<const ops::MoeExpertSlotDev*>(slots_dev.data());

    // The H2D gate runs on its own in-order queue, like the production staging
    // pump; the dequant must not observe the slots before it completes.
    sycl::queue copy_queue(ctx.queue().get_context(), ctx.queue().get_device(),
                           sycl::property_list{sycl::property::queue::in_order()});
    sycl::event ready = copy_queue.memcpy(
        slots_dev.data(), slots_host.data(), slots_host.size() * sizeof(ops::MoeExpertSlotDev));
    ops::moe_expert_xmx_prefill_chunk(ctx, scratch, chunk_a, 0, x_dev.data_as<bf16>(),
                                      acc_a_dev.data_as<float>(), H, I, Block, &ready);
    // Same chunk on bank 1 without a ready event: identical bytes out.
    ops::moe_expert_xmx_prefill_chunk(ctx, scratch, chunk_a, 1, x_dev.data_as<bf16>(),
                                      acc_b_dev.data_as<float>(), H, I, Block, nullptr);

    // Case C: B=1 chunk that reuses expert 2's descriptor at slots[2].
    ops::MoeExpertXmxChunk chunk_c;
    chunk_c.num_experts = 1;
    chunk_c.top_k = TopK;
    chunk_c.rows_per_band = RowsPerBand;
    chunk_c.member_row_base = bases_c_dev.data_as<int>();
    chunk_c.member_token_ids = tokens_c_dev.data_as<int>();
    chunk_c.member_ranks = ranks_c_dev.data_as<int>();
    chunk_c.slots = reinterpret_cast<const ops::MoeExpertSlotDev*>(slots_dev.data()) + 2;
    ops::moe_expert_xmx_prefill_chunk(ctx, scratch, chunk_c, 1, x_dev.data_as<bf16>(),
                                      acc_c_dev.data_as<float>(), H, I, Block, nullptr);

    ctx.synchronize();
    copy_queue.wait_and_throw();

    std::vector<float> acc_a(acc_elements);
    std::vector<float> acc_b(acc_elements);
    std::vector<float> acc_c(acc_elements);
    ctx.memcpy_d2h(acc_a.data(), acc_a_dev.data(), acc_elements * sizeof(float));
    ctx.memcpy_d2h(acc_b.data(), acc_b_dev.data(), acc_elements * sizeof(float));
    ctx.memcpy_d2h(acc_c.data(), acc_c_dev.data(), acc_elements * sizeof(float));

    const std::vector<int> counts_a{16, 5};
    const std::vector<int> counts_c{7};
    check_chunk(acc_a, tokens_a, ranks_a, counts_a,
                reference_rows(fixtures, 0, counts_a, tokens_a, x_host), "caseA(bank0,ready)");
    check_chunk(acc_b, tokens_a, ranks_a, counts_a,
                reference_rows(fixtures, 0, counts_a, tokens_a, x_host), "caseB(bank1)");
    check_chunk(acc_c, tokens_c, ranks_c, counts_c,
                reference_rows(fixtures, 2, counts_c, tokens_c, x_host), "caseC(B=1,flat)");
    expect(acc_a == acc_b, "bank 0 and bank 1 results differ");

    ops::moe_expert_xmx_scratch_destroy(ctx, scratch);
    expect(scratch.max_chunk_rows == 0 && scratch.a1 == nullptr && scratch.impl == nullptr,
           "scratch_destroy left state behind");
}

}  // namespace

int main() {
    try {
        run_pipeline_test();
    } catch (const std::exception& error) {
        std::cerr << "AilaMoeExpertXmxTests: failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "AilaMoeExpertXmxTests: passed\n";
    return 0;
}
