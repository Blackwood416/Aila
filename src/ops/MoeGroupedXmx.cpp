#include "MoeExpertOps.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <stdexcept>

namespace ops {
namespace {
using namespace sycl::ext::oneapi::experimental::matrix;
constexpr float nf4[16] = {
    -1.f, -.6961928009986877f, -.5250730514526367f, -.39491748809814453f,
    -.28444138169288635f, -.18477343022823334f, -.09105003625154495f, 0.f,
    .07958029955625534f, .16093020141124725f, .24611230194568634f, .33791524171829224f,
    .4407079219818115f, .5626170039176941f, .7229568362236023f, 1.f};

template<bool Nested>
inline float scale(const float* absmax, const MoeNestedScaleDev& nested, size_t i) {
    if constexpr (Nested) {
        const float product = sycl::fma(nested.nested_quant_map[nested.codes[i]],
            nested.nested_absmax[i / nested.blocksize], 0.f);
        return sycl::fma(product, 1.f, nested.offset);
    } else {
        return absmax[i];
    }
}

// [HYPOTHESIS] M8/N32/K32, SG8, four subgroups. Each subgroup computes
// an 8x8 fragment; gate/up share A and retain separate FP32 accumulators.
// Packed weights are decoded only into a small work-group-local B tile.
template<bool GateUp, bool Nested>
sycl::event launch_grouped_xmx(
    Context& ctx, const MoeExpertSlotDev* slots, const MoeExpertTile* tiles,
    int num_tiles, const int* tokens, const int* ranks, const float* quant_map,
    const bf16* input, bf16* h, float* down, int top_k, int hidden, int intermediate,
    const sycl::event* ready) {
    constexpr int BM = 8, BN = 32, BK = 32, WG = 32;
    const int K = GateUp ? hidden : intermediate;
    const int N = GateUp ? intermediate : hidden;
    if (num_tiles <= 0 || K <= 0 || K % 64 || N <= 0 || top_k <= 0)
        throw std::invalid_argument("grouped XMX requires tiles, positive dimensions and K%64=0");
    const int grid_n = (N + BN - 1) / BN;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (ready) cgh.depends_on(*ready);
        sycl::local_accessor<bf16, 1> a(BM * BK, cgh);
        sycl::local_accessor<bf16, 1> b(BN * BK, cgh);
        sycl::local_accessor<bf16, 1> u((GateUp ? BN * BK : 1), cgh);
        sycl::local_accessor<float, 1> c(BM * BN, cgh);
        sycl::local_accessor<float, 1> cu((GateUp ? BM * BN : 1), cgh);
        sycl::local_accessor<float, 1> lut(16, cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(num_tiles) * grid_n * WG, WG),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
                const int lid = int(item.get_local_id(0));
                const int sg_id = lid / 8;
                auto sg = item.get_sub_group();
                const int group = int(item.get_group(0));
                const auto tile = tiles[group / grid_n];
                const auto slot = slots[tile.first];
                const int n0 = (group % grid_n) * BN;
                const auto packed = GateUp ? slot.gate_up_packed : slot.down_packed;
                const auto absmax = GateUp ? slot.gate_up_absmax : slot.down_absmax;
                const auto nested = GateUp ? slot.gate_up_nested : slot.down_nested;
                if (lid < 16) lut[lid] = quant_map ? quant_map[lid] : nf4[lid];
                joint_matrix<sycl::sub_group, float, use::accumulator, 8, 8> acc, up;
                joint_matrix_fill(sg, acc, 0.f);
                if constexpr (GateUp) joint_matrix_fill(sg, up, 0.f);
                item.barrier(sycl::access::fence_space::local_space);
                for (int kb = 0; kb < K; kb += BK) {
                    for (int i = lid; i < BM * BK; i += WG) {
                        const int r = i / BK, k = kb + i % BK;
                        bf16 v = bf16(0.f);
                        if (r < tile.rows) {
                            const int idx = tile.first + r;
                            const size_t row = GateUp ? size_t(tokens[idx]) :
                                size_t(tokens[idx]) * top_k + ranks[idx];
                            v = input[row * K + k];
                        }
                        a[i] = v;
                    }
                    // One uint32 load supplies eight adjacent K weights.
                    for (int i = lid; i < BN * BK / 8; i += WG) {
                        const int col = i / (BK / 8), k = kb + (i % (BK / 8)) * 8;
                        const int row = n0 + col;
                        uint32_t p = 0, pu = 0;
                        float s = 0.f, su = 0.f;
                        if (row < N) {
                            p = *reinterpret_cast<const uint32_t*>(packed + size_t(row) * (K / 2) + k / 2);
                            s = scale<Nested>(absmax, nested, size_t(row) * (K / 64) + k / 64);
                            if constexpr (GateUp) {
                                pu = *reinterpret_cast<const uint32_t*>(packed + size_t(row + N) * (K / 2) + k / 2);
                                su = scale<Nested>(absmax, nested, size_t(row + N) * (K / 64) + k / 64);
                            }
                        }
                        for (int v = 0; v < 8; ++v) {
                            const int shift = (v / 2) * 8 + (v % 2 == 0 ? 4 : 0);
                            b[i * 8 + v] = bf16(lut[(p >> shift) & 15] * s);
                            if constexpr (GateUp) u[i * 8 + v] = bf16(lut[(pu >> shift) & 15] * su);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int k = 0; k < BK; k += 16) {
                        joint_matrix<sycl::sub_group, bf16, use::a, 8, 16, layout::row_major> ja;
                        joint_matrix<sycl::sub_group, bf16, use::b, 16, 8, layout::col_major> jb;
                        joint_matrix_load(sg, ja, a.get_multi_ptr<sycl::access::decorated::no>() + k, BK);
                        joint_matrix_load(sg, jb, b.get_multi_ptr<sycl::access::decorated::no>() + sg_id * 8 * BK + k, BK);
                        joint_matrix_mad(sg, acc, ja, jb, acc);
                        if constexpr (GateUp) {
                            joint_matrix_load(sg, jb, u.get_multi_ptr<sycl::access::decorated::no>() + sg_id * 8 * BK + k, BK);
                            joint_matrix_mad(sg, up, ja, jb, up);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                joint_matrix_store(sg, acc, c.get_multi_ptr<sycl::access::decorated::no>() + sg_id * 64, 8, layout::row_major);
                if constexpr (GateUp)
                    joint_matrix_store(sg, up, cu.get_multi_ptr<sycl::access::decorated::no>() + sg_id * 64, 8, layout::row_major);
                item.barrier(sycl::access::fence_space::local_space);
                for (int i = lid; i < BM * BN; i += WG) {
                    const int r = (i % 64) / 8;
                    const int n = n0 + (i / 64) * 8 + i % 8;
                    if (r < tile.rows && n < N) {
                        const int idx = tile.first + r;
                        const size_t out = (size_t(tokens[idx]) * top_k + ranks[idx]) * N + n;
                        if constexpr (GateUp) h[out] = bf16((c[i] / (1.f + sycl::exp(-c[i]))) * cu[i]);
                        else down[out] = c[i];
                    }
                }
            });
    });
}
} // namespace

sycl::event moe_grouped_xmx_gate_up(
    Context& ctx, const MoeExpertSlotDev* slots, const MoeExpertTile* tiles,
    int num_tiles, const int* tokens, const int* ranks, const float* quant_map,
    const bf16* x, bf16* h, int top_k, int hidden, int intermediate,
    bool use_nested, const sycl::event* ready) {
    if (use_nested) return launch_grouped_xmx<true, true>(ctx, slots, tiles, num_tiles,
        tokens, ranks, quant_map, x, h, nullptr, top_k, hidden, intermediate, ready);
    return launch_grouped_xmx<true, false>(ctx, slots, tiles, num_tiles,
        tokens, ranks, quant_map, x, h, nullptr, top_k, hidden, intermediate, ready);
}

sycl::event moe_grouped_xmx_down(
    Context& ctx, const MoeExpertSlotDev* slots, const MoeExpertTile* tiles,
    int num_tiles, const int* tokens, const int* ranks, const float* quant_map,
    const bf16* h, float* down, int top_k, int hidden, int intermediate,
    bool use_nested, const sycl::event* ready) {
    if (use_nested) return launch_grouped_xmx<false, true>(ctx, slots, tiles, num_tiles,
        tokens, ranks, quant_map, h, nullptr, down, top_k, hidden, intermediate, ready);
    return launch_grouped_xmx<false, false>(ctx, slots, tiles, num_tiles,
        tokens, ranks, quant_map, h, nullptr, down, top_k, hidden, intermediate, ready);
}
} // namespace ops
