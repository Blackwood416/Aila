#include "MoeExpertXmx.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "utils/EnvUtils.hpp"

namespace ops {
namespace {

using dt = dnnl::memory::data_type;

constexpr int kMaxExpertsPerChunk = 16;
constexpr int kPlanBaseSlots = kMaxExpertsPerChunk + 1;
// Distinct (B, M) pairs held alive at once.  A layer needs 10-25 and a whole
// 14K prefill visits a few hundred across 48 layers, so the cap must be large
// enough that the eviction path - which drains the queue - never fires inside
// a run.  512 prims is tens of MB of JIT/handles, well inside the reserve.
constexpr size_t kMaxCachedShapes = 512;

// Verbatim copy of MoeGroupedXmx.cpp:13-22: nested NF4 scale decode.
template <bool Nested>
inline float expert_scale(const float* absmax, const MoeNestedScaleDev& nested, size_t i) {
    if constexpr (Nested) {
        const float product = sycl::fma(nested.nested_quant_map[nested.codes[i]],
            nested.nested_absmax[i / nested.blocksize], 0.f);
        return sycl::fma(product, 1.f, nested.offset);
    } else {
        return absmax[i];
    }
}

// The slot descriptor carries both the nested quantization and the flat absmax
// view; the device-side branch keeps every work-item arithmetic identical to
// expert_scale above.
inline float slot_scale(const MoeNestedScaleDev& nested, const float* absmax, size_t i) {
    return nested.codes != nullptr ? expert_scale<true>(absmax, nested, i)
                                   : expert_scale<false>(absmax, nested, i);
}

// Batched oneDNN matmul cache, keyed by (B, M) like Bnb4BitLinear.cpp:2996-3025
// / Bnb4GroupedMoE.  The USM memory handles are built once per entry and bank;
// they must outlive every queued execution (Bnb4BitLinear.hpp:351-355), so the
// entry owns them and is only dropped after a queue drain.
struct XmxPrim {
    dnnl::matmul mm1;
    dnnl::matmul mm2;
    dnnl::memory::desc s1, w1, d1, s2, w2, d2;
    std::unordered_map<int, dnnl::memory> args1[2];
    std::unordered_map<int, dnnl::memory> args2[2];
};

struct XmxCache {
    std::map<uint64_t, XmxPrim> prims;
};

XmxCache& cache_of(MoeExpertXmxScratch& s) {
    return *static_cast<XmxCache*>(s.impl);
}

dnnl::memory usm_memory(Context& ctx, const dnnl::memory::desc& md, void* ptr) {
    return dnnl::sycl_interop::make_memory(md, ctx.engine(),
                                           dnnl::sycl_interop::memory_kind::usm, ptr);
}

// NF4 -> bf16 dequant of a whole chunk: 1 work-item = 1 uint32 (8 codes) plus a
// vec8 bf16 store, wgu written [B][2I][H] and wdn [B][H][I] in the packed [N][K]
// linear order (no transpose; the GEMM takes the strided view).
sycl::event submit_dequant(Context& ctx, const MoeExpertXmxScratch& s,
                           const MoeExpertXmxChunk& chunk, int bank, int hidden,
                           int intermediate, int blocksize, const sycl::event* ready) {
    const MoeExpertSlotDev* slots = chunk.slots;
    bf16* wgu = static_cast<bf16*>(s.wgu_bf16[bank]);
    bf16* wdn = static_cast<bf16*>(s.wdn_bf16[bank]);
    const size_t gu_elems = size_t(2) * size_t(intermediate) * size_t(hidden);
    const size_t dn_elems = size_t(hidden) * size_t(intermediate);
    const size_t gu_words = gu_elems / 8;
    const size_t dn_words = dn_elems / 8;
    const size_t per_expert = gu_words + dn_words;
    const size_t block_size = size_t(blocksize);
    const int experts = chunk.num_experts;
    const bool have_ready = ready != nullptr && *ready != sycl::event{};
    const sycl::event ready_event = have_ready ? *ready : sycl::event{};
    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (have_ready) cgh.depends_on(ready_event);
        cgh.parallel_for(sycl::range<1>(size_t(experts) * per_expert), [=](sycl::id<1> idx) {
            constexpr float qm[16] = {
                -1.f, -.6961928009986877f, -.5250730514526367f, -.39491748809814453f,
                -.28444138169288635f, -.18477343022823334f, -.09105003625154495f, 0.f,
                .07958029955625534f, .16093020141124725f, .24611230194568634f, .33791524171829224f,
                .4407079219818115f, .5626170039176941f, .7229568362236023f, 1.f};
            const size_t i = idx[0];
            const int expert = static_cast<int>(i / per_expert);
            size_t word = i - size_t(expert) * per_expert;
            const bool gate_up = word < gu_words;
            if (!gate_up) word -= gu_words;
            const MoeExpertSlotDev& slot = slots[expert];
            const MoeNestedScaleDev& nested = gate_up ? slot.gate_up_nested : slot.down_nested;
            const float* absmax = gate_up ? slot.gate_up_absmax : slot.down_absmax;
            const uint8_t* packed = (gate_up ? slot.gate_up_packed : slot.down_packed) + word * 4;
            const uint32_t codes = *reinterpret_cast<const uint32_t*>(packed);
            const float scale = slot_scale(nested, absmax, (word * 8) / block_size);
            sycl::vec<bf16, 8> decoded;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t byte = (codes >> (8 * b)) & 0xFFu;
                decoded[2 * b] = bf16(qm[byte >> 4] * scale);
                decoded[2 * b + 1] = bf16(qm[byte & 15u] * scale);
            }
            bf16* out = gate_up ? wgu + size_t(expert) * gu_elems
                                : wdn + size_t(expert) * dn_elems;
            *reinterpret_cast<sycl::vec<bf16, 8>*>(out + word * 8) = decoded;
        });
    });
}

// a1[b][m][h] = x[token(member b, row m)][h]; padded rows are zero filled.
sycl::event submit_gather(Context& ctx, const MoeExpertXmxScratch& s,
                          const MoeExpertXmxChunk& chunk, const bf16* x, int hidden,
                          const sycl::event& dep) {
    const int* bases = chunk.member_row_base;
    const int* tokens = chunk.member_token_ids;
    const int experts = chunk.num_experts;
    const int rows_per_band = chunk.rows_per_band;
    bf16* a1 = static_cast<bf16*>(s.a1);
    const size_t vecs_per_row = size_t(hidden) / 8;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.depends_on(dep);
        cgh.parallel_for(
            sycl::range<1>(size_t(experts) * size_t(rows_per_band) * vecs_per_row),
            [=](sycl::id<1> idx) {
                const size_t row = idx[0] / vecs_per_row;
                const size_t k8 = idx[0] - row * vecs_per_row;
                const int expert = static_cast<int>(row / size_t(rows_per_band));
                const int m = static_cast<int>(row - size_t(expert) * size_t(rows_per_band));
                const int begin = bases[expert];
                using v8 = sycl::vec<bf16, 8>;
                v8 value = v8(bf16(0.f));
                if (m < bases[expert + 1] - begin) {
                    const int token = tokens[begin + m];
                    value = *reinterpret_cast<const v8*>(x + size_t(token) * size_t(hidden) +
                                                         k8 * 8);
                }
                *reinterpret_cast<v8*>(a1 + row * size_t(hidden) + k8 * 8) = value;
            });
    });
}

// f32 counterpart of moe_silu_gate_up_mul_flat (MoeExpertOps.cpp:5303): same
// math, same launch shape and the same gate=[0, I) / up=[I, 2I) column split,
// but the gate_up tile is the f32 c1 buffer, so the SwiGLU input keeps the XMX
// accumulator's precision.  a2 stays bf16 (the down GEMM's A input does).
sycl::event submit_swiglu_f32(Context& ctx, const float* gate_up, bf16* out,
                              int total_rows, int intermediate) {
    const size_t total_elements =
        static_cast<size_t>(total_rows) * static_cast<size_t>(intermediate);
    constexpr size_t wg = 256;
    const size_t num_wgs = (total_elements + wg - 1) / wg;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t idx = item.get_global_id(0);
                if (idx < total_elements) {
                    const size_t t = idx / intermediate;
                    const size_t i = idx % intermediate;
                    const size_t row_stride = static_cast<size_t>(2) * intermediate;
                    const float g_f = gate_up[t * row_stride + i];
                    const float u_f = gate_up[t * row_stride + intermediate + i];
                    const float silu_g = g_f / (1.0f + sycl::native::exp(-g_f));
                    out[idx] = bf16(silu_g * u_f);
                }
            });
    });
}

// Row b * M + m -> down_acc[(token * top_k + rank) * hidden]; padding rows are
// skipped, every real row is written exactly once.
sycl::event submit_scatter(Context& ctx, const MoeExpertXmxScratch& s,
                           const MoeExpertXmxChunk& chunk, float* down_acc, int hidden,
                           const sycl::event& dep) {
    const int* bases = chunk.member_row_base;
    const int* tokens = chunk.member_token_ids;
    const int* ranks = chunk.member_ranks;
    const int experts = chunk.num_experts;
    const int rows_per_band = chunk.rows_per_band;
    const int top_k = chunk.top_k;
    const float* c2 = static_cast<const float*>(s.c2);
    const size_t vecs_per_row = size_t(hidden) / 4;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.depends_on(dep);
        cgh.parallel_for(
            sycl::range<1>(size_t(experts) * size_t(rows_per_band) * vecs_per_row),
            [=](sycl::id<1> idx) {
                const size_t row = idx[0] / vecs_per_row;
                const size_t k4 = idx[0] - row * vecs_per_row;
                const int expert = static_cast<int>(row / size_t(rows_per_band));
                const int m = static_cast<int>(row - size_t(expert) * size_t(rows_per_band));
                const int begin = bases[expert];
                if (m >= bases[expert + 1] - begin) return;
                const int act = begin + m;
                const size_t dst_row =
                    size_t(tokens[act]) * size_t(top_k) + size_t(ranks[act]);
                using f4 = sycl::vec<float, 4>;
                *reinterpret_cast<f4*>(down_acc + dst_row * size_t(hidden) + k4 * 4) =
                    *reinterpret_cast<const f4*>(c2 + row * size_t(hidden) + k4 * 4);
            });
    });
}

XmxPrim& ensure_xmx_prim(Context& ctx, MoeExpertXmxScratch& s, int num_experts, int rows_per_band,
                         int hidden, int intermediate) {
    XmxCache& cache = cache_of(s);
    // The c1 dtype is part of the key: an f32 gate_up dst descriptor must never
    // be served to a bf16 c1 buffer (or the other way round).
    const bool c1_f32 = s.c1_f32;
    const uint64_t key = (uint64_t(c1_f32 ? 1u : 0u) << 63) |
                         (uint64_t(static_cast<uint32_t>(num_experts)) << 32) |
                         uint64_t(static_cast<uint32_t>(rows_per_band));
    const auto found = cache.prims.find(key);
    if (found != cache.prims.end()) return found->second;

    if (cache.prims.size() >= kMaxCachedShapes) {
        // Dropping primitives also drops their memory handles; that is only
        // safe once the queue is drained (Bnb4BitLinear.hpp:351-355).
        ctx.synchronize();
        cache.prims.erase(cache.prims.begin());
    }

    const int64_t b = num_experts;
    const int64_t m = rows_per_band;
    const int64_t h = hidden;
    const int64_t i = intermediate;
    const int64_t two_i = 2 * i;
    XmxPrim prim;
    // Same descriptors as the probe: [B, M, K] A, [B, K, N] W and [B, M, N] C,
    // with W stored as the packed [N][K] linear buffer (K contiguous).
    prim.s1 = dnnl::memory::desc({b, m, h}, dt::bf16, {m * h, h, 1});
    prim.w1 = dnnl::memory::desc({b, h, two_i}, dt::bf16, {two_i * h, 1, h});
    prim.d1 = dnnl::memory::desc({b, m, two_i}, c1_f32 ? dt::f32 : dt::bf16,
                                 {m * two_i, two_i, 1});
    prim.s2 = dnnl::memory::desc({b, m, i}, dt::bf16, {m * i, i, 1});
    prim.w2 = dnnl::memory::desc({b, i, h}, dt::bf16, {h * i, 1, i});
    prim.d2 = dnnl::memory::desc({b, m, h}, dt::f32, {m * h, h, 1});
    prim.mm1 = dnnl::matmul(dnnl::matmul::primitive_desc(ctx.engine(), prim.s1, prim.w1, prim.d1));
    prim.mm2 = dnnl::matmul(dnnl::matmul::primitive_desc(ctx.engine(), prim.s2, prim.w2, prim.d2));
    for (int bank = 0; bank < 2; ++bank) {
        prim.args1[bank] = {
            {DNNL_ARG_SRC, usm_memory(ctx, prim.s1, s.a1)},
            {DNNL_ARG_WEIGHTS, usm_memory(ctx, prim.w1, s.wgu_bf16[bank])},
            {DNNL_ARG_DST, usm_memory(ctx, prim.d1, s.c1)}};
        prim.args2[bank] = {
            {DNNL_ARG_SRC, usm_memory(ctx, prim.s2, s.a2)},
            {DNNL_ARG_WEIGHTS, usm_memory(ctx, prim.w2, s.wdn_bf16[bank])},
            {DNNL_ARG_DST, usm_memory(ctx, prim.d2, s.c2)}};
    }
    return cache.prims.emplace(key, std::move(prim)).first->second;
}

void release_all(MoeExpertXmxScratch& s) {
    s.wgu_bf16[0] = nullptr;
    s.wgu_bf16[1] = nullptr;
    s.wdn_bf16[0] = nullptr;
    s.wdn_bf16[1] = nullptr;
    s.a1 = nullptr;
    s.c1 = nullptr;
    s.a2 = nullptr;
    s.c2 = nullptr;
    s.d_plan = nullptr;
    s.max_chunk_rows = 0;
    s.c1_f32 = false;
    s.hidden = 0;
    s.intermediate = 0;
}

}  // namespace

bool moe_expert_xmx_scratch_create(Context& ctx, MoeExpertXmxScratch& s, int max_chunk_rows,
                                   int hidden, int intermediate) {
    moe_expert_xmx_scratch_destroy(ctx, s);
    if (max_chunk_rows <= 0 || hidden <= 0 || intermediate <= 0) {
        throw std::invalid_argument(
            "moe_expert_xmx_scratch_create: rows and dimensions must be positive");
    }
    // AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32 (int, default 0): non-zero keeps the
    // gate_up destination in f32.  Read per create and latched into the
    // scratch: the production backend creates the scratch once per model load,
    // so this is a once-per-process decision in practice, while tests and
    // scratch recreation can still select the mode per scratch.  The f32 c1
    // also selects the f32-input SwiGLU in moe_expert_xmx_prefill_chunk.
    const bool c1_f32 = aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32", 0) != 0;
    const int rows = static_cast<int>((static_cast<int64_t>(max_chunk_rows) + 15) / 16 * 16);
    const size_t gu_bytes =
        size_t(kMaxExpertsPerChunk) * 2 * size_t(intermediate) * size_t(hidden) * sizeof(bf16);
    const size_t dn_bytes =
        size_t(kMaxExpertsPerChunk) * size_t(hidden) * size_t(intermediate) * sizeof(bf16);
    const size_t c1_bytes = size_t(rows) * 2 * size_t(intermediate) *
                            (c1_f32 ? sizeof(float) : sizeof(bf16));
    // Locals are owned by the rollback lambda until the scratch commits.
    void* wgu0 = nullptr;
    void* wgu1 = nullptr;
    void* wdn0 = nullptr;
    void* wdn1 = nullptr;
    void* a1 = nullptr;
    void* c1 = nullptr;
    void* a2 = nullptr;
    void* c2 = nullptr;
    void* d_plan = nullptr;
    XmxCache* cache = nullptr;
    auto rollback = [&]() {
        if (cache != nullptr) delete cache;
        if (d_plan != nullptr) ctx.free_device(d_plan);
        if (c2 != nullptr) ctx.free_device(c2);
        if (a2 != nullptr) ctx.free_device(a2);
        if (c1 != nullptr) ctx.free_device(c1);
        if (a1 != nullptr) ctx.free_device(a1);
        if (wdn1 != nullptr) ctx.free_device(wdn1);
        if (wdn0 != nullptr) ctx.free_device(wdn0);
        if (wgu1 != nullptr) ctx.free_device(wgu1);
        if (wgu0 != nullptr) ctx.free_device(wgu0);
        release_all(s);
    };
    try {
        wgu0 = ctx.alloc_device(gu_bytes);
        wgu1 = ctx.alloc_device(gu_bytes);
        wdn0 = ctx.alloc_device(dn_bytes);
        wdn1 = ctx.alloc_device(dn_bytes);
        a1 = ctx.alloc_device(size_t(rows) * size_t(hidden) * sizeof(bf16));
        c1 = ctx.alloc_device(c1_bytes);
        a2 = ctx.alloc_device(size_t(rows) * size_t(intermediate) * sizeof(bf16));
        c2 = ctx.alloc_device(size_t(rows) * size_t(hidden) * sizeof(float));
        d_plan = ctx.alloc_device((size_t(kPlanBaseSlots) + 2 * size_t(rows)) * sizeof(int));
        cache = new XmxCache();
    } catch (...) {
        rollback();
        return false;
    }
    s.wgu_bf16[0] = wgu0;
    s.wgu_bf16[1] = wgu1;
    s.wdn_bf16[0] = wdn0;
    s.wdn_bf16[1] = wdn1;
    s.a1 = a1;
    s.c1 = c1;
    s.a2 = a2;
    s.c2 = c2;
    s.d_plan = d_plan;
    s.impl = cache;
    s.max_chunk_rows = rows;
    s.c1_f32 = c1_f32;
    s.hidden = hidden;
    s.intermediate = intermediate;
    return true;
}

void moe_expert_xmx_scratch_destroy(Context& ctx, MoeExpertXmxScratch& s) {
    // oneDNN handles and USM blocks may still be referenced by queued work; the
    // project drains before freeing either (Context::~Context does the same and
    // also swallows a device fault here).
    try {
        ctx.synchronize();
    } catch (...) {
    }
    delete static_cast<XmxCache*>(s.impl);
    s.impl = nullptr;
    if (s.wgu_bf16[0] != nullptr) ctx.free_device(s.wgu_bf16[0]);
    if (s.wgu_bf16[1] != nullptr) ctx.free_device(s.wgu_bf16[1]);
    if (s.wdn_bf16[0] != nullptr) ctx.free_device(s.wdn_bf16[0]);
    if (s.wdn_bf16[1] != nullptr) ctx.free_device(s.wdn_bf16[1]);
    if (s.a1 != nullptr) ctx.free_device(s.a1);
    if (s.c1 != nullptr) ctx.free_device(s.c1);
    if (s.a2 != nullptr) ctx.free_device(s.a2);
    if (s.c2 != nullptr) ctx.free_device(s.c2);
    if (s.d_plan != nullptr) ctx.free_device(s.d_plan);
    release_all(s);
}

sycl::event moe_expert_xmx_prefill_chunk(Context& ctx, MoeExpertXmxScratch& scratch,
                                         const MoeExpertXmxChunk& chunk, int bank,
                                         const bf16* x, float* down_acc, int hidden,
                                         int intermediate, int blocksize,
                                         const sycl::event* ready) {
    if (scratch.impl == nullptr || scratch.wgu_bf16[0] == nullptr ||
        scratch.wdn_bf16[0] == nullptr) {
        throw std::runtime_error("moe_expert_xmx_prefill_chunk: scratch is not created");
    }
    if (chunk.num_experts <= 0 || chunk.num_experts > kMaxExpertsPerChunk) {
        throw std::invalid_argument("moe_expert_xmx_prefill_chunk: chunk.num_experts must be 1..16");
    }
    if (chunk.top_k <= 0) {
        throw std::invalid_argument(
            "moe_expert_xmx_prefill_chunk: chunk.top_k must be positive (down_acc row = "
            "token * top_k + rank)");
    }
    if (chunk.rows_per_band <= 0 || chunk.rows_per_band % 16 != 0) {
        throw std::invalid_argument(
            "moe_expert_xmx_prefill_chunk: chunk.rows_per_band must be a multiple of 16");
    }
    if (chunk.slots == nullptr || chunk.member_row_base == nullptr ||
        chunk.member_token_ids == nullptr || chunk.member_ranks == nullptr || x == nullptr ||
        down_acc == nullptr) {
        throw std::invalid_argument("moe_expert_xmx_prefill_chunk: null device pointer in chunk");
    }
    if (bank < 0 || bank > 1) {
        throw std::invalid_argument("moe_expert_xmx_prefill_chunk: bank must be 0 or 1");
    }
    if (size_t(chunk.num_experts) * size_t(chunk.rows_per_band) > size_t(scratch.max_chunk_rows)) {
        throw std::invalid_argument(
            "moe_expert_xmx_prefill_chunk: B * rows_per_band exceeds scratch.max_chunk_rows");
    }
    if (hidden != scratch.hidden || intermediate != scratch.intermediate) {
        throw std::invalid_argument(
            "moe_expert_xmx_prefill_chunk: shape differs from the scratch's geometry");
    }
    if (hidden % 8 != 0 || intermediate % 8 != 0 || blocksize <= 0 || blocksize % 8 != 0 ||
        hidden % blocksize != 0 || intermediate % blocksize != 0) {
        throw std::invalid_argument(
            "moe_expert_xmx_prefill_chunk: hidden/intermediate must divide by 8 and by blocksize");
    }

    XmxPrim& prim =
        ensure_xmx_prim(ctx, scratch, chunk.num_experts, chunk.rows_per_band, hidden, intermediate);
    const int rows = chunk.num_experts * chunk.rows_per_band;

    sycl::event last = submit_dequant(ctx, scratch, chunk, bank, hidden, intermediate, blocksize, ready);
    last = submit_gather(ctx, scratch, chunk, x, hidden, last);
    last = dnnl::sycl_interop::execute(prim.mm1, ctx.stream(), prim.args1[bank], {last});
    if (scratch.c1_f32) {
        // SwiGLU reads the unrounded f32 gate_up tile; a2 is still bf16.
        last = submit_swiglu_f32(ctx, static_cast<const float*>(scratch.c1),
                                 static_cast<bf16*>(scratch.a2), rows, intermediate);
        last = dnnl::sycl_interop::execute(prim.mm2, ctx.stream(), prim.args2[bank], {last});
    } else {
        // SwiGLU on every padded row (zero rows stay zero).  It carries no
        // event, so the in-order queue is what orders it after mm1 and before
        // mm2.
        moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(scratch.c1),
                                  static_cast<bf16*>(scratch.a2), rows, intermediate);
        last = dnnl::sycl_interop::execute(prim.mm2, ctx.stream(), prim.args2[bank], {});
    }
    last = submit_scatter(ctx, scratch, chunk, down_acc, hidden, last);
    return last;
}

}  // namespace ops
