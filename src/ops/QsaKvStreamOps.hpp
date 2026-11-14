#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include <cstdint>

namespace ops {

// Per-layer residency map for streamed QSA KV caches (RAM authoritative copy +
// fixed-size VRAM slot pool with device-side CLOCK eviction).
struct QsaKvStreamMap {
    Tensor page_table;  // [n_blocks] s32: logical block -> slot, -1 unmapped, -2 claimed
    Tensor slot_block;  // [n_slots]  s32: physical slot -> logical block, or -1 free
    Tensor slot_stamp;  // [n_slots]  s32: last epoch that touched the slot
    Tensor slot_ref;    // [n_slots]  s32: CLOCK second-chance reference bit
    Tensor miss_block;  // [n_slots]  s32: missed logical block ids claimed this call
    Tensor miss_slot;   // [n_slots]  s32: victim physical slots chosen this call
    Tensor ctl;         // [16]       s32: [0]=epoch [1]=hand [2]=misses [3]=overflow, [4..9]=u64 counters
    int n_blocks = 0;
    int n_slots = 0;

    bool valid() const noexcept {
        return page_table.valid() && slot_block.valid() && slot_stamp.valid() &&
               slot_ref.valid() && miss_block.valid() && miss_slot.valid() &&
               ctl.valid() && n_blocks > 0 && n_slots > 0;
    }
};

// Optional dual-write target for decode_prepare_qgkv_packed_partial when KV is streamed:
// writes always go to host_k/host_v at logical active_pos, and also to VRAM slots if
// page_table[active_pos / 4] >= 0.
struct QsaKvStreamWriteTarget {
    void* host_k = nullptr;
    void* host_v = nullptr;
    float* host_k_scale = nullptr;
    float* host_v_scale = nullptr;
    const int32_t* page_table = nullptr;
    int host_capacity = 0;
};

struct QsaKvStreamCounters {
    uint64_t misses = 0;
    uint64_t lookups = 0;
    uint64_t calls = 0;
    bool overflow = false;
};

void qsa_kv_stream_init_map(Context& ctx, QsaKvStreamMap& map, int capacity, int n_slots);
void qsa_kv_stream_reset(Context& ctx, QsaKvStreamMap& map);

void qsa_kv_stream_resolve(
    Context& ctx,
    QsaKvStreamMap& map,
    Tensor& slots_k,
    Tensor& slots_v,
    Tensor* slots_k_scale,
    Tensor* slots_v_scale,
    const Tensor& host_k,
    const Tensor& host_v,
    const Tensor* host_k_scale,
    const Tensor* host_v_scale,
    const int32_t* logical_blocks,
    int num_rows,
    int start_pos,
    int block_topk,
    int compress_ratio,
    Tensor& resolved_blocks_out,
    Tensor* resolved_tail_indices_out = nullptr);

void qsa_kv_stream_stage_from_host(
    Context& ctx,
    Tensor& stage_k,
    Tensor& stage_v,
    Tensor* stage_k_scale,
    Tensor* stage_v_scale,
    const Tensor& host_k,
    const Tensor& host_v,
    const Tensor* host_k_scale,
    const Tensor* host_v_scale,
    int stage_len);

QsaKvStreamCounters qsa_kv_stream_counters(Context& ctx, const QsaKvStreamMap& map);

} // namespace ops
