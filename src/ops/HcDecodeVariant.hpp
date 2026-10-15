#pragma once

#include "core/Context.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// Experimental decode hyper-connection kernels (E5 round, 2026-10-15).
//
// Same arithmetic as ops::hc_decode_down_inject / ops::hc_decode_up_mix:
// identical per-lane element order for every FMA and the same
// reduce_over_group sequence, so the BF16 rounding points are unchanged.
// The only difference is that the vec8 weight/activation loads are software
// pipelined (explicit load-ahead of `prefetch_depth` iterations) to raise
// memory-level parallelism on the latency-bound rows=1 shapes.
//
// Default: NOT wired into any production path. Callers must verify bitwise
// parity against the production entry before promoting, and the promotion
// gate is charter section 5.3 (>=3% at both shapes, plus >=5% projected e2e).

void hc_decode_down_inject_pl(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups,
    int prefetch_depth = 2, int work_group = 256);

void hc_decode_up_mix_pl(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups, int prefetch_depth = 2);

} // namespace ops
