#pragma once

// AILA_Q4E_MOE_COPY_VIA_COPYQ (read once, default off): route the per-layer
// CPU-complement host copies (activation x D2H, CPU result H2D, router-weight
// H2D) through an in-order copy queue instead of the compute queue.
//
// Receipts (tmp/q4e-structural-20261016/chain-probe3..5.log): a host-pointer
// copy issued on the compute queue costs 22-79 us and closes the UR/L0 dynamic
// batch, so every later submit pays the drained round trip (~20-28 us instead
// of 5-8 us). The same copy on a second in-order queue costs 8.0 us and keeps
// the batch open; a kernel taking a cross-queue event costs 8.5 us and a
// deferred cross-queue wait 12.8 us.

#include "core/Context.hpp"
#include "utils/EnvUtils.hpp"

#include <cstddef>
#include <vector>

namespace aila {
namespace q4e {

inline bool moe_copy_via_copyq() {
    static const bool enabled =
        aila::env::read_flag("AILA_Q4E_MOE_COPY_VIA_COPYQ", false);
    return enabled;
}

// Submits one host-pointer copy. A null copy queue keeps the exact previous
// behavior (host-pointer copy on the in-order compute queue).
//
// With a copy queue the copy takes a compute-queue barrier dependency, so it
// may only start after every command the compute queue has submitted so far
// completed. That single fence covers both ordering directions the CPU
// complement needs: a D2H read never observes device bytes its producer kernel
// has not written yet, and an H2D write never overtakes a previously submitted
// kernel that still reads the destination. The barrier is a device-side command
// (it does not close the L0 batch, unlike waits/status queries/same-queue
// host-pointer copies). The consumer side stays explicit: kernels receive the
// returned event as a dependency and the CPU worker waits it through
// job.activation.
inline sycl::event submit_host_copy(Context& ctx, sycl::queue* copy_q, void* dst,
                                    const void* src, size_t bytes,
                                    const sycl::event* order_fence = nullptr) {
    if (!copy_q) return ctx.queue().memcpy(dst, src, bytes);
    if (order_fence)
        return copy_q->memcpy(dst, src, bytes,
                              std::vector<sycl::event>{*order_fence});
    return copy_q->memcpy(
        dst, src, bytes,
        std::vector<sycl::event>{ctx.queue().ext_oneapi_submit_barrier()});
}

// One compute-queue fence shared by a batch of copies that all need the same
// ordering (the coalesced CPU-result rank spans). Returns {} without a copy
// queue, and the caller must then pass a null fence pointer back.
inline sycl::event host_copy_fence(Context& ctx, sycl::queue* copy_q) {
    if (!copy_q) return {};
    return ctx.queue().ext_oneapi_submit_barrier();
}

// Folds several real events into one event for kernels that take a single
// `const sycl::event*` dependency. Callers only push events they actually
// submitted (sycl::event{} compares unequal to itself on this adapter, so a
// default-constructed event is never used as a presence test here).
inline sycl::event join_dependencies(Context& ctx,
                                     const std::vector<sycl::event>& deps) {
    if (deps.empty()) return {};
    if (deps.size() == 1) return deps.front();
    return ctx.queue().ext_oneapi_submit_barrier(deps);
}

} // namespace q4e
} // namespace aila
