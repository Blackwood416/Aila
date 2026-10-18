// P0 probe for the proposed Qwen4Exp whole-token graph:
//   route publish (D->H) -> CPU expert work -> completion flag (H->D)
//   -> copy-from-mapped kernel -> bitwise verify -> completion notify (D->H)
//
// The probe deliberately uses Aila's Context and Limited SYCL Graph path.
// It does not use queue.memcpy inside the graph: the mapped host result is
// transferred by a kernel, matching the Strata WDDM-safe pattern.
//
// No main-project source is modified. The GPU run is opt-in through run.ps1.

#include "core/Context.hpp"
#include "engine/Engine.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"
#include "ops/GatedDeltaNetOps.hpp"
#include "ops/MoeExpertOps.hpp"

#include <sycl/sycl.hpp>
#include <sycl/bit_cast.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ex = sycl::ext::oneapi::experimental;
namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

// The backend and engine friend this exact global type name.
struct Qwen4ExpGenerationTestAccess {
    static Context& Ctx(InferenceEngine& engine) { return *engine.ctx_; }
    static Qwen4ExpBnb4Backend& Backend(InferenceEngine& engine) {
        return dynamic_cast<Qwen4ExpBnb4Backend&>(*engine.backend_);
    }
    static const ops::DeltaNetGeometry& FrontGeometry(
        Qwen4ExpBnb4Backend& b) {
        return b.layers_.at(0).geom;
    }
    static size_t LayerCount(Qwen4ExpBnb4Backend& b) {
        return b.layers_.size();
    }
    static aila::Qwen4ExpAdaptiveCache& Adaptive(
        Qwen4ExpBnb4Backend& b) {
        return b.adaptive_cache_;
    }
    static const aila::LayerNestedMetadata& NestedMeta(
        Qwen4ExpBnb4Backend& b, int layer) {
        return b.host_arena_.GetLayerNestedMetadata(layer);
    }
    static Tensor& MoeH(Qwen4ExpBnb4Backend& b) { return b.buf_.moe_h; }
    static Tensor& ResidentDown(Qwen4ExpBnb4Backend& b) {
        return b.resident_down_;
    }
    static int ExpertsPerLayer(Qwen4ExpBnb4Backend& b) {
        return b.cfg_.num_experts;
    }
    static int TopK(Qwen4ExpBnb4Backend& b) {
        return b.cfg_.num_experts_per_tok;
    }
    static int HiddenSize(Qwen4ExpBnb4Backend& b) {
        return b.hidden_size_;
    }
    static int IntermediateSize(Qwen4ExpBnb4Backend& b) {
        return b.cfg_.moe_intermediate_size;
    }
    static Tensor& FrontHidden(Qwen4ExpBnb4Backend& b) {
        return b.buf_.hidden4;
    }
    static Tensor& FrontX(Qwen4ExpBnb4Backend& b) { return b.buf_.x; }
    static Tensor& FrontScores(Qwen4ExpBnb4Backend& b) {
        return b.buf_.router_scores;
    }
    static Tensor& FrontInject(Qwen4ExpBnb4Backend& b) {
        return b.buf_.hc_inject_f32;
    }
    static const int* FrontIds(Qwen4ExpBnb4Backend& b) {
        return b.router_host_ids_;
    }
    static const float* FrontWeights(Qwen4ExpBnb4Backend& b) {
        return b.router_host_weights_;
    }
    static void FrontRoute(Qwen4ExpBnb4Backend& b,
                           ops::DeltaNetState& state,
                           int* route_ids, float* route_weights) {
        auto& ctx = *b.context_;
        auto& layer = b.layers_.at(0);
        if (!layer.is_linear)
            throw std::runtime_error("P1 requires a linear layer 0");
        b.RunHyperConnection(ctx, layer.attn_hc, 1, true);
        layer.linear_all_proj.ForwardDecodeRows(
            ctx, b.linear_scratch_, b.buf_.x, b.buf_.linear_all, 1);
        const auto* p = b.buf_.linear_all.data_as<bf16>();
        ops::DeltaNetWeights weights{
            layer.linear_conv1d_weight->data_as<bf16>(),
            layer.linear_norm_weight->data_as<float>(),
            layer.linear_A_log->data_as<float>(),
            layer.linear_dt_bias->data_as<bf16>()};
        ops::run_linear_delta_decode_gpu(
            ctx, layer.geom, weights, state, p, p + b.linear_qkv_dim_,
            p + b.linear_qkv_dim_ + b.linear_z_dim_,
            p + b.linear_qkv_dim_ + b.linear_z_dim_ + b.linear_kv_heads_,
            b.buf_.attn_out.data_as<bf16>());
        layer.linear_o_proj.ForwardDecodeRows(
            ctx, b.linear_scratch_, b.buf_.attn_out, b.buf_.y, 1);
        b.RunHyperConnectionCombined(
            ctx, layer.mlp_hc, 1, true, b.buf_.y.data_as<bf16>(),
            b.buf_.hc_inject_f32.data_as<float>());
        b.DenseLinear(
            ctx, layer.router_gate, layer.router_w,
            b.buf_.x, b.buf_.router_scores, 1, b.cfg_.num_experts,
            b.hidden_size_);
        ops::router_softmax_topk_device(
            ctx, b.buf_.router_scores.data_as<bf16>(), nullptr,
            route_ids, route_weights, 1,
            b.cfg_.num_experts, b.cfg_.num_experts_per_tok, true);
    }
    static void Front(Qwen4ExpBnb4Backend& b, ops::DeltaNetState& state) {
        FrontRoute(b, state, b.router_host_ids_, b.router_host_weights_);
    }
};

namespace {

constexpr int kTopK = 10;
constexpr int kResultWidth = 8;
constexpr int kResultCount = kTopK * kResultWidth;
constexpr uint32_t kDefaultSpinMax = 20000;
constexpr uint32_t kPollFlushEvery = 1024;

// Intel cache-control hints. These are the load/store half of the Strata
// doorbell fix: system-scope atomics alone can remain in the L1/L3 cache on Arc.
using UncachedRead = decltype(oex::properties(
    iex::read_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));
using UncachedWrite = decltype(oex::properties(
    iex::write_hint<iex::cache_control<
        iex::cache_mode::uncached,
        oex::cache_level::L1,
        oex::cache_level::L3>>));

template <class T>
inline T uncached_load(const T* p) {
    oex::annotated_ptr<T, UncachedRead> a(const_cast<T*>(p));
    return a[0];
}

template <class T>
inline void uncached_store(T* p, T v) {
    oex::annotated_ptr<T, UncachedWrite> a(p);
    a[0] = v;
}

struct alignas(64) ControlRing {
    uint32_t route_seq;
    uint32_t cpu_flag;
    uint32_t result_seq;
    uint32_t done_seq;
    uint32_t timeout_count;
    uint32_t stale_count;
    uint32_t mismatch_count;
    uint32_t host_route_error_count;
    uint64_t timeout_spins;
    uint64_t reserved[3];
};

struct alignas(64) RouteRing {
    int32_t ids[kTopK];
    float weights[kTopK];
    uint32_t reserved[12];
};

struct alignas(64) ResultRing {
    float values[kResultCount];
    uint32_t reserved[16];
};

struct alignas(64) HostRing {
    ControlRing ctrl;
    RouteRing route;
    ResultRing result;
};

struct alignas(64) DeviceEpoch {
    uint32_t seq;
    uint32_t reserved[15];
};

static_assert(sizeof(ControlRing) == 64);
static_assert(sizeof(RouteRing) == 128);
static_assert(sizeof(ResultRing) == 384);
static_assert(sizeof(HostRing) == 576);
static_assert(sizeof(DeviceEpoch) == 64);

struct PolledRing {
    HostRing* ptr = nullptr;
    ze_context_handle_t ze_ctx = nullptr;
    bool ze_owned = false;
};

PolledRing allocate_polled_ring(Context& ctx) {
    PolledRing out;
    out.ze_ctx =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            ctx.queue().get_context());
    ze_host_mem_alloc_desc_t desc{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
        ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
    void* storage = nullptr;
    const ze_result_t result = zeMemAllocHost(
        out.ze_ctx, &desc, sizeof(HostRing), alignof(HostRing), &storage);
    out.ze_owned = result == ZE_RESULT_SUCCESS && storage != nullptr;
    out.ptr = out.ze_owned
        ? static_cast<HostRing*>(storage)
        : static_cast<HostRing*>(ctx.alloc_host(sizeof(HostRing)));
    if (reinterpret_cast<uintptr_t>(out.ptr) % alignof(HostRing) != 0)
        throw std::runtime_error("host ring is not 64-byte aligned");
    std::memset(out.ptr, 0, sizeof(HostRing));
    return out;
}

void free_polled_ring(Context& ctx, PolledRing& ring) {
    if (!ring.ptr) return;
    if (ring.ze_owned)
        zeMemFree(ring.ze_ctx, ring.ptr);
    else
        ctx.free_host(ring.ptr);
    ring.ptr = nullptr;
    ring.ze_owned = false;
}

inline uint32_t expected_id(uint32_t seq, uint32_t k) {
    return (seq * 37u + k * 53u) & 511u;
}

inline uint32_t expected_weight_bits(uint32_t seq, uint32_t k) {
    return (seq * 13u + k * 7u) & 255u;
}

inline float expected_weight(uint32_t seq, uint32_t k) {
    return static_cast<float>(expected_weight_bits(seq, k));
}

inline float expected_result(uint32_t seq, uint32_t k, uint32_t j) {
    const uint32_t id = expected_id(seq, k);
    const uint32_t base = (seq & 1023u) * 4096u;
    return static_cast<float>(base + id * kResultWidth + j);
}

inline uint32_t bit_u32(float v) {
    return sycl::bit_cast<uint32_t>(v);
}

inline volatile uint32_t* volatile_u32(uint32_t* p) {
    return reinterpret_cast<volatile uint32_t*>(p);
}

inline volatile float* volatile_float(float* p) {
    return reinterpret_cast<volatile float*>(p);
}

void submit_route_publish(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq + 1u;
            epoch->seq = seq;
            for (uint32_t k = 0; k < kTopK; ++k) {
                uncached_store(&ring->route.ids[k],
                               static_cast<int32_t>(expected_id(seq, k)));
                uncached_store(&ring->route.weights[k],
                               expected_weight(seq, k));
            }
            // Payload is uncached and ordered before the sequence publication.
            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            uncached_store(&ring->ctrl.route_seq, seq);
        });
    });
}

void submit_real_route_publish(sycl::queue& q, HostRing* ring,
                               DeviceEpoch* epoch, const int* ids,
                               const float* weights) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq + 1u;
            epoch->seq = seq;
            for (uint32_t k = 0; k < kTopK; ++k) {
                uncached_store(&ring->route.ids[k],
                               static_cast<int32_t>(ids[k]));
                uncached_store(&ring->route.weights[k], weights[k]);
            }
            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            uncached_store(&ring->ctrl.route_seq, seq);
        });
    });
}

void submit_resident_classify(sycl::queue& q, const int* ids,
                              const int* slot_table, int* selected,
                              int layer, int experts, int top_k) {
    q.parallel_for(
        sycl::range<1>(16),
        [=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
            const int rank = static_cast<int>(index[0]);
            if (rank < top_k) {
                const int expert = ids[rank];
                selected[rank] =
                    expert >= 0 && expert < experts
                        ? slot_table[layer * experts + expert]
                        : -1;
            }
        });
}

void submit_device_wait(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                        uint32_t spin_max) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq;
            uint32_t flag = 0;
            uint32_t spins = 0;
            for (;;) {
                // Keep the acquire fence inside the retry loop. Besides ordering,
                // it prevents the compiler from hoisting a non-volatile load out
                // of the polling loop.
                sycl::atomic_fence(sycl::memory_order::acquire,
                                   sycl::memory_scope::system);
                flag = uncached_load(&ring->ctrl.cpu_flag);
                if (flag == seq) break;
                if (++spins >= spin_max) break;
            }
            if (flag != seq) {
                uncached_store(&ring->ctrl.timeout_count,
                               uncached_load(&ring->ctrl.timeout_count) + 1u);
                uncached_store(&ring->ctrl.timeout_spins,
                               static_cast<uint64_t>(spins));
                return;
            }
            // The matching flag establishes publication of result_seq/result.
            sycl::atomic_fence(sycl::memory_order::acquire,
                               sycl::memory_scope::system);

            bool stale = uncached_load(&ring->ctrl.result_seq) != seq;
            if (!stale) {
                for (uint32_t k = 0; k < kTopK; ++k) {
                    for (uint32_t j = 0; j < kResultWidth; ++j) {
                        const uint32_t idx = k * kResultWidth + j;
                        const float got =
                            uncached_load(&ring->result.values[idx]);
                        const float expected = expected_result(seq, k, j);
                        if (bit_u32(got) != bit_u32(expected)) {
                            stale = true;
                            break;
                        }
                    }
                    if (stale) break;
                }
            }
            if (stale) {
                uncached_store(&ring->ctrl.stale_count,
                               uncached_load(&ring->ctrl.stale_count) + 1u);
            }
        });
    });
}

// The graph must not contain queue.memcpy: a copy-engine node can split the
// WDDM submission. This kernel reads mapped pinned host USM directly.
void submit_copy_from_mapped(sycl::queue& q, HostRing* ring, float* device_result) {
    q.parallel_for(sycl::range<1>(kResultCount), [=](sycl::id<1> id) {
        const size_t i = id[0];
        device_result[i] = uncached_load(&ring->result.values[i]);
    });
}

void submit_bitwise_verify(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                           const float* device_result) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = epoch->seq;
            bool mismatch = uncached_load(&ring->ctrl.result_seq) != seq;
            if (!mismatch) {
                for (uint32_t k = 0; k < kTopK; ++k) {
                    for (uint32_t j = 0; j < kResultWidth; ++j) {
                        const uint32_t idx = k * kResultWidth + j;
                        const float got = device_result[idx];
                        const float expected = expected_result(seq, k, j);
                        if (bit_u32(got) != bit_u32(expected)) {
                            mismatch = true;
                            break;
                        }
                    }
                    if (mismatch) break;
                }
            }
            if (mismatch) {
                uncached_store(&ring->ctrl.mismatch_count,
                               uncached_load(&ring->ctrl.mismatch_count) + 1u);
            }
            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            uncached_store(&ring->ctrl.done_seq, seq);
        });
    });
}

double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = p * static_cast<double>(v.size() - 1);
    const size_t i = static_cast<size_t>(pos + 0.5);
    return v[std::min(i, v.size() - 1)];
}

void burn_cpu_us(int cpu_us) {
    if (cpu_us <= 0) return;
    const auto deadline = Clock::now() + std::chrono::microseconds(cpu_us);
    while (Clock::now() < deadline) {
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

sycl::info::event_command_status release_graph_submission(
    const sycl::event& graph_event) {
    // A driver entry is required to discharge the L0 dynamic batch. Querying
    // the graph event is the release primitive already measured in Aila's
    // submit probes; it neither waits nor changes graph semantics.
    return graph_event
        .get_info<sycl::info::event::command_execution_status>();
}

enum class PollResult { Reached, Timeout, QueueEnded, Error };

PollResult poll_host_word(sycl::queue& q, volatile uint32_t* word,
                          uint32_t want, int timeout_ms, std::string& error) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    uint32_t polls = 0;
    while (*word != want) {
        if (*word > want) {
            error = "ring sequence advanced past expected value";
            return PollResult::Error;
        }
        if ((++polls & (kPollFlushEvery - 1u)) == 0u) {
            try {
                if (q.ext_oneapi_empty() && *word != want) {
                    error = "queue became empty before expected ring value";
                    return PollResult::QueueEnded;
                }
            } catch (const sycl::exception& e) {
                error = std::string("queue query failed: ") + e.what();
                return PollResult::Error;
            }
        }
        if (Clock::now() >= deadline) {
            std::ostringstream os;
            os << "host timeout waiting for ring value " << want
               << " (last=" << *word << ", polls=" << polls << ')';
            error = os.str();
            return PollResult::Timeout;
        }
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return PollResult::Reached;
}

bool host_route_matches(HostRing* ring, uint32_t seq, int timeout_ms,
                        bool synthetic_route, std::string& error) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    uint32_t polls = 0;
    for (;;) {
        // On A770 the sequence word can become visible before its payload.
        // The host initialized these lines with memset, so volatile alone can
        // keep hitting the CPU's stale cache line while the GPU writes through
        // uncached PCIe memory. Invalidate the route lines, enter the driver,
        // then re-read the payload.
        _mm_clflush(&ring->route.ids[0]);
        _mm_clflush(&ring->route.ids[kTopK - 1]);
        _mm_clflush(&ring->route.weights[0]);
        _mm_clflush(&ring->route.weights[kTopK - 1]);
        uint32_t bad_k = kTopK;
        for (uint32_t k = 0; k < kTopK; ++k) {
            const int32_t got_id = ring->route.ids[k];
            const float got_weight = ring->route.weights[k];
            const bool bad = synthetic_route
                ? got_id != static_cast<int32_t>(expected_id(seq, k)) ||
                      bit_u32(got_weight) !=
                          bit_u32(expected_weight(seq, k))
                : got_id < 0 || got_id >= 512 ||
                      !std::isfinite(got_weight) || got_weight < 0.0f ||
                      got_weight > 1.0f;
            if (bad) {
                bad_k = k;
                break;
            }
        }
        if (bad_k == kTopK) {
            std::atomic_thread_fence(std::memory_order_acquire);
            return true;
        }
        if (Clock::now() >= deadline) {
            std::ostringstream os;
            os << "route payload timeout at k=" << bad_k << " seq=" << seq
               << " polls=" << (polls + 1u);
            error = os.str();
            return false;
        }
        ++polls;
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

void write_host_result(HostRing* ring, uint32_t value_seq, uint32_t publish_seq) {
    volatile float* values = volatile_float(ring->result.values);
    for (uint32_t k = 0; k < kTopK; ++k) {
        for (uint32_t j = 0; j < kResultWidth; ++j) {
            values[k * kResultWidth + j] =
                expected_result(value_seq, k, j);
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *volatile_u32(&ring->ctrl.result_seq) = publish_seq;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    *volatile_u32(&ring->ctrl.cpu_flag) = publish_seq;
}

struct Handoff {
    uint32_t seq = 0;
    double launch_to_route_us = 0.0;
    double route_to_flag_us = 0.0;
    double route_to_done_us = 0.0;
    double replay_wall_us = 0.0;
    uint32_t timeout_delta = 0;
    uint32_t stale_delta = 0;
    uint32_t mismatch_delta = 0;
};

Handoff service_replay(Context& ctx,
                       ex::command_graph<ex::graph_state::executable>& graph,
                       HostRing* ring,
                       uint32_t expected_seq,
                       int cpu_us,
                       int host_timeout_ms,
                       bool stale_injection,
                       bool synthetic_route = true) {
    sycl::queue& q = ctx.queue();
    auto& ctrl = ring->ctrl;
    const uint32_t timeout_before = *volatile_u32(&ctrl.timeout_count);
    const uint32_t stale_before = *volatile_u32(&ctrl.stale_count);
    const uint32_t mismatch_before = *volatile_u32(&ctrl.mismatch_count);

    const auto t_launch = Clock::now();
    auto graph_event = q.ext_oneapi_graph(graph);
    release_graph_submission(graph_event);

    std::string error;
    const PollResult route_poll =
        poll_host_word(q, volatile_u32(&ctrl.route_seq), expected_seq,
                       host_timeout_ms, error);
    if (route_poll != PollResult::Reached) {
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error("route " + error);
    }
    const auto t_route = Clock::now();
    if (!host_route_matches(ring, expected_seq, host_timeout_ms,
                            synthetic_route, error)) {
        *volatile_u32(&ctrl.host_route_error_count) =
            *volatile_u32(&ctrl.host_route_error_count) + 1u;
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error(error);
    }

    const uint32_t value_seq =
        stale_injection ? expected_seq - 1u : expected_seq;
    burn_cpu_us(cpu_us);
    write_host_result(ring, value_seq, expected_seq);
    const auto t_flag = Clock::now();

    const PollResult done_poll =
        poll_host_word(q, volatile_u32(&ctrl.done_seq), expected_seq,
                       host_timeout_ms, error);
    if (done_poll != PollResult::Reached) {
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error("done " + error);
    }
    const auto t_done = Clock::now();
    graph_event.wait_and_throw();
    const auto t_end = Clock::now();

    Handoff out;
    out.seq = expected_seq;
    out.launch_to_route_us = us_between(t_launch, t_route);
    out.route_to_flag_us = us_between(t_route, t_flag);
    out.route_to_done_us = us_between(t_route, t_done);
    out.replay_wall_us = us_between(t_launch, t_end);
    out.timeout_delta =
        *volatile_u32(&ctrl.timeout_count) - timeout_before;
    out.stale_delta = *volatile_u32(&ctrl.stale_count) - stale_before;
    out.mismatch_delta =
        *volatile_u32(&ctrl.mismatch_count) - mismatch_before;
    return out;
}

struct ArmStats {
    int cpu_us = 0;
    size_t measured = 0;
    bool fresh_ok = true;
    bool stale_detected = false;
    std::vector<double> launch_to_route;
    std::vector<double> route_to_flag;
    std::vector<double> route_to_done;
    std::vector<double> wall;
};

void print_arm(const ArmStats& s) {
    const bool ok = s.fresh_ok && s.stale_detected && s.measured >= 1000;
    std::cout << "[TOKEN GRAPH ARM] cpu_us=" << s.cpu_us
              << " replays=" << s.measured
              << " ok=" << (ok ? 1 : 0)
              << " fresh_bitexact=" << (s.fresh_ok ? 1 : 0)
              << " stale_detected=" << (s.stale_detected ? 1 : 0)
              << " launch_to_route_us_median="
              << std::fixed << std::setprecision(3)
              << percentile(s.launch_to_route, 0.50)
              << " launch_to_route_us_p99="
              << percentile(s.launch_to_route, 0.99)
              << " route_to_flag_us_median="
              << percentile(s.route_to_flag, 0.50)
              << " route_to_flag_us_p99="
              << percentile(s.route_to_flag, 0.99)
              << " handoff_us_median="
              << percentile(s.route_to_done, 0.50)
              << " handoff_us_p99="
              << percentile(s.route_to_done, 0.99)
              << " replay_wall_us_median="
              << percentile(s.wall, 0.50)
              << " replay_wall_us_p99="
              << percentile(s.wall, 0.99)
              << '\n';
}

std::vector<int> parse_cpu_sweep(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const int value = std::stoi(item);
        if (value < 0) throw std::runtime_error("cpu-us values must be >= 0");
        out.push_back(value);
    }
    if (out.empty()) throw std::runtime_error("cpu-us sweep is empty");
    return out;
}

struct Options {
    uint32_t replays = 1000;
    int warmup = 10;
    uint32_t spin_max = kDefaultSpinMax;
    int host_timeout_ms = 2000;
    std::vector<int> cpu_sweep{0, 50, 200, 500};
    std::string real_front_model;
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--replays") {
            o.replays = static_cast<uint32_t>(std::stoul(next("--replays")));
        } else if (arg == "--warmup") {
            o.warmup = std::stoi(next("--warmup"));
        } else if (arg == "--spin-max") {
            o.spin_max = static_cast<uint32_t>(std::stoul(next("--spin-max")));
        } else if (arg == "--host-timeout-ms") {
            o.host_timeout_ms = std::stoi(next("--host-timeout-ms"));
        } else if (arg == "--cpu-us") {
            o.cpu_sweep = parse_cpu_sweep(next("--cpu-us"));
        } else if (arg == "--real-front-model") {
            o.real_front_model = next("--real-front-model");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: probe_token_graph.exe [--replays N] [--warmup N]\n"
                   "       [--cpu-us 0,50,200,500] [--spin-max N]\n"
                   "       [--host-timeout-ms N] [--real-front-model DIR]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (o.replays < 1000)
        throw std::runtime_error("--replays must be >= 1000 for P0");
    if (o.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (!o.spin_max) throw std::runtime_error("--spin-max must be > 0");
    if (o.host_timeout_ms <= 0)
        throw std::runtime_error("--host-timeout-ms must be > 0");
    return o;
}

bool same_bytes(const std::vector<uint8_t>& a,
                const std::vector<uint8_t>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

std::vector<uint8_t> snapshot_front(Context& ctx,
                                    Qwen4ExpBnb4Backend& b,
                                    Tensor& state, Tensor& history) {
    std::vector<uint8_t> bytes;
    auto device = [&](const void* p, size_t n) {
        const size_t offset = bytes.size();
        bytes.resize(offset + n);
        ctx.memcpy_d2h(bytes.data() + offset, p, n);
    };
    device(state.data(), state.size_bytes());
    device(history.data(), history.size_bytes());
    device(Qwen4ExpGenerationTestAccess::FrontHidden(b).data(), 4 * 2560 * 2);
    device(Qwen4ExpGenerationTestAccess::FrontX(b).data(), 2560 * 2);
    device(Qwen4ExpGenerationTestAccess::FrontScores(b).data(), 512 * 2);
    device(Qwen4ExpGenerationTestAccess::FrontInject(b).data(),
           4 * sizeof(float));
    return bytes;
}

int run_real_front_model(const Options& options) {
    InferenceEngine engine;
    if (!engine.init(options.real_front_model, 512))
        throw std::runtime_error("real-front model init failed");
    Context& ctx = Qwen4ExpGenerationTestAccess::Ctx(engine);
    sycl::queue& q = ctx.queue();
    auto& b = Qwen4ExpGenerationTestAccess::Backend(engine);
    const auto& geom = Qwen4ExpGenerationTestAccess::FrontGeometry(b);
    const bool limited =
        q.get_device().has(sycl::aspect::ext_oneapi_limited_graph);
    const bool full = q.get_device().has(sycl::aspect::ext_oneapi_graph);
    if (!limited && !full)
        throw std::runtime_error("static graph unsupported");

    auto state = Tensor::allocate(
        ctx, {int64_t(geom.v_heads) * geom.head_k_dim * geom.head_v_dim},
        dnnl::memory::data_type::f32);
    auto history = Tensor::allocate(
        ctx, {int64_t(geom.conv_rows) * geom.qkv_dim},
        dnnl::memory::data_type::f32);
    std::vector<float> initial_state(state.numel());
    std::vector<float> initial_history(history.numel());
    std::vector<bf16> input(4 * 2560);

    auto reset = [&](int seed) {
        for (size_t j = 0; j < initial_state.size(); ++j) {
            initial_state[j] =
                float(int((j * 17 + seed * 31) % 101) - 50) * .0002f;
        }
        for (size_t j = 0; j < initial_history.size(); ++j) {
            initial_history[j] =
                float(int((j * 7 + seed * 13) % 101) - 50) * .002f;
        }
        for (size_t j = 0; j < input.size(); ++j) {
            input[j] =
                bf16(float(int((j * 11 + seed * 3) % 103) - 51) * .015f);
        }
        ctx.memcpy_h2d(state.data(), initial_state.data(), state.size_bytes());
        ctx.memcpy_h2d(history.data(), initial_history.data(),
                       history.size_bytes());
        ctx.memcpy_h2d(
            Qwen4ExpGenerationTestAccess::FrontHidden(b).data(),
            input.data(), input.size() * sizeof(bf16));
        return ops::DeltaNetState{state.data_as<float>(),
                                  history.data_as<float>(), seed % 3, false};
    };

    // Warm every lazy allocation/path outside the capture.
    auto ds = reset(0);
    Qwen4ExpGenerationTestAccess::Front(b, ds);
    ctx.synchronize();

    ds = reset(0);
    Qwen4ExpGenerationTestAccess::Front(b, ds);
    ctx.synchronize();
    const auto reference = snapshot_front(ctx, b, state, history);
    std::array<int, kTopK> reference_ids{};
    std::array<float, kTopK> reference_weights{};
    for (int k = 0; k < kTopK; ++k) {
        reference_ids[k] =
            Qwen4ExpGenerationTestAccess::FrontIds(b)[k];
        reference_weights[k] =
            Qwen4ExpGenerationTestAccess::FrontWeights(b)[k];
    }

    ds = reset(0);
    ctx.synchronize();
    for (int warm = 0; warm < 4; ++warm) {
        Qwen4ExpGenerationTestAccess::Front(b, ds);
        ctx.synchronize();
    }
    std::vector<double> direct_us;
    direct_us.reserve(30);
    for (int repeat = 0; repeat < 30; ++repeat) {
        const auto start = Clock::now();
        Qwen4ExpGenerationTestAccess::Front(b, ds);
        ctx.synchronize();
        direct_us.push_back(us_between(start, Clock::now()));
    }
    std::cout << "[REAL FRONT DIRECT] median_us="
              << percentile(direct_us, 0.50)
              << " p10_us=" << percentile(direct_us, 0.10)
              << " p90_us=" << percentile(direct_us, 0.90) << '\n';

    ds = reset(0);
    ctx.synchronize();
    PolledRing polled = allocate_polled_ring(ctx);
    HostRing* ring = polled.ptr;
    DeviceEpoch* epoch =
        static_cast<DeviceEpoch*>(ctx.alloc_device(sizeof(DeviceEpoch)));
    float* device_result =
        static_cast<float*>(ctx.alloc_device(sizeof(float) * kResultCount));
    int* graph_route_ids = static_cast<int*>(
        ctx.alloc_device(sizeof(int) * kTopK));
    float* graph_route_weights = static_cast<float*>(
        ctx.alloc_device(sizeof(float) * kTopK));
    auto& adaptive = Qwen4ExpGenerationTestAccess::Adaptive(b);
    const int top_k = Qwen4ExpGenerationTestAccess::TopK(b);
    const int experts = Qwen4ExpGenerationTestAccess::ExpertsPerLayer(b);
    const int hidden = Qwen4ExpGenerationTestAccess::HiddenSize(b);
    const int intermediate =
        Qwen4ExpGenerationTestAccess::IntermediateSize(b);
    if (!adaptive.IsAllocated() || !adaptive.DeviceSlotTable() ||
        top_k < 1 || top_k > 16)
        throw std::runtime_error("adaptive resident graph state unavailable");
    const auto& nested_meta =
        Qwen4ExpGenerationTestAccess::NestedMeta(b, 0);
    std::vector<ops::MoeExpertSlotDev> resident_descriptors(
        static_cast<size_t>(adaptive.NumGpuSlots()));
    for (int slot = 0; slot < adaptive.NumGpuSlots(); ++slot) {
        if (adaptive.GetResidentEntry(slot).expert_id >= 0)
            resident_descriptors[slot] =
                adaptive.GetResidentSlotDev(slot, &nested_meta);
    }
    auto* resident_descriptors_dev =
        static_cast<ops::MoeExpertSlotDev*>(ctx.alloc_device(
            resident_descriptors.size() *
            sizeof(ops::MoeExpertSlotDev)));
    int* selected_dev =
        static_cast<int*>(ctx.alloc_device(sizeof(int) * 16));
    int* moe_test_ids_dev =
        static_cast<int*>(ctx.alloc_device(sizeof(int) * top_k));
    bf16* x_copy =
        static_cast<bf16*>(ctx.alloc_device(sizeof(bf16) * hidden));
    bf16* h_reference =
        static_cast<bf16*>(ctx.alloc_device(
            sizeof(bf16) * top_k * intermediate));
    float* down_reference =
        static_cast<float*>(ctx.alloc_device(
            sizeof(float) * top_k * hidden));
    ctx.memcpy_h2d(resident_descriptors_dev,
                   resident_descriptors.data(),
                   resident_descriptors.size() *
                       sizeof(ops::MoeExpertSlotDev));
    std::array<int, 16> moe_test_ids{};
    std::vector<bool> used(static_cast<size_t>(experts), false);
    int chosen = 0;
    const int resident_target = std::min(5, top_k);
    for (int slot = 0;
         slot < adaptive.NumGpuSlots() && chosen < resident_target; ++slot) {
        const auto entry = adaptive.GetResidentEntry(slot);
        if (entry.layer != 0 || entry.expert_id < 0 ||
            entry.expert_id >= experts || used[entry.expert_id])
            continue;
        moe_test_ids[chosen++] = entry.expert_id;
        used[entry.expert_id] = true;
    }
    for (int expert = 0;
         expert < experts && chosen < top_k; ++expert) {
        if (used[expert] || adaptive.GetSlot(0, expert) >= 0) continue;
        moe_test_ids[chosen++] = expert;
        used[expert] = true;
    }
    for (int expert = 0;
         expert < experts && chosen < top_k; ++expert) {
        if (used[expert]) continue;
        moe_test_ids[chosen++] = expert;
        used[expert] = true;
    }
    ctx.memcpy_h2d(moe_test_ids_dev, moe_test_ids.data(),
                   sizeof(int) * top_k);
    q.memset(epoch, 0, sizeof(DeviceEpoch)).wait_and_throw();
    q.memset(device_result, 0, sizeof(float) * kResultCount).wait_and_throw();
    q.memset(graph_route_ids, 0, sizeof(int) * kTopK).wait_and_throw();
    q.memset(graph_route_weights, 0, sizeof(float) * kTopK).wait_and_throw();
    q.memset(selected_dev, -1, sizeof(int) * 16).wait_and_throw();

    std::cout << "[REAL FRONT GRAPH] model=" << options.real_front_model
              << " layers=" << Qwen4ExpGenerationTestAccess::LayerCount(b)
              << " allocation="
              << (polled.ze_owned ? "zeMemAllocHost_uncached_bias"
                                  : "malloc_host_fallback")
              << '\n';

    const size_t h_graph_bytes =
        sizeof(bf16) * size_t(top_k) * intermediate;
    const size_t down_graph_bytes =
        sizeof(float) * size_t(top_k) * hidden;
    q.memset(Qwen4ExpGenerationTestAccess::MoeH(b).data(),
             0x7f, h_graph_bytes);
    q.memset(Qwen4ExpGenerationTestAccess::ResidentDown(b).data(),
             0x7f, down_graph_bytes);
    q.memset(h_reference, 0x7f, h_graph_bytes);
    q.memset(down_reference, 0x7f, down_graph_bytes);
    ctx.synchronize();

    bool all_ok = true;
    {
        ex::command_graph<ex::graph_state::modifiable> recording(
            q.get_context(), q.get_device());
        recording.begin_recording(q);
        try {
            Qwen4ExpGenerationTestAccess::FrontRoute(
                b, ds, graph_route_ids, graph_route_weights);
            submit_real_route_publish(
                q, ring, epoch, graph_route_ids, graph_route_weights);
            submit_resident_classify(
                q, moe_test_ids_dev, adaptive.DeviceSlotTable(), selected_dev,
                0, experts, top_k);
            ops::moe_resident_gate_up(
                ctx, resident_descriptors_dev, selected_dev,
                Qwen4ExpGenerationTestAccess::FrontX(b).data_as<bf16>(),
                Qwen4ExpGenerationTestAccess::MoeH(b).data_as<bf16>(),
                top_k, hidden, intermediate, true, true);
            ops::moe_resident_down(
                ctx, resident_descriptors_dev, selected_dev,
                Qwen4ExpGenerationTestAccess::MoeH(b).data_as<bf16>(),
                Qwen4ExpGenerationTestAccess::ResidentDown(b)
                    .data_as<float>(),
                top_k, hidden, intermediate, true, true);
            submit_device_wait(q, ring, epoch, options.spin_max);
            submit_copy_from_mapped(q, ring, device_result);
            submit_bitwise_verify(q, ring, epoch, device_result);
            recording.end_recording(q);
        } catch (...) {
            recording.end_recording(q);
            throw;
        }
        auto executable = recording.finalize();
        std::cout << "[REAL FRONT GRAPH] capture=ok\n";

        uint32_t expected_seq = 1;
        const Handoff first = service_replay(
            ctx, executable, ring, expected_seq, 0, options.host_timeout_ms,
            false, false);
        if (first.timeout_delta || first.stale_delta ||
            first.mismatch_delta) {
            throw std::runtime_error("real-front handshake failed");
        }
        ctx.synchronize();
        const auto actual = snapshot_front(ctx, b, state, history);
        const bool front_parity = same_bytes(reference, actual);
        bool route_parity = true;
        for (int k = 0; k < kTopK; ++k) {
            route_parity &=
                ring->route.ids[k] == reference_ids[k] &&
                bit_u32(ring->route.weights[k]) ==
                    bit_u32(reference_weights[k]);
        }

        std::array<int, 16> selected_host{};
        ctx.memcpy_d2h(selected_host.data(), selected_dev,
                       sizeof(int) * selected_host.size());
        int resident_count = 0;
        for (int k = 0; k < top_k; ++k)
            resident_count += selected_host[k] >= 0 ? 1 : 0;
        q.memcpy(x_copy,
                 Qwen4ExpGenerationTestAccess::FrontX(b).data(),
                 sizeof(bf16) * hidden)
            .wait_and_throw();
        ops::moe_resident_gate_up(
            ctx, resident_descriptors_dev, selected_dev, x_copy,
            h_reference, top_k, hidden, intermediate, true, true);
        ops::moe_resident_down(
            ctx, resident_descriptors_dev, selected_dev, h_reference,
            down_reference, top_k, hidden, intermediate, true, true);
        ctx.synchronize();
        std::vector<float> graph_down(top_k * hidden);
        std::vector<float> direct_down(top_k * hidden);
        ctx.memcpy_d2h(
            graph_down.data(),
            Qwen4ExpGenerationTestAccess::ResidentDown(b).data(),
            down_graph_bytes);
        ctx.memcpy_d2h(direct_down.data(), down_reference, down_graph_bytes);
        const bool moe_down_parity =
            std::memcmp(graph_down.data(), direct_down.data(),
                        down_graph_bytes) == 0;
        std::cout << "[REAL FRONT GRAPH] front_parity="
                  << (front_parity ? 1 : 0)
                  << " route_parity=" << (route_parity ? 1 : 0)
                  << " resident_count=" << resident_count
                  << " moe_down_parity=" << (moe_down_parity ? 1 : 0)
                  << " handoff_us=" << first.route_to_done_us << '\n';
        all_ok &= front_parity && route_parity && moe_down_parity &&
                  resident_count > 0;

        for (int cpu_us : options.cpu_sweep) {
            ArmStats stats;
            stats.cpu_us = cpu_us;
            for (int w = 0; w < options.warmup; ++w) {
                ++expected_seq;
                const Handoff h = service_replay(
                    ctx, executable, ring, expected_seq, cpu_us,
                    options.host_timeout_ms, false, false);
                if (h.timeout_delta || h.stale_delta || h.mismatch_delta)
                    throw std::runtime_error(
                        "real-front warmup handshake failure");
            }
            for (uint32_t r = 0; r < options.replays; ++r) {
                ++expected_seq;
                const Handoff h = service_replay(
                    ctx, executable, ring, expected_seq, cpu_us,
                    options.host_timeout_ms, false, false);
                stats.launch_to_route.push_back(h.launch_to_route_us);
                stats.route_to_flag.push_back(h.route_to_flag_us);
                stats.route_to_done.push_back(h.route_to_done_us);
                stats.wall.push_back(h.replay_wall_us);
                if (h.timeout_delta || h.stale_delta || h.mismatch_delta) {
                    stats.fresh_ok = false;
                    break;
                }
            }
            stats.measured = stats.route_to_done.size();
            ++expected_seq;
            const Handoff stale = service_replay(
                ctx, executable, ring, expected_seq, cpu_us,
                options.host_timeout_ms, true, false);
            stats.stale_detected =
                stale.timeout_delta == 0 && stale.stale_delta >= 1 &&
                stale.mismatch_delta >= 1;
            print_arm(stats);
            all_ok &= stats.fresh_ok && stats.stale_detected &&
                      stats.measured >= 1000;
        }
        ctx.synchronize();
    }

    ctx.free_device(down_reference);
    ctx.free_device(h_reference);
    ctx.free_device(x_copy);
    ctx.free_device(moe_test_ids_dev);
    ctx.free_device(selected_dev);
    ctx.free_device(resident_descriptors_dev);
    ctx.free_device(graph_route_weights);
    ctx.free_device(graph_route_ids);
    ctx.free_device(device_result);
    ctx.free_device(epoch);
    free_polled_ring(ctx, polled);
    std::cout << "[REAL FRONT GRAPH] verdict all_ok=" << (all_ok ? 1 : 0)
              << '\n';
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!options.real_front_model.empty())
            return run_real_front_model(options);
        Context ctx;
        sycl::queue& q = ctx.queue();
        const sycl::device dev = q.get_device();
        const bool limited =
            dev.has(sycl::aspect::ext_oneapi_limited_graph);
        const bool full = dev.has(sycl::aspect::ext_oneapi_graph);

        std::cout << "[TOKEN GRAPH CAPS] limited=" << (limited ? 1 : 0)
                  << " full=" << (full ? 1 : 0)
                  << " device=" << dev.get_info<sycl::info::device::name>()
                  << '\n';
        if (!limited && !full) {
            std::cout << "[TOKEN GRAPH SKIP] static graph unsupported\n";
            return 2;
        }

        auto ze_ctx =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                q.get_context());
        ze_host_mem_alloc_desc_t host_desc{
            ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
            ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* ring_storage = nullptr;
        const ze_result_t ring_result = zeMemAllocHost(
            ze_ctx, &host_desc, sizeof(HostRing), alignof(HostRing),
            &ring_storage);
        const bool ring_is_ze_uncached =
            ring_result == ZE_RESULT_SUCCESS && ring_storage != nullptr;
        HostRing* ring = ring_is_ze_uncached
            ? static_cast<HostRing*>(ring_storage)
            : static_cast<HostRing*>(ctx.alloc_host(sizeof(HostRing)));
        if (reinterpret_cast<uintptr_t>(ring) % alignof(HostRing) != 0) {
            throw std::runtime_error("host ring is not 64-byte aligned");
        }
        DeviceEpoch* epoch =
            static_cast<DeviceEpoch*>(ctx.alloc_device(sizeof(DeviceEpoch)));
        float* device_result =
            static_cast<float*>(ctx.alloc_device(sizeof(float) * kResultCount));
        std::memset(ring, 0, sizeof(HostRing));
        q.memset(epoch, 0, sizeof(DeviceEpoch)).wait_and_throw();
        q.memset(device_result, 0, sizeof(float) * kResultCount)
            .wait_and_throw();

        std::cout << "[TOKEN GRAPH CONFIG] replays=" << options.replays
                  << " warmup=" << options.warmup
                  << " spin_max=" << options.spin_max
                  << " host_timeout_ms=" << options.host_timeout_ms
                  << " cpu_sweep=";
        for (size_t i = 0; i < options.cpu_sweep.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << options.cpu_sweep[i];
        }
        std::cout << '\n';
        std::cout
            << "[TOKEN GRAPH RING] allocation="
            << (ring_is_ze_uncached
                    ? "zeMemAllocHost_uncached_bias"
                    : "aila_context_alloc_host_fallback")
            << " ring_bytes=" << sizeof(HostRing)
            << " device_access=uncached_l1_l3"
               " copy=copy_from_mapped_kernel"
               " bounded_device_spin=1"
               " release=event_status_then_queue_empty_fallback\n";

        ctx.synchronize();
        uint32_t expected_seq = 0;
        bool all_ok = true;
        {
            ex::command_graph<ex::graph_state::modifiable> recording(
                q.get_context(), q.get_device());
            recording.begin_recording(q);
            try {
                submit_route_publish(q, ring, epoch);
                submit_device_wait(q, ring, epoch, options.spin_max);
                submit_copy_from_mapped(q, ring, device_result);
                submit_bitwise_verify(q, ring, epoch, device_result);
                recording.end_recording(q);
            } catch (...) {
                recording.end_recording(q);
                throw;
            }
            auto executable = recording.finalize();
            std::cout << "[TOKEN GRAPH CAPTURE] status=ok nodes=4\n";

            for (int cpu_us : options.cpu_sweep) {
                ArmStats stats;
                stats.cpu_us = cpu_us;

                for (int w = 0; w < options.warmup; ++w) {
                    ++expected_seq;
                    const Handoff h = service_replay(
                        ctx, executable, ring, expected_seq, cpu_us,
                        options.host_timeout_ms, false);
                    if (h.timeout_delta || h.stale_delta || h.mismatch_delta) {
                        std::cerr
                            << "[TOKEN GRAPH WARMUP FAIL] seq=" << expected_seq
                            << " cpu_us=" << cpu_us
                            << " timeout_delta=" << h.timeout_delta
                            << " stale_delta=" << h.stale_delta
                            << " mismatch_delta=" << h.mismatch_delta
                            << " timeout_spins="
                            << *reinterpret_cast<volatile uint64_t*>(
                                   &ring->ctrl.timeout_spins)
                            << " route_us=" << h.route_to_flag_us
                            << " done_us=" << h.route_to_done_us << '\n';
                        throw std::runtime_error(
                            "warmup exposed timeout/stale/mismatch");
                    }
                }

                for (uint32_t r = 0; r < options.replays; ++r) {
                    ++expected_seq;
                    const Handoff h = service_replay(
                        ctx, executable, ring, expected_seq, cpu_us,
                        options.host_timeout_ms, false);
                    stats.launch_to_route.push_back(h.launch_to_route_us);
                    stats.route_to_flag.push_back(h.route_to_flag_us);
                    stats.route_to_done.push_back(h.route_to_done_us);
                    stats.wall.push_back(h.replay_wall_us);
                    if (h.timeout_delta || h.stale_delta || h.mismatch_delta) {
                        stats.fresh_ok = false;
                        break;
                    }
                }
                stats.measured = stats.route_to_done.size();

                // One deliberate stale-result replay proves the detector is live.
                ++expected_seq;
                const Handoff stale = service_replay(
                    ctx, executable, ring, expected_seq, cpu_us,
                    options.host_timeout_ms, true);
                stats.stale_detected =
                    stale.timeout_delta == 0 && stale.stale_delta >= 1 &&
                    stale.mismatch_delta >= 1;

                print_arm(stats);
                if (!stats.fresh_ok || !stats.stale_detected ||
                    stats.measured < 1000) {
                    all_ok = false;
                }
            }

            std::cout << "[TOKEN GRAPH STATUS] timeouts="
                      << *volatile_u32(&ring->ctrl.timeout_count)
                      << " stale="
                      << *volatile_u32(&ring->ctrl.stale_count)
                      << " mismatches="
                      << *volatile_u32(&ring->ctrl.mismatch_count)
                      << " host_route_errors="
                      << *volatile_u32(&ring->ctrl.host_route_error_count)
                      << '\n';
            std::cout << "[TOKEN GRAPH VERDICT] all_ok=" << (all_ok ? 1 : 0)
                      << " bit_exact=" << (all_ok ? 1 : 0)
                      << " stale_detector=" << (all_ok ? 1 : 0) << '\n';
        }

        ctx.synchronize();
        ctx.free_device(device_result);
        ctx.free_device(epoch);
        if (ring_is_ze_uncached)
            zeMemFree(ze_ctx, ring);
        else
            ctx.free_host(ring);
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[TOKEN GRAPH ERROR] " << e.what() << '\n';
        return 1;
    }
}
