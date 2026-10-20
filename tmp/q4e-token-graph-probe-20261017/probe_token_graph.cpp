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
#include <functional>
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
    static aila::Qwen4ExpExpertExecutor& Executor(
        Qwen4ExpBnb4Backend& b) {
        return b.expert_executor_;
    }
    static aila::Qwen4ExpHostArena& HostArena(Qwen4ExpBnb4Backend& b) {
        return b.host_arena_;
    }
    static Qwen4ExpLayer& Layer(Qwen4ExpBnb4Backend& b, int layer) {
        return b.layers_.at(static_cast<size_t>(layer));
    }
    static Qwen4ExpBuffers& Buffers(Qwen4ExpBnb4Backend& b) {
        return b.buf_;
    }
    static Bnb4BitLinearScratch& LinearScratch(Qwen4ExpBnb4Backend& b) {
        return b.linear_scratch_;
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
    static int SharedIntermediateSize(Qwen4ExpBnb4Backend& b) {
        return b.cfg_.shared_expert_intermediate_size;
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
    static void LayerFrontRoute(Qwen4ExpBnb4Backend& b, int layer_index,
                                ops::DeltaNetState& state,
                                int* route_ids, float* route_weights) {
        auto& ctx = *b.context_;
        auto& layer = b.layers_.at(static_cast<size_t>(layer_index));
        if (!layer.is_linear)
            throw std::runtime_error("two-layer P3 requires GDN layers");
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
    static void SharedExpert(Qwen4ExpBnb4Backend& b, int layer_index) {
        auto& ctx = *b.context_;
        auto& layer = b.layers_.at(static_cast<size_t>(layer_index));
        b.DenseLinear(ctx, layer.shared_expert_gate,
                      layer.shared_expert_gate_w, b.buf_.x,
                      b.buf_.shared_gate_buf, 1, 1, b.hidden_size_);
        if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                ctx, layer.shared_gate_up_proj, b.buf_.x,
                b.buf_.shared_inter_buf,
                b.cfg_.shared_expert_intermediate_size)) {
            layer.shared_gate_up_proj.ForwardDecodeRows(
                ctx, b.linear_scratch_, b.buf_.x,
                b.buf_.shared_gate_up_buf, 1);
            ops::moe_silu_gate_up_mul_flat(
                ctx, b.buf_.shared_gate_up_buf.data_as<bf16>(),
                b.buf_.shared_inter_buf.data_as<bf16>(), 1,
                b.cfg_.shared_expert_intermediate_size);
        }
        layer.shared_down_proj.ForwardDecodeRows(
            ctx, b.linear_scratch_, b.buf_.shared_inter_buf,
            b.buf_.shared_out_buf, 1);
    }
    static void ResidentExperts(
        Qwen4ExpBnb4Backend& b,
        const ops::MoeExpertSlotDev* resident_descriptors,
        const int* selected, float* resident_down) {
        auto& ctx = *b.context_;
        ops::moe_resident_gate_up(
            ctx, resident_descriptors, selected,
            b.buf_.x.data_as<bf16>(), b.buf_.moe_h.data_as<bf16>(),
            b.cfg_.num_experts_per_tok, b.hidden_size_,
            b.cfg_.moe_intermediate_size, true, true);
        ops::moe_resident_down(
            ctx, resident_descriptors, selected,
            b.buf_.moe_h.data_as<bf16>(), resident_down,
            b.cfg_.num_experts_per_tok, b.hidden_size_,
            b.cfg_.moe_intermediate_size, true, true);
    }
    static void FinalBridge(Qwen4ExpBnb4Backend& b) {
        auto& ctx = *b.context_;
        ops::hc_inject_add(
            ctx, b.buf_.hidden4.data_as<bf16>(),
            b.buf_.y.data_as<bf16>(),
            b.buf_.hc_inject_f32.data_as<float>(), 1,
            b.hidden_size_, 4);
    }
    static void Front(Qwen4ExpBnb4Backend& b, ops::DeltaNetState& state) {
        FrontRoute(b, state, b.router_host_ids_, b.router_host_weights_);
    }
};

namespace {

constexpr int kTopK = 10;
constexpr int kResultWidth = 8;
constexpr int kResultCount = kTopK * kResultWidth;
constexpr int kHidden = 2560;
constexpr int kMaxPacketRows = 16;
constexpr int kPacketVectorElements = 4;
constexpr size_t kPacketValueCount =
    static_cast<size_t>(kMaxPacketRows) * kHidden;
constexpr size_t kPacketRowVectors =
    kHidden / kPacketVectorElements;
constexpr size_t kPacketRankBytes = sizeof(int) * kMaxPacketRows;
constexpr uint32_t kPacketTailSentinel = 0xCDCDCDCDu;
constexpr uint32_t kDefaultSpinMax = 20000;
constexpr uint32_t kPollFlushEvery = 1024;
static_assert(kHidden % kPacketVectorElements == 0);

enum class PacketCopyMode {
    Scalar,
    WgStride,
};

inline const char* packet_copy_mode_name(PacketCopyMode mode) {
    switch (mode) {
    case PacketCopyMode::Scalar: return "scalar";
    case PacketCopyMode::WgStride: return "wg_stride";
    }
    return "unknown";
}

// Consumed by packet copy and merge; only kGateValid permits data movement.
enum : uint32_t {
    kGatePending = 0,
    kGateValid = 1,
    kGateTimeout = 2,
    kGateStale = 3,
    kGateChannelMismatch = 4,
    kGatePacketMismatch = 5,
    kGatePacketShape = 6,
};

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
using Vec4U32 = sycl::vec<uint32_t, kPacketVectorElements>;
static_assert(sizeof(Vec4U32) == 16);

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
    uint32_t channel_mismatch_count;
    uint32_t packet_seq_mismatch_count;
    uint32_t packet_shape_error_count;
    uint32_t gate_abort_count;
    uint64_t reserved[1];
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

struct alignas(64) CpuPacketRing {
    uint32_t ready_seq;
    uint32_t rows;
    uint32_t reserved[14];
    int32_t rank_to_row[kMaxPacketRows];
    float values[kMaxPacketRows * kHidden];
    bf16 x[kHidden];
};

struct alignas(64) DeviceEpoch {
    uint32_t seq;
    uint32_t gate_state;
    // First fail-closed reason in a multi-handoff replay. The first route
    // publication clears it; later handoffs never turn a failed epoch valid.
    uint32_t sticky_abort;
    // Published by the device wait after packet validation. The wg_stride
    // copy uses this instead of rereading packet->rows from host memory.
    uint32_t packet_rows;
    uint32_t reserved[12];
};

static_assert(sizeof(ControlRing) == 64);
static_assert(sizeof(RouteRing) == 128);
static_assert(sizeof(ResultRing) == 384);
static_assert(sizeof(HostRing) == 576);
static_assert(sizeof(CpuPacketRing) == 169088);
static_assert(sizeof(DeviceEpoch) == 64);

inline uint32_t device_load_u32(const uint32_t* ptr) {
    sycl::atomic_ref<
        uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
        value(*const_cast<uint32_t*>(ptr));
    return value.load(sycl::memory_order::acquire);
}

inline void device_store_u32(uint32_t* ptr, uint32_t value) {
    sycl::atomic_ref<
        uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
        target(*ptr);
    target.store(value, sycl::memory_order::release);
}

inline bool epoch_gate_valid(const DeviceEpoch* epoch) {
    return !epoch ||
        (device_load_u32(&epoch->gate_state) == kGateValid &&
         device_load_u32(&epoch->sticky_abort) == 0);
}

bool validate_cpu_packet_shape(const int* rank_to_row, int rows, int top_k,
                               std::string& error) {
    if (top_k < 1 || top_k > kMaxPacketRows) {
        error = "CPU packet top_k is out of range";
        return false;
    }
    if (rows < 0 || rows > kMaxPacketRows || rows > top_k) {
        error = "CPU packet rows are out of range";
        return false;
    }
    if (!rank_to_row) {
        error = "CPU packet rank map is null";
        return false;
    }

    uint32_t seen_rows = 0;
    int mapped_rows = 0;
    for (int rank = 0; rank < top_k; ++rank) {
        const int row = rank_to_row[rank];
        if (row == -1) continue;
        if (row < 0 || row >= rows) {
            error = "CPU packet rank maps outside rows";
            return false;
        }
        const uint32_t bit = 1u << static_cast<uint32_t>(row);
        if (seen_rows & bit) {
            error = "CPU packet rank map has duplicate rows";
            return false;
        }
        seen_rows |= bit;
        ++mapped_rows;
    }
    if (mapped_rows != rows) {
        error = "CPU packet row count does not match rank map";
        return false;
    }
    return true;
}

bool validate_cpu_packet_shape(const CpuPacketRing* packet, int top_k,
                               std::string& error) {
    if (!packet) {
        error = "CPU packet is null";
        return false;
    }
    return validate_cpu_packet_shape(
        packet->rank_to_row, static_cast<int>(packet->rows), top_k, error);
}

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

void submit_route_publish(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                          bool clear_abort = true) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = device_load_u32(&epoch->seq) + 1u;
            device_store_u32(&epoch->seq, seq);
            if (clear_abort)
                device_store_u32(&epoch->sticky_abort, 0);
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
                               const float* weights,
                               bool clear_abort = true) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = device_load_u32(&epoch->seq) + 1u;
            device_store_u32(&epoch->seq, seq);
            if (clear_abort)
                device_store_u32(&epoch->sticky_abort, 0);
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

void submit_copy_x_to_mapped(sycl::queue& q, const bf16* x,
                             CpuPacketRing* packet) {
    auto* dst = reinterpret_cast<uint16_t*>(packet->x);
    q.parallel_for(sycl::range<1>(kHidden), [=](sycl::id<1> id) {
        const uint16_t value =
            reinterpret_cast<const uint16_t*>(x)[id[0]];
        uncached_store(dst + id[0], value);
    });
}

void submit_resident_classify(Context& ctx, const int* ids,
                              const int* slot_table, int* selected,
                              int experts, int top_k) {
    ops::moe_resident_classify_device(
        ctx, ids, slot_table, selected, top_k, experts);
}

void submit_device_wait(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                        CpuPacketRing* packet, int top_k,
                        uint32_t spin_max, bool store_packet_rows = false) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = device_load_u32(&epoch->seq);
            const uint32_t sticky =
                device_load_u32(&epoch->sticky_abort);
            if (sticky != 0) {
                device_store_u32(&epoch->gate_state, sticky);
                return;
            }
            device_store_u32(&epoch->gate_state, kGatePending);

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
                if (flag > seq) break;
                if (++spins >= spin_max) break;
            }
            if (flag != seq) {
                uint32_t* reason_counter = flag > seq
                    ? &ring->ctrl.channel_mismatch_count
                    : &ring->ctrl.timeout_count;
                uncached_store(
                    reason_counter,
                    uncached_load(reason_counter) + 1u);
                if (flag <= seq) {
                    uncached_store(&ring->ctrl.timeout_spins,
                                   static_cast<uint64_t>(spins));
                }
                uncached_store(
                    &ring->ctrl.gate_abort_count,
                    uncached_load(&ring->ctrl.gate_abort_count) + 1u);
                const uint32_t abort_reason =
                    flag > seq ? kGateChannelMismatch : kGateTimeout;
                device_store_u32(
                    &epoch->sticky_abort, abort_reason);
                device_store_u32(
                    &epoch->gate_state, abort_reason);
                return;
            }
            // The matching flag establishes publication of result_seq/result.
            sycl::atomic_fence(sycl::memory_order::acquire,
                               sycl::memory_scope::system);

            const uint32_t route_seq =
                uncached_load(&ring->ctrl.route_seq);
            if (route_seq != seq) {
                uncached_store(
                    &ring->ctrl.channel_mismatch_count,
                    uncached_load(&ring->ctrl.channel_mismatch_count) + 1u);
                uncached_store(
                    &ring->ctrl.gate_abort_count,
                    uncached_load(&ring->ctrl.gate_abort_count) + 1u);
                device_store_u32(
                    &epoch->sticky_abort, kGateChannelMismatch);
                device_store_u32(
                    &epoch->gate_state, kGateChannelMismatch);
                return;
            }

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
                uncached_store(
                    &ring->ctrl.gate_abort_count,
                    uncached_load(&ring->ctrl.gate_abort_count) + 1u);
                device_store_u32(&epoch->sticky_abort, kGateStale);
                device_store_u32(&epoch->gate_state, kGateStale);
                return;
            }

            if (packet) {
                const uint32_t packet_seq =
                    uncached_load(&packet->ready_seq);
                const uint32_t rows = uncached_load(&packet->rows);
                if (packet_seq != seq) {
                    uncached_store(
                        &ring->ctrl.packet_seq_mismatch_count,
                        uncached_load(
                            &ring->ctrl.packet_seq_mismatch_count) + 1u);
                    uncached_store(
                        &ring->ctrl.gate_abort_count,
                        uncached_load(&ring->ctrl.gate_abort_count) + 1u);
                    device_store_u32(
                        &epoch->sticky_abort, kGatePacketMismatch);
                    device_store_u32(
                        &epoch->gate_state, kGatePacketMismatch);
                    return;
                }

                uint32_t seen_rows = 0;
                uint32_t mapped_rows = 0;
                bool shape_ok =
                    top_k >= 1 && top_k <= kMaxPacketRows &&
                    rows <= static_cast<uint32_t>(kMaxPacketRows) &&
                    rows <= static_cast<uint32_t>(top_k);
                for (int rank = 0; shape_ok && rank < top_k; ++rank) {
                    const int row =
                        uncached_load(&packet->rank_to_row[rank]);
                    if (row == -1) continue;
                    if (row < 0 || row >= static_cast<int>(rows)) {
                        shape_ok = false;
                        break;
                    }
                    const uint32_t bit =
                        1u << static_cast<uint32_t>(row);
                    if (seen_rows & bit) {
                        shape_ok = false;
                        break;
                    }
                    seen_rows |= bit;
                    ++mapped_rows;
                }
                if (shape_ok && mapped_rows != rows) shape_ok = false;
                if (!shape_ok) {
                    uncached_store(
                        &ring->ctrl.packet_shape_error_count,
                        uncached_load(
                            &ring->ctrl.packet_shape_error_count) + 1u);
                    uncached_store(
                        &ring->ctrl.gate_abort_count,
                        uncached_load(&ring->ctrl.gate_abort_count) + 1u);
                    device_store_u32(
                        &epoch->sticky_abort, kGatePacketShape);
                    device_store_u32(
                        &epoch->gate_state, kGatePacketShape);
                    return;
                }
                if (store_packet_rows)
                    device_store_u32(&epoch->packet_rows, rows);
            }

            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::device);
            device_store_u32(&epoch->gate_state, kGateValid);
        });
    });
}

// The graph must not contain queue.memcpy: a copy-engine node can split the
// WDDM submission. This kernel reads mapped pinned host USM directly.
void submit_copy_from_mapped(sycl::queue& q, HostRing* ring,
                             DeviceEpoch* epoch, float* device_result) {
    q.parallel_for(sycl::range<1>(kResultCount), [=](sycl::id<1> id) {
        const size_t i = id[0];
        device_result[i] = device_load_u32(&epoch->gate_state) == kGateValid
            ? uncached_load(&ring->result.values[i])
            : 0.0f;
    });
}

void submit_copy_cpu_packet_scalar(sycl::queue& q, CpuPacketRing* packet,
                                   DeviceEpoch* epoch, int* packet_dev,
                                   int top_k) {
    q.parallel_for(
        sycl::range<1>(kMaxPacketRows +
                       size_t(kMaxPacketRows) * kHidden),
        [=](sycl::id<1> id) {
            const size_t i = id[0];
            const bool valid =
                device_load_u32(&epoch->gate_state) == kGateValid;
            if (i < kMaxPacketRows) {
                packet_dev[i] =
                    valid && static_cast<int>(i) < top_k
                    ? uncached_load(&packet->rank_to_row[i])
                    : -1;
            } else if (valid) {
                auto* values = reinterpret_cast<float*>(
                    packet_dev + kMaxPacketRows);
                values[i - kMaxPacketRows] =
                    uncached_load(&packet->values[i - kMaxPacketRows]);
            }
        });
}

// One work-group owns one destination row. The rank map remains scalar; only
// the active payload rows are read from uncached host memory. The inactive
// rows are restored to a device-side sentinel without touching host payload.
void submit_copy_cpu_packet_wg_stride(
    sycl::queue& q, CpuPacketRing* packet, DeviceEpoch* epoch,
    int* packet_dev, int top_k, size_t packet_wg) {
    q.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(size_t(kMaxPacketRows) * packet_wg),
            sycl::range<1>(packet_wg)),
        [=](sycl::nd_item<1> item) {
            const size_t global = item.get_global_id(0);
            const bool valid = epoch_gate_valid(epoch);
            if (global < kMaxPacketRows) {
                packet_dev[global] =
                    valid && static_cast<int>(global) < top_k
                    ? uncached_load(&packet->rank_to_row[global])
                    : -1;
            }
            if (!valid) return;

            const size_t group = item.get_group(0);
            const size_t local = item.get_local_id(0);
            const auto* src =
                reinterpret_cast<const Vec4U32*>(packet->values);
            auto* dst =
                reinterpret_cast<Vec4U32*>(packet_dev + kMaxPacketRows);
            const size_t row_base = group * kPacketRowVectors;
            const uint32_t active_rows =
                device_load_u32(&epoch->packet_rows);
            if (group < active_rows) {
                for (size_t vec = local; vec < kPacketRowVectors;
                     vec += packet_wg) {
                    dst[row_base + vec] =
                        uncached_load(src + row_base + vec);
                }
            } else {
                Vec4U32 sentinel;
#pragma unroll
                for (int lane = 0; lane < kPacketVectorElements; ++lane)
                    sentinel[lane] = kPacketTailSentinel;
                for (size_t vec = local; vec < kPacketRowVectors;
                     vec += packet_wg) {
                    dst[row_base + vec] = sentinel;
                }
            }
        });
}

void submit_copy_cpu_packet(
    sycl::queue& q, CpuPacketRing* packet, DeviceEpoch* epoch,
    int* packet_dev, int top_k,
    PacketCopyMode mode = PacketCopyMode::Scalar,
    size_t packet_wg = 256) {
    if (mode == PacketCopyMode::WgStride) {
        submit_copy_cpu_packet_wg_stride(
            q, packet, epoch, packet_dev, top_k, packet_wg);
    } else {
        submit_copy_cpu_packet_scalar(
            q, packet, epoch, packet_dev, top_k);
    }
}

void submit_gated_ordered_merge(sycl::queue& q, const float* down_acc,
                                const float* weights, bf16* y, int out_features,
                                int top_k, const int* cpu_packet,
                                const DeviceEpoch* epoch) {
    constexpr size_t wg_size = 256;
    const size_t num_wgs =
        (static_cast<size_t>(out_features) + wg_size - 1) / wg_size;
    const auto* cpu_ranks = cpu_packet;
    const auto* cpu_down =
        reinterpret_cast<const float*>(cpu_ranks + kMaxPacketRows);

    q.parallel_for(
        sycl::nd_range<1>(num_wgs * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            // A failed gate is a hard no-op: never read the CPU packet and
            // never overwrite the previous layer output.
            if (!epoch_gate_valid(epoch)) return;
            const size_t row = item.get_global_id(0);
            if (row >= static_cast<size_t>(out_features)) return;

            float total_val = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                const int cpu_row = cpu_ranks[k];
                const float val = cpu_row >= 0
                    ? cpu_down[size_t(cpu_row) * out_features + row]
                    : down_acc[size_t(k) * out_features + row];
                total_val += weights[k] * val;
            }
            y[row] = bf16(total_val);
        });
}

void submit_zero_bytes(sycl::queue& q, void* dst, size_t bytes) {
    auto* out = static_cast<uint8_t*>(dst);
    q.parallel_for(sycl::range<1>(bytes),
                   [=](sycl::id<1> id) { out[id[0]] = 0; });
}

void submit_copy_bytes(sycl::queue& q, void* dst, const void* src,
                       size_t bytes) {
    auto* out = static_cast<uint8_t*>(dst);
    const auto* in = static_cast<const uint8_t*>(src);
    q.parallel_for(sycl::range<1>(bytes),
                   [=](sycl::id<1> id) { out[id[0]] = in[id[0]]; });
}

// Probe-local mirror of Qwen4ExpBnb4Backend.cpp::add_shared_expert_to_y.
// Keeping the vec8 BF16 store order preserves the production rounding.
void submit_add_shared_expert_to_y(sycl::queue& q, bf16* y,
                                   const bf16* shared_out,
                                   const bf16* shared_gate,
                                   const DeviceEpoch* epoch = nullptr) {
    using vec8 = sycl::vec<bf16, 8>;
    constexpr int dim8 = kHidden / 8;
    constexpr int wg = 64;
    constexpr size_t num_wgs = (dim8 + wg - 1) / wg;
    q.parallel_for(
        sycl::nd_range<1>(num_wgs * wg, wg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            if (!epoch_gate_valid(epoch)) return;
            const size_t d8 = item.get_global_id(0);
            if (d8 >= static_cast<size_t>(dim8)) return;
            const float g = static_cast<float>(shared_gate[0]);
            const float gate_act = 1.0f / (1.0f + sycl::native::exp(-g));
            const size_t elem = d8 * 8;
            const vec8 s_vec =
                *reinterpret_cast<const vec8*>(shared_out + elem);
            vec8 y_vec = *reinterpret_cast<const vec8*>(y + elem);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                y_vec[v] = bf16(static_cast<float>(y_vec[v]) +
                                gate_act * static_cast<float>(s_vec[v]));
            }
            *reinterpret_cast<vec8*>(y + elem) = y_vec;
        });
}

void submit_gated_final_bridge(sycl::queue& q,
                               Qwen4ExpBnb4Backend& b,
                               const DeviceEpoch* epoch) {
    auto& buf = Qwen4ExpGenerationTestAccess::Buffers(b);
    auto* hidden = buf.hidden4.data_as<bf16>();
    const auto* y = buf.y.data_as<bf16>();
    const auto* inj = buf.hc_inject_f32.data_as<float>();
    using vec8 = sycl::vec<bf16, 8>;
    constexpr int group_dim = kHidden;
    constexpr int groups = 4;
    constexpr size_t wg = 64;
    constexpr size_t dim8 = group_dim / 8;
    constexpr size_t wgs_per_group = (dim8 + wg - 1) / wg;
    const size_t total_groups = groups;
    q.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(total_groups, wgs_per_group * wg),
            sycl::range<2>(1, wg)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
            if (!epoch_gate_valid(epoch)) return;
            const size_t rg = item.get_group(0);
            const size_t i = item.get_global_id(1);
            if (i >= dim8) return;
            const float inj_val = static_cast<float>(inj[rg]);
            const size_t offset = rg * group_dim + i * 8;
            vec8 vh = *reinterpret_cast<const vec8*>(hidden + offset);
            const vec8 vy = *reinterpret_cast<const vec8*>(y + i * 8);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                vh[v] = bf16(static_cast<float>(vh[v]) +
                              inj_val * static_cast<float>(vy[v]));
            }
            *reinterpret_cast<vec8*>(hidden + offset) = vh;
        });
}

void submit_bitwise_verify(sycl::queue& q, HostRing* ring, DeviceEpoch* epoch,
                           const float* device_result) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            const uint32_t seq = device_load_u32(&epoch->seq);
            bool mismatch =
                device_load_u32(&epoch->gate_state) != kGateValid ||
                uncached_load(&ring->ctrl.result_seq) != seq;
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

void write_host_packet(CpuPacketRing* packet, uint32_t value_seq,
                       uint32_t publish_seq, const int* rank_to_row,
                       int rows) {
    if (!packet || !rank_to_row) return;
    for (int rank = 0; rank < kMaxPacketRows; ++rank)
        packet->rank_to_row[rank] = rank_to_row[rank];
    volatile float* values =
        reinterpret_cast<volatile float*>(packet->values);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < kHidden; ++col) {
            values[row * kHidden + col] =
                float((value_seq * 131u + uint32_t(row * kHidden + col)) %
                      8191u) *
                0.0001f;
        }
    }
    packet->rows = static_cast<uint32_t>(rows);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    *volatile_u32(&packet->ready_seq) = publish_seq;
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
    int cpu_rows = 0;
    double launch_to_route_us = 0.0;
    double route_to_flag_us = 0.0;
    double route_to_done_us = 0.0;
    double replay_wall_us = 0.0;
    uint32_t timeout_delta = 0;
    uint32_t stale_delta = 0;
    uint32_t mismatch_delta = 0;
    uint32_t gate_abort_delta = 0;
};

Handoff service_replay(Context& ctx,
                       ex::command_graph<ex::graph_state::executable>& graph,
                       HostRing* ring,
                       uint32_t expected_seq,
                       int cpu_us,
                       int host_timeout_ms,
                       bool stale_injection,
                       bool synthetic_route = true,
                       CpuPacketRing* cpu_packet = nullptr,
                       const int* cpu_packet_ranks = nullptr,
                       int cpu_packet_rows = 0,
                       const std::function<int(uint32_t)>&
                           cpu_packet_fill = {}) {
    sycl::queue& q = ctx.queue();
    auto& ctrl = ring->ctrl;
    const uint32_t timeout_before = *volatile_u32(&ctrl.timeout_count);
    const uint32_t stale_before = *volatile_u32(&ctrl.stale_count);
    const uint32_t mismatch_before = *volatile_u32(&ctrl.mismatch_count);
    const uint32_t gate_abort_before =
        *volatile_u32(&ctrl.gate_abort_count);

    if (cpu_packet && !cpu_packet_fill) {
        std::string packet_error;
        if (!validate_cpu_packet_shape(
                cpu_packet_ranks, cpu_packet_rows, kTopK, packet_error)) {
            throw std::runtime_error("CPU packet rejected before launch: " +
                                     packet_error);
        }
    }

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
    int actual_cpu_rows = cpu_packet_rows;
    if (cpu_packet) {
        if (cpu_packet_fill) {
            actual_cpu_rows = cpu_packet_fill(value_seq);
            cpu_packet->rows = static_cast<uint32_t>(actual_cpu_rows);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *volatile_u32(&cpu_packet->ready_seq) = expected_seq;
        } else {
            write_host_packet(cpu_packet, value_seq, expected_seq,
                              cpu_packet_ranks, cpu_packet_rows);
        }
        std::string packet_error;
        if (!validate_cpu_packet_shape(cpu_packet, kTopK, packet_error)) {
            std::cerr << "[TOKEN GRAPH PACKET GATE] seq=" << expected_seq
                      << " " << packet_error << '\n';
        }
    }
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
    out.cpu_rows = actual_cpu_rows;
    out.launch_to_route_us = us_between(t_launch, t_route);
    out.route_to_flag_us = us_between(t_route, t_flag);
    out.route_to_done_us = us_between(t_route, t_done);
    out.replay_wall_us = us_between(t_launch, t_end);
    out.timeout_delta =
        *volatile_u32(&ctrl.timeout_count) - timeout_before;
    out.stale_delta = *volatile_u32(&ctrl.stale_count) - stale_before;
    out.mismatch_delta =
        *volatile_u32(&ctrl.mismatch_count) - mismatch_before;
    out.gate_abort_delta =
        *volatile_u32(&ctrl.gate_abort_count) - gate_abort_before;
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

std::array<int, 2> parse_layer_pair(const std::string& text) {
    std::array<int, 2> out{};
    std::stringstream ss(text);
    std::string item;
    int count = 0;
    while (std::getline(ss, item, ',')) {
        if (item.empty() || count >= 2)
            throw std::runtime_error("--layers requires exactly two integers");
        out[count++] = std::stoi(item);
    }
    if (count != 2 || out[0] < 0 || out[1] < 0)
        throw std::runtime_error("--layers requires two non-negative integers");
    return out;
}

PacketCopyMode parse_packet_copy_mode(const std::string& text) {
    if (text == "scalar") return PacketCopyMode::Scalar;
    if (text == "wg_stride") return PacketCopyMode::WgStride;
    throw std::runtime_error(
        "--packet-copy-mode must be scalar or wg_stride");
}

struct Options {
    uint32_t replays = 1000;
    int warmup = 10;
    uint32_t spin_max = kDefaultSpinMax;
    int host_timeout_ms = 2000;
    std::vector<int> cpu_sweep{0, 50, 200, 500};
    std::string real_front_model;
    bool two_layer = false;
    std::array<int, 2> layers{4, 5};
    PacketCopyMode packet_copy_mode = PacketCopyMode::Scalar;
    size_t packet_wg = 256;
};

Options parse_options(int argc, char** argv) {
    Options o;
    bool replays_explicit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--replays") {
            o.replays = static_cast<uint32_t>(std::stoul(next("--replays")));
            replays_explicit = true;
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
        } else if (arg == "--two-layer") {
            o.two_layer = true;
        } else if (arg == "--layers") {
            o.layers = parse_layer_pair(next("--layers"));
        } else if (arg == "--packet-copy-mode") {
            o.packet_copy_mode =
                parse_packet_copy_mode(next("--packet-copy-mode"));
        } else if (arg == "--packet-wg") {
            o.packet_wg = static_cast<size_t>(
                std::stoul(next("--packet-wg")));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: probe_token_graph.exe [--replays N] [--warmup N]\n"
                   "       [--cpu-us 0,50,200,500] [--spin-max N]\n"
                   "       [--host-timeout-ms N] [--real-front-model DIR]\n"
                   "       [--two-layer] [--layers 4,5]\n"
                   "       [--packet-copy-mode scalar|wg_stride]\n"
                   "       [--packet-wg N] (wg_stride only, default 256)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (o.two_layer) {
        if (o.real_front_model.empty())
            throw std::runtime_error("--two-layer requires --real-front-model");
        if (!replays_explicit) o.replays = 4;
        if (o.replays < 1 || o.replays > 16)
            throw std::runtime_error("--two-layer --replays must be in [1, 16]");
    } else if (o.replays < 1000) {
        throw std::runtime_error("--replays must be >= 1000 for P0");
    }
    if (o.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (!o.spin_max) throw std::runtime_error("--spin-max must be > 0");
    if (o.host_timeout_ms <= 0)
        throw std::runtime_error("--host-timeout-ms must be > 0");
    if (o.packet_wg < 32 || o.packet_wg > 1024)
        throw std::runtime_error("--packet-wg must be in [32, 1024]");
    if (!o.two_layer &&
        o.packet_copy_mode != PacketCopyMode::Scalar)
        throw std::runtime_error(
            "--packet-copy-mode wg_stride requires --two-layer");
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

constexpr int kHidden4 = 10240;
constexpr size_t kCpuPacketDeviceBytes =
    sizeof(int) * kMaxPacketRows +
    sizeof(float) * kMaxPacketRows * kHidden;

struct LayerCaptureDev {
    Tensor hidden;
    Tensor route_ids;
    Tensor route_weights;
    Tensor selected;
    Tensor moe_h;
    Tensor resident_down;
    Tensor packet;
    Tensor routed_y;
    Tensor y_shared;
    Tensor shared_gate;
    Tensor shared_inter;
    Tensor shared_out;
};

struct LayerLane {
    int layer = -1;
    Tensor state;
    Tensor history;
    Tensor conv_head;
    Tensor route_ids;
    Tensor route_weights;
    Tensor selected;
    Tensor packet;
    LayerCaptureDev capture;
};

struct HostLayerSnapshot {
    std::vector<uint8_t> hidden;
    std::vector<uint8_t> state;
    std::vector<uint8_t> history;
    std::vector<uint8_t> conv_head;
    std::vector<uint8_t> route_ids;
    std::vector<uint8_t> route_weights;
    std::vector<uint8_t> selected;
    std::vector<uint8_t> moe_h;
    std::vector<uint8_t> resident_down;
    std::vector<uint8_t> packet;
    std::vector<uint8_t> routed_y;
    std::vector<uint8_t> y_shared;
    std::vector<uint8_t> shared_gate;
    std::vector<uint8_t> shared_inter;
    std::vector<uint8_t> shared_out;
    int cpu_rows = 0;
};

void allocate_layer_capture(Context& ctx, LayerCaptureDev& capture,
                            int intermediate, int top_k) {
    capture.hidden = Tensor::allocate(
        ctx, {kHidden4}, dnnl::memory::data_type::bf16);
    capture.route_ids = Tensor::allocate(
        ctx, {kTopK}, dnnl::memory::data_type::s32);
    capture.route_weights = Tensor::allocate(
        ctx, {kTopK}, dnnl::memory::data_type::f32);
    capture.selected = Tensor::allocate(
        ctx, {kMaxPacketRows}, dnnl::memory::data_type::s32);
    capture.moe_h = Tensor::allocate(
        ctx, {int64_t(top_k), int64_t(intermediate)},
        dnnl::memory::data_type::bf16);
    capture.resident_down = Tensor::allocate(
        ctx, {int64_t(top_k), int64_t(kHidden)},
        dnnl::memory::data_type::f32);
    capture.packet = Tensor::allocate(
        ctx, {int64_t(kCpuPacketDeviceBytes)},
        dnnl::memory::data_type::u8);
    capture.routed_y = Tensor::allocate(
        ctx, {kHidden}, dnnl::memory::data_type::bf16);
    capture.y_shared = Tensor::allocate(
        ctx, {kHidden}, dnnl::memory::data_type::bf16);
    capture.shared_gate = Tensor::allocate(
        ctx, {1}, dnnl::memory::data_type::bf16);
    capture.shared_inter = Tensor::allocate(
        ctx, {int64_t(intermediate)}, dnnl::memory::data_type::bf16);
    capture.shared_out = Tensor::allocate(
        ctx, {kHidden}, dnnl::memory::data_type::bf16);
}

void allocate_layer_lane(Context& ctx, LayerLane& lane, int layer,
                         const ops::DeltaNetGeometry& geom,
                         int intermediate, int top_k) {
    lane.layer = layer;
    lane.state = Tensor::allocate(
        ctx, {int64_t(geom.v_heads) * geom.head_k_dim * geom.head_v_dim},
        dnnl::memory::data_type::f32);
    lane.history = Tensor::allocate(
        ctx, {int64_t(geom.conv_rows) * geom.qkv_dim},
        dnnl::memory::data_type::f32);
    lane.conv_head = Tensor::allocate(
        ctx, {ops::delta_net_conv_head_device_slots},
        dnnl::memory::data_type::s32);
    lane.route_ids = Tensor::allocate(
        ctx, {kTopK}, dnnl::memory::data_type::s32);
    lane.route_weights = Tensor::allocate(
        ctx, {kTopK}, dnnl::memory::data_type::f32);
    lane.selected = Tensor::allocate(
        ctx, {kMaxPacketRows}, dnnl::memory::data_type::s32);
    lane.packet = Tensor::allocate(
        ctx, {int64_t(kCpuPacketDeviceBytes)},
        dnnl::memory::data_type::u8);
    allocate_layer_capture(ctx, lane.capture, intermediate, top_k);
}

void reset_layer_lane(Context& ctx, LayerLane& lane) {
    std::vector<float> initial_state(lane.state.numel());
    std::vector<float> initial_history(lane.history.numel());
    for (size_t j = 0; j < initial_state.size(); ++j) {
        initial_state[j] =
            float(int((j * 17 + lane.layer * 31 + 5) % 101) - 50) *
            0.0002f;
    }
    for (size_t j = 0; j < initial_history.size(); ++j) {
        initial_history[j] =
            float(int((j * 7 + lane.layer * 13 + 3) % 101) - 50) *
            0.002f;
    }
    const std::array<int, 3> initial_head{
        2, 2, 0};
    ctx.memcpy_h2d(lane.state.data(), initial_state.data(),
                   lane.state.size_bytes());
    ctx.memcpy_h2d(lane.history.data(), initial_history.data(),
                   lane.history.size_bytes());
    ctx.memcpy_h2d(lane.conv_head.data(), initial_head.data(),
                   lane.conv_head.size_bytes());
    ctx.queue()
        .memset(lane.selected.data(), 0xff,
                lane.selected.size_bytes())
        .wait_and_throw();
}

void seed_hidden4(Context& ctx, Qwen4ExpBnb4Backend& b, int seed) {
    std::array<bf16, kHidden4> input{};
    for (int j = 0; j < kHidden4; ++j) {
        input[j] =
            bf16(float(int((j * 11 + seed * 3) % 103) - 51) * .015f);
    }
    ctx.memcpy_h2d(
        Qwen4ExpGenerationTestAccess::FrontHidden(b).data(),
        input.data(), sizeof(input));
}

std::vector<uint8_t> read_tensor_bytes(Context& ctx, Tensor& tensor) {
    std::vector<uint8_t> bytes(tensor.size_bytes());
    if (!bytes.empty())
        ctx.memcpy_d2h(bytes.data(), tensor.data(), bytes.size());
    return bytes;
}

HostLayerSnapshot snapshot_layer(Context& ctx, LayerLane& lane,
                                 int cpu_rows) {
    HostLayerSnapshot out;
    out.hidden = read_tensor_bytes(ctx, lane.capture.hidden);
    out.state = read_tensor_bytes(ctx, lane.state);
    out.history = read_tensor_bytes(ctx, lane.history);
    out.conv_head = read_tensor_bytes(ctx, lane.conv_head);
    out.route_ids = read_tensor_bytes(ctx, lane.capture.route_ids);
    out.route_weights =
        read_tensor_bytes(ctx, lane.capture.route_weights);
    out.selected = read_tensor_bytes(ctx, lane.capture.selected);
    out.moe_h = read_tensor_bytes(ctx, lane.capture.moe_h);
    out.resident_down =
        read_tensor_bytes(ctx, lane.capture.resident_down);
    out.packet = read_tensor_bytes(ctx, lane.capture.packet);
    out.routed_y = read_tensor_bytes(ctx, lane.capture.routed_y);
    out.y_shared = read_tensor_bytes(ctx, lane.capture.y_shared);
    out.shared_gate =
        read_tensor_bytes(ctx, lane.capture.shared_gate);
    out.shared_inter =
        read_tensor_bytes(ctx, lane.capture.shared_inter);
    out.shared_out = read_tensor_bytes(ctx, lane.capture.shared_out);
    out.cpu_rows = cpu_rows;
    return out;
}

bool same_bytes(const std::vector<uint8_t>& a,
                const std::vector<uint8_t>& b);

void compare_field(bool& all, const char* field, int token, int layer,
                   const std::vector<uint8_t>& direct,
                   const std::vector<uint8_t>& graph) {
    if (same_bytes(direct, graph)) return;
    all = false;
    std::cerr << "[TWO LAYER PARITY FAIL] token=" << token
              << " layer=" << layer << " field=" << field
              << " direct_bytes=" << direct.size()
              << " graph_bytes=" << graph.size() << '\n';
}

struct LayerParity {
    bool hidden = true;
    bool state = true;
    bool history = true;
    bool conv_head = true;
    bool routes = true;
    bool selected = true;
    bool moe_h = true;
    bool resident_down = true;
    bool packet = true;
    bool routed_y = true;
    bool shared = true;
    bool y_shared = true;
    bool cpu_rows = true;
};

bool packet_snapshots_match_wg_stride(
    const HostLayerSnapshot& direct,
    const HostLayerSnapshot& graph, std::string& detail) {
    if (direct.cpu_rows != graph.cpu_rows) {
        detail = "region=active_rows direct_rows=" +
            std::to_string(direct.cpu_rows) +
            " graph_rows=" + std::to_string(graph.cpu_rows);
        return false;
    }
    if (direct.cpu_rows < 0 || direct.cpu_rows > kMaxPacketRows) {
        detail = "region=active_rows invalid_rows=" +
            std::to_string(direct.cpu_rows);
        return false;
    }

    const size_t active_values =
        static_cast<size_t>(direct.cpu_rows) * kHidden;
    const size_t active_bytes = active_values * sizeof(float);
    const size_t required_bytes = kPacketRankBytes + active_bytes;
    const size_t full_packet_bytes =
        kPacketRankBytes + kPacketValueCount * sizeof(float);
    if (direct.packet.size() < required_bytes ||
        graph.packet.size() < full_packet_bytes ||
        direct.packet.size() < full_packet_bytes) {
        detail = "region=buffer_size direct_bytes=" +
            std::to_string(direct.packet.size()) +
            " graph_bytes=" + std::to_string(graph.packet.size()) +
            " active_required_bytes=" +
            std::to_string(required_bytes) +
            " full_required_bytes=" +
            std::to_string(full_packet_bytes);
        return false;
    }

    for (size_t rank = 0; rank < kMaxPacketRows; ++rank) {
        int32_t direct_rank = 0;
        int32_t graph_rank = 0;
        std::memcpy(
            &direct_rank, direct.packet.data() +
                rank * sizeof(int32_t), sizeof(direct_rank));
        std::memcpy(
            &graph_rank, graph.packet.data() +
                rank * sizeof(int32_t), sizeof(graph_rank));
        if (direct_rank != graph_rank) {
            detail = "region=rank_map rank=" +
                std::to_string(rank) +
                " byte_offset=" +
                std::to_string(rank * sizeof(int32_t)) +
                " direct=" + std::to_string(direct_rank) +
                " graph=" + std::to_string(graph_rank);
            return false;
        }
    }

    for (size_t i = 0; i < active_values; ++i) {
        uint32_t direct_bits = 0;
        uint32_t graph_bits = 0;
        const size_t value_offset = kPacketRankBytes + i * sizeof(float);
        std::memcpy(
            &direct_bits, direct.packet.data() + value_offset,
            sizeof(direct_bits));
        std::memcpy(
            &graph_bits, graph.packet.data() + value_offset,
            sizeof(graph_bits));
        if (direct_bits != graph_bits) {
            const size_t row = i / kHidden;
            const size_t column = i % kHidden;
            std::ostringstream msg;
            msg << "region=active_row row=" << row
                << " column=" << column
                << " byte_offset=" << value_offset
                << " direct_bits=0x" << std::hex
                << std::setw(8) << std::setfill('0') << direct_bits
                << " graph_bits=0x" << std::setw(8)
                << std::setfill('0') << graph_bits;
            detail = msg.str();
            return false;
        }
    }

    // Inactive rows are not copied from host memory. They must instead retain
    // the device sentinel; this catches any out-of-bounds payload write.
    for (size_t i = active_values; i < kPacketValueCount; ++i) {
        uint32_t graph_bits = 0;
        const size_t value_offset = kPacketRankBytes + i * sizeof(float);
        std::memcpy(
            &graph_bits, graph.packet.data() + value_offset,
            sizeof(graph_bits));
        if (graph_bits != kPacketTailSentinel) {
            const size_t row = i / kHidden;
            const size_t column = i % kHidden;
            std::ostringstream msg;
            msg << "region=inactive_tail row=" << row
                << " column=" << column
                << " byte_offset=" << value_offset
                << " expected_sentinel=0x" << std::hex
                << std::setw(8) << std::setfill('0')
                << kPacketTailSentinel << " graph_bits=0x"
                << std::setw(8) << std::setfill('0') << graph_bits;
            detail = msg.str();
            return false;
        }
    }

    std::ostringstream msg;
    msg << "rank_map=exact active_rows=" << direct.cpu_rows
        << " active_values=" << active_values
        << " active_bytes=" << active_bytes
        << " inactive_bytes="
        << (kPacketValueCount - active_values) * sizeof(float)
        << " tail_sentinel=exact";
    detail = msg.str();
    return true;
}

void compare_layer_snapshots(int token, int layer,
                             const HostLayerSnapshot& direct,
                             const HostLayerSnapshot& graph,
                             LayerParity& parity,
                             PacketCopyMode packet_copy_mode =
                                 PacketCopyMode::Scalar) {
    compare_field(parity.hidden, "hidden4", token, layer,
                  direct.hidden, graph.hidden);
    compare_field(parity.state, "linear_state", token, layer,
                  direct.state, graph.state);
    compare_field(parity.history, "linear_conv_state", token, layer,
                  direct.history, graph.history);
    compare_field(parity.conv_head, "conv_head_dev", token, layer,
                  direct.conv_head, graph.conv_head);
    compare_field(parity.routes, "route", token, layer,
                  direct.route_ids, graph.route_ids);
    compare_field(parity.routes, "route_weights", token, layer,
                  direct.route_weights, graph.route_weights);
    compare_field(parity.selected, "selected", token, layer,
                  direct.selected, graph.selected);
    compare_field(parity.moe_h, "moe_h", token, layer,
                  direct.moe_h, graph.moe_h);
    compare_field(parity.resident_down, "resident_down", token, layer,
                  direct.resident_down, graph.resident_down);
    if (packet_copy_mode == PacketCopyMode::Scalar) {
        compare_field(parity.packet, "cpu_packet", token, layer,
                      direct.packet, graph.packet);
    } else {
        std::string packet_detail;
        if (!packet_snapshots_match_wg_stride(
                direct, graph, packet_detail)) {
            parity.packet = false;
            std::cerr << "[TWO LAYER PARITY FAIL] token=" << token
                      << " layer=" << layer << " field=cpu_packet "
                      << packet_detail << '\n';
        } else {
            std::cout << "[TWO LAYER PACKET PARITY] token=" << token
                      << " layer=" << layer << ' ' << packet_detail
                      << '\n';
        }
    }
    compare_field(parity.routed_y, "routed_merge", token, layer,
                  direct.routed_y, graph.routed_y);
    compare_field(parity.shared, "shared_gate", token, layer,
                  direct.shared_gate, graph.shared_gate);
    compare_field(parity.shared, "shared_inter", token, layer,
                  direct.shared_inter, graph.shared_inter);
    compare_field(parity.shared, "shared_out", token, layer,
                  direct.shared_out, graph.shared_out);
    compare_field(parity.y_shared, "y_after_shared", token, layer,
                  direct.y_shared, graph.y_shared);
    if (direct.cpu_rows != graph.cpu_rows) {
        parity.cpu_rows = false;
        std::cerr << "[TWO LAYER PARITY FAIL] token=" << token
                  << " layer=" << layer << " field=cpu_rows"
                  << " direct=" << direct.cpu_rows
                  << " graph=" << graph.cpu_rows << '\n';
    }
}

struct PacketServiceResult {
    int cpu_rows = 0;
    bool shape_ok = true;
    std::string error;
};

PacketServiceResult fill_cpu_packet_for_layer(
    aila::Qwen4ExpExpertExecutor& executor,
    const aila::Qwen4ExpHostArena& arena,
    aila::Qwen4ExpAdaptiveCache& adaptive, int layer,
    const int* route_ids, int top_k, CpuPacketRing* packet) {
    PacketServiceResult out;
    std::array<int, kMaxPacketRows> cold_ids{};
    std::array<int, kMaxPacketRows> cold_ranks{};
    int cold_count = 0;
    for (int rank = 0; rank < top_k; ++rank) {
        const int expert = route_ids[rank];
        if (adaptive.GetSlot(layer, expert) >= 0) continue;
        cold_ids[cold_count] = expert;
        cold_ranks[cold_count] = rank;
        ++cold_count;
    }
    if (cold_count == 0) {
        std::fill_n(packet->rank_to_row, kMaxPacketRows, -1);
        std::memset(packet->values, 0,
                    sizeof(float) * kMaxPacketRows * kHidden);
        packet->rows = 0;
        out.cpu_rows = 0;
    } else {
        out.cpu_rows = executor.RunGraphColdPacket(
            arena, layer, cold_ids.data(), cold_ranks.data(),
            cold_count, packet->x, packet->rank_to_row,
            packet->values);
        packet->rows = static_cast<uint32_t>(out.cpu_rows);
        if (out.cpu_rows < kMaxPacketRows) {
            std::memset(
                packet->values +
                    size_t(out.cpu_rows) * kHidden, 0,
                sizeof(float) * size_t(kMaxPacketRows - out.cpu_rows) *
                    kHidden);
        }
    }
    out.shape_ok =
        validate_cpu_packet_shape(packet, top_k, out.error);
    return out;
}

void publish_cpu_packet(CpuPacketRing* packet, uint32_t seq) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    *volatile_u32(&packet->ready_seq) = seq;
}

void submit_capture_layer_tail(sycl::queue& q, LayerLane& lane,
                               Qwen4ExpBnb4Backend& b, int top_k,
                               int intermediate) {
    auto& buf = Qwen4ExpGenerationTestAccess::Buffers(b);
    submit_copy_bytes(
        q, lane.capture.hidden.data(), buf.hidden4.data(),
        sizeof(bf16) * kHidden4);
    submit_copy_bytes(
        q, lane.capture.route_ids.data(), lane.route_ids.data(),
        sizeof(int) * kTopK);
    submit_copy_bytes(
        q, lane.capture.route_weights.data(), lane.route_weights.data(),
        sizeof(float) * kTopK);
    submit_copy_bytes(
        q, lane.capture.selected.data(), lane.selected.data(),
        sizeof(int) * kMaxPacketRows);
    submit_copy_bytes(
        q, lane.capture.moe_h.data(), buf.moe_h.data(),
        sizeof(bf16) * size_t(top_k) * intermediate);
    submit_copy_bytes(
        q, lane.capture.resident_down.data(),
        Qwen4ExpGenerationTestAccess::ResidentDown(b).data(),
        sizeof(float) * size_t(top_k) * kHidden);
    submit_copy_bytes(
        q, lane.capture.packet.data(), lane.packet.data(),
        kCpuPacketDeviceBytes);
    submit_copy_bytes(
        q, lane.capture.y_shared.data(), buf.y.data(),
        sizeof(bf16) * kHidden);
    submit_copy_bytes(
        q, lane.capture.shared_gate.data(), buf.shared_gate_buf.data(),
        sizeof(bf16));
    submit_copy_bytes(
        q, lane.capture.shared_inter.data(), buf.shared_inter_buf.data(),
        sizeof(bf16) * intermediate);
    submit_copy_bytes(
        q, lane.capture.shared_out.data(), buf.shared_out_buf.data(),
        sizeof(bf16) * kHidden);
}

ops::DeltaNetState layer_state(LayerLane& lane, int host_head) {
    return ops::DeltaNetState{
        lane.state.data_as<float>(), lane.history.data_as<float>(),
        host_head, false, lane.conv_head.data_as<int>()};
}

void submit_graph_layer(
    Context& ctx, Qwen4ExpBnb4Backend& b, LayerLane& lane,
    HostRing* ring, DeviceEpoch* epoch, CpuPacketRing* cpu_packet,
    const ops::MoeExpertSlotDev* resident_descriptors,
    const int* layer_slot_table, int top_k, int intermediate,
    int experts, uint32_t spin_max, bool clear_abort,
    PacketCopyMode packet_copy_mode, size_t packet_wg) {
    sycl::queue& q = ctx.queue();
    auto& buf = Qwen4ExpGenerationTestAccess::Buffers(b);
    auto state = layer_state(lane, 2);
    Qwen4ExpGenerationTestAccess::LayerFrontRoute(
        b, lane.layer, state, lane.route_ids.data_as<int>(),
        lane.route_weights.data_as<float>());
    submit_copy_x_to_mapped(q, buf.x.data_as<bf16>(), cpu_packet);
    submit_real_route_publish(
        q, ring, epoch, lane.route_ids.data_as<int>(),
        lane.route_weights.data_as<float>(), clear_abort);
    submit_resident_classify(
        ctx, lane.route_ids.data_as<int>(), layer_slot_table,
        lane.selected.data_as<int>(), experts, top_k);
    submit_zero_bytes(
        q, buf.moe_h.data(),
        sizeof(bf16) * size_t(top_k) * intermediate);
    submit_zero_bytes(
        q, Qwen4ExpGenerationTestAccess::ResidentDown(b).data(),
        sizeof(float) * size_t(top_k) * kHidden);
    Qwen4ExpGenerationTestAccess::ResidentExperts(
        b, resident_descriptors, lane.selected.data_as<int>(),
        Qwen4ExpGenerationTestAccess::ResidentDown(b).data_as<float>());
    Qwen4ExpGenerationTestAccess::SharedExpert(b, lane.layer);
    submit_device_wait(
        q, ring, epoch, cpu_packet, top_k, spin_max,
        packet_copy_mode == PacketCopyMode::WgStride);
    submit_copy_cpu_packet(
        q, cpu_packet, epoch, lane.packet.data_as<int>(), top_k,
        packet_copy_mode, packet_wg);
    submit_gated_ordered_merge(
        q, Qwen4ExpGenerationTestAccess::ResidentDown(b)
               .data_as<float>(),
        lane.route_weights.data_as<float>(), buf.y.data_as<bf16>(),
        kHidden, top_k, lane.packet.data_as<int>(), epoch);
    submit_copy_bytes(
        q, lane.capture.routed_y.data(), buf.y.data(),
        sizeof(bf16) * kHidden);
    submit_add_shared_expert_to_y(
        q, buf.y.data_as<bf16>(), buf.shared_out_buf.data_as<bf16>(),
        buf.shared_gate_buf.data_as<bf16>(), epoch);
    submit_gated_final_bridge(q, b, epoch);
    submit_capture_layer_tail(
        q, lane, b, top_k, intermediate);
}

HostLayerSnapshot run_direct_layer(
    Context& ctx, Qwen4ExpBnb4Backend& b, LayerLane& lane,
    CpuPacketRing* cpu_packet,
    const ops::MoeExpertSlotDev* resident_descriptors,
    const int* layer_slot_table, int top_k, int intermediate,
    int experts) {
    sycl::queue& q = ctx.queue();
    auto& buf = Qwen4ExpGenerationTestAccess::Buffers(b);
    auto state = layer_state(lane, 2);
    Qwen4ExpGenerationTestAccess::LayerFrontRoute(
        b, lane.layer, state, lane.route_ids.data_as<int>(),
        lane.route_weights.data_as<float>());
    ctx.synchronize();

    std::array<int, kMaxPacketRows> route_ids{};
    ctx.memcpy_d2h(route_ids.data(), lane.route_ids.data(),
                   sizeof(int) * kTopK);
    q.memcpy(cpu_packet->x, buf.x.data(), sizeof(bf16) * kHidden)
        .wait_and_throw();
    PacketServiceResult service = fill_cpu_packet_for_layer(
        Qwen4ExpGenerationTestAccess::Executor(b),
        Qwen4ExpGenerationTestAccess::HostArena(b),
        Qwen4ExpGenerationTestAccess::Adaptive(b), lane.layer,
        route_ids.data(), top_k, cpu_packet);
    if (!service.shape_ok)
        throw std::runtime_error(
            "direct CPU packet shape invalid: " + service.error);

    q.memcpy(lane.packet.data(), cpu_packet->rank_to_row,
             sizeof(int) * kMaxPacketRows)
        .wait_and_throw();
    q.memcpy(
         reinterpret_cast<uint8_t*>(lane.packet.data()) +
             sizeof(int) * kMaxPacketRows,
         cpu_packet->values,
         sizeof(float) * kMaxPacketRows * kHidden)
        .wait_and_throw();
    submit_resident_classify(
        ctx, lane.route_ids.data_as<int>(), layer_slot_table,
        lane.selected.data_as<int>(), experts, top_k);
    submit_zero_bytes(
        q, buf.moe_h.data(),
        sizeof(bf16) * size_t(top_k) * intermediate);
    submit_zero_bytes(
        q, Qwen4ExpGenerationTestAccess::ResidentDown(b).data(),
        sizeof(float) * size_t(top_k) * kHidden);
    Qwen4ExpGenerationTestAccess::ResidentExperts(
        b, resident_descriptors, lane.selected.data_as<int>(),
        Qwen4ExpGenerationTestAccess::ResidentDown(b).data_as<float>());
    Qwen4ExpGenerationTestAccess::SharedExpert(b, lane.layer);
    submit_gated_ordered_merge(
        q, Qwen4ExpGenerationTestAccess::ResidentDown(b)
               .data_as<float>(),
        lane.route_weights.data_as<float>(), buf.y.data_as<bf16>(),
        kHidden, top_k, lane.packet.data_as<int>(), nullptr);
    submit_copy_bytes(
        q, lane.capture.routed_y.data(), buf.y.data(),
        sizeof(bf16) * kHidden);
    submit_add_shared_expert_to_y(
        q, buf.y.data_as<bf16>(), buf.shared_out_buf.data_as<bf16>(),
        buf.shared_gate_buf.data_as<bf16>());
    Qwen4ExpGenerationTestAccess::FinalBridge(b);
    submit_capture_layer_tail(
        q, lane, b, top_k, intermediate);
    ctx.synchronize();
    return snapshot_layer(ctx, lane, service.cpu_rows);
}

struct TwoLayerServiceResult {
    std::array<int, 2> cpu_rows{};
    std::array<double, 2> route_to_flag_us{};
    double launch_to_route_us = 0.0;
    double replay_wall_us = 0.0;
    uint32_t timeout_delta = 0;
    uint32_t stale_delta = 0;
    uint32_t mismatch_delta = 0;
    uint32_t gate_abort_delta = 0;
    uint32_t host_route_error_delta = 0;
    uint32_t packet_shape_error_delta = 0;
};

TwoLayerServiceResult service_two_layer_replay(
    Context& ctx,
    ex::command_graph<ex::graph_state::executable>& graph,
    HostRing* ring, uint32_t expected_first_seq, int cpu_us,
    int host_timeout_ms, const std::array<int, 2>& layers,
    CpuPacketRing* cpu_packet, int top_k,
    aila::Qwen4ExpExpertExecutor& executor,
    const aila::Qwen4ExpHostArena& arena,
    aila::Qwen4ExpAdaptiveCache& adaptive,
    bool stale_first_handoff = false) {
    sycl::queue& q = ctx.queue();
    auto& ctrl = ring->ctrl;
    const uint32_t timeout_before = *volatile_u32(&ctrl.timeout_count);
    const uint32_t stale_before = *volatile_u32(&ctrl.stale_count);
    const uint32_t mismatch_before =
        *volatile_u32(&ctrl.mismatch_count);
    const uint32_t gate_abort_before =
        *volatile_u32(&ctrl.gate_abort_count);
    const uint32_t host_route_error_before =
        *volatile_u32(&ctrl.host_route_error_count);
    const uint32_t packet_shape_before =
        *volatile_u32(&ctrl.packet_shape_error_count);

    const auto t_launch = Clock::now();
    auto graph_event = q.ext_oneapi_graph(graph);
    release_graph_submission(graph_event);

    TwoLayerServiceResult out;
    std::string error;
    for (int index = 0; index < 2; ++index) {
        const uint32_t seq = expected_first_seq + uint32_t(index);
        const PollResult route_poll =
            poll_host_word(q, volatile_u32(&ctrl.route_seq), seq,
                           host_timeout_ms, error);
        if (route_poll != PollResult::Reached) {
            try { graph_event.wait_and_throw(); } catch (...) {}
            throw std::runtime_error("two-layer route " +
                                     std::to_string(index) + " " + error);
        }
        const auto t_route = Clock::now();
        if (index == 0)
            out.launch_to_route_us = us_between(t_launch, t_route);
        if (!host_route_matches(
                ring, seq, host_timeout_ms, false, error)) {
            *volatile_u32(&ctrl.host_route_error_count) =
                *volatile_u32(&ctrl.host_route_error_count) + 1u;
            try { graph_event.wait_and_throw(); } catch (...) {}
            throw std::runtime_error(
                "two-layer route payload " + std::to_string(index) +
                " " + error);
        }

        std::array<int, kMaxPacketRows> route_ids{};
        for (int k = 0; k < kTopK; ++k)
            route_ids[k] = ring->route.ids[k];
        burn_cpu_us(cpu_us);
        PacketServiceResult service = fill_cpu_packet_for_layer(
            executor, arena, adaptive, layers[index], route_ids.data(),
            top_k, cpu_packet);
        out.cpu_rows[index] = service.cpu_rows;
        publish_cpu_packet(cpu_packet, seq);
        write_host_result(ring,
                          index == 0 && stale_first_handoff ? seq - 1u : seq,
                          seq);
        const auto t_flag = Clock::now();
        out.route_to_flag_us[index] = us_between(t_route, t_flag);
    }

    const uint32_t final_seq = expected_first_seq + 1u;
    const PollResult done_poll =
        poll_host_word(q, volatile_u32(&ctrl.done_seq), final_seq,
                       host_timeout_ms, error);
    if (done_poll != PollResult::Reached) {
        try { graph_event.wait_and_throw(); } catch (...) {}
        throw std::runtime_error("two-layer done " + error);
    }
    const auto t_done = Clock::now();
    graph_event.wait_and_throw();
    out.replay_wall_us = us_between(t_launch, Clock::now());
    out.timeout_delta =
        *volatile_u32(&ctrl.timeout_count) - timeout_before;
    out.stale_delta = *volatile_u32(&ctrl.stale_count) - stale_before;
    out.mismatch_delta =
        *volatile_u32(&ctrl.mismatch_count) - mismatch_before;
    out.gate_abort_delta =
        *volatile_u32(&ctrl.gate_abort_count) - gate_abort_before;
    out.host_route_error_delta =
        *volatile_u32(&ctrl.host_route_error_count) -
        host_route_error_before;
    out.packet_shape_error_delta =
        *volatile_u32(&ctrl.packet_shape_error_count) -
        packet_shape_before;
    (void)t_done;
    return out;
}

int run_two_layer_model(const Options& options) {
    InferenceEngine engine;
    if (!engine.init(options.real_front_model, 512))
        throw std::runtime_error("two-layer model init failed");
    Context& ctx = Qwen4ExpGenerationTestAccess::Ctx(engine);
    sycl::queue& q = ctx.queue();
    auto& b = Qwen4ExpGenerationTestAccess::Backend(engine);
    if (!q.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !q.get_device().has(sycl::aspect::ext_oneapi_graph))
        throw std::runtime_error("static graph unsupported");

    const int top_k = Qwen4ExpGenerationTestAccess::TopK(b);
    const int experts = Qwen4ExpGenerationTestAccess::ExpertsPerLayer(b);
    const int hidden = Qwen4ExpGenerationTestAccess::HiddenSize(b);
    const int intermediate =
        Qwen4ExpGenerationTestAccess::IntermediateSize(b);
    const int shared_intermediate =
        Qwen4ExpGenerationTestAccess::SharedIntermediateSize(b);
    if (top_k != kTopK || hidden != kHidden ||
        intermediate != 640 || shared_intermediate != 640)
        throw std::runtime_error(
            "P3 requires Qwen4Exp 10/2560/640/640 geometry");
    if (options.layers[0] == options.layers[1])
        throw std::runtime_error("--layers must name two distinct layers");
    for (int layer : options.layers) {
        if (layer < 0 ||
            layer >= static_cast<int>(Qwen4ExpGenerationTestAccess::LayerCount(b)))
            throw std::runtime_error("layer index out of range");
        if (!Qwen4ExpGenerationTestAccess::Layer(b, layer).is_linear)
            throw std::runtime_error("P3 layers must both be GDN");
    }

    auto& adaptive = Qwen4ExpGenerationTestAccess::Adaptive(b);
    auto& executor = Qwen4ExpGenerationTestAccess::Executor(b);
    auto& arena = Qwen4ExpGenerationTestAccess::HostArena(b);
    if (!adaptive.IsAllocated() || !adaptive.DeviceSlotTable() ||
        adaptive.NumLayers() <= std::max(options.layers[0], options.layers[1]))
        throw std::runtime_error("adaptive resident table unavailable");
    if (!executor.cpu_overlap_enabled() || !arena.IsAllocated() ||
        arena.NumLayers() <=
            std::max(options.layers[0], options.layers[1]))
        throw std::runtime_error("CPU arena or worker unavailable");
    if (executor.max_cpu_credits() < top_k)
        throw std::runtime_error(
            "P3 requires AILA_Q4E_CPU_MAX_EXPERTS >= top_k");

    const auto& geom0 = Qwen4ExpGenerationTestAccess::Layer(b, options.layers[0]).geom;
    const auto& geom1 = Qwen4ExpGenerationTestAccess::Layer(b, options.layers[1]).geom;
    if (geom0.conv_rows != 3 || geom1.conv_rows != 3 ||
        geom0.qkv_dim != 10240 || geom1.qkv_dim != 10240)
        throw std::runtime_error("P3 requires the default GDN ring geometry");
    if (!ops::supports_default_grouped_linear_delta_fastpath(
            geom0.q_heads, geom0.v_heads, geom0.head_k_dim,
            geom0.head_v_dim, geom0.kernel, geom0.conv_rows) ||
        !ops::supports_default_grouped_linear_delta_fastpath(
            geom1.q_heads, geom1.v_heads, geom1.head_k_dim,
            geom1.head_v_dim, geom1.kernel, geom1.conv_rows))
        throw std::runtime_error(
            "P3 requires the default grouped GDN fastpath");

    std::array<LayerLane, 2> direct_lanes{};
    std::array<LayerLane, 2> graph_lanes{};
    for (int index = 0; index < 2; ++index) {
        const auto& geom =
            Qwen4ExpGenerationTestAccess::Layer(
                b, options.layers[index])
                .geom;
        allocate_layer_lane(
            ctx, direct_lanes[index], options.layers[index], geom,
            intermediate, top_k);
        allocate_layer_lane(
            ctx, graph_lanes[index], options.layers[index], geom,
            intermediate, top_k);
        reset_layer_lane(ctx, direct_lanes[index]);
        reset_layer_lane(ctx, graph_lanes[index]);
    }
    seed_hidden4(ctx, b, 0);
    ctx.synchronize();

    // Build one global descriptor table. A slot's descriptor must use the
    // layer metadata that owns that slot, not the current probe layer.
    std::vector<ops::MoeExpertSlotDev> resident_descriptors(
        static_cast<size_t>(adaptive.NumGpuSlots()));
    int valid_resident_slots = 0;
    for (int slot = 0; slot < adaptive.NumGpuSlots(); ++slot) {
        const auto entry = adaptive.GetResidentEntry(slot);
        if (entry.layer < 0 || entry.expert_id < 0) continue;
        const auto& meta =
            Qwen4ExpGenerationTestAccess::NestedMeta(
                b, entry.layer);
        resident_descriptors[slot] =
            adaptive.GetResidentSlotDev(slot, &meta);
        ++valid_resident_slots;
    }
    auto* resident_descriptors_dev =
        static_cast<ops::MoeExpertSlotDev*>(ctx.alloc_device(
            resident_descriptors.size() *
            sizeof(ops::MoeExpertSlotDev)));
    q.memcpy(resident_descriptors_dev, resident_descriptors.data(),
             resident_descriptors.size() *
                 sizeof(ops::MoeExpertSlotDev))
        .wait_and_throw();

    PolledRing polled = allocate_polled_ring(ctx);
    HostRing* ring = polled.ptr;
    void* packet_storage = nullptr;
    ze_host_mem_alloc_desc_t packet_desc{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
        ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
    if (zeMemAllocHost(
            polled.ze_ctx, &packet_desc, sizeof(CpuPacketRing),
            alignof(CpuPacketRing), &packet_storage) !=
            ZE_RESULT_SUCCESS ||
        !packet_storage)
        throw std::runtime_error(
            "two-layer uncached CPU packet allocation failed");
    auto* cpu_packet = static_cast<CpuPacketRing*>(packet_storage);
    std::memset(cpu_packet, 0, sizeof(CpuPacketRing));
    DeviceEpoch* epoch =
        static_cast<DeviceEpoch*>(ctx.alloc_device(sizeof(DeviceEpoch)));
    float* device_result = static_cast<float*>(
        ctx.alloc_device(sizeof(float) * kResultCount));
    q.memset(epoch, 0, sizeof(DeviceEpoch)).wait_and_throw();
    q.memset(device_result, 0, sizeof(float) * kResultCount)
        .wait_and_throw();

    std::cout << "[TWO LAYER GRAPH CONFIG] model="
              << options.real_front_model
              << " layers=" << options.layers[0] << ','
              << options.layers[1] << " tokens=" << options.replays
              << " top_k=" << top_k << " hidden=" << hidden
              << " intermediate=" << intermediate
              << " resident_slots=" << adaptive.NumGpuSlots()
              << " valid_slots=" << valid_resident_slots
              << " cpu_max_experts=" << executor.max_cpu_credits()
              << " cpu_us="
              << (options.cpu_sweep.empty() ? 0 : options.cpu_sweep.front())
              << " spin_max=" << options.spin_max
              << " host_timeout_ms=" << options.host_timeout_ms
              << " conv_head=2"
              << " allocation="
              << (polled.ze_owned
                      ? "zeMemAllocHost_uncached_bias"
                      : "malloc_host_fallback")
              << '\n';
    if (options.packet_copy_mode == PacketCopyMode::WgStride) {
        std::cout << "[TWO LAYER PACKET COPY] mode="
                  << packet_copy_mode_name(options.packet_copy_mode)
                  << " wg=" << options.packet_wg << '\n';
    }

    // Warm BNB4 decode scratch and the CPU worker outside capture.
    for (int index = 0; index < 2; ++index) {
        (void)run_direct_layer(
            ctx, b, direct_lanes[index], cpu_packet,
            resident_descriptors_dev,
            adaptive.DeviceSlotTable() +
                direct_lanes[index].layer * experts,
            top_k, intermediate, experts);
    }
    for (int index = 0; index < 2; ++index)
        reset_layer_lane(ctx, direct_lanes[index]);
    seed_hidden4(ctx, b, 0);
    ctx.synchronize();

    const size_t token_count = options.replays;
    std::vector<std::array<HostLayerSnapshot, 2>> direct_snapshots(
        token_count);
    for (size_t token = 0; token < token_count; ++token) {
        for (int index = 0; index < 2; ++index) {
            direct_snapshots[token][index] = run_direct_layer(
                ctx, b, direct_lanes[index], cpu_packet,
                resident_descriptors_dev,
                adaptive.DeviceSlotTable() +
                    direct_lanes[index].layer * experts,
                top_k, intermediate, experts);
        }
    }

    // Graph clone remains at the exact seeded state. Capture only after the
    // direct reference has warmed every path.
    for (int index = 0; index < 2; ++index)
        reset_layer_lane(ctx, graph_lanes[index]);
    seed_hidden4(ctx, b, 0);
    ctx.synchronize();

    auto executable = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(
            q.get_context(), q.get_device());
        recording.begin_recording(q);
        try {
            for (int index = 0; index < 2; ++index) {
                submit_graph_layer(
                    ctx, b, graph_lanes[index], ring, epoch,
                    cpu_packet, resident_descriptors_dev,
                    adaptive.DeviceSlotTable() +
                        graph_lanes[index].layer * experts,
                    top_k, intermediate, experts, options.spin_max,
                    index == 0, options.packet_copy_mode,
                    options.packet_wg);
            }
            submit_copy_from_mapped(
                q, ring, epoch, device_result);
            submit_bitwise_verify(
                q, ring, epoch, device_result);
            recording.end_recording(q);
        } catch (...) {
            recording.end_recording(q);
            throw;
        }
        return recording.finalize();
    }();
    std::cout << "[TWO LAYER GRAPH] capture=ok layers="
              << options.layers[0] << ',' << options.layers[1]
              << " waits=2\n";

    seed_hidden4(ctx, b, 0);
    ctx.synchronize();
    uint32_t expected_first_seq = 1;
    LayerParity layer4_parity;
    LayerParity layer5_parity;
    bool head_sequence_ok = true;
    uint64_t graph_cpu_rows = 0;
    uint64_t direct_cpu_rows = 0;
    uint32_t service_faults = 0;
    std::ostringstream head_path;
    head_path << "2";

    std::vector<std::array<HostLayerSnapshot, 2>> graph_snapshots(
        token_count);
    for (size_t token = 0; token < token_count; ++token) {
        const TwoLayerServiceResult service =
            service_two_layer_replay(
                ctx, executable, ring, expected_first_seq,
                options.cpu_sweep.empty() ? 0 : options.cpu_sweep.front(),
                options.host_timeout_ms, options.layers, cpu_packet,
                top_k, executor, arena, adaptive);
        graph_cpu_rows +=
            static_cast<uint64_t>(service.cpu_rows[0]) +
            static_cast<uint64_t>(service.cpu_rows[1]);
        service_faults += service.timeout_delta;
        service_faults += service.stale_delta;
        service_faults += service.mismatch_delta;
        service_faults += service.gate_abort_delta;
        service_faults += service.host_route_error_delta;
        service_faults += service.packet_shape_error_delta;
        for (int index = 0; index < 2; ++index) {
            graph_snapshots[token][index] =
                snapshot_layer(ctx, graph_lanes[index],
                               service.cpu_rows[index]);
        }
        compare_layer_snapshots(
            static_cast<int>(token), options.layers[0],
            direct_snapshots[token][0], graph_snapshots[token][0],
            layer4_parity, options.packet_copy_mode);
        compare_layer_snapshots(
            static_cast<int>(token), options.layers[1],
            direct_snapshots[token][1], graph_snapshots[token][1],
            layer5_parity, options.packet_copy_mode);

        int actual_head = -1;
        for (int index = 0; index < 2; ++index) {
            const auto& direct = direct_snapshots[token][index].conv_head;
            const auto& graph = graph_snapshots[token][index].conv_head;
            if (direct.size() < 3 * sizeof(int) ||
                graph.size() < 3 * sizeof(int)) {
                head_sequence_ok = false;
                continue;
            }
            int direct_head[3]{};
            int graph_head[3]{};
            std::memcpy(direct_head, direct.data(),
                        sizeof(direct_head));
            std::memcpy(graph_head, graph.data(),
                        sizeof(graph_head));
            const int expected =
                (2 + static_cast<int>(token) + 1) % 3;
            if (direct_head[0] != expected ||
                graph_head[0] != expected ||
                direct_head[1] != (expected + 2) % 3 ||
                graph_head[1] != (expected + 2) % 3 ||
                direct_head[2] != 0 || graph_head[2] != 0)
                head_sequence_ok = false;
            if (index == 0) actual_head = direct_head[0];
        }
        if (actual_head >= 0)
            head_path << "->" << actual_head;
        direct_cpu_rows +=
            static_cast<uint64_t>(direct_snapshots[token][0].cpu_rows) +
            static_cast<uint64_t>(direct_snapshots[token][1].cpu_rows);
        std::cout << "[TWO LAYER TOKEN] token=" << token
                  << " l4_cpu_rows=" << service.cpu_rows[0]
                  << " l5_cpu_rows=" << service.cpu_rows[1]
                  << " l4_handoff_us="
                  << service.route_to_flag_us[0]
                  << " l5_handoff_us="
                  << service.route_to_flag_us[1]
                  << " replay_us=" << service.replay_wall_us << '\n';
        expected_first_seq += 2u;
    }

    const bool all_parity =
        layer4_parity.hidden && layer5_parity.hidden &&
        layer4_parity.state && layer5_parity.state &&
        layer4_parity.history && layer5_parity.history &&
        layer4_parity.routes && layer5_parity.routes &&
        layer4_parity.selected && layer5_parity.selected &&
        layer4_parity.moe_h && layer5_parity.moe_h &&
        layer4_parity.resident_down && layer5_parity.resident_down &&
        layer4_parity.packet && layer5_parity.packet &&
        layer4_parity.routed_y && layer5_parity.routed_y &&
        layer4_parity.shared && layer5_parity.shared &&
        layer4_parity.y_shared && layer5_parity.y_shared &&
        layer4_parity.cpu_rows && layer5_parity.cpu_rows;
    const bool service_ok =
        service_faults == 0 && graph_cpu_rows > 0 &&
        direct_cpu_rows > 0;
    const bool ring_ok =
        *volatile_u32(&ring->ctrl.timeout_count) == 0 &&
        *volatile_u32(&ring->ctrl.stale_count) == 0 &&
        *volatile_u32(&ring->ctrl.mismatch_count) == 0 &&
        *volatile_u32(&ring->ctrl.channel_mismatch_count) == 0 &&
        *volatile_u32(&ring->ctrl.packet_seq_mismatch_count) == 0 &&
        *volatile_u32(&ring->ctrl.packet_shape_error_count) == 0 &&
        *volatile_u32(&ring->ctrl.gate_abort_count) == 0 &&
        *volatile_u32(&ring->ctrl.host_route_error_count) == 0;

    std::cout << "[TWO LAYER GRAPH PARITY]"
              << " hidden_l4=" << (layer4_parity.hidden ? 1 : 0)
              << " hidden_l5=" << (layer5_parity.hidden ? 1 : 0)
              << " state="
              << (layer4_parity.state && layer5_parity.state ? 1 : 0)
              << " history="
              << (layer4_parity.history && layer5_parity.history ? 1 : 0)
              << " heads="
              << (layer4_parity.conv_head &&
                          layer5_parity.conv_head && head_sequence_ok
                      ? 1
                      : 0)
              << " routes="
              << (layer4_parity.routes && layer5_parity.routes ? 1 : 0)
              << " selected="
              << (layer4_parity.selected && layer5_parity.selected
                      ? 1
                      : 0)
              << " moe_h="
              << (layer4_parity.moe_h && layer5_parity.moe_h ? 1 : 0)
              << " resident_down="
              << (layer4_parity.resident_down &&
                          layer5_parity.resident_down
                      ? 1
                      : 0)
              << " packet="
              << (layer4_parity.packet && layer5_parity.packet ? 1 : 0)
              << " routed_merge="
              << (layer4_parity.routed_y && layer5_parity.routed_y
                      ? 1
                      : 0)
              << " shared="
              << (layer4_parity.shared && layer5_parity.shared ? 1 : 0)
              << " y_shared="
              << (layer4_parity.y_shared && layer5_parity.y_shared
                      ? 1
                      : 0)
              << " cpu_rows="
              << (layer4_parity.cpu_rows && layer5_parity.cpu_rows
                      ? 1
                      : 0)
              << '\n';
    std::cout << "[TWO LAYER GRAPH STATUS] timeouts="
              << *volatile_u32(&ring->ctrl.timeout_count)
              << " stale=" << *volatile_u32(&ring->ctrl.stale_count)
              << " mismatches="
              << *volatile_u32(&ring->ctrl.mismatch_count)
              << " channel_mismatches="
              << *volatile_u32(&ring->ctrl.channel_mismatch_count)
              << " packet_seq_mismatches="
              << *volatile_u32(
                     &ring->ctrl.packet_seq_mismatch_count)
              << " packet_shape_errors="
              << *volatile_u32(&ring->ctrl.packet_shape_error_count)
              << " gate_aborts="
              << *volatile_u32(&ring->ctrl.gate_abort_count)
              << " host_route_errors="
              << *volatile_u32(&ring->ctrl.host_route_error_count)
              << " graph_cpu_rows=" << graph_cpu_rows
              << " direct_cpu_rows=" << direct_cpu_rows
              << " heads=" << head_path.str() << '\n';

    auto read_current_hidden = [&]() {
        std::vector<uint8_t> bytes(sizeof(bf16) * kHidden4);
        ctx.memcpy_d2h(
            bytes.data(),
            Qwen4ExpGenerationTestAccess::FrontHidden(b).data(),
            bytes.size());
        return bytes;
    };
    const auto hidden_before_abort = read_current_hidden();
    const TwoLayerServiceResult abort_probe =
        service_two_layer_replay(
            ctx, executable, ring, expected_first_seq,
            options.cpu_sweep.empty() ? 0 : options.cpu_sweep.front(),
            options.host_timeout_ms, options.layers, cpu_packet, top_k,
            executor, arena, adaptive, true);
    const auto hidden_after_abort = read_current_hidden();
    submit_gated_final_bridge(q, b, epoch);
    ctx.synchronize();
    const auto hidden_after_standalone_final =
        read_current_hidden();
    const bool standalone_final_gate_ok =
        same_bytes(hidden_after_abort, hidden_after_standalone_final);
    const bool abort_write_gate_ok =
        abort_probe.gate_abort_delta >= 1 &&
        abort_probe.mismatch_delta >= 1 &&
        standalone_final_gate_ok;
    std::cout << "[TWO LAYER ABORT GATE]"
              << " gate_aborts=" << abort_probe.gate_abort_delta
              << " mismatches=" << abort_probe.mismatch_delta
              << " front_committed="
              << (same_bytes(hidden_before_abort, hidden_after_abort) ? 0 : 1)
              << " standalone_final_gate="
              << (standalone_final_gate_ok ? 1 : 0)
              << " session_poisoned=1\n";

    const bool all_ok =
        all_parity && head_sequence_ok && service_ok && ring_ok &&
        abort_write_gate_ok;
    std::cout << "[TWO LAYER GRAPH] verdict all_ok="
              << (all_ok ? 1 : 0)
              << " bit_exact=" << (all_parity ? 1 : 0)
              << " head_sequence=" << (head_sequence_ok ? 1 : 0)
              << " service_ok=" << (service_ok ? 1 : 0)
              << " ring_ok=" << (ring_ok ? 1 : 0)
              << " abort_write_gate=" << (abort_write_gate_ok ? 1 : 0)
              << '\n';

    ctx.synchronize();
    ctx.free_device(device_result);
    ctx.free_device(epoch);
    ctx.free_device(resident_descriptors_dev);
    zeMemFree(polled.ze_ctx, packet_storage);
    free_polled_ring(ctx, polled);
    return all_ok ? 0 : 1;
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
    void* packet_storage = nullptr;
    ze_host_mem_alloc_desc_t packet_desc{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
        ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
    if (zeMemAllocHost(polled.ze_ctx, &packet_desc,
                       sizeof(CpuPacketRing),
                       alignof(CpuPacketRing), &packet_storage) !=
            ZE_RESULT_SUCCESS ||
        !packet_storage)
        throw std::runtime_error("uncached CPU packet ring allocation failed");
    auto* cpu_packet = static_cast<CpuPacketRing*>(packet_storage);
    std::memset(cpu_packet, 0, sizeof(CpuPacketRing));
    DeviceEpoch* epoch =
        static_cast<DeviceEpoch*>(ctx.alloc_device(sizeof(DeviceEpoch)));
    float* device_result =
        static_cast<float*>(ctx.alloc_device(sizeof(float) * kResultCount));
    int* graph_route_ids = static_cast<int*>(
        ctx.alloc_device(sizeof(int) * kTopK));
    float* graph_route_weights = static_cast<float*>(
        ctx.alloc_device(sizeof(float) * kTopK));
    auto& adaptive = Qwen4ExpGenerationTestAccess::Adaptive(b);
    auto& graph_executor = Qwen4ExpGenerationTestAccess::Executor(b);
    auto& graph_arena = Qwen4ExpGenerationTestAccess::HostArena(b);
    const int top_k = Qwen4ExpGenerationTestAccess::TopK(b);
    const int experts = Qwen4ExpGenerationTestAccess::ExpertsPerLayer(b);
    const int hidden = Qwen4ExpGenerationTestAccess::HiddenSize(b);
    const int intermediate =
        Qwen4ExpGenerationTestAccess::IntermediateSize(b);
    if (!adaptive.IsAllocated() || !adaptive.DeviceSlotTable() ||
        top_k < 1 || top_k > 16)
        throw std::runtime_error("adaptive resident graph state unavailable");
    if (!graph_executor.cpu_overlap_enabled() ||
        !graph_arena.IsAllocated())
        throw std::runtime_error("graph CPU worker or arena unavailable");
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
    const size_t cpu_packet_bytes =
        sizeof(int) * kMaxPacketRows +
        sizeof(float) * kMaxPacketRows * kHidden;
    int* graph_packet_dev =
        static_cast<int*>(ctx.alloc_device(cpu_packet_bytes));
    int* direct_packet_dev =
        static_cast<int*>(ctx.alloc_device(cpu_packet_bytes));
    bf16* merge_y =
        static_cast<bf16*>(ctx.alloc_device(sizeof(bf16) * hidden));
    bf16* merge_y_reference =
        static_cast<bf16*>(ctx.alloc_device(sizeof(bf16) * hidden));
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
    q.memset(epoch, 0, sizeof(DeviceEpoch)).wait_and_throw();
    q.memset(device_result, 0, sizeof(float) * kResultCount).wait_and_throw();
    q.memset(graph_route_ids, 0, sizeof(int) * kTopK).wait_and_throw();
    q.memset(graph_route_weights, 0, sizeof(float) * kTopK).wait_and_throw();
    q.memset(selected_dev, -1, sizeof(int) * 16).wait_and_throw();
    q.memset(merge_y, 0, sizeof(bf16) * hidden).wait_and_throw();
    q.memset(merge_y_reference, 0, sizeof(bf16) * hidden).wait_and_throw();

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

    auto real_cpu_fill = [&](uint32_t) -> int {
        std::array<int, 16> cold_ids{};
        std::array<int, 16> cold_ranks{};
        int cold_count = 0;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = ring->route.ids[rank];
            if (adaptive.GetSlot(0, expert) >= 0) continue;
            cold_ids[cold_count] = expert;
            cold_ranks[cold_count] = rank;
            ++cold_count;
        }
        if (!cold_count) {
            std::fill_n(cpu_packet->rank_to_row, kMaxPacketRows, -1);
            return 0;
        }
        return graph_executor.RunGraphColdPacket(
            graph_arena, 0, cold_ids.data(), cold_ranks.data(),
            cold_count, cpu_packet->x, cpu_packet->rank_to_row,
            cpu_packet->values);
    };

    bool all_ok = true;
    {
        ex::command_graph<ex::graph_state::modifiable> recording(
            q.get_context(), q.get_device());
        recording.begin_recording(q);
        try {
            Qwen4ExpGenerationTestAccess::FrontRoute(
                b, ds, graph_route_ids, graph_route_weights);
            submit_copy_x_to_mapped(
                q,
                Qwen4ExpGenerationTestAccess::FrontX(b).data_as<bf16>(),
                cpu_packet);
            submit_real_route_publish(
                q, ring, epoch, graph_route_ids, graph_route_weights);
            submit_resident_classify(
                ctx, graph_route_ids, adaptive.DeviceSlotTable(),
                selected_dev, experts, top_k);
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
            submit_device_wait(
                q, ring, epoch, cpu_packet, top_k, options.spin_max);
            submit_copy_cpu_packet(
                q, cpu_packet, epoch, graph_packet_dev, top_k);
            submit_gated_ordered_merge(
                q,
                Qwen4ExpGenerationTestAccess::ResidentDown(b)
                    .data_as<float>(),
                graph_route_weights, merge_y, hidden, top_k,
                graph_packet_dev, epoch);
            submit_copy_from_mapped(q, ring, epoch, device_result);
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
            false, false, cpu_packet, nullptr, 0, real_cpu_fill);
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
        q.memcpy(direct_packet_dev, cpu_packet->rank_to_row,
                 sizeof(int) * kMaxPacketRows)
            .wait_and_throw();
        q.memcpy(reinterpret_cast<uint8_t*>(direct_packet_dev) +
                     sizeof(int) * kMaxPacketRows,
                 cpu_packet->values,
                 sizeof(float) * kMaxPacketRows * kHidden)
            .wait_and_throw();
        ops::moe_ordered_accumulate_down_rows(
            ctx,
            Qwen4ExpGenerationTestAccess::ResidentDown(b)
                .data_as<float>(),
            graph_route_weights, merge_y_reference, 1, top_k, hidden,
            nullptr, nullptr, nullptr, direct_packet_dev);
        ctx.synchronize();
        std::vector<bf16> graph_merge_y(hidden);
        std::vector<bf16> direct_merge_y(hidden);
        ctx.memcpy_d2h(graph_merge_y.data(), merge_y,
                       sizeof(bf16) * hidden);
        ctx.memcpy_d2h(direct_merge_y.data(), merge_y_reference,
                       sizeof(bf16) * hidden);
        const bool packet_merge_parity =
            std::memcmp(graph_merge_y.data(), direct_merge_y.data(),
                        sizeof(bf16) * hidden) == 0;
        std::cout << "[REAL FRONT GRAPH] front_parity="
                  << (front_parity ? 1 : 0)
                  << " route_parity=" << (route_parity ? 1 : 0)
                  << " resident_count=" << resident_count
                  << " cpu_rows=" << first.cpu_rows
                  << " moe_down_parity=" << (moe_down_parity ? 1 : 0)
                  << " packet_merge_parity="
                  << (packet_merge_parity ? 1 : 0)
                  << " handoff_us=" << first.route_to_done_us << '\n';
        all_ok &= front_parity && route_parity && moe_down_parity &&
                  packet_merge_parity && first.cpu_rows > 0 &&
                  resident_count + first.cpu_rows == top_k;

        for (int cpu_us : options.cpu_sweep) {
            ArmStats stats;
            stats.cpu_us = cpu_us;
            for (int w = 0; w < options.warmup; ++w) {
                ++expected_seq;
                const Handoff h = service_replay(
                    ctx, executable, ring, expected_seq, cpu_us,
                    options.host_timeout_ms, false, false, cpu_packet,
                    nullptr, 0, real_cpu_fill);
                if (h.timeout_delta || h.stale_delta || h.mismatch_delta)
                    throw std::runtime_error(
                        "real-front warmup handshake failure");
            }
            for (uint32_t r = 0; r < options.replays; ++r) {
                ++expected_seq;
                const Handoff h = service_replay(
                    ctx, executable, ring, expected_seq, cpu_us,
                    options.host_timeout_ms, false, false, cpu_packet,
                    nullptr, 0, real_cpu_fill);
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
                options.host_timeout_ms, true, false, cpu_packet,
                nullptr, 0, real_cpu_fill);
            stats.stale_detected =
                stale.timeout_delta == 0 && stale.stale_delta >= 1 &&
                stale.mismatch_delta >= 1 &&
                stale.gate_abort_delta >= 1;
            print_arm(stats);
            all_ok &= stats.fresh_ok && stats.stale_detected &&
                      stats.measured >= 1000;
        }
        ctx.synchronize();
    }

    std::cout << "[REAL FRONT GRAPH STATUS] timeouts="
              << *volatile_u32(&ring->ctrl.timeout_count)
              << " stale="
              << *volatile_u32(&ring->ctrl.stale_count)
              << " mismatches="
              << *volatile_u32(&ring->ctrl.mismatch_count)
              << " channel_mismatches="
              << *volatile_u32(&ring->ctrl.channel_mismatch_count)
              << " packet_seq_mismatches="
              << *volatile_u32(&ring->ctrl.packet_seq_mismatch_count)
              << " packet_shape_errors="
              << *volatile_u32(&ring->ctrl.packet_shape_error_count)
              << " gate_aborts="
              << *volatile_u32(&ring->ctrl.gate_abort_count)
              << " host_route_errors="
              << *volatile_u32(&ring->ctrl.host_route_error_count)
              << '\n';
    std::cout << "[REAL FRONT GRAPH] verdict all_ok=" << (all_ok ? 1 : 0)
              << '\n';

    ctx.free_device(merge_y_reference);
    ctx.free_device(merge_y);
    ctx.free_device(direct_packet_dev);
    ctx.free_device(graph_packet_dev);
    ctx.free_device(down_reference);
    ctx.free_device(h_reference);
    ctx.free_device(x_copy);
    ctx.free_device(selected_dev);
    ctx.free_device(resident_descriptors_dev);
    ctx.free_device(graph_route_weights);
    ctx.free_device(graph_route_ids);
    ctx.free_device(device_result);
    ctx.free_device(epoch);
    zeMemFree(polled.ze_ctx, cpu_packet);
    free_polled_ring(ctx, polled);
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.two_layer)
            return run_two_layer_model(options);
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
               " fail_closed_gate=valid_abort"
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
                submit_device_wait(
                    q, ring, epoch, nullptr, kTopK, options.spin_max);
                submit_copy_from_mapped(q, ring, epoch, device_result);
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
                    stale.mismatch_delta >= 1 &&
                    stale.gate_abort_delta >= 1;

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
                      << " channel_mismatches="
                      << *volatile_u32(&ring->ctrl.channel_mismatch_count)
                      << " packet_seq_mismatches="
                      << *volatile_u32(
                             &ring->ctrl.packet_seq_mismatch_count)
                      << " packet_shape_errors="
                      << *volatile_u32(&ring->ctrl.packet_shape_error_count)
                      << " gate_aborts="
                      << *volatile_u32(&ring->ctrl.gate_abort_count)
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
