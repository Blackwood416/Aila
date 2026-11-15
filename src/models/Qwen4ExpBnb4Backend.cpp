#include "Qwen4ExpBnb4Backend.hpp"
#include "../ops/CpuTilePool.hpp"
#include "../memory/SessionMemory.hpp"
#include "../ops/QsaIndexerOps.hpp"
#include "../ops/Qwen4ExpAttentionPair.hpp"
#include "../ops/HcDecodeVariant.hpp"
#include "Qwen4ExpTokenGraphStageProf.hpp"
#include "models/Qwen4ExpHotsetPolicy.hpp"
#include "models/Qwen4ExpMoeCopyQueue.hpp"
#include "Qwen4ExpDecodeFusion.hpp"
#include "../utils/EnvUtils.hpp"
#include "../utils/DxgiMemory.hpp"
#include "../profile/Profiling.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;
using bf16 = sycl::ext::oneapi::bfloat16;
using aila::StagingBatch;
using aila::Qwen4ExpAdaptiveCache;
using aila::Qwen4ExpHostArena;

namespace {

template <typename T, typename Compare>
void stable_sort_with_scratch(T* arr, size_t n, T* scratch, Compare comp) {
    if (n <= 1) return;
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t i = 0; i < n; i += 2 * width) {
            size_t mid = std::min(i + width, n);
            size_t end = std::min(i + 2 * width, n);
            size_t p = i, q = mid, k = i;
            while (p < mid && q < end) {
                if (comp(arr[q], arr[p])) {
                    scratch[k++] = arr[q++];
                } else {
                    scratch[k++] = arr[p++];
                }
            }
            while (p < mid) scratch[k++] = arr[p++];
            while (q < end) scratch[k++] = arr[q++];
        }
        std::copy(scratch, scratch + n, arr);
    }
}

void apply_silu_div4(Context& ctx, Tensor& buf, size_t count) {
    if ((count % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        vec8* ptr8 = reinterpret_cast<vec8*>(buf.data());
        size_t count8 = count / 8;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count8), [=](sycl::id<1> idx) {
                vec8 v_in = ptr8[idx];
                vec8 v_out;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float v = static_cast<float>(v_in[k]) * 0.25f;
                    float s = 1.0f / (1.0f + sycl::native::exp(-v));
                    v_out[k] = bf16(v * s);
                }
                ptr8[idx] = v_out;
            });
        });
        return;
    }
    bf16* ptr = static_cast<bf16*>(buf.data());
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            float v = static_cast<float>(ptr[idx]) * 0.25f;
            float s = 1.0f / (1.0f + sycl::native::exp(-v));
            ptr[idx] = bf16(v * s);
        });
    });
}

void apply_sigmoid(Context& ctx, Tensor& buf, size_t count) {
    if ((count % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        vec8* ptr8 = reinterpret_cast<vec8*>(buf.data());
        size_t count8 = count / 8;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count8), [=](sycl::id<1> idx) {
                vec8 v_in = ptr8[idx];
                vec8 v_out;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float v = static_cast<float>(v_in[k]);
                    float s = 1.0f / (1.0f + sycl::native::exp(-v));
                    v_out[k] = bf16(s);
                }
                ptr8[idx] = v_out;
            });
        });
        return;
    }
    bf16* ptr = static_cast<bf16*>(buf.data());
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            float v = static_cast<float>(ptr[idx]);
            float s = 1.0f / (1.0f + sycl::native::exp(-v));
            ptr[idx] = bf16(s);
        });
    });
}

void apply_two_sigmoid_div4(Context& ctx, Tensor& in_buf, Tensor& out_f32, size_t count) {
    const bf16* in_ptr = static_cast<const bf16*>(in_buf.data());
    float* out_ptr = static_cast<float*>(out_f32.data());
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            float v = static_cast<float>(in_ptr[idx]) * 0.25f;
            float s = 1.0f / (1.0f + sycl::native::exp(-v));
            out_ptr[idx] = 2.0f * s;
        });
    });
}

void add_shared_expert_to_y(Context& ctx, Tensor& y, Tensor& shared_out, Tensor& shared_gate, int rows, int hidden_size) {
    bf16* y_ptr = static_cast<bf16*>(y.data());
    const bf16* s_ptr = static_cast<const bf16*>(shared_out.data());
    const bf16* g_ptr = static_cast<const bf16*>(shared_gate.data());

    if ((hidden_size % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = hidden_size / 8;
        constexpr int wg = 64;
        const size_t num_wgs_per_row = (dim8 + wg - 1) / wg;

        if (rows == 1) {
            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_wgs_per_row * wg), sycl::range<1>(wg)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                        const size_t d8 = item.get_global_id(0);
                        if (d8 < static_cast<size_t>(dim8)) {
                            const float g_val = static_cast<float>(g_ptr[0]);
                            const float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
                            const size_t elem_offset = d8 * 8;
                            const vec8 s_vec = *reinterpret_cast<const vec8*>(s_ptr + elem_offset);
                            vec8 y_vec = *reinterpret_cast<const vec8*>(y_ptr + elem_offset);
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                float s_val = static_cast<float>(s_vec[v]);
                                float y_val = static_cast<float>(y_vec[v]);
                                y_vec[v] = bf16(y_val + gate_act * s_val);
                            }
                            *reinterpret_cast<vec8*>(y_ptr + elem_offset) = y_vec;
                        }
                    });
            });
            return;
        }

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(rows), num_wgs_per_row * wg),
                                  sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t r = item.get_group(0);
                    const size_t d8 = item.get_global_id(1);
                    if (d8 < static_cast<size_t>(dim8)) {
                        const float g_val = static_cast<float>(g_ptr[r]);
                        const float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
                        const size_t elem_offset = r * hidden_size + d8 * 8;
                        const vec8 s_vec = *reinterpret_cast<const vec8*>(s_ptr + elem_offset);
                        vec8 y_vec = *reinterpret_cast<const vec8*>(y_ptr + elem_offset);
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            float s_val = static_cast<float>(s_vec[v]);
                            float y_val = static_cast<float>(y_vec[v]);
                            y_vec[v] = bf16(y_val + gate_act * s_val);
                        }
                        *reinterpret_cast<vec8*>(y_ptr + elem_offset) = y_vec;
                    }
                });
        });
        return;
    }

    const size_t total = static_cast<size_t>(rows) * hidden_size;
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t r = idx[0] / hidden_size;
            float g_val = static_cast<float>(g_ptr[r]);
            float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
            float s_val = static_cast<float>(s_ptr[idx[0]]);
            float y_val = static_cast<float>(y_ptr[idx[0]]);
            y_ptr[idx[0]] = bf16(y_val + gate_act * s_val);
        });
    });
}

void dump_hidden_row(Context& ctx, const std::string& path, const Tensor& hidden4, int row_idx, int hidden_dim = 10240) {
    std::vector<bf16> host_buf(hidden_dim);
    const bf16* src = static_cast<const bf16*>(hidden4.data()) + static_cast<size_t>(row_idx) * hidden_dim;
    ctx.memcpy_d2h(host_buf.data(), src, hidden_dim * sizeof(bf16));
    std::ofstream os(path, std::ios::binary);
    if (os) {
        os.write(reinterpret_cast<const char*>(host_buf.data()), hidden_dim * sizeof(bf16));
    }
}

} // namespace

// D1 (opt-in, default off): decode layer-boundary experiments.
// The adaptive decode CPU cold route can be classified and started as soon as
// the router wait returns (instead of inside RunResidentDecode below), giving
// the CPU worker the shared-expert submission block as extra lead time. The
// stash is written in Forward and consumed once by RunResidentDecode for the
// same layer; any mismatch fails closed instead of silently changing rows.
namespace {
static thread_local int d1_stash_layer = -1;
static thread_local int d1_stash_cpu_count = -1;
static thread_local bool d1_stash_valid = false;
static thread_local std::array<int, 16> d1_stash_ids{};
static thread_local std::array<int, 16> d1_stash_ranks{};

// R1 expert-XMX plan staging: one compact block per chunk. The staging buffer
// is sized for this many chunks because a long prefill now splits hot experts
// into <=256-row bands (AILA_Q4E_PREFILL_EXPERT_XMX_MAX_M), which can push a
// single layer past the old 128-chunk bound. The per-chunk block (row bases
// plus slot descriptors) is small next to the shared rows/ranks region.
constexpr int kMaxXmxPlanChunks = 256;
}  // namespace

// ============================================================================
// Session
// ============================================================================

Qwen4ExpBnb4Backend::Session::Session(
    Context& ctx, int capacity,
    std::shared_ptr<const SessionOwner> owner,
    std::size_t maxSnapshots, uint64_t instanceId,
    aila::TokenGraphRuntime* tokenGraphRuntime,
    int kvStreamResidentTokens)
    : context_(&ctx), owner_(std::move(owner)),
      token_graph_runtime_(tokenGraphRuntime),
      session_instance_id_(instanceId),
      capacity_(capacity), snapshots_(maxSnapshots) {
    // INT8 KV is opt-in and default-off; both switches must be present.
    const bool int8_kv = aila::env::read_flag("AILA_KV_QUANT", false) &&
                         aila::env::kv_quant_format_is_int8();
    if (aila::env::read_flag("AILA_KV_QUANT", false)) {
        AILA_LOG_INFO("[Q4E KV] AILA_KV_QUANT=1 format=%s",
                      int8_kv ? "int8" : "fp8(ignored: Qwen4Exp keeps bf16 KV)");
    }
    const bool token_graph_requested =
        aila::env::read_flag("AILA_Q4E_TOKEN_GRAPH", false);
    graph_gdn_state_enabled_ = aila::env::read_flag(
        "AILA_Q4E_GRAPH_GDN_STATE", token_graph_requested);
    if (graph_gdn_state_enabled_) {
        gdn_conv_heads_dev_ = Tensor::allocate(
            ctx, {48, ops::delta_net_conv_head_device_slots},
            dnnl::memory::data_type::s32);
        gdn_conv_heads_host_.assign(48 * ops::delta_net_conv_head_device_slots, 0);
        ctx.queue().memset(gdn_conv_heads_dev_.data(), 0,
                           gdn_conv_heads_dev_.size_bytes());
    }
    layerCaches_.resize(48);
    for (int i = 0; i < 48; ++i) {
        bool is_linear = ((i + 1) % 4 != 0);
        auto& cache = layerCaches_[i];
        if (is_linear) {
            cache.linear_state = Tensor::allocate(ctx, {48, 128, 128}, dnnl::memory::data_type::f32);
            ctx.queue().memset(cache.linear_state.data(), 0, cache.linear_state.size_bytes());
            cache.host_linear_state.assign(48 * 128 * 128, 0.0f);

            cache.linear_conv_state = Tensor::allocate(ctx, {3, 10240}, dnnl::memory::data_type::f32);
            ctx.queue().memset(cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
            cache.host_linear_conv_state.assign(3 * 10240, 0.0f);
            cache.linear_conv_head = 0;
            cache.device_state_dirty = false;
            if (graph_gdn_state_enabled_) {
                cache.linear_conv_head_dev =
                    gdn_conv_heads_dev_.data_as<int>() +
                    i * ops::delta_net_conv_head_device_slots;
            }
        } else {
            const auto kv_dtype = int8_kv ? dnnl::memory::data_type::s8
                                          : dnnl::memory::data_type::bf16;
            const auto kv_stream_plan = aila::memory::qwen4exp_kv_stream_plan(capacity, kvStreamResidentTokens);
            if (kv_stream_plan.enabled) {
                cache.kv_stream_enabled = true;
                kv_host_int8_ = int8_kv;
                const int slot_tokens = kv_stream_plan.residentTokens;
                cache.k = Tensor::allocate(ctx, {2, slot_tokens, 256}, kv_dtype);
                cache.v = Tensor::allocate(ctx, {2, slot_tokens, 256}, kv_dtype);
                int seg_tokens = std::clamp(
                    aila::env::read_int("AILA_Q4E_KV_HOST_SEGMENT_TOKENS", 16384),
                    4096, std::max(4096, capacity));
                seg_tokens = ((seg_tokens + 3) / 4) * 4;
                const int min_seg = (((capacity + ops::QsaKvHostSegments::kMaxSegments - 1) /
                                      ops::QsaKvHostSegments::kMaxSegments + 3) / 4) * 4;
                seg_tokens = std::max(seg_tokens, min_seg);
                seg_tokens = std::min(seg_tokens, capacity);
                const int total_segs = (capacity + seg_tokens - 1) / seg_tokens;
                cache.kv_host_segment_tokens = seg_tokens;
                cache.kv_host_total_segments = total_segs;
                cache.kv_host_last_segment_tokens = capacity - (total_segs - 1) * seg_tokens;
                if (int8_kv) {
                    cache.k_scale = Tensor::allocate(ctx, {2, slot_tokens}, dnnl::memory::data_type::f32);
                    cache.v_scale = Tensor::allocate(ctx, {2, slot_tokens}, dnnl::memory::data_type::f32);
                }
                ops::qsa_kv_stream_init_map(ctx, cache.kv_stream_map, capacity, kv_stream_plan.slotCount);
                if (aila::env::read_flag("AILA_QSA_COMPRESSED_K", true)) {
                    cache.indexer_k_tail_only = true;
                    cache.indexer_k = Tensor::allocate(ctx, {8, 128}, dnnl::memory::data_type::bf16);
                } else {
                    cache.indexer_k = Tensor::allocate_host(ctx, {capacity, 128}, dnnl::memory::data_type::bf16);
                }
            } else {
                cache.k = Tensor::allocate(ctx, {2, capacity, 256}, kv_dtype);
                cache.v = Tensor::allocate(ctx, {2, capacity, 256}, kv_dtype);
                if (int8_kv) {
                    cache.k_scale = Tensor::allocate(ctx, {2, capacity}, dnnl::memory::data_type::f32);
                    cache.v_scale = Tensor::allocate(ctx, {2, capacity}, dnnl::memory::data_type::f32);
                }
                cache.indexer_k = Tensor::allocate(ctx, {capacity, 128}, dnnl::memory::data_type::bf16);
            }
            const int max_blocks = std::max(1, capacity / 4);
            cache.indexer_compressed_k = Tensor::allocate(ctx, {max_blocks, 128}, dnnl::memory::data_type::bf16);
            cache.num_compressed_blocks = 0;
        }
    }
    EnsureKvHostCapacity(ctx, 1);
}

void Qwen4ExpBnb4Backend::Session::EnsureKvHostCapacity(Context& ctx, int requiredTokens) {
    const int needed = std::min(capacity_, std::max(1, requiredTokens));
    if (needed <= kv_host_allocated_tokens_) return;
    int max_covered = capacity_;
    for (auto& cache : layerCaches_) {
        if (!cache.kv_stream_enabled) continue;
        const auto kv_dtype = kv_host_int8_ ? dnnl::memory::data_type::s8
                                            : dnnl::memory::data_type::bf16;
        const int seg_tokens = cache.kv_host_segment_tokens;
        const int total_segs = cache.kv_host_total_segments;
        const int target_segs = std::min(total_segs, (needed + seg_tokens - 1) / seg_tokens);
        while (static_cast<int>(cache.host_k_segs.size()) < target_segs) {
            const int s = static_cast<int>(cache.host_k_segs.size());
            const int rows = (s + 1 == total_segs) ? cache.kv_host_last_segment_tokens : seg_tokens;
            cache.host_k_segs.push_back(Tensor::allocate_host(ctx, {2, rows, 256}, kv_dtype));
            cache.host_v_segs.push_back(Tensor::allocate_host(ctx, {2, rows, 256}, kv_dtype));
            if (kv_host_int8_) {
                Tensor ks = Tensor::allocate_host(ctx, {2, rows}, dnnl::memory::data_type::f32);
                Tensor vs = Tensor::allocate_host(ctx, {2, rows}, dnnl::memory::data_type::f32);
                std::memset(ks.data(), 0, ks.size_bytes());
                std::memset(vs.data(), 0, vs.size_bytes());
                cache.host_k_scale_segs.push_back(std::move(ks));
                cache.host_v_scale_segs.push_back(std::move(vs));
            }
        }
        const int covered = (target_segs == total_segs) ? capacity_ : (target_segs * seg_tokens);
        max_covered = std::min(max_covered, covered);
    }
    kv_host_allocated_tokens_ = max_covered;
}

Qwen4ExpBnb4Backend::Session::~Session() {
    if (token_graph_runtime_) {
        (void)token_graph_runtime_->RetireSession(this);
    }
    if (context_) {
        (void)context_->drain_noexcept();
    }
}

bool Qwen4ExpBnb4Backend::GdnDeviceHeadReadable(
    const Qwen4ExpLayerCache& cache) noexcept {
    return cache.linear_conv_head_dev != nullptr &&
           (cache.linear_conv_head_device_authority ||
            cache.linear_conv_head_device_synced);
}

void Qwen4ExpBnb4Backend::ValidateGdnHeadPath(
    const Session& session, int count) const {
    if (!session.graph_gdn_state_enabled_) return;
    if (count != 1) {
        for (const auto& cache : session.layerCaches_) {
            if (cache.linear_conv_head_device_authority) {
                throw std::invalid_argument(
                    "Qwen4Exp device-authoritative GDN state cannot switch to "
                    "a multi-row/prefill forward before an explicit head "
                    "synchronization");
            }
        }
        return;
    }
    int linear_rows = 0;
    int device_authority_rows = 0;
    for (const auto& cache : session.layerCaches_) {
        if (cache.linear_conv_head_dev == nullptr) continue;
        ++linear_rows;
        device_authority_rows += cache.linear_conv_head_device_authority ? 1 : 0;
    }
    if (device_authority_rows != 0 &&
        device_authority_rows != linear_rows) {
        throw std::invalid_argument(
            "Qwen4Exp graph GDN state has mixed host/device authority");
    }
}

void Qwen4ExpBnb4Backend::SynchronizeGdnDeviceHeadsToHost(
    Context& ctx, Session& session, bool preserve_token_graph) {
    if (!session.gdn_conv_heads_dev_.valid()) return;
    if (!preserve_token_graph) {
        (void)AbortTokenGraph();
        session.InvalidateTokenGraphState();
    }
    sycl::event last_d2h{};
    bool have_last_d2h = false;
    for (size_t i = 0; i < session.layerCaches_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        if (cache.linear_conv_head_dev == nullptr ||
            !cache.linear_conv_head_device_authority) {
            continue;
        }
        int* host_row = session.gdn_conv_heads_host_.data() +
            i * ops::delta_net_conv_head_device_slots;
        last_d2h = ctx.memcpy_d2h_async(
            host_row, cache.linear_conv_head_dev, sizeof(int));
        have_last_d2h = true;
    }
    if (have_last_d2h) last_d2h.wait_and_throw();
    for (size_t i = 0; i < session.layerCaches_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        if (cache.linear_conv_head_dev == nullptr ||
            !cache.linear_conv_head_device_authority) {
            continue;
        }
        int* host_row = session.gdn_conv_heads_host_.data() +
            i * ops::delta_net_conv_head_device_slots;
        cache.linear_conv_head = host_row[0];
        cache.device_state_dirty = true;
        cache.linear_conv_head_device_authority = false;
        cache.linear_conv_head_device_synced = true;
    }
}

void Qwen4ExpBnb4Backend::PublishGdnHostHeadsToDevice(
    Context& ctx, Session& session) {
    if (!session.gdn_conv_heads_dev_.valid()) return;
    (void)AbortTokenGraph();
    session.InvalidateTokenGraphState();
    std::fill(session.gdn_conv_heads_host_.begin(),
              session.gdn_conv_heads_host_.end(), 0);
    for (size_t i = 0; i < session.layerCaches_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        if (cache.linear_conv_head_dev == nullptr) continue;
        int* host_row = session.gdn_conv_heads_host_.data() +
            i * ops::delta_net_conv_head_device_slots;
        host_row[ops::delta_net_conv_head_device_head_slot] =
            cache.linear_conv_head;
        host_row[ops::delta_net_conv_head_device_previous_slot] =
            cache.linear_conv_head;
        host_row[ops::delta_net_conv_head_device_arrivals_slot] = 0;
    }
    ctx.memcpy_h2d(session.gdn_conv_heads_dev_.data(),
                   session.gdn_conv_heads_host_.data(),
                   session.gdn_conv_heads_dev_.size_bytes());
    for (auto& cache : session.layerCaches_) {
        if (cache.linear_conv_head_dev == nullptr) continue;
        cache.linear_conv_head_device_authority = false;
        cache.linear_conv_head_device_synced = true;
    }
}

void Qwen4ExpBnb4Backend::PrepareGdnGraphState(
    Context& ctx, Session& session) {
    if (!session.gdn_conv_heads_dev_.valid()) {
        throw std::invalid_argument(
            "Qwen4Exp graph GDN state was not enabled for this session");
    }
    (void)AbortTokenGraph();
    for (const auto& cache : session.layerCaches_) {
        if (cache.linear_conv_head_device_authority) {
            throw std::invalid_argument(
                "Qwen4Exp graph GDN capture requires host head authority; "
                "synchronize the device heads first");
        }
    }
    PublishGdnHostHeadsToDevice(ctx, session);
}

void Qwen4ExpBnb4Backend::MarkGdnGraphStateDirty(Session& session) {
    for (auto& cache : session.layerCaches_) {
        if (cache.linear_conv_head_dev == nullptr) continue;
        cache.linear_conv_head_device_authority = true;
        cache.linear_conv_head_device_synced = false;
        cache.device_state_dirty = true;
    }
}

// ============================================================================
// Backend Construction & Lifecycle
// ============================================================================

Qwen4ExpBnb4Backend::Qwen4ExpBnb4Backend()
    : sessionOwner_(std::make_shared<SessionOwner>()) {}

Qwen4ExpBnb4Backend::~Qwen4ExpBnb4Backend() {
    (void)token_graph_runtime_.wait_drained_noexcept();
    const bool idle_confirmed =
        context_ && context_->drain_noexcept();
    if (!ReleaseWholeTokenGraph(
            context_, defaultSession_.get(), idle_confirmed)) {
        AILA_LOG_ERROR(
            "[Q4E Graph] whole-token executable drain failed during teardown");
    }
    if (!ReleaseQsaLayerGraph(context_, idle_confirmed)) {
        AILA_LOG_ERROR(
            "[Q4E Graph] P10 executable drain failed during backend teardown");
    }
    if (context_ && xmx_scratch_.impl) ops::moe_expert_xmx_scratch_destroy(*context_, xmx_scratch_);
    if (context_) {
        for (int b = 0; b < 2; ++b) {
            if (xmx_plan_pinned_[b]) context_->free_host(xmx_plan_pinned_[b]);
            if (xmx_plan_dev_[b]) context_->free_device(xmx_plan_dev_[b]);
            xmx_plan_pinned_[b] = nullptr;
            xmx_plan_dev_[b] = nullptr;
        }
    }
    if (export_enabled_ && !export_path_.empty()) {
        // P3 calibration: write the collected decode/prefill counts as a full
        // 24576-key profile in the legacy format before the caches go away.
        try {
            export_policy_.Save(export_path_, export_identity_);
            AILA_LOG_INFO("[Q4E Hotset] adaptive export written: %s",
                export_path_.c_str());
        } catch (const std::exception& e) {
            AILA_LOG_ERROR("[Q4E Hotset] adaptive export failed: %s", e.what());
        }
        export_enabled_ = false;
        export_path_.clear();
    }
    expert_executor_.AbortResidentCold();
    free_pinned_host_buffers();
    if (resident_selected_host_ && context_) context_->free_host(resident_selected_host_);
    if (copy_q_) {
        try { copy_q_->wait(); } catch (...) {}
        copy_q_.reset();
    }
    adaptive_cache_.Release();
    host_arena_.Release();
    if (resident_embedding_) {
        if (resident_embedding_large_pages_) {
            if (context_) context_->free_host(resident_embedding_);
        } else {
#ifdef _WIN32
            VirtualFree(resident_embedding_, 0, MEM_RELEASE);
#else
            std::free(resident_embedding_);
#endif
        }
        resident_embedding_ = nullptr;
    }
}

void Qwen4ExpBnb4Backend::PrepareDecodeVerification(Context& ctx) {
    if (context_ != &ctx) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: verification requires the model Context");
    }
    // Called under the execution-domain lock before verification, never in its hot loop.
    // Keep the bundle alive until backend teardown has drained this Context.
    if (!verification_attention_pair_bundle_)
        verification_attention_pair_bundle_.emplace(ops::prepare_q4e_attention_pair(ctx));
    // Multi-row (2..4 token) CPU verifier scratch, allocated once here so the
    // verification window itself performs zero device allocations. Only the
    // arena verifier consumes these; ordinary decode never touches them.
    const int top_k = cfg_.num_experts_per_tok;
    if (!verify_down_.valid())
        verify_down_ = Tensor::allocate(ctx, {4LL * top_k * hidden_size_}, dnnl::memory::data_type::f32);
    if (!verify_weights_.valid())
        verify_weights_ = Tensor::allocate(ctx, {4LL * top_k}, dnnl::memory::data_type::f32);
}

void Qwen4ExpBnb4Backend::free_pinned_host_buffers() {
    if (!context_) return;
    // Buffer growth already checks Context errors before entering here. Teardown
    // must also release raw buffers after a forward fault without rethrowing the
    // latched async error from a destructor. Always drain the actual queue first.
    try { context_->queue().wait_and_throw(); } catch (...) {}
    if (router_host_ids_) context_->free_host(router_host_ids_);
    if (router_host_weights_) context_->free_host(router_host_weights_);
    router_host_ids_ = nullptr;
    router_host_weights_ = nullptr;
    if (embed_host_ids_) context_->free_host(embed_host_ids_);
    if (embed_host_rows_) context_->free_host(embed_host_rows_);
    embed_host_ids_ = nullptr;
    embed_host_rows_ = nullptr;
    // The drain above retired the row upload; the event must not outlive the
    // pinned blocks it was reading.
    embed_h2d_event_ = sycl::event{};
    if (batch_slots_host_) context_->free_host(batch_slots_host_);
    if (batch_slots_dev_) context_->free_device(batch_slots_dev_);
    if (batch_weights_dev_) context_->free_device(batch_weights_dev_);
    batch_slots_host_ = nullptr;
    batch_slots_dev_ = nullptr;
    batch_weights_dev_ = nullptr;

    if (expert_offsets_host_) context_->free_host(expert_offsets_host_);
    if (expert_counts_host_) context_->free_host(expert_counts_host_);
    if (expert_tiles_host_) context_->free_host(expert_tiles_host_);
    if (expert_curr_pos_host_) context_->free_host(expert_curr_pos_host_);
    if (expert_eight_pos_host_) context_->free_host(expert_eight_pos_host_);
    if (expert_quad_pos_host_) context_->free_host(expert_quad_pos_host_);
    if (expert_pair_pos_host_) context_->free_host(expert_pair_pos_host_);
    if (expert_single_pos_host_) context_->free_host(expert_single_pos_host_);
    if (expert_map_token_host_) context_->free_host(expert_map_token_host_);
    if (expert_map_rank_host_) context_->free_host(expert_map_rank_host_);
    if (expert_map_slot_host_) context_->free_host(expert_map_slot_host_);
    if (expert_map_token_dev_) context_->free_device(expert_map_token_dev_);
    if (expert_map_rank_dev_) context_->free_device(expert_map_rank_dev_);
    if (expert_map_slot_dev_) context_->free_device(expert_map_slot_dev_);
    expert_offsets_host_ = nullptr;
    expert_counts_host_ = nullptr;
    expert_tiles_host_ = nullptr;
    expert_curr_pos_host_ = nullptr;
    expert_eight_pos_host_ = nullptr;
    expert_quad_pos_host_ = nullptr;
    expert_pair_pos_host_ = nullptr;
    expert_single_pos_host_ = nullptr;
    expert_map_token_host_ = nullptr;
    expert_map_rank_host_ = nullptr;
    expert_map_slot_host_ = nullptr;
    expert_map_token_dev_ = nullptr;
    expert_map_rank_dev_ = nullptr;
    expert_map_slot_dev_ = nullptr;
    for (int b = 0; b < 2; ++b) {
        if (xmx_plan_pinned_[b]) context_->free_host(xmx_plan_pinned_[b]);
        if (xmx_plan_dev_[b]) context_->free_device(xmx_plan_dev_[b]);
        xmx_plan_pinned_[b] = nullptr;
        xmx_plan_dev_[b] = nullptr;
    }
}

// ============================================================================
// Load
// ============================================================================

bool Qwen4ExpBnb4Backend::load(
    Context& ctx, ModelWeights& weights, const ModelSpec& spec,
    int max_seq_len, std::string* error_message) {
    context_ = &ctx;
    // Opt-in load-phase timing (C7 deep dive): the ~116 s startup needs a
    // phase breakdown before any fill mechanism (batched reads, DirectStorage)
    // can be judged.
    const bool load_timing = aila::env::read_flag("AILA_Q4E_LOAD_TIMING", false);
    auto load_since = std::chrono::steady_clock::now();
    auto load_mark = [&](const char* what) {
        if (!load_timing) return;
        const auto now = std::chrono::steady_clock::now();
        AILA_LOG_WARN("[Q4E LoadTiming] %s: %.3f s", what,
                      std::chrono::duration<double>(now - load_since).count());
        load_since = now;
    };
    moe_compact_tile_ = 0; // adaptive prefill uses the existing ordered rank buffer
    if (moe_compact_tile_ && (moe_compact_tile_ != 256 && moe_compact_tile_ != 512 && moe_compact_tile_ != 1024))
        throw std::invalid_argument("AILA_Q4E_MOE_COMPACT_ACTS must be 0, 256, 512 or 1024");
    if (moe_compact_tile_ && (!aila::env::read_flag("AILA_Q4E_EXPERT_MAJOR_PREFILL", true) ||
        aila::env::read_int("AILA_Q4E_DEMAND_MICROBATCH",0) <= 0 ||
        aila::env::read_int("AILA_Q4E_GROUPED_XMX_MIN_ROWS",0) > 0 ||
        aila::env::read_flag("AILA_Q4E_CPU_EXPERT_DISPATCH",false)))
        throw std::invalid_argument("Compact MoE requires GPU ALU expert-major demand microbatches");
    prefill_attention_tile_ = aila::env::read_int("AILA_Q4E_PREFILL_ATTN_TILE", 0);
    if (prefill_attention_tile_ != 0 && prefill_attention_tile_ != 128 &&
        prefill_attention_tile_ != 256 && prefill_attention_tile_ != 512)
        throw std::invalid_argument("AILA_Q4E_PREFILL_ATTN_TILE must be 0, 128, 256 or 512");
    grouped_xmx_supported_ = ctx.queue().get_device().get_info<sycl::info::device::name>().find("Arc(TM) A770") != std::string::npos;
    expert_histogram_enabled_ = aila::env::read_flag("AILA_Q4E_EXPERT_HISTOGRAM", false);
    // Round 3 measurements on tools/prefill-bench.ps1 at 7878 tokens:
    //   mode 0 (recurrence)          25,272 ms  311.7 tok/s   <- old default
    //   mode 1 (FP32 chunk control)  24,645 ms  319.7 tok/s
    //   mode 2 (BF16 XMX products)   24,798 ms  317.7 tok/s
    // The chunked path replaces the 5.24 s `run_linear_delta_prefill_gpu_batched` recurrence that
    // the section 64 trace showed dominating GDN. Mode 1 is the fastest and is the default now;
    // the flag still selects 0 or 2.
    gdn_chunk_mode_ = std::clamp(aila::env::read_int("AILA_Q4E_GDN_CHUNK_MODE", 1), 0, 2);
    spec_ = spec;
    cfg_ = spec.qwen4exp_text;
    hidden_size_ = cfg_.hidden_size;
    max_seq_len_ = max_seq_len;

    full_q_heads_ = cfg_.num_attention_heads;
    full_kv_heads_ = cfg_.num_key_value_heads;
    full_head_dim_ = cfg_.head_dim;
    full_q_dim_ = full_q_heads_ * full_head_dim_;
    full_kv_dim_ = full_kv_heads_ * full_head_dim_;
    full_q_proj_dim_ = 2 * full_q_dim_;
    full_fused_qkv_dim_ = full_q_proj_dim_ + 2 * full_kv_dim_;

    linear_q_heads_ = cfg_.linear_num_key_heads;
    linear_kv_heads_ = cfg_.linear_num_value_heads;
    linear_head_dim_ = cfg_.linear_key_head_dim;
    linear_q_dim_ = linear_q_heads_ * linear_head_dim_;
    linear_kv_dim_ = linear_kv_heads_ * cfg_.linear_value_head_dim;
    linear_qkv_dim_ = 2 * linear_q_dim_ + linear_kv_dim_;
    linear_z_dim_ = linear_kv_dim_;
    linear_all_dim_ = linear_qkv_dim_ + linear_z_dim_ + 2 * linear_kv_heads_;

    auto upload_bnb = [&](const std::string& weight_name) {
        if (weights.has_host(weight_name)) weights.upload(weight_name, ctx);
        if (weights.has_host(weight_name + ".absmax")) weights.upload(weight_name + ".absmax", ctx);
        if (weights.has_host(weight_name + ".quant_map")) weights.upload(weight_name + ".quant_map", ctx);
        if (weights.has_host(weight_name + ".nested_absmax")) weights.upload(weight_name + ".nested_absmax", ctx);
        if (weights.has_host(weight_name + ".nested_quant_map")) weights.upload(weight_name + ".nested_quant_map", ctx);
        if (weights.has_host(weight_name + ".quant_state.bitsandbytes__nf4")) weights.upload(weight_name + ".quant_state.bitsandbytes__nf4", ctx);
    };

    auto erase_bnb = [&](const std::string& weight_name) {
        if (weights.has(weight_name)) weights.erase(weight_name);
        if (weights.has(weight_name + ".absmax")) weights.erase(weight_name + ".absmax");
        if (weights.has(weight_name + ".nested_absmax")) weights.erase(weight_name + ".nested_absmax");
        if (weights.has(weight_name + ".nested_quant_map")) weights.erase(weight_name + ".nested_quant_map");
        if (weights.has(weight_name + ".quant_state.bitsandbytes__nf4")) weights.erase(weight_name + ".quant_state.bitsandbytes__nf4");
    };

    static const bool s_q4e_async_dequant = aila::env::read_flag("AILA_Q4E_ASYNC_DEQUANT", true);
    auto init_quant_linear = [&](const std::string& name, Bnb4BitLinear& linear) -> bool {
        upload_bnb(name);
        Bnb4BitWeightRef ref;
        std::string local_error;
        if (!LoadBnb4BitWeightRef(ctx, weights, name, ref, &local_error)) {
            if (error_message) *error_message = local_error;
            return false;
        }
        if (!linear.init(ctx, ref, &local_error)) {
            if (error_message) *error_message = local_error;
            return false;
        }
        if (s_q4e_async_dequant) {
            linear.set_allow_async_dequant(true);
        }
        static const bool s_q4e_prefill_xmx =
            aila::env::read_flag("AILA_Q4E_PREFILL_XMX", false) ||
            aila::env::read_flag("AILA_Q4E_FUSED_PREFILL", false);
        static const bool s_direct_prefill_io =
            aila::env::read_flag("AILA_Q4E_DIRECT_PREFILL_IO", true);
        linear.set_direct_prefill_io(s_direct_prefill_io);
        if (s_q4e_prefill_xmx) {
            linear.set_allow_dense_xmx(true);
        }
        return true;
    };

    auto ensure_f32 = [&](const std::string& name) -> Tensor* {
        if (!weights.has(name)) {
            weights.upload(name, ctx);
        }
        Tensor& src = weights.get(name);
        if (src.dtype() == dnnl::memory::data_type::f32) {
            return &src;
        }
        size_t n = static_cast<size_t>(src.numel());
        std::vector<bf16> host_bf16(n);
        std::vector<float> host_f32(n);
        ctx.memcpy_d2h(host_bf16.data(), src.data(), n * sizeof(bf16));
        for (size_t i = 0; i < n; ++i) {
            host_f32[i] = static_cast<float>(host_bf16[i]);
        }
        Tensor dst = Tensor::allocate(ctx, src.shape(), dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(dst.data(), host_f32.data(), n * sizeof(float));
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    auto plus_one_norm = [&](const std::string& name) -> Tensor* {
        if (!weights.has(name)) {
            weights.upload(name, ctx);
        }
        Tensor& src = weights.get(name);
        size_t n = static_cast<size_t>(src.numel());
        std::vector<bf16> host(n);
        ctx.memcpy_d2h(host.data(), src.data(), n * sizeof(bf16));
        std::vector<float> host_f32(n);
        for (size_t i = 0; i < n; ++i) {
            host[i] = bf16(static_cast<float>(host[i]) + 1.0f);
        }
        Tensor dst = Tensor::allocate(ctx, src.shape(), dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(dst.data(), host.data(), n * sizeof(bf16));
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    AILA_LOG_INFO("[Qwen4Exp] Uploading model embed_tokens and lm_head...");
    const std::string embed_name = "model.language_model.embed_tokens.weight";
    static const bool s_host_embed = aila::env::read_flag("AILA_Q4E_HOST_EMBED", true);
    embed_host_ = s_host_embed ? weights.host_view(embed_name) : HostTensorView{};
    if (!(embed_host_.valid() && embed_host_.dtype == DT_BF16 && embed_host_.shape.size() == 2 &&
          embed_host_.shape[1] == hidden_size_)) {
        embed_host_ = {};
        embed_weight_ = &weights.upload(embed_name, ctx);
    }

    static const bool s_lm_head_q8 = aila::env::read_flag("AILA_Q4E_LM_HEAD_Q8", false);
    const HostTensorView lm_head_host = weights.has_host("lm_head.weight") ? weights.host_view("lm_head.weight") : HostTensorView{};
    const bool use_q8_head = s_lm_head_q8 && hidden_size_ == 2560 && cfg_.vocab_size == 248320 &&
        lm_head_host.valid() && lm_head_host.dtype == DT_BF16 && lm_head_host.shape.size() == 2 &&
        lm_head_host.shape[0] == cfg_.vocab_size && lm_head_host.shape[1] == hidden_size_ &&
        ops::bf16_q8_head_supported(ctx, hidden_size_, cfg_.vocab_size);
    Tensor& raw_lm_head = weights.upload("lm_head.weight", ctx);
    if (use_q8_head) {
        if (raw_lm_head.context() != &ctx || raw_lm_head.dtype() != dnnl::memory::data_type::bf16 ||
            raw_lm_head.ndim() != 2 || raw_lm_head.shape(0) != cfg_.vocab_size || raw_lm_head.shape(1) != hidden_size_)
            throw std::runtime_error("Qwen4Exp Q8 head requires raw BF16 [248320,2560] in model Context");
        lm_head_q8_weight_ = Tensor::allocate(ctx, {cfg_.vocab_size, hidden_size_}, dnnl::memory::data_type::u8);
        lm_head_q8_scale_ = Tensor::allocate(ctx, {cfg_.vocab_size, hidden_size_ / 64}, dnnl::memory::data_type::f32);
        // Four activation rows so multi-row verification (rows 2..4) quantizes
        // every row in one pass instead of reusing one row's scratch.
        lm_head_q8_input_ = Tensor::allocate(ctx, {4 * hidden_size_}, dnnl::memory::data_type::u8);
        lm_head_q8_input_scale_ = Tensor::allocate(ctx, {4 * (hidden_size_ / 64)}, dnnl::memory::data_type::f32);
        ops::quantize_bf16_weights_q8_64(ctx, raw_lm_head, lm_head_q8_weight_, lm_head_q8_scale_);
        ctx.synchronize(); // Raw BF16 cannot be freed until conversion drains.
        weights.erase("lm_head.weight"); // Keeps host mmap; no transposed BF16 or oneDNN head.
        lmHeadQ8Enabled_ = true;
        AILA_LOG_INFO("[Q4E LM Head Q8 Load] actual_enabled=1 hidden=%d vocab=%d block=64 weight_bytes=%llu scale_bytes=%llu saved_head_bytes=%llu host_original_bf16=1 experimental=1",
            hidden_size_, cfg_.vocab_size,
            static_cast<unsigned long long>(lm_head_q8_weight_.size_bytes()),
            static_cast<unsigned long long>(lm_head_q8_scale_.size_bytes()),
            static_cast<unsigned long long>(size_t(hidden_size_) * cfg_.vocab_size * sizeof(bf16) -
                lm_head_q8_weight_.size_bytes() - lm_head_q8_scale_.size_bytes()));
    } else {
        if (s_lm_head_q8) AILA_LOG_INFO("[Q4E LM Head Q8 Load] requested=1 actual_enabled=0 unsupported_shape_or_context=1");
        lm_head_weight_transposed_ = Tensor::allocate(ctx, {hidden_size_, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
        ops::transpose(ctx, raw_lm_head, lm_head_weight_transposed_);
        ctx.synchronize();
        weights.erase("lm_head.weight");
        lm_head_.init(ctx, lm_head_weight_transposed_, hidden_size_, cfg_.vocab_size, /*preprocessed=*/true, /*prepare_gemv=*/false);
    }

    // Hyper-Connection Mixer
    AILA_LOG_INFO("[Qwen4Exp] Loading hyper_connection_mixer...");
    auto load_hc = [&](const std::string& hc_prefix, Qwen4ExpHyperConnectionLayer& hc, bool with_inject) {
        weights.upload(hc_prefix + "hc_norm.weight", ctx);
        hc.hc_norm_weight = ensure_f32(hc_prefix + "hc_norm.weight");
        hc.mix_down_w = &weights.upload(hc_prefix + "input_mix_weight_down.weight", ctx);
        hc.input_mix_weight_down.init(ctx, *hc.mix_down_w, 10240, 320, false);
        hc.mix_up_w = &weights.upload(hc_prefix + "input_mix_weight_up.weight", ctx);
        hc.input_mix_weight_up.init(ctx, *hc.mix_up_w, 320, 10240, false);
        if (with_inject) {
            hc.inject_w = &weights.upload(hc_prefix + "block_inject_weight.weight", ctx);
            hc.block_inject_weight.init(ctx, *hc.inject_w, 10240, 4, false);
        }
    };
    load_hc("model.language_model.hyper_connection_mixer.", mixer_, false);

    // Layers
    AILA_LOG_INFO("[Qwen4Exp] Loading %d layers...", cfg_.num_hidden_layers);
    layers_.resize(cfg_.num_hidden_layers);
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "model.language_model.layers." + std::to_string(i) + ".";

        bool is_linear = true;
        if (i < static_cast<int>(cfg_.layer_types.size())) {
            is_linear = (cfg_.layer_types[i] == "linear_attention");
        } else {
            is_linear = ((i + 1) % 4 != 0);
        }
        layer.is_linear = is_linear;

        load_hc(prefix + "attn_hyper_connection.", layer.attn_hc, true);
        load_hc(prefix + "mlp_hyper_connection.", layer.mlp_hc, true);

        // Attention
        if (is_linear) {
            layer.geom = ops::DeltaNetGeometry::create(
                linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
                cfg_.linear_conv_kernel_dim, cfg_.rms_norm_eps, ops::GateActivation::Sigmoid);

            if (!init_quant_linear(prefix + "linear_attn.out_proj.weight", layer.linear_o_proj)) {
                return false;
            }

            Bnb4BitLinear linear_qkv_proj, linear_z_proj, linear_a_proj, linear_b_proj;
            if (!init_quant_linear(prefix + "linear_attn.in_proj_qkv.weight", linear_qkv_proj) ||
                !init_quant_linear(prefix + "linear_attn.in_proj_z.weight", linear_z_proj) ||
                !init_quant_linear(prefix + "linear_attn.in_proj_a.weight", linear_a_proj) ||
                !init_quant_linear(prefix + "linear_attn.in_proj_b.weight", linear_b_proj)) {
                return false;
            }
            std::string fused_error;
            if (!layer.linear_all_proj.init_fused_rows(ctx, linear_qkv_proj, linear_z_proj, linear_a_proj, linear_b_proj, &fused_error)) {
                if (error_message) *error_message = fused_error;
                return false;
            }
            erase_bnb(prefix + "linear_attn.in_proj_qkv.weight");
            erase_bnb(prefix + "linear_attn.in_proj_z.weight");
            erase_bnb(prefix + "linear_attn.in_proj_a.weight");
            erase_bnb(prefix + "linear_attn.in_proj_b.weight");

            weights.upload(prefix + "linear_attn.norm.weight", ctx);
            layer.linear_norm_weight = ensure_f32(prefix + "linear_attn.norm.weight");
            weights.upload(prefix + "linear_attn.A_log", ctx);
            layer.linear_A_log = ensure_f32(prefix + "linear_attn.A_log");
            layer.linear_dt_bias = &weights.upload(prefix + "linear_attn.dt_bias", ctx);
            layer.linear_conv1d_weight = &weights.upload(prefix + "linear_attn.conv1d.weight", ctx);
        } else {
            if (!init_quant_linear(prefix + "self_attn.o_proj.weight", layer.o_proj)) {
                return false;
            }
            Bnb4BitLinear q_proj, k_proj, v_proj;
            if (!init_quant_linear(prefix + "self_attn.q_proj.weight", q_proj) ||
                !init_quant_linear(prefix + "self_attn.k_proj.weight", k_proj) ||
                !init_quant_linear(prefix + "self_attn.v_proj.weight", v_proj)) {
                return false;
            }
            std::string fused_error;
            if (!layer.qkv_proj.init_fused_rows(ctx, q_proj, k_proj, v_proj, &fused_error)) {
                if (error_message) *error_message = fused_error;
                return false;
            }
            erase_bnb(prefix + "self_attn.q_proj.weight");
            erase_bnb(prefix + "self_attn.k_proj.weight");
            erase_bnb(prefix + "self_attn.v_proj.weight");

            weights.upload(prefix + "self_attn.q_norm.weight", ctx);
            layer.q_norm_weight = plus_one_norm(prefix + "self_attn.q_norm.weight");
            weights.upload(prefix + "self_attn.k_norm.weight", ctx);
            layer.k_norm_weight = plus_one_norm(prefix + "self_attn.k_norm.weight");

            // QSA Indexer (M2 & W2)
            const std::string index_qk_name = prefix + "self_attn.indexer.index_qk_proj.weight";
            if (weights.has_host(index_qk_name) || weights.has(index_qk_name)) {
                const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                layer.index_qk_weight = &weights.upload(index_qk_name, ctx);
                layer.index_qk_proj.init(ctx, *layer.index_qk_weight, hidden_size_, indexer_out_dim, false);
                const int indexer_k_dim = cfg_.indexer_kv_heads * cfg_.indexer_head_dim;
                const size_t k_offset = static_cast<size_t>(cfg_.indexer_n_heads * cfg_.indexer_head_dim) * hidden_size_;
                layer.index_k_weight = Tensor::view(
                    ctx, layer.index_qk_weight->data_as<bf16>() + k_offset,
                    {indexer_k_dim, hidden_size_});
                layer.index_k_proj.init(ctx, layer.index_k_weight, hidden_size_, indexer_k_dim, false);
                layer.index_q_norm_weight = plus_one_norm(prefix + "self_attn.indexer.q_layernorm.weight");
                layer.index_k_norm_weight = plus_one_norm(prefix + "self_attn.indexer.k_layernorm.weight");
            }
        }

        // MoE Router & Shared Expert
        layer.router_w = &weights.upload(prefix + "mlp.gate.weight", ctx);
        layer.router_gate.init(ctx, *layer.router_w, hidden_size_, cfg_.num_experts, false);

        layer.shared_expert_gate_w = &weights.upload(prefix + "mlp.shared_expert_gate.weight", ctx);
        layer.shared_expert_gate.init(ctx, *layer.shared_expert_gate_w, hidden_size_, 1, false);

        Bnb4BitLinear shared_gate_proj, shared_up_proj;
        if (!init_quant_linear(prefix + "mlp.shared_expert.gate_proj.weight", shared_gate_proj) ||
            !init_quant_linear(prefix + "mlp.shared_expert.up_proj.weight", shared_up_proj) ||
            !init_quant_linear(prefix + "mlp.shared_expert.down_proj.weight", layer.shared_down_proj)) {
            return false;
        }
        std::string fused_gate_up_error;
        if (!layer.shared_gate_up_proj.init_fused_rows(ctx, shared_gate_proj, shared_up_proj, &fused_gate_up_error)) {
            if (error_message) *error_message = fused_gate_up_error;
            return false;
        }
        erase_bnb(prefix + "mlp.shared_expert.gate_proj.weight");
        erase_bnb(prefix + "mlp.shared_expert.up_proj.weight");
    }

    // PLE (Layer 1): staging slots hold one forward slice (up to prefill chunk),
    // not the full retained context, avoiding >2.5 GB VRAM/pinned bloat at 262K.
    const int ple_chunk_cap = std::min(
        max_seq_len_,
        std::max(8192, aila::env::read_int("AILA_Q35_PREFILL_CHUNK", 8192)));
    AILA_LOG_INFO("[Qwen4Exp] Initializing PLE (Layer 1, staging_cap=%d)...", ple_chunk_cap);
    if (!ple_.init(ctx, weights, 1, ple_chunk_cap, cfg_.rms_norm_eps, error_message)) {
        return false;
    }

    // Expert Cache: AILA_Q4E_EXPERT_CACHE_MB, or by default the VRAM left after static weights
    // minus headroom for session state, prefill workspace and the desktop. On a 16 GB A770
    // driving a display, ~1 GB free after warmup already slowed every kernel ~2x (WDDM paging);
    // ~2 GB free did not.
    size_t expert_cache_bytes = 0;
    // Keep the admission accounting on the same KV layout the session builder
    // allocates: bf16, or the opt-in int8 codes plus the two per-token F32
    // scale tensors. A single source lives in SessionMemory.
    const bool int8_kv = aila::env::read_flag("AILA_KV_QUANT", false) &&
                         aila::env::kv_quant_format_is_int8();
    const size_t kv_element_bytes = int8_kv ? 1 : sizeof(bf16);
    const int expert_cache_mb = aila::env::read_int("AILA_Q4E_EXPERT_CACHE_MB", 0);
    const bool mtp_enabled = aila::env::read_flag("AILA_Q4E_MTP_DRAFT", false) ||
                             aila::env::read_flag("AILA_Q4E_SERVER_MTP", false);
    const auto kv_stream_plan = aila::memory::qwen4exp_kv_stream_plan(max_seq_len_);
    const bool dxgi_budget_enabled = aila::env::read_flag("AILA_Q4E_DXGI_BUDGET", true);
    const auto dxgi_info = dxgi_budget_enabled ? aila::QueryDxgiLocalVram() : aila::DxgiLocalVramInfo{};
    size_t auto_vram_ceiling = 0;
    size_t auto_non_kv_reserve = 0;
    resolved_kv_resident_tokens_ = kv_stream_plan.enabled ? kv_stream_plan.residentTokens : 0;
    if (expert_cache_mb > 0) {
        expert_cache_bytes = static_cast<size_t>(expert_cache_mb) << 20;
    } else {
        size_t vram = ctx.queue().get_device().get_info<sycl::info::device::global_mem_size>();
        if (ctx.device_allocation_limit() > 0) vram = std::min(vram, ctx.device_allocation_limit());
        const size_t full_layers = static_cast<size_t>(
            std::count_if(layers_.begin(), layers_.end(), [](const Qwen4ExpLayer& l) { return !l.is_linear; }));
        const size_t linear_layers = layers_.size() - full_layers;
        const size_t device_kv_cap = kv_stream_plan.enabled
            ? static_cast<size_t>(kv_stream_plan.residentTokens)
            : static_cast<size_t>(max_seq_len_);
        const auto kv_layout = aila::memory::qwen4exp_kv_bytes(
            cfg_, device_kv_cap, kv_element_bytes);
        const size_t kv_bytes = kv_layout.kv;
        const size_t kv_scale_bytes = kv_layout.scales;
        const size_t kv_stream_map_bytes = kv_stream_plan.enabled
            ? full_layers * aila::memory::qwen4exp_kv_stream_map_bytes_per_layer(
                kv_stream_plan.blockCount, kv_stream_plan.slotCount)
            : 0;
        const size_t indexer_k_bytes = full_layers * cfg_.indexer_head_dim * sizeof(bf16) * static_cast<size_t>(max_seq_len_);
        const bool tail_only_indexer_k = kv_stream_plan.enabled && aila::env::read_flag("AILA_QSA_COMPRESSED_K", true);
        const size_t device_indexer_k_bytes = tail_only_indexer_k
            ? (full_layers * 8u * cfg_.indexer_head_dim * sizeof(bf16))
            : (kv_stream_plan.enabled ? 0u : indexer_k_bytes);
        const size_t host_indexer_k_bytes = (kv_stream_plan.enabled && !tail_only_indexer_k) ? indexer_k_bytes : 0u;
        const size_t indexer_compressed_k_bytes = full_layers * (static_cast<size_t>(max_seq_len_) / 4) * cfg_.indexer_head_dim * sizeof(bf16);
        const size_t recurrent_bytes = linear_layers * (48 * 128 * 128 + 3 * 10240) * sizeof(float);
        const size_t non_kv_session_state = device_indexer_k_bytes + indexer_compressed_k_bytes + recurrent_bytes;
        const size_t total_session_state = kv_bytes + kv_scale_bytes + kv_stream_map_bytes + non_kv_session_state;
        const size_t max_reserve =
            static_cast<size_t>(std::max(0, aila::env::read_int("AILA_Q4E_VRAM_RESERVE_MB", 1400))) << 20;
        const int prefill_chunk_cap = std::min(
            max_seq_len_,
            std::max(64, aila::env::read_int("AILA_Q35_PREFILL_CHUNK", 8192)));
        // Account for the full MTP footprint (705 MiB draft weights/slots/checkpoint +
        // chunk-capped hiddenRows + MTP KV/session tensors) and prefill runtime buffers
        // allocated lazily on the first prefill chunk so AILA_Q4E_VRAM_RESERVE_MB
        // represents actual free WDDM headroom rather than getting eaten by workspace.
        size_t mtp_reserve = 0;
        if (mtp_enabled) {
            const int mtp_cap = std::clamp(
                aila::env::read_int("AILA_Q4E_MTP_MAX_CONTEXT", 2048),
                2,
                std::max(2, max_seq_len_));
            const int mtp_hidden_rows_cap = std::min(mtp_cap, 512);
            const size_t mtp_hidden_rows_bytes =
                static_cast<size_t>(mtp_hidden_rows_cap) * 10240u * sizeof(bf16);
            const size_t mtp_session_bytes =
                static_cast<size_t>(mtp_cap) * 2304u +
                static_cast<size_t>(std::max(1, mtp_cap / 4)) * 256u +
                10240u * sizeof(bf16);
            // Include Qwen4ExpMtpRuntime persistent device tensors:
            // checkpoint (118,038,528 B) + linearInputs (48*4*16480*2 B) +
            // logits (4*vocab*2 B) + verify scratch (~2 MiB).
            const size_t mtp_runtime_bytes =
                size_t{118038528u} +
                size_t{48u} * 4u * 16480u * sizeof(bf16) +
                size_t{4u} * static_cast<size_t>(cfg_.vocab_size) * sizeof(bf16) +
                (static_cast<size_t>(2) << 20);
            mtp_reserve = (static_cast<size_t>(705) << 20) + mtp_hidden_rows_bytes +
                mtp_session_bytes + mtp_runtime_bytes;
        }
        // Estimate peak workspace allocated by ensure_runtime_buffers(prefill_chunk_cap)
        // plus WarmupPrefillKernels (linear_scratch_.weight_bf16 + xmx_scratch_)
        // and PLE staging_cap.
        const int ws_attn_cap = prefill_attention_tile_ ? std::min(prefill_chunk_cap, prefill_attention_tile_) : prefill_chunk_cap;
        constexpr int ws_dense_limit = 2052;
        const int ws_dense_cap = std::min(ws_attn_cap, ws_dense_limit);
        const int ws_dense_kv_cap = std::min(max_seq_len_, ws_dense_limit);
        const int ws_idx_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
        const int ws_idx_q_dim = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
        const size_t ws_qsa_base = std::max({
            size_t(ws_attn_cap) * size_t(full_fused_qkv_dim_) * sizeof(bf16),
            size_t(full_q_heads_) * size_t(ws_dense_cap) * size_t(ws_dense_cap) * sizeof(float),
            size_t(full_q_heads_) * size_t(ws_dense_cap) * size_t(ws_dense_kv_cap) * sizeof(float)});
        const size_t ws_qsa_tail =
            size_t(ws_attn_cap) * (size_t(full_q_dim_) * 2u + size_t(full_kv_dim_) * 2u +
                                   size_t(ws_idx_out_dim) + size_t(ws_idx_q_dim)) * sizeof(bf16);
        const size_t ws_kv_stage_half = kv_stream_plan.enabled
            ? size_t(full_kv_heads_) * size_t(max_seq_len_) * size_t(full_head_dim_) * kv_element_bytes
            : 0u;
        const size_t ws_kv_stage_scales = (kv_stream_plan.enabled && int8_kv)
            ? 2u * size_t(full_kv_heads_) * size_t(max_seq_len_) * sizeof(float)
            : 0u;
        const size_t ws_stage_peak = kv_stream_plan.enabled
            ? (std::max(ws_qsa_base, ws_kv_stage_half) + ws_qsa_tail + ws_kv_stage_half + ws_kv_stage_scales)
            : 0u;
        const size_t ws_gdn_peak = (gdn_chunk_mode_ > 0 && grouped_xmx_supported_)
            ? (size_t(ws_attn_cap) * size_t(linear_all_dim_) * sizeof(bf16) +
               size_t(ws_attn_cap) * size_t(linear_kv_heads_) * 514u * sizeof(float))
            : size_t(ws_attn_cap) * size_t(linear_all_dim_) * sizeof(bf16);
        const size_t ws_phase_peak = std::max({
            ws_qsa_base + ws_qsa_tail,
            ws_stage_peak,
            ws_gdn_peak,
            size_t(prefill_chunk_cap) * (20480u + 320u) * sizeof(bf16),
            size_t(prefill_chunk_cap) * size_t(cfg_.num_experts_per_tok) * size_t(hidden_size_) * sizeof(float)});
        const bool ws_xmx_c1_f32 =
            aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_W_F16", 0) != 0 ||
            aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32", 0) != 0;
        const size_t ws_xmx_scratch_bytes = prefill_expert_xmx_
            ? (static_cast<size_t>(ws_xmx_c1_f32 ? 385 : 375) << 20)
            : 0u;
        const size_t ws_linear_scratch_bytes =
            size_t(linear_all_dim_) * size_t(hidden_size_) * sizeof(bf16);
        const size_t ws_ple_staging_bytes =
            2u * size_t(prefill_chunk_cap) * size_t(hidden_size_) * sizeof(bf16);
        const bool ws_has_qsa = layers_.size() > 3 && layers_[3].index_qk_weight != nullptr;
        const int ws_comp_ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
        const int ws_block_topk = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / ws_comp_ratio;
        const int ws_decode_scores_cap = ws_has_qsa
            ? std::min(max_seq_len_, std::max(4096, ws_block_topk * ws_comp_ratio + 128))
            : max_seq_len_;
        const int ws_decode_tiles = ws_has_qsa
            ? std::max(68, (ws_decode_scores_cap + 127) / 128)
            : ((max_seq_len_ + 127) / 128);
        const size_t ws_persistent_bufs =
            size_t(prefill_chunk_cap) * (10240u + 2u * size_t(hidden_size_)) * sizeof(bf16) +
            size_t(ws_attn_cap) * size_t(full_q_dim_) * sizeof(bf16) +
            size_t(full_q_heads_) * size_t(ws_decode_scores_cap) * sizeof(float) +
            size_t(full_q_heads_) * size_t(ws_decode_tiles) * size_t(full_head_dim_ == 256 ? 272 : 144) * sizeof(float) +
            ws_xmx_scratch_bytes + ws_linear_scratch_bytes + ws_ple_staging_bytes +
            (static_cast<size_t>(380) << 20); // qsa_tile_* + indexer_* + expert_*_dev + measured L0/oneDNN JIT runtime headroom
        const size_t workspace_reserve = ws_phase_peak + ws_persistent_bufs;

        size_t effective_vram_ceiling = vram > max_reserve ? (vram - max_reserve) : 0;
        if (dxgi_info.valid && ctx.device_allocation_limit() == 0) {
            const uint64_t own_alloc = ctx.current_allocated_bytes();
            const uint64_t other_used = dxgi_info.currentUsageBytes > own_alloc
                ? (dxgi_info.currentUsageBytes - own_alloc) : 0;
            const size_t dxgi_avail_proc = dxgi_info.budgetBytes > other_used
                ? static_cast<size_t>(dxgi_info.budgetBytes - other_used) : 0;
            constexpr size_t kDxgiDriverSafetyMargin = static_cast<size_t>(512) << 20;
            const size_t dxgi_ceiling = dxgi_avail_proc > kDxgiDriverSafetyMargin
                ? (dxgi_avail_proc - kDxgiDriverSafetyMargin) : 0;
            effective_vram_ceiling = std::min(effective_vram_ceiling, dxgi_ceiling);
            AILA_LOG_INFO(
                "[Qwen4Exp] DXGI VRAM Budget: hw=%.0f MB budget=%.0f MB other_used=%.0f MB avail=%.0f MB ceiling=%.0f MB",
                vram / (1024.0 * 1024.0),
                dxgi_info.budgetBytes / (1024.0 * 1024.0),
                other_used / (1024.0 * 1024.0),
                dxgi_info.availableBytes / (1024.0 * 1024.0),
                effective_vram_ceiling / (1024.0 * 1024.0));
        }

        const size_t reserve = total_session_state + mtp_reserve + workspace_reserve;
        const size_t min_bytes = static_cast<size_t>(1024) << 20;
        const size_t used = ctx.current_allocated_bytes() + reserve;
        expert_cache_bytes = effective_vram_ceiling > used + min_bytes
            ? effective_vram_ceiling - used
            : min_bytes;
        auto_vram_ceiling = effective_vram_ceiling;
        auto_non_kv_reserve = non_kv_session_state + mtp_reserve + workspace_reserve;

        AILA_LOG_INFO(
            "[Qwen4Exp] Memory Admission Accounting: Total VRAM=%.0f MB, Static Weights=%.1f MB, "
            "Session State=%.1f MB (KV=%.1f MB element_bytes=%zu kv_scales=%.1f MB, IndexerK=%.1f MB, "
            "IndexerCompressedK=%.1f MB, Recurrent=%.1f MB), MTP=%.1f MB, Workspace=%.1f MB, "
            "Reserve Headroom=%.0f MB, Expert Cache=%.0f MB",
            vram / (1024.0 * 1024.0),
            ctx.current_allocated_bytes() / (1024.0 * 1024.0),
            total_session_state / (1024.0 * 1024.0),
            kv_bytes / (1024.0 * 1024.0),
            kv_element_bytes,
            kv_scale_bytes / (1024.0 * 1024.0),
            device_indexer_k_bytes / (1024.0 * 1024.0),
            indexer_compressed_k_bytes / (1024.0 * 1024.0),
            recurrent_bytes / (1024.0 * 1024.0),
            mtp_reserve / (1024.0 * 1024.0),
            workspace_reserve / (1024.0 * 1024.0),
            max_reserve / (1024.0 * 1024.0),
            expert_cache_bytes / (1024.0 * 1024.0));
        if (kv_stream_plan.enabled) {
            const size_t host_kv_bytes = aila::memory::qwen4exp_kv_bytes(
                cfg_, static_cast<size_t>(max_seq_len_), kv_element_bytes).total();
            AILA_LOG_INFO(
                "[Qwen4Exp] KV Streaming: enabled=1 initial_resident_tokens=%d n_slots=%d logical_blocks=%d "
                "device_kv_mb=%.1f host_kv_mb=%.1f host_indexer_k_mb=%.1f",
                kv_stream_plan.residentTokens, kv_stream_plan.slotCount, kv_stream_plan.blockCount,
                (kv_bytes + kv_scale_bytes + kv_stream_map_bytes) / (1024.0 * 1024.0),
                host_kv_bytes / (1024.0 * 1024.0),
                host_indexer_k_bytes / (1024.0 * 1024.0));
        }
    }
    // GPU experts and native RAM experts are complementary owners. Keep
    // embed_tokens in pageable host memory by default (large-page lock is opt-in)
    // so 1.18 GiB does not stay pinned against PLE's mmap working set.
    if (embed_host_.valid()) {
        static const bool s_embed_large_pages =
            aila::env::read_flag("AILA_Q4E_EMBED_LARGE_PAGES", false);
        if (s_embed_large_pages) {
            resident_embedding_ = ctx.alloc_large_host(embed_host_.bytes);
            resident_embedding_large_pages_ = true;
        } else {
#ifdef _WIN32
            resident_embedding_ = VirtualAlloc(nullptr, embed_host_.bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!resident_embedding_) throw std::bad_alloc();
#else
            resident_embedding_ = std::malloc(embed_host_.bytes);
            if (!resident_embedding_) throw std::bad_alloc();
#endif
            resident_embedding_large_pages_ = false;
        }
        std::memcpy(resident_embedding_, embed_host_.ptr, embed_host_.bytes);
        embed_host_.ptr = static_cast<const uint8_t*>(resident_embedding_);
    }
    aila::AdaptiveCacheConfig cache_cfg;
    cache_cfg.num_layers=cfg_.num_hidden_layers;
    cache_cfg.num_experts_per_layer=cfg_.num_experts;
    cache_cfg.hidden_size=hidden_size_;
    cache_cfg.moe_intermediate_size=cfg_.moe_intermediate_size;
    int staging_slots=aila::env::read_int_raw("AILA_Q4E_STAGING_SLOTS", 16);
    cache_cfg.staging_slots_per_buffer=std::clamp(
        staging_slots, 1, aila::AdaptiveCacheConfig::kMaxStagingSlots);
    const size_t slot_bytes=size_t(3)*hidden_size_*cfg_.moe_intermediate_size*33/64;
    // The 32K session and prefill projections reserve more VRAM than the CLI's
    // 4K smoke. Fit the cache after those reservations, including both banks.
    // The hard cap only guards against pathological budgets; raise it when a
    // smaller KV layout (AILA_KV_QUANT_FORMAT=int8) frees device bytes.
    // 3,456 = 72 slots/layer * 48 layers, so the auto VRAM budget (not the
    // guard) decides residency; the slot count still clamps to
    // expert_cache_bytes/slot_bytes minus the staging banks below.
    const int default_max_gpu_slots = mtp_enabled ? 3072 : (lmHeadQ8Enabled_ ? 3456 : 3360);
    const int max_gpu_slots = aila::env::read_int("AILA_Q4E_MAX_GPU_SLOTS", default_max_gpu_slots);
    // Reserve the staging banks by their configured depth (2 buffers).
    const int staging_reserve_slots = 2 * cache_cfg.staging_slots_per_buffer;
    cache_cfg.num_gpu_slots=std::min(max_gpu_slots,int(expert_cache_bytes/slot_bytes)-staging_reserve_slots);
    if (cache_cfg.num_gpu_slots<=0) {
        if (error_message) *error_message="Qwen4Exp resident cache exceeds the admitted VRAM budget";
        return false;
    }
    const auto hotset_profile_path=aila::env::read_string("AILA_Q4E_HOTSET_PROFILE","");
    const bool profile_hotset=!hotset_profile_path.empty();
    const auto hotset_export_path=aila::env::read_string("AILA_Q4E_HOTSET_EXPORT","");
    const bool export_hotset=!hotset_export_path.empty();
    std::vector<std::pair<int,int>> initial_ranked_profile;
    if (profile_hotset || export_hotset) {
        // Build the same source-byte identity used by the legacy cache. The
        // adaptive arena's expert payload is always compact nested NF4.
        std::vector<aila::ExpertLayerSource> profile_sources(cfg_.num_hidden_layers);
        for (int layer=0;layer<cfg_.num_hidden_layers;++layer) {
            const std::string prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.";
            auto& source=profile_sources[layer];
            std::string detail;
            const auto gate_up_base=prefix+"gate_up_proj",down_base=prefix+"down_proj";
            const auto gate_up_name=weights.has_host(gate_up_base)?gate_up_base:gate_up_base+".weight";
            const auto down_name=weights.has_host(down_base)?down_base:down_base+".weight";
            if (!LoadBnb4BitHostRef(weights,gate_up_name,source.gate_up,&detail,hidden_size_) ||
                !LoadBnb4BitHostRef(weights,down_name,source.down,&detail,cfg_.moe_intermediate_size)) {
                if (error_message) *error_message="Cannot validate AILA_Q4E_HOTSET_PROFILE identity: "+detail;
                return false;
            }
            source.valid=true;
        }
        const std::string identity=Qwen4ExpHotsetPolicy::MakeIdentity(profile_sources,
            cfg_.num_hidden_layers,cfg_.num_experts,hidden_size_,cfg_.moe_intermediate_size,64,true);
        if (profile_hotset) {
            Qwen4ExpHotsetPolicy policy;
            policy.Init(cfg_.num_hidden_layers,cfg_.num_experts);
            try {
                policy.Load(hotset_profile_path,identity);
            } catch (const std::exception& e) {
                if (error_message) *error_message=e.what();
                return false;
            }
            const size_t resident_rank_count=std::min<size_t>(
                static_cast<size_t>(cache_cfg.num_gpu_slots),policy.ranking.size());
            initial_ranked_profile.reserve(resident_rank_count);
            for (size_t i=0;i<resident_rank_count;++i) {
                const int key=policy.ranking[i];
                initial_ranked_profile.emplace_back(key/cfg_.num_experts,key%cfg_.num_experts);
            }
            AILA_LOG_INFO("[Q4E Adaptive Residency] profile=%s identity=%s profile_total_keys=%zu "
                          "ranked_keys=%zu admitted_slots=%d kv_element_bytes=%zu",
                hotset_profile_path.c_str(),identity.c_str(),policy.ranking.size(),
                initial_ranked_profile.size(),cache_cfg.num_gpu_slots,kv_element_bytes);
        }
        // P3: opt-in routing collector for offline hotset re-ranking. The
        // adaptive cache never re-ranks online, so a calibration run exports
        // the full 24576-key profile at teardown in the legacy format.
        if (export_hotset) {
            export_policy_.Init(cfg_.num_hidden_layers,cfg_.num_experts);
            export_identity_=identity;
            export_path_=hotset_export_path;
            export_enabled_=true;
            AILA_LOG_INFO("[Q4E Hotset] adaptive export enabled: path=%s decode+prefill counts -> full profile",
                export_path_.c_str());
        }
    }
    load_mark("weights+embedding prep (before adaptive allocate)");
    if (!adaptive_cache_.Allocate(ctx,cache_cfg,error_message)) return false;
    if (kv_stream_plan.enabled && auto_vram_ceiling > 0 &&
        aila::env::read_string("AILA_Q4E_KV_RESIDENT_TOKENS", "").empty()) {
        const size_t committed = ctx.current_allocated_bytes() + auto_non_kv_reserve;
        const size_t budget_for_kv = auto_vram_ceiling > committed ? (auto_vram_ceiling - committed) : 0;
        resolved_kv_resident_tokens_ = aila::memory::qwen4exp_kv_stream_select_auto_resident_tokens(
            cfg_, max_seq_len_, budget_for_kv, kv_element_bytes);
        const auto auto_plan = aila::memory::qwen4exp_kv_stream_plan(max_seq_len_, resolved_kv_resident_tokens_);
        const size_t auto_kv_bytes = aila::memory::qwen4exp_kv_bytes(
            cfg_, static_cast<size_t>(auto_plan.residentTokens), kv_element_bytes).total();
        AILA_LOG_INFO(
            "[Qwen4Exp] KV Streaming AutoTier: budget_for_kv_mb=%.1f resident_tokens=%d n_slots=%d device_kv_mb=%.1f",
            budget_for_kv / (1024.0 * 1024.0),
            auto_plan.residentTokens, auto_plan.slotCount,
            auto_kv_bytes / (1024.0 * 1024.0));
    }
    auto gpu_owners=adaptive_cache_.InitializeResidency(
        profile_hotset ? &initial_ranked_profile : nullptr);
    load_mark("adaptive_cache.Allocate+InitializeResidency");
    aila::HostArenaConfig arena_cfg;
    arena_cfg.num_layers=cfg_.num_hidden_layers; arena_cfg.num_experts_per_layer=cfg_.num_experts;
    arena_cfg.hidden_size=hidden_size_; arena_cfg.moe_intermediate_size=cfg_.moe_intermediate_size;
    if (!host_arena_.Allocate(ctx,arena_cfg,error_message,gpu_owners)) return false;
    load_mark("host_arena.Allocate");
    AILA_LOG_INFO("[Qwen4Exp] Loading native RAM complement: %zu experts, %.2f GiB; GPU-only experts=%d",
        host_arena_.ResidentExperts(),host_arena_.TotalAllocatedBytes()/double(1ULL<<30),adaptive_cache_.NumGpuSlots());
    if (!host_arena_.LoadFromModelWeights(weights,error_message)) return false;
    load_mark("host_arena.LoadFromModelWeights");
    adaptive_cache_.FinishInitialLoad(ctx.queue());
    load_mark("FinishInitialLoad");
    copy_q_=std::make_unique<sycl::queue>(ctx.queue().get_context(),ctx.queue().get_device(),
        [](sycl::exception_list exceptions) noexcept {
            for (const auto& e : exceptions) {
                try {
                    if (e) std::rethrow_exception(e);
                } catch (const std::exception& ex) {
                    std::fprintf(stderr, "[copy_q_ async exception] %s\n", ex.what());
                } catch (...) {
                    std::fprintf(stderr, "[copy_q_ async exception] unknown\n");
                }
            }
        },
        sycl::property_list{sycl::property::queue::in_order()});
    prefill_expert_uses_.assign(cfg_.num_experts,0);
    prefill_unique_.reserve(cfg_.num_experts);
    adaptive_prefill_slots_.resize(cfg_.num_experts);
    AILA_LOG_INFO("[Qwen4Exp] Adaptive residency ready: expert_mmap_fallback=0 small_host_staging_mib=%.1f",
        adaptive_cache_.TotalStagingBytes()/double(1ULL<<20));

    ensure_runtime_buffers(ctx, 1);
    load_mark("runtime buffers (ensure_runtime_buffers)");
    static const int s_resident_decode = aila::env::read_int("AILA_Q4E_RESIDENT_DECODE", 0);
    cold_mode_ = aila::env::read_int("AILA_Q4E_COLD_MODE",0);
    if(profile_hotset && !cold_mode_)cold_mode_=1;
    resident_prefill_ = aila::env::read_flag("AILA_Q4E_RESIDENT_PREFILL",false);
    if(cold_mode_<0 || cold_mode_>2)throw std::invalid_argument("Cold mode must be 0/1/2");
    resident_decode_mode_ = cold_mode_ ? 2 : s_resident_decode;
    if (resident_decode_mode_ < 0 || resident_decode_mode_ > 2 ||
        (resident_decode_mode_ && aila::env::read_flag("AILA_Q4E_CPU_EXPERT_DISPATCH", false))) {
        if (error_message) *error_message = "Resident decode currently supports GPU-only mode 0/1/2";
        return false;
    }
    resident_down_ = Tensor::allocate(ctx, {cfg_.num_experts_per_tok, hidden_size_}, dnnl::memory::data_type::f32);
    resident_weights_ = Tensor::allocate(ctx, {cfg_.num_experts_per_tok}, dnnl::memory::data_type::f32);
    AILA_LOG_INFO("[Q4E ResidentDecode] mode=%d adaptive_pipeline=1", resident_decode_mode_);
    expert_executor_.init(ctx,1,cfg_.num_experts_per_tok,hidden_size_,cfg_.moe_intermediate_size);
    expert_executor_.set_copy_queue(copy_q_.get());

    // With K-only indexer projection (A2) the engine's 2-token warmup completes no QSA block, so
    // it launches no QsaIndexerOps kernel. The first request would then pay that module's JIT
    // (~1.2 s on A770) inside its first prefill; compress one dummy block here to move it to load.
    for (const auto& layer : layers_) {
        if (layer.is_linear || !layer.index_k_norm_weight) continue;
        const int k_dim = cfg_.indexer_kv_heads * cfg_.indexer_head_dim;
        const int ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
        Tensor dummy_k = Tensor::allocate(ctx, {ratio, k_dim}, dnnl::memory::data_type::bf16);
        Tensor dummy_compressed = Tensor::allocate(ctx, {1, k_dim}, dnnl::memory::data_type::bf16);
        ctx.queue().memset(dummy_k.data(), 0, static_cast<size_t>(ratio) * k_dim * sizeof(bf16));
        ops::qsa_compress_key_blocks(ctx, dummy_k, *layer.index_k_norm_weight, dummy_compressed,
                                     0, 1, cfg_.rope.rope_theta, false, nullptr, nullptr, nullptr,
                                     0, 0, cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1],
                                     cfg_.rope.mrope_section[2]);
        ctx.synchronize();
        break;
    }
    const bool s_expert_major = true;
    const bool s_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);
    const bool s_pred_ids_only = aila::env::read_flag("AILA_Q4E_PRED_IDS_ONLY", true);
    const bool s_dense_xmx = aila::env::read_flag("AILA_Q4E_PREFILL_XMX", false) ||
                             aila::env::read_flag("AILA_Q4E_FUSED_PREFILL", false);
    const bool s_direct_io = aila::env::read_flag("AILA_Q4E_DIRECT_PREFILL_IO", true);
    const char* prefill_mode = s_expert_major ? "S4 (expert-major W4/G3)" : (s_moe_rows ? "S1 (batched-rows)" : "baseline (token-by-token)");
    AILA_LOG_INFO("[Qwen4Exp] Prefill MoE mode: %s, Router pred: %s",
                  prefill_mode, s_pred_ids_only ? "ids-only" : "full-softmax");
    AILA_LOG_INFO("[Qwen4Exp] Prefill Dense Linear: %s (direct_io=%s)",
                  s_dense_xmx ? "XMX Joint Matrix" : "oneDNN on-demand",
                  s_direct_io ? "enabled" : "disabled");
    AILA_LOG_INFO("[Qwen4Exp] Backend successfully loaded (48 layers, max_seq_len=%d)", max_seq_len_);
    AILA_LOG_INFO("[Q4E ExecutionConfig] preset=%s chunk=%d native_ram_mib=%.1f slots=%d dense_xmx=%d direct_io=%d phase_scratch=%d staging_experts=%d checkpoint=%s attn_tile=%d compact_acts=%d",
        aila::env::read_string("AILA_Q4E_ACTIVE_PRESET", "none").c_str(),
        aila::env::read_int("AILA_Q35_PREFILL_CHUNK", 512),
        (host_arena_.TotalAllocatedBytes() + (resident_embedding_ ? aila::memory::LargePageAllocationSize(embed_host_.bytes) : 0)) / double(1ULL<<20), adaptive_cache_.NumGpuSlots(),
        int(s_dense_xmx), int(s_direct_io), int(aila::env::read_flag("AILA_Q4E_PHASE_SCRATCH", false)),
        adaptive_cache_.StagingSlotsPerBuffer(),
        aila::env::read_string("AILA_Q4E_INLINE_CHECKPOINT", "auto-30/26").c_str(), prefill_attention_tile_, moe_compact_tile_);
    return true;
}

// ============================================================================
// Runtime Buffers
// ============================================================================

void Qwen4ExpBnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    // Optional preallocation floor: pay the chunk-8192 phase-scratch growth and
    // first-touch once (warmup/first call) instead of inside the first real
    // request, where it costs ~55-90 s of host/alloc time on this machine.
    static const int s_runtime_min_cap =
        std::max(0, aila::env::read_int("AILA_Q4E_RUNTIME_MIN_CAP", 0));
    if (s_runtime_min_cap > seq_len) seq_len = std::min(s_runtime_min_cap, max_seq_len_);
    if (seq_len <= runtime_seq_capacity_) return;
    // The resize replaces resident scratch pointers. Invalidate before the
    // first mutation so even a failed allocation cannot leave an old key valid.
    ++runtime_generation_;
    int cap = std::max(seq_len, runtime_seq_capacity_ == 0 ? std::min(max_seq_len_, 64) : runtime_seq_capacity_ * 2);
    cap = std::min(cap, max_seq_len_);
    if (cap < seq_len) cap = seq_len;
    // Growth is outside decode's steady-state path. Drain readers before any
    // old tensor/view backing store is replaced (including oneDNN consumers).
    ctx.synchronize();
    static const bool s_phase_scratch = aila::env::read_flag("AILA_Q4E_PHASE_SCRATCH", false);
    const bool s_expert_major = true;
    constexpr int kDensePrefillLimit = 2052;
    const int attn_cap = prefill_attention_tile_ ? std::min(cap, prefill_attention_tile_) : cap;
    const int dense_cap = std::min(attn_cap, kDensePrefillLimit);
    const int dense_kv_cap = std::min(max_seq_len_, kDensePrefillLimit);
    const int down_rows = moe_compact_tile_ ? std::max(moe_compact_tile_, std::min(cap,4)*cfg_.num_experts_per_tok)
                                             : cap*cfg_.num_experts_per_tok;
    if (moe_compact_tile_) {
        buf_.moe_window_sum = Tensor::allocate(ctx,{int64_t(cap)*hidden_size_*int64_t(sizeof(int64_t))},dnnl::memory::data_type::u8);
        if(!buf_.moe_compact_invalid.valid())buf_.moe_compact_invalid=Tensor::allocate(ctx,{1},dnnl::memory::data_type::s32);
        buf_.moe_inverse_map = Tensor::allocate(ctx,{cap,cfg_.num_experts_per_tok},dnnl::memory::data_type::s32);
    } else { buf_.moe_window_sum=Tensor{};buf_.moe_inverse_map=Tensor{}; }
    const auto align256 = [](size_t n) { return (n + 255u) & ~size_t(255u); };
    const bool alias_prefill_phases = s_phase_scratch && cap > 64;
    const size_t hc_n_bytes = align256(size_t(cap) * 10240u * sizeof(bf16));
    const size_t hc_down_bytes = align256(size_t(cap) * 320u * sizeof(bf16));
    const size_t hc_up_bytes = align256(size_t(cap) * 10240u * sizeof(bf16));
    const size_t hc_phase_bytes = hc_n_bytes + hc_down_bytes + hc_up_bytes;
    const size_t moe_down_bytes = align256(size_t(down_rows) * size_t(hidden_size_) * sizeof(float));
    const size_t moe_router_bytes = align256(size_t(cap) * 512u * sizeof(bf16));
    const size_t moe_h_bytes = align256(size_t(cap) * size_t(cfg_.num_experts_per_tok) * size_t(cfg_.moe_intermediate_size) * sizeof(bf16));
    const size_t moe_sh_gu_bytes = align256(size_t(cap) * 1280u * sizeof(bf16));
    const size_t moe_sh_in_bytes = align256(size_t(cap) * 640u * sizeof(bf16));
    const size_t moe_sh_out_bytes = align256(size_t(cap) * size_t(hidden_size_) * sizeof(bf16));
    const size_t moe_phase_bytes = moe_down_bytes +
        (alias_prefill_phases ? (moe_router_bytes + moe_h_bytes + moe_sh_gu_bytes + moe_sh_in_bytes + moe_sh_out_bytes) : 0u);
    const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
    const int indexer_q_dim = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
    const auto kv_stream_plan = aila::memory::qwen4exp_kv_stream_plan(max_seq_len_);
    const bool int8_kv = aila::env::read_flag("AILA_KV_QUANT", false) &&
                         aila::env::kv_quant_format_is_int8();
    const size_t kv_elem_bytes = int8_kv ? 1u : sizeof(bf16);
    const size_t kv_stage_k_max = (alias_prefill_phases && kv_stream_plan.enabled)
        ? align256(size_t(full_kv_heads_) * size_t(max_seq_len_) * size_t(full_head_dim_) * kv_elem_bytes)
        : 0u;
    const size_t kv_stage_v_max = kv_stage_k_max;
    const size_t kv_stage_scale_max = (alias_prefill_phases && kv_stream_plan.enabled && int8_kv)
        ? align256(size_t(full_kv_heads_) * size_t(max_seq_len_) * sizeof(float))
        : 0u;
    const size_t kv_stage_dense_pair = (alias_prefill_phases && kv_stream_plan.enabled)
        ? (align256(size_t(full_kv_heads_) * size_t(dense_kv_cap) * size_t(full_head_dim_) * kv_elem_bytes) * 2u +
           (int8_kv ? align256(size_t(full_kv_heads_) * size_t(dense_kv_cap) * sizeof(float)) * 2u : 0u))
        : 0u;
    const size_t qsa_base_bytes_raw = align256(std::max({
        size_t(attn_cap) * size_t(full_fused_qkv_dim_) * sizeof(bf16),
        size_t(full_q_heads_) * size_t(dense_cap) * size_t(dense_cap) * sizeof(float),
        size_t(full_q_heads_) * size_t(dense_cap) * size_t(dense_kv_cap) * sizeof(float)}));
    const size_t qsa_base_bytes = std::max(qsa_base_bytes_raw, kv_stage_k_max);
    const size_t qsa_q_bytes = align256(size_t(attn_cap) * size_t(full_q_dim_) * sizeof(bf16));
    const size_t qsa_z_bytes = align256(size_t(attn_cap) * size_t(full_q_dim_) * sizeof(bf16));
    const size_t qsa_k_bytes = align256(size_t(attn_cap) * size_t(full_kv_dim_) * sizeof(bf16));
    const size_t qsa_v_bytes = align256(size_t(attn_cap) * size_t(full_kv_dim_) * sizeof(bf16));
    const size_t qsa_idx_qk_bytes = align256(size_t(attn_cap) * size_t(indexer_out_dim) * sizeof(bf16));
    const size_t qsa_idx_q_bytes = align256(size_t(attn_cap) * size_t(indexer_q_dim) * sizeof(bf16));
    const size_t qsa_tail_bytes = qsa_q_bytes + qsa_z_bytes + qsa_k_bytes + qsa_v_bytes + qsa_idx_qk_bytes + qsa_idx_q_bytes;
    const size_t qsa_stage_tail_bytes = (alias_prefill_phases && kv_stream_plan.enabled)
        ? std::max(kv_stage_v_max + 2u * kv_stage_scale_max, kv_stage_dense_pair)
        : 0u;
    buf_.kv_stream_stage_in_phase_scratch = (alias_prefill_phases && kv_stream_plan.enabled);
    buf_.kv_stream_stage_k_max_bytes = kv_stage_k_max;
    buf_.kv_stream_stage_v_max_bytes = kv_stage_v_max;
    buf_.kv_stream_stage_scale_bytes = kv_stage_scale_max;
    buf_.kv_stream_stage_v_offset = qsa_base_bytes + qsa_tail_bytes;
    buf_.kv_stream_stage_k_scale_offset = buf_.kv_stream_stage_v_offset + kv_stage_v_max;
    buf_.kv_stream_stage_v_scale_offset = buf_.kv_stream_stage_k_scale_offset + kv_stage_scale_max;
    if (s_phase_scratch) {
        // Disjoint per-layer lifetimes on the in-order compute queue:
        //   (0) Layer-1 PLE intermediates (key/val/norm/gv)
        //   (1) Hyper-Connection mixer intermediates (hc_N, hc_mix_down, hc_mix_up)
        //   (2) Attention (full_qkv / dense scores + q/z/k/v/indexer_qk/indexer_q, or GDN linear_all + raw + prepared)
        //   (3) MoE (moe_down_acc + router_scores + moe_h + shared expert intermediates)
        const size_t gdn_lin_bytes = align256(size_t(attn_cap)*linear_all_dim_*sizeof(bf16));
        const size_t gdn_raw_bytes = size_t(attn_cap)*size_t(linear_kv_heads_)*128u*sizeof(float);
        const size_t gdn_prep_bytes = size_t(attn_cap)*size_t(linear_kv_heads_)*386u*sizeof(float);
        const size_t bytes[] = {
            size_t(attn_cap)*full_fused_qkv_dim_*sizeof(bf16),
            (gdn_chunk_mode_ > 0 && grouped_xmx_supported_)
                ? gdn_lin_bytes + gdn_raw_bytes + gdn_prep_bytes
                : size_t(attn_cap)*linear_all_dim_*sizeof(bf16),
            size_t(full_q_heads_)*dense_cap*dense_cap*sizeof(float),
            size_t(full_q_heads_)*dense_cap*dense_kv_cap*sizeof(float),
            s_expert_major ? moe_phase_bytes : 0,
            alias_prefill_phases ? hc_phase_bytes : 0,
            alias_prefill_phases ? (qsa_base_bytes + qsa_tail_bytes + qsa_stage_tail_bytes) : 0};
        size_t peak=0,total=0;for(size_t n:bytes){peak=std::max(peak,n);total+=n;}
        buf_.phase_scratch=Tensor::allocate(ctx,{int64_t(peak)},dnnl::memory::data_type::u8);
        AILA_LOG_INFO("[Q4E PhaseScratch] cap=%d bytes=%zu separate_bytes=%zu saved_bytes=%zu",
            cap,peak,total,total-peak);
    }
    auto phase = [&](const std::vector<int64_t>& shape,dnnl::memory::data_type type) {
        return s_phase_scratch ? Tensor::view(ctx,buf_.phase_scratch.data(),shape,type)
                               : Tensor::allocate(ctx,shape,type);
    };
    auto phase_at = [&](size_t byte_offset, const std::vector<int64_t>& shape, dnnl::memory::data_type type) {
        return alias_prefill_phases
            ? Tensor::view(ctx, buf_.phase_scratch.data_as<uint8_t>() + byte_offset, shape, type)
            : Tensor::allocate(ctx, shape, type);
    };

    buf_.hidden4 = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_N = phase_at(0, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_mix_down = phase_at(hc_n_bytes, {cap, 320}, dnnl::memory::data_type::bf16);
    buf_.hc_mix_up = phase_at(hc_n_bytes + hc_down_bytes, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_inject_buf = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::bf16);
    buf_.hc_inject_f32 = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::f32);
    buf_.hc_inject_prev_f32 = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::f32);
    buf_.x = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);
    buf_.y = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    // Full Attention buffers
    const size_t off_qsa_q = qsa_base_bytes;
    const size_t off_qsa_z = off_qsa_q + qsa_q_bytes;
    const size_t off_qsa_k = off_qsa_z + qsa_z_bytes;
    const size_t off_qsa_v = off_qsa_k + qsa_k_bytes;
    const size_t off_qsa_idx_qk = off_qsa_v + qsa_v_bytes;
    const size_t off_qsa_idx_q = off_qsa_idx_qk + qsa_idx_qk_bytes;
    buf_.full_qkv = phase({attn_cap, full_fused_qkv_dim_}, dnnl::memory::data_type::bf16);
    buf_.q = phase_at(off_qsa_q, {attn_cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.z = phase_at(off_qsa_z, {attn_cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.k = phase_at(off_qsa_k, {attn_cap, full_kv_dim_}, dnnl::memory::data_type::bf16);
    buf_.v = phase_at(off_qsa_v, {attn_cap, full_kv_dim_}, dnnl::memory::data_type::bf16);
    buf_.attn_out = Tensor::allocate(ctx, {attn_cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    const bool has_qsa_buf = layers_.size() > 3 && layers_[3].index_qk_weight != nullptr;
    const int buf_comp_ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
    const int buf_block_topk = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / buf_comp_ratio;
    const int decode_scores_cap = has_qsa_buf
        ? std::min(max_seq_len_, std::max(4096, buf_block_topk * buf_comp_ratio + 128))
        : max_seq_len_;
    buf_.decode_scores = Tensor::allocate(ctx, {full_q_heads_, decode_scores_cap}, dnnl::memory::data_type::f32);
    constexpr int kDecodeExactTile = 128;
    const int partial_stride = (full_head_dim_ == 256) ? 272 : 144;
    const int max_tiles = has_qsa_buf
        ? std::max(68, (decode_scores_cap + kDecodeExactTile - 1) / kDecodeExactTile)
        : ((max_seq_len_ + kDecodeExactTile - 1) / kDecodeExactTile);
    buf_.decode_attn_partials = Tensor::allocate(
        ctx,
        {(int64_t)full_q_heads_, (int64_t)max_tiles, (int64_t)partial_stride},
        dnnl::memory::data_type::f32);
    // D1: Limit dense attention scratchpads to the maximum dense prefill length (2052).
    // Prefill exceeding this threshold runs sparse QSA attention via decode_scores and
    // decode_attn_partials, eliminating the 6 GiB - 96 GiB incr_scores memory explosion.
    buf_.scores = phase({full_q_heads_, dense_cap, dense_cap}, dnnl::memory::data_type::f32);
    buf_.incr_scores = phase({full_q_heads_, dense_cap, dense_kv_cap}, dnnl::memory::data_type::f32);

    // QSA Indexer buffers (M2, A3, A4)
    buf_.indexer_qk = phase_at(off_qsa_idx_qk, {attn_cap, indexer_out_dim}, dnnl::memory::data_type::bf16);
    buf_.indexer_q = phase_at(off_qsa_idx_q, {attn_cap, indexer_q_dim}, dnnl::memory::data_type::bf16);
    if (kv_stream_plan.enabled && aila::env::read_flag("AILA_QSA_COMPRESSED_K", true)) {
        buf_.indexer_k_stage = Tensor::allocate(ctx, {attn_cap + 4, cfg_.indexer_head_dim}, dnnl::memory::data_type::bf16);
    } else {
        buf_.indexer_k_stage = Tensor{};
    }
    const int max_complete_blocks = (max_seq_len_ + 3) / 4;
    buf_.indexer_block_scores = Tensor::allocate(ctx, {max_complete_blocks}, dnnl::memory::data_type::f32);
    buf_.indexer_active_tokens = Tensor::allocate(ctx, {std::max(max_seq_len_, cfg_.indexer_budget + cfg_.indexer_compress_ratio)}, dnnl::memory::data_type::s32);
    const int block_topk = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / (cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4);
    buf_.indexer_active_blocks = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);
    // Round 3 measurements on tools/prefill-bench.ps1 at 7878 tokens show the tile size is
    // worth real time because each tile is three kernel launches (block scores, top-k, sparse
    // attention) and a 64-row tile makes 123 launches per QSA layer:
    //   tile 64 (old default)  26,233 ms  300.3 tok/s
    //   tile 128               25,733 ms  306.1 tok/s
    //   tile 256               25,397 ms  310.2 tok/s
    //   tile 512               25,172 ms  313.0 tok/s
    //   tile 1024             25,128 ms  313.5 tok/s
    //   tile 4096              25,533 ms  308.5 tok/s
    // so the curve peaks near 1024. The buffer is bounded by a byte budget rather than a fixed
    // tile count, which keeps qsa_tile_block_scores at 16 MB whatever max_seq is; at max_seq 16384
    // that yields 1024 rows and at max_seq 4096 it yields 4096 rows, clamped to 1024.
    static const int s_qsa_logits_budget =
        aila::env::read_int("AILA_QSA_LOGITS_BUDGET_BYTES", 16 * 1024 * 1024);
    const size_t row_bytes = static_cast<size_t>(max_complete_blocks) * sizeof(float);
    int default_tile = 64;
    if (row_bytes > 0) {
        default_tile = std::clamp(static_cast<int>(s_qsa_logits_budget / row_bytes), 16, 1024);
    }
    qsa_query_tile_ = aila::env::read_int("AILA_QSA_QUERY_TILE", default_tile);
    buf_.qsa_tile_block_scores = Tensor::allocate(ctx, {qsa_query_tile_, max_complete_blocks}, dnnl::memory::data_type::f32);
    // The per-tile selector buffers are sized for the largest query tile the
    // prefill loop may derive from the score budget (<= 1024 rows), not just for
    // the configured tile: at long context the score fit shrinks the tile, but
    // the buffer row count must stay independent of that choice.
    const int max_qsa_tile_rows = std::max(qsa_query_tile_, 1024);
    buf_.qsa_tile_active_blocks = Tensor::allocate(ctx, {max_qsa_tile_rows, block_topk + 4}, dnnl::memory::data_type::s32);
    buf_.qsa_tile_valid_counts = Tensor::allocate(ctx, {max_qsa_tile_rows, 2}, dnnl::memory::data_type::s32);
    if (kv_stream_plan.enabled && !buf_.qsa_resolved_blocks.valid()) {
        buf_.qsa_resolved_blocks = Tensor::allocate(ctx, {4, block_topk + 4}, dnnl::memory::data_type::s32);
        buf_.qsa_resolved_tail_indices = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::s32);
    }


    // Linear Attention buffers
    buf_.linear_all = phase({attn_cap, linear_all_dim_}, dnnl::memory::data_type::bf16);
    static const bool s_attention_nf4_q8 = aila::env::read_flag("AILA_Q4E_ATTN_NF4_Q8", false);
    attentionNf4Q8Enabled_ = s_attention_nf4_q8;
    if (attentionNf4Q8Enabled_) {
        const int projection_input = std::max({hidden_size_, linear_kv_dim_, full_q_dim_});
        const int quantized_capacity = ((projection_input + 63) / 64) * 64;
        if (!buf_.attn_xq.valid() || buf_.attn_xq.numel() < quantized_capacity) {
            // Shared by successive decode projections on the in-order queue;
            // retain these fixed-size banks through prefill growth and trimming.
            buf_.attn_xq = Tensor::allocate(ctx, {quantized_capacity}, dnnl::memory::data_type::u8);
            buf_.attn_xs = Tensor::allocate(ctx, {quantized_capacity / 64}, dnnl::memory::data_type::f32);
        }
    }
    if (gdn_chunk_mode_ > 0 && grouped_xmx_supported_) {
        void* ext_raw = nullptr;
        void* ext_prep = nullptr;
        const size_t lin_bytes = (size_t(attn_cap) * linear_all_dim_ * sizeof(bf16) + 255u) & ~size_t(255u);
        const size_t raw_bytes = size_t(attn_cap) * size_t(linear_kv_heads_) * 128u * sizeof(float);
        const size_t prep_bytes = size_t(attn_cap) * size_t(linear_kv_heads_) * 386u * sizeof(float);
        if (s_phase_scratch && buf_.phase_scratch.valid() &&
            lin_bytes + raw_bytes + prep_bytes <= buf_.phase_scratch.size_bytes()) {
            ext_raw = buf_.phase_scratch.data_as<uint8_t>() + lin_bytes;
            ext_prep = buf_.phase_scratch.data_as<uint8_t>() + lin_bytes + raw_bytes;
        }
        gdn_chunk_scratch_.Reserve(ctx, attn_cap, linear_kv_heads_, ext_raw, ext_prep);
    }

    // MoE buffers (aliased into the upper tail of phase_scratch after moe_down_acc when cap > 64)
    const size_t off_router = moe_down_bytes;
    const size_t off_moe_h = off_router + moe_router_bytes;
    const size_t off_sh_gu = off_moe_h + moe_h_bytes;
    const size_t off_sh_in = off_sh_gu + moe_sh_gu_bytes;
    const size_t off_sh_out = off_sh_in + moe_sh_in_bytes;
    buf_.router_scores = phase_at(off_router, {cap, 512}, dnnl::memory::data_type::bf16);
    free_pinned_host_buffers();
    router_host_ids_ = static_cast<int*>(ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
    router_host_weights_ = static_cast<float*>(ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(float)));
    embed_host_ids_ = static_cast<int*>(ctx.alloc_host(static_cast<size_t>(cap) * sizeof(int)));
    if (embed_host_.valid()) {
        embed_host_rows_ = static_cast<uint8_t*>(ctx.alloc_host(static_cast<size_t>(cap) * hidden_size_ * sizeof(bf16)));
    }
    const bool s_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);

    if (s_expert_major || s_moe_rows) {
        batch_weights_dev_ = static_cast<float*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(float)));
    }
    if (s_moe_rows && !s_expert_major) {
        batch_slots_host_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
        batch_slots_dev_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
    }
    buf_.moe_h = phase_at(off_moe_h, {cap, cfg_.num_experts_per_tok, cfg_.moe_intermediate_size}, dnnl::memory::data_type::bf16);
    static const bool s_gpu_nf4_q8 = aila::env::read_flag("AILA_Q4E_GPU_NF4_Q8", false);
    moeFrontFusedGpu_ = ctx.queue().get_device().is_gpu();
    gpuNf4Q8Supported_ = s_gpu_nf4_q8 && hidden_size_ == 2560 && cfg_.moe_intermediate_size == 640 &&
        cfg_.num_experts_per_tok >= 1 && cfg_.num_experts_per_tok <= 16 &&
        ctx.queue().get_device().get_info<sycl::info::device::max_work_group_size>() >= 1024;
    if (gpuNf4Q8Supported_ && !buf_.moe_x_q8.valid()) {
        // Independent of the prefill capacity: these tiny buffers persist through
        // prompt growth/trim and introduce no steady-state decode allocations.
        buf_.moe_x_q8 = Tensor::allocate(ctx, {hidden_size_}, dnnl::memory::data_type::u8);
        buf_.moe_x_q8_scale = Tensor::allocate(ctx, {hidden_size_ / 64}, dnnl::memory::data_type::f32);
        buf_.moe_h_q8 = Tensor::allocate(ctx, {cfg_.num_experts_per_tok, cfg_.moe_intermediate_size}, dnnl::memory::data_type::u8);
        buf_.moe_h_q8_scale = Tensor::allocate(ctx, {cfg_.num_experts_per_tok, cfg_.moe_intermediate_size / 64}, dnnl::memory::data_type::f32);
    }
    buf_.shared_gate_buf = Tensor::allocate(ctx, {cap, 1}, dnnl::memory::data_type::bf16);
    buf_.shared_gate_up_buf = phase_at(off_sh_gu, {cap, 1280}, dnnl::memory::data_type::bf16);
    buf_.shared_inter_buf = phase_at(off_sh_in, {cap, 640}, dnnl::memory::data_type::bf16);
    buf_.shared_out_buf = phase_at(off_sh_out, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    if (s_expert_major) {
        buf_.moe_down_acc = phase({down_rows, hidden_size_}, dnnl::memory::data_type::f32);

        expert_offsets_host_ = static_cast<int*>(ctx.alloc_host((cfg_.num_experts + 1) * sizeof(int)));
        expert_counts_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        // At most one descriptor per activation, including every padded expert tail.
        expert_tiles_host_ = static_cast<ops::MoeExpertTile*>(ctx.alloc_host(
            static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertTile)));
        expert_tiles_dev_ = Tensor::allocate(ctx, {cap, cfg_.num_experts_per_tok, 2}, dnnl::memory::data_type::s32);
        expert_curr_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_eight_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_quad_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_pair_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_single_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_map_token_host_ = static_cast<int*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_rank_host_ = static_cast<int*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_slot_host_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
        expert_map_token_dev_ = static_cast<int*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_rank_dev_ = static_cast<int*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_slot_dev_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));

        // R1 expert-XMX path (docs/plans/qwen38-r1-prefill-expert-xmx-plan-2026-10-16.md):
        // per-chunk plan staging (double buffered so host rewrites cannot race
        // the async H2D) plus the flag-only ops scratch. Everything here is
        // inert when AILA_Q4E_PREFILL_EXPERT_XMX is unset.
        {
            // Adopted default-on 2026-10-16 (owner waiver): the expert-XMX path
            // measured TTFT 63.99s vs 69.6-71.2s ALU (-8..-10%) and its
            // deviation is accumulation-order bound (KL 0.019 vs the 0.01
            // contract; two XMX variants differ 0.0099; fp16 storage tightened
            // the kernel 2.1x with no e2e change). AILA_Q4E_PREFILL_EXPERT_XMX=0
            // returns to the ALU path.
            //
            // Round 2 measurements on tools/prefill-bench.ps1, all at the shipped
            // 8192 chunk, show the choice depends entirely on the row count M:
            //   M=2026  XMX 13,396 ms vs ALU 11,838 ms   XMX loses 13%
            //   M=2995  XMX 15,641 ms vs ALU 15,412 ms   XMX loses  1%
            //   M=4952  XMX 19,971 ms vs ALU 22,483 ms   XMX wins  13%
            //   M=7878  XMX 26,159 ms vs ALU 33,042 ms   XMX wins  26%
            // A small M starves the systolic units, which is why an earlier
            // chunk-2048 measurement concluded this path was a net loss. With the
            // 8192 chunk the large-M case is common, so the path is enabled and
            // MIN_M (below) keeps small chunks on the ALU kernels.
            static const bool s_prefill_expert_xmx =
                aila::env::read_flag("AILA_Q4E_PREFILL_EXPERT_XMX", true);
            const size_t bases_bytes = size_t(17) * sizeof(int);
            const size_t rows_bytes = size_t(cap) * cfg_.num_experts_per_tok * sizeof(int);
            const size_t ranks_off = bases_bytes + rows_bytes;
            const size_t slots_off = (ranks_off + rows_bytes + 63u) & ~size_t(63u);
            const size_t slots_bytes = size_t(16) * sizeof(ops::MoeExpertSlotDev);
            const size_t per_chunk_hdr = ((bases_bytes + slots_bytes + 127u) & ~size_t(63u));
            xmx_plan_bytes_ = int(slots_off + slots_bytes + size_t(kMaxXmxPlanChunks) * per_chunk_hdr);
            xmx_tokens_off_ = int(bases_bytes);
            xmx_ranks_off_ = int(ranks_off);
            xmx_slots_off_ = int(slots_off);
            if (s_prefill_expert_xmx) {
                xmx_plan_pinned_[0] = static_cast<unsigned char*>(ctx.alloc_host(size_t(xmx_plan_bytes_)));
                xmx_plan_pinned_[1] = static_cast<unsigned char*>(ctx.alloc_host(size_t(xmx_plan_bytes_)));
                if (!xmx_plan_pinned_[0]) xmx_plan_host_[0].assign(size_t(xmx_plan_bytes_), 0);
                else xmx_plan_host_[0].clear();
                if (!xmx_plan_pinned_[1]) xmx_plan_host_[1].assign(size_t(xmx_plan_bytes_), 0);
                else xmx_plan_host_[1].clear();
                xmx_plan_dev_[0] = static_cast<unsigned char*>(ctx.alloc_device(size_t(xmx_plan_bytes_)));
                xmx_plan_dev_[1] = static_cast<unsigned char*>(ctx.alloc_device(size_t(xmx_plan_bytes_)));
                const bool scratch_ok = (xmx_scratch_.impl != nullptr) ||
                    ops::moe_expert_xmx_scratch_create(ctx, xmx_scratch_, 4096,
                        hidden_size_, cfg_.moe_intermediate_size);
                if (xmx_plan_dev_[0] && xmx_plan_dev_[1] && scratch_ok) {
                    prefill_expert_xmx_ = true;
                    const int c1_f32 = aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32", 0);
                    const int w_f16 = aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_W_F16", 0);
                    AILA_LOG_INFO(
                        "[Q4E PrefillExpertXmx] enabled=1 max_chunk_rows=4096 channels=%d/%d "
                        "plan_bytes=%d c1_f32=%d w_f16=%d", hidden_size_, cfg_.moe_intermediate_size,
                        xmx_plan_bytes_, (c1_f32 != 0 || w_f16 != 0) ? 1 : 0, w_f16 != 0 ? 1 : 0);
                } else {
                    AILA_LOG_WARN("[Q4E PrefillExpertXmx] unavailable (plan or scratch allocation "
                                  "failed); falling back to the ALU path");
                }
            }
        }
    } else {
        buf_.moe_down_acc = Tensor{};
    }

    if (!buf_.logits.valid()) {
        buf_.logits = Tensor::allocate(ctx, {1, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
    }

    runtime_seq_capacity_ = cap;
}

// ============================================================================
// Forward Implementation
// ============================================================================

void Qwen4ExpBnb4Backend::CommitStreamedIndexerTail(
    Context& ctx, Session& session, int prevLen, int committedLen) {
    if (committedLen <= prevLen) return;
    const int ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
    const int baseBlock = prevLen / ratio;
    const int committedBlock = committedLen / ratio;
    const int rem = committedLen % ratio;
    if (committedBlock <= baseBlock || rem <= 0) return;
    const int srcOff = (committedBlock - baseBlock) * ratio;
    for (size_t i = 0; i < session.layerCaches_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        if (!cache.indexer_k_tail_only || !cache.indexer_k.valid()) continue;
        ctx.queue().memcpy(
            cache.indexer_k.data(),
            cache.indexer_k.data_as<bf16>() + static_cast<size_t>(srcOff) * cfg_.indexer_head_dim,
            static_cast<size_t>(rem) * cfg_.indexer_head_dim * sizeof(bf16));
    }
}

void Qwen4ExpBnb4Backend::TrimDecodeRuntimeBuffers(Context& ctx) {
    static const int explicit_trim_cap = aila::env::read_int_raw("AILA_Q4E_DECODE_TRIM_CAP", 0);
    if (explicit_trim_cap < 0)
        throw std::invalid_argument("AILA_Q4E_DECODE_TRIM_CAP requires 0 or a capacity of at least 64");
    int trim_cap = explicit_trim_cap;
    const char* trim_reason = "explicit";
    aila::DxgiLocalVramInfo dxgi_before{};
    if (trim_cap == 0) {
        static const bool s_auto_trim = aila::env::read_flag("AILA_Q4E_AUTO_DECODE_TRIM", true);
        if (!s_auto_trim || runtime_seq_capacity_ <= 256) return;
        dxgi_before = aila::QueryDxgiLocalVram();
        const uint64_t trigger_bytes =
            static_cast<uint64_t>(std::max(128, aila::env::read_int("AILA_Q4E_DECODE_TRIM_TRIGGER_MB", 2048))) << 20;
        if (!dxgi_before.valid || dxgi_before.availableBytes >= trigger_bytes) return;
        trim_cap = 64;
        trim_reason = "dxgi_low_headroom";
    }
    if (trim_cap < 64)
        throw std::invalid_argument("AILA_Q4E_DECODE_TRIM_CAP requires 0 or a capacity of at least 64");
    if (runtime_seq_capacity_ <= trim_cap) return;

    // This boundary follows the last explicit prefill forward, including its
    // logits/checkpoint consumers. Keep KV, recurrent state, weights, logits,
    // oneDNN scratch and resident tables; replace only capacity-sized runtime
    // scratch and its host mirrors. The next large prefill grows normally.
    expert_executor_.AbortResidentCold();
    ctx.synchronize();
    if (!dxgi_before.valid) dxgi_before = aila::QueryDxgiLocalVram();
    const int old_cap = runtime_seq_capacity_;
    const uint64_t bytes_before = ctx.current_allocated_bytes();
    runtime_seq_capacity_ = 0;
    ensure_runtime_buffers(ctx, trim_cap);
    const uint64_t bytes_after = ctx.current_allocated_bytes();
    const auto dxgi_after = aila::QueryDxgiLocalVram();
    AILA_LOG_INFO("[Q4E DecodeTrim] reason=%s cap=%d->%d usm_mb=%.1f->%.1f freed_mb=%.1f dxgi_used_mb=%.0f->%.0f/%.0f dxgi_avail_mb=%.0f->%.0f",
        trim_reason, old_cap, runtime_seq_capacity_,
        bytes_before / (1024.0 * 1024.0), bytes_after / (1024.0 * 1024.0),
        (bytes_before > bytes_after ? bytes_before - bytes_after : 0) / (1024.0 * 1024.0),
        dxgi_before.valid ? dxgi_before.currentUsageBytes / (1024.0 * 1024.0) : 0.0,
        dxgi_after.valid ? dxgi_after.currentUsageBytes / (1024.0 * 1024.0) : 0.0,
        dxgi_after.valid ? dxgi_after.dedicatedVideoMemoryBytes / (1024.0 * 1024.0) : 0.0,
        dxgi_before.valid ? dxgi_before.availableBytes / (1024.0 * 1024.0) : 0.0,
        dxgi_after.valid ? dxgi_after.availableBytes / (1024.0 * 1024.0) : 0.0);
}

void Qwen4ExpBnb4Backend::WarmupPrefillKernels(Context& ctx) {
    try {
        if (!buf_.phase_scratch.valid() || layers_.empty()) {
            AILA_LOG_WARN("[Warmup] Q4E prefill kernel warmup skipped (runtime buffers absent)");
            return;
        }
        const size_t scratch_bytes = buf_.phase_scratch.size_bytes();
        unsigned char* scratch = buf_.phase_scratch.data_as<uint8_t>();
        int dense_warmed = 0;
        int xmx_warmed = 0;

        // 1. Dense NF4 oneDNN prefill GEMM at M = 128 across all distinct Bnb4BitLinear shapes.
        //    Warming linear_all_proj (2560->16480) first pre-allocates linear_scratch_.weight_bf16
        //    to its maximum 84.4 MB size during warmup instead of inside GenerateMessages.
        constexpr int kWarmSeq = 128;
        Bnb4BitLinear* projections[5] = {
            &layers_[0].linear_all_proj,
            &layers_[0].linear_o_proj,
            &layers_[0].shared_gate_up_proj,
            &layers_[0].shared_down_proj,
            layers_.size() > 3 ? &layers_[3].qkv_proj : nullptr
        };
        for (Bnb4BitLinear* projection : projections) {
            if (!projection || projection->in_features() <= 0 || projection->out_features() <= 0) continue;
            const size_t in_bytes =
                size_t(kWarmSeq) * size_t(projection->in_features()) * sizeof(bf16);
            const size_t out_bytes =
                size_t(kWarmSeq) * size_t(projection->out_features()) * sizeof(bf16);
            const size_t out_off = (in_bytes + 255u) & ~size_t(255u);
            if (out_off + out_bytes > scratch_bytes) continue;
            ctx.queue().memset(scratch, 0, out_off + out_bytes);
            Tensor warm_input = Tensor::view(ctx, scratch,
                {int64_t(kWarmSeq), int64_t(projection->in_features())},
                dnnl::memory::data_type::bf16);
            Tensor warm_output = Tensor::view(ctx, scratch + out_off,
                {int64_t(kWarmSeq), int64_t(projection->out_features())},
                dnnl::memory::data_type::bf16);
            projection->forward(ctx, linear_scratch_, warm_input, warm_output, kWarmSeq);
            ++dense_warmed;
        }

        if (prefill_expert_xmx_ && xmx_scratch_.impl != nullptr) {
            // Pre-populate the (B, M) oneDNN primitives the prefill chunk
            // scheduler targets so no JIT compilation happens mid-prefill.
            ops::moe_expert_xmx_warmup_primitives(ctx, xmx_scratch_, hidden_size_,
                cfg_.moe_intermediate_size);
        }

        // 2. Expert-XMX prefill chunk (AILA_Q4E_PREFILL_EXPERT_XMX).
        if (prefill_expert_xmx_ && xmx_scratch_.impl != nullptr &&
            adaptive_cache_.IsAllocated() && xmx_plan_dev_[0] != nullptr &&
            host_arena_.IsAllocated()) {
            const int warm_top_k = 1;
            const int rows_per_band = 192;
            const int hidden = hidden_size_;
            const int intermediate = cfg_.moe_intermediate_size;
            const size_t x_bytes = size_t(rows_per_band) * size_t(hidden) * sizeof(bf16);
            const size_t down_bytes =
                size_t(rows_per_band) * size_t(warm_top_k) * size_t(hidden) * sizeof(float);
            const size_t down_off = (x_bytes + 255u) & ~size_t(255u);
            int warm_layer = -1;
            int warm_slot = -1;
            for (int l = 0; l < static_cast<int>(layers_.size()) && warm_slot < 0; ++l) {
                for (int e = 0; e < cfg_.num_experts; ++e) {
                    const int slot = adaptive_cache_.GetSlot(l, e);
                    if (slot >= 0) { warm_layer = l; warm_slot = slot; break; }
                }
            }
            if (warm_slot >= 0 && down_off + down_bytes <= scratch_bytes) {
                unsigned char* plan_host = xmx_plan_pinned_[0] ? xmx_plan_pinned_[0] : xmx_plan_host_[0].data();
                const size_t plan_bytes = size_t(xmx_slots_off_) + sizeof(ops::MoeExpertSlotDev);
                std::memset(plan_host, 0, plan_bytes);
                auto* plan_bases = reinterpret_cast<int*>(plan_host);
                auto* plan_tokens = reinterpret_cast<int*>(plan_host + xmx_tokens_off_);
                auto* plan_ranks = reinterpret_cast<int*>(plan_host + xmx_ranks_off_);
                auto* plan_slots =
                    reinterpret_cast<ops::MoeExpertSlotDev*>(plan_host + xmx_slots_off_);
                plan_bases[0] = 0;
                plan_bases[1] = 1;
                plan_tokens[0] = 0;
                plan_ranks[0] = 0;
                plan_slots[0] = adaptive_cache_.GetResidentSlotDev(
                    warm_slot, &host_arena_.GetLayerNestedMetadata(warm_layer));
                ctx.queue().memcpy(xmx_plan_dev_[0], plan_host, plan_bytes);
                ops::MoeExpertXmxChunk chunk{};
                chunk.num_experts = 1;
                chunk.top_k = warm_top_k;
                chunk.rows_per_band = rows_per_band;
                chunk.member_row_base = reinterpret_cast<const int*>(xmx_plan_dev_[0]);
                chunk.member_token_ids =
                    reinterpret_cast<const int*>(xmx_plan_dev_[0] + xmx_tokens_off_);
                chunk.member_ranks =
                    reinterpret_cast<const int*>(xmx_plan_dev_[0] + xmx_ranks_off_);
                chunk.slots = reinterpret_cast<const ops::MoeExpertSlotDev*>(
                    xmx_plan_dev_[0] + xmx_slots_off_);
                bf16* warm_x = reinterpret_cast<bf16*>(scratch);
                float* warm_down = reinterpret_cast<float*>(scratch + down_off);
                ctx.queue().memset(warm_x, 0, x_bytes);
                ctx.queue().memset(warm_down, 0, down_bytes);
                (void)ops::moe_expert_xmx_prefill_chunk(ctx, xmx_scratch_, chunk, 0, warm_x,
                                                        warm_down, hidden, intermediate, 64,
                                                        nullptr, nullptr);
                xmx_warmed = 1;
            }
        }

        ctx.synchronize();
        AILA_LOG_INFO("[Warmup] Q4E prefill kernels warmed: dense_onednn=%d (M=%d weight_scratch=%zu B) "
                      "expert_xmx=%d phase_scratch=%zu B",
                      dense_warmed, kWarmSeq, size_t(linear_scratch_.weight_bf16.size_bytes()),
                      xmx_warmed, scratch_bytes);
    } catch (const std::exception& error) {
        AILA_LOG_WARN("[Warmup] Q4E prefill kernel warmup skipped: %s", error.what());
        try { ctx.queue().wait_and_throw(); } catch (...) {}
    }
}

sycl::event Qwen4ExpBnb4Backend::RunResidentDecode(Context& ctx, int layer, bool fuse_shared, int count,
                                                   bool verify_rows) {
    if (adaptive_cache_.IsAllocated()) {
        constexpr int kMaxTopK = 16;
        const int top_k = cfg_.num_experts_per_tok, intermediate = cfg_.moe_intermediate_size;
        const bool nested = true;
        // Adaptive descriptors are produced only by MakeResidentSlot, which
        // stamps nested blocksize 256; the kernels drop the runtime division.
        const bool nested_fixed256 = true;
        const auto& nested_meta = host_arena_.GetLayerNestedMetadata(layer);

        auto* h_base = buf_.moe_h.data_as<bf16>();
        auto* down = resident_down_.data_as<float>();
        auto* x_base = buf_.x.data_as<bf16>();
        auto* y_base = buf_.y.data_as<bf16>();
        auto* sh_out_base = fuse_shared ? buf_.shared_out_buf.data_as<bf16>() : nullptr;
        auto* sh_gate_base = fuse_shared ? buf_.shared_gate_buf.data_as<bf16>() : nullptr;

        sycl::event last{};
        struct ResidentColdWork {
            aila::Qwen4ExpExpertExecutor& executor;
            bool active = false;
            ~ResidentColdWork() noexcept {
                if (active) executor.AbortResidentCold();
            }
        } cpu_work{expert_executor_};
        // P1 release gate. With the L0 v1 regular command list, submitted
        // kernels stay in the batch until the next SYCL wait or event status
        // query. Sampling the last event once before the CPU join lets the
        // queued resident/cold kernels execute during the condvar wait
        // (synthetic probe: 29.2 -> 25.4 ms/token; e2e gate pending).
        static const bool s_join_release =
            aila::env::read_int_raw("AILA_Q4E_JOIN_RELEASE", 1) != 0;

        // Opt-in multi-row CPU verification. The native host arena is the RAM
        // complement of the GPU hotset, so every routed miss can be served by
        // the bounded CPU worker in one rows-token batch while the resident
        // hotset runs on the GPU. Enabled only through AILA_Q4E_ARENA_VERIFY_ROWS
        // with the PrepareDecodeVerification scratch present (MTP path) and a
        // complete arena/GPU cover for every routed rank; otherwise the
        // per-token loop below is byte-for-byte unchanged.
        bool arena_verify = verify_rows && count >= 2 && count <= 4 && !decodeHcReferenceForTest_ &&
            verify_down_.valid() && verify_weights_.valid() &&
            expert_executor_.CanExecuteArenaVerifyRows(host_arena_, count, top_k);
        if (arena_verify)
            for (int t = 0; t < count && arena_verify; ++t)
                for (int k = 0; k < top_k; ++k) {
                    const int e = router_host_ids_[t * top_k + k];
                    if (adaptive_cache_.GetSlot(layer, e) < 0 && !host_arena_.GetExpertPointer(layer, e)) {
                        arena_verify = false;
                        break;
                    }
                }
        if (arena_verify) {
            expert_executor_.StageResidentInput(x_base, count);
            float* verify_down = verify_down_.data_as<float>();
            expert_executor_.StartArenaVerifyRows(host_arena_, layer, count, router_host_ids_, sycl::event{});
            ctx.queue().memcpy(verify_weights_.data_as<float>(), router_host_weights_,
                               size_t(count) * top_k * sizeof(float));
            for (int t = 0; t < count; ++t) {
                const int* t_ids = router_host_ids_ + t * top_k;
                const bf16* x_t = x_base + size_t(t) * hidden_size_;
                std::array<int, kMaxTopK> hit_slots{}, hit_ranks{};
                int hit_count = 0;
                for (int k = 0; k < top_k; ++k) {
                    const int slot = adaptive_cache_.GetSlot(layer, t_ids[k]);
                    if (slot >= 0) { hit_slots[hit_count] = slot; hit_ranks[hit_count] = k; ++hit_count; }
                }
                if (cold_epoch_enabled_) adaptive_cache_.ObserveRouting(layer, t_ids, top_k);
                if (export_enabled_ && cold_epoch_enabled_) export_policy_.Observe(layer, t_ids, 1, top_k, false);
                if (!hit_count) continue;
                std::array<ops::MoeExpertSlotDev, kMaxTopK> hot{};
                for (int i = 0; i < hit_count; ++i)
                    hot[i] = adaptive_cache_.GetResidentSlotDev(hit_slots[i], &nested_meta);
                float* t_down = verify_down + size_t(t) * top_k * hidden_size_;
                last = ops::moe_decode_gate_up_swiglu_grouped(ctx, hot.data(), nullptr, x_t, h_base,
                    hit_count, hidden_size_, intermediate, 64, adaptive_cache_.ResidentReady(), nullptr,
                    nested, nested_fixed256);
                last = ops::moe_decode_down_to_rank_f32(ctx, hot.data(), hit_ranks.data(), h_base, t_down,
                    hit_count, top_k, intermediate, hidden_size_, 64, nullptr, nullptr, &last, nested,
                    nested_fixed256);
                adaptive_cache_.RecordResidentConsumer(last);
            }
            const auto cpu_ready = expert_executor_.JoinArenaVerifyRows(verify_down);
            if (cpu_ready != sycl::event{}) last = cpu_ready;
            last = ops::moe_ordered_accumulate_down_rows(ctx, verify_down, verify_weights_.data_as<float>(),
                y_base, count, top_k, hidden_size_, sh_out_base, sh_gate_base, &last);
            if (resident_probe_) resident_probe_(resident_probe_data_, *this, layer);
            return last;
        }


        // G1 (opt-in, AILA_Q4E_VERIFY_CPU_COLD): a 2..4 row verify window
        // serves its cold experts through the same single-row arena CPU batch
        // as the count==1 T1 body, one row at a time and capped per row by
        // max_cpu_credits_ (AILA_Q4E_CPU_MAX_EXPERTS) exactly like T1. The
        // GPU resident hotset and the GPU-staged overflow stay byte-for-byte
        // the T1 path, so each verify row resolves every rank on the same
        // device (and same slot bytes) that a sequential T1 decode of that
        // row would. Stage all rows once so every per-row job indexes the
        // same pinned host image.
        static const bool s_verify_cpu_cold = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_COLD", true);
        const bool verify_cpu_cold = s_verify_cpu_cold && verify_rows && count >= 2 && count <= 4 &&
            count <= expert_executor_.max_cpu_tokens() && expert_executor_.cpu_overlap_enabled();
        if (verify_cpu_cold && !verify_x_already_staged_) expert_executor_.StageResidentInput(x_base, count);

        // G1b (opt-in, AILA_Q4E_VERIFY_CPU_BATCH): same cold prefixes as G1,
        // but the whole window runs as ONE CPU job so the coordinator pays one
        // join instead of one per row. Every row is classified before any GPU
        // work (GetSlot is a pure lookup and ObserveRouting only updates
        // counters), the per-row max_cpu_credits_ cap and the >cap GPU-staging
        // overflow are unchanged, and the loop below re-checks each row's
        // prefix and fails closed on any drift. Any unmet condition drops back
        // to the untouched per-row G1 path.
        static const bool s_verify_cpu_batch = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_BATCH", true);
        bool verify_cpu_batch = s_verify_cpu_batch && verify_cpu_cold && !d1_stash_valid &&
            expert_executor_.CanExecuteArenaColdRows(count);
        bool batch_all_hits = false;
        std::array<std::array<int, kMaxTopK>, 4> pre_hit_slots{}, pre_hit_ranks{}, pre_miss_experts{}, pre_miss_ranks{};
        std::array<int, 4> pre_hit_count{}, pre_miss_count{};
        bool pre_classified = false;
        std::array<int, 4> batch_begin{}, batch_count{};
        if (verify_cpu_batch) {
            pre_classified = true;
            int batch_total = 0;
            for (int t = 0; t < count; ++t) {
                const int* b_ids = router_host_ids_ + t * top_k;
                int b_hit = 0, b_miss = 0;
                for (int k = 0; k < top_k; ++k) {
                    const int exp_id = b_ids[k];
                    const int slot = adaptive_cache_.GetSlot(layer, exp_id);
                    if (slot >= 0) {
                        pre_hit_slots[t][b_hit] = slot;
                        pre_hit_ranks[t][b_hit] = k;
                        ++b_hit;
                    } else {
                        pre_miss_experts[t][b_miss] = exp_id;
                        pre_miss_ranks[t][b_miss] = k;
                        ++b_miss;
                    }
                }
                pre_hit_count[t] = b_hit;
                pre_miss_count[t] = b_miss;
                for (int c = 0; c < b_miss && verify_cpu_batch; ++c)
                    verify_cpu_batch = host_arena_.GetExpertPointer(layer, pre_miss_experts[t][c]) != nullptr;
                batch_count[t] = std::min(b_miss, expert_executor_.max_cpu_credits());
                batch_begin[t] = batch_total;
                batch_total += batch_count[t];
            }
            if (batch_total == 0 && verify_cpu_batch && verify_down_.valid() &&
                expert_executor_.cpu_out_host_ptr() != nullptr) {
                verify_cpu_batch = false;
                batch_all_hits = true;
            } else if (batch_total < 1 || batch_total > ops::cpu::kMaxExpertBatchTasks) {
                verify_cpu_batch = false;
            }
            if (verify_cpu_batch) {
                // Strict row-major partition: the flattened list is exactly the
                // per-row miss prefix the loop below would send one row at a
                // time, so every (row, expert) task keeps its own bytes.
                std::array<int, ops::cpu::kMaxExpertBatchTasks> flat_experts{}, flat_ranks{};
                int offset = 0;
                for (int t = 0; t < count; ++t)
                    for (int c = 0; c < batch_count[t]; ++c, ++offset) {
                        flat_experts[offset] = pre_miss_experts[t][c];
                        flat_ranks[offset] = pre_miss_ranks[t][c];
                    }
                expert_executor_.StartArenaColdRows(host_arena_, layer, count, flat_experts.data(),
                    flat_ranks.data(), batch_begin.data(), batch_count.data(), batch_total,
                    verify_x_already_staged_ ? nullptr : x_base);
                cpu_work.active = true;
            }
        }

        const bool batch_split_gpu_join = (verify_cpu_batch || batch_all_hits) && verify_down_.valid();
        const bool batch_pack_gpu_hot = batch_split_gpu_join;
        std::array<ops::MoeExpertSlotDev, 16> combined_hot{};
        std::array<int, 16> combined_tokens{}, combined_ranks{};
        int combined_hot_count = 0;
        bool batch_any_staging = false;
        bool batch_any_device_live = false;
        std::array<bool, 4> batch_device_live{};
        std::array<sycl::event, 4> batch_device_last{};
        std::array<int, 4> batch_row_cpu_count{};

        for (int t = 0; t < count; ++t) {
            const int* t_ids = router_host_ids_ + t * top_k;
            const float* t_weights = router_host_weights_ + t * top_k;
            const bf16* x_t = x_base + static_cast<size_t>(t) * hidden_size_;
            bf16* y_t = y_base + static_cast<size_t>(t) * hidden_size_;
            const bf16* sh_out_t = sh_out_base ? sh_out_base + static_cast<size_t>(t) * hidden_size_ : nullptr;
            const bf16* sh_gate_t = sh_gate_base ? sh_gate_base + t : nullptr;
            float* row_down = batch_split_gpu_join
                ? (verify_down_.data_as<float>() + static_cast<size_t>(t) * top_k * hidden_size_)
                : down;

            // 1. Separate hits and misses via adaptive cache
            int hit_count = 0;
            int miss_count = 0;
            std::array<int, kMaxTopK> hit_slots{};
            std::array<int, kMaxTopK> hit_ranks{};
            std::array<int, kMaxTopK> miss_experts{};
            std::array<int, kMaxTopK> miss_ranks{};

            if (pre_classified) {
                hit_count = pre_hit_count[t];
                miss_count = pre_miss_count[t];
                for (int i = 0; i < hit_count; ++i) {
                    hit_slots[i] = pre_hit_slots[t][i];
                    hit_ranks[i] = pre_hit_ranks[t][i];
                }
                for (int i = 0; i < miss_count; ++i) {
                    miss_experts[i] = pre_miss_experts[t][i];
                    miss_ranks[i] = pre_miss_ranks[t][i];
                }
            } else {
                for (int k = 0; k < top_k; ++k) {
                    const int exp_id = t_ids[k];
                    const int slot = adaptive_cache_.GetSlot(layer, exp_id);
                    if (slot >= 0) {
                        hit_slots[hit_count] = slot;
                        hit_ranks[hit_count] = k;
                        hit_count++;
                    } else {
                        miss_experts[miss_count] = exp_id;
                        miss_ranks[miss_count] = k;
                        miss_count++;
                    }
                }
            }

            // 2. Observe routing for frequency tracking & stats
            if (cold_epoch_enabled_) {
                adaptive_cache_.ObserveRouting(layer, t_ids, top_k, 1.0f);
            }
            if (export_enabled_ && cold_epoch_enabled_) export_policy_.Observe(layer, t_ids, 1, top_k, false);

            // Native CPU sources remain owned until Join below. CPU work can
            // overlap the already submitted shared expert and resident kernels.
            if (d1_stash_valid && d1_stash_layer != layer) d1_stash_valid=false;
            int cpu_count=0;
            bool direct_stage_for_row=direct_arena_stage_;
            // The multi-row gate falls back to the unchanged all-GPU staging
            // path whenever a routed miss has no native RAM owner, mirroring
            // the per-rank ownership the count==1 branch would have required.
            bool cpu_cold_ok = true;
            if (verify_cpu_cold && count > 1 && !pre_classified)
                for (int c = 0; c < miss_count && cpu_cold_ok; ++c)
                    cpu_cold_ok = host_arena_.GetExpertPointer(layer, miss_experts[c]) != nullptr;
            if (d1_stash_valid && d1_stash_layer == layer) {
                // D1: the CPU batch was already started at the router wait.
                // It must have taken exactly this miss prefix; otherwise the
                // early classification would silently change which rows the
                // CPU computes, so fail closed instead.
                cpu_count = d1_stash_cpu_count;
                if (cpu_count<0 || cpu_count>miss_count)
                    throw std::logic_error("Qwen4Exp: early CPU route prefix out of range");
                for (int c=0;c<cpu_count;++c)
                    if (d1_stash_ids[c]!=miss_experts[c] || d1_stash_ranks[c]!=miss_ranks[c])
                        throw std::logic_error("Qwen4Exp: early CPU route classification changed");
                d1_stash_valid=false;
            } else if (verify_cpu_batch) {
                cpu_count = batch_count[t];
            } else if (batch_all_hits) {
                cpu_count = 0;
            } else if (expert_executor_.cpu_overlap_enabled() &&
                       (count==1 || (verify_cpu_cold && cpu_cold_ok))) {
                // x_base equals x_t for count==1, so the single-row call is
                // byte-for-byte the T1 call; multi-row calls index the row
                // staged by StageResidentInput above.
                int cpu_target = miss_count;
                const bool direct_cold_split = cold_split_this_forward_ &&
                    cold_split_num_ > 0 && direct_arena_stage_;
                if (direct_cold_split && miss_count > 0) {
                    // Smooth the target over the token instead of applying
                    // floor(miss_count * share) independently per layer.
                    cold_split_credit_ += miss_count * cold_split_num_;
                    const int gpu_share = std::min(miss_count, cold_split_credit_ >> 8);
                    cold_split_credit_ &= 255;
                    cpu_target = miss_count - gpu_share;
                    direct_stage_for_row = true;
                    static bool split_logged = false;
                    if (!split_logged) {
                        split_logged = true;
                        AILA_LOG_INFO("[Q4E DirectColdSplit] enabled=1 share=%d/256 "
                                      "scope=token cpu_max=%d direct_dma=1",
                            cold_split_num_, expert_executor_.max_cpu_credits());
                    }
                }
                cpu_count = expert_executor_.StartArenaCold(host_arena_,layer,miss_experts.data(),miss_ranks.data(),cpu_target,
                    verify_x_already_staged_ ? nullptr : x_base, t, count);
            }
            cpu_work.active=verify_cpu_batch || cpu_count>0;
            const int gpu_cold_count=miss_count-cpu_count;
            StagingBatch staging{};
            if (gpu_cold_count>0) {
                sycl::queue& dma_q = copy_q_ ? *copy_q_ : ctx.queue();
                staging = adaptive_cache_.StageColdExpertsAsync(
                    host_arena_, dma_q, layer, miss_experts.data()+cpu_count, gpu_cold_count, ping_pong_index_,
                    &expert_executor_, direct_stage_for_row);
                ping_pong_index_ ^= 1;
            }

            // 4. Build GPU descriptors and launch XMX fused kernels
            // 4a. Execute hot hits (immediate GPU execution)
            if (hit_count > 0 && batch_pack_gpu_hot && combined_hot_count + hit_count <= 16 &&
                staging.count == 0 && !batch_any_staging) {
                for (int i = 0; i < hit_count; ++i) {
                    combined_hot[static_cast<size_t>(combined_hot_count)] =
                        adaptive_cache_.GetResidentSlotDev(hit_slots[i], &nested_meta);
                    combined_tokens[static_cast<size_t>(combined_hot_count)] = t;
                    combined_ranks[static_cast<size_t>(combined_hot_count)] = hit_ranks[i];
                    ++combined_hot_count;
                }
            } else if (hit_count > 0) {
                if (combined_hot_count > 0) {
                    last = ops::moe_decode_gate_up_swiglu_grouped(
                        ctx, combined_hot.data(), nullptr,
                        x_base, h_base, combined_hot_count,
                        hidden_size_, intermediate, 64, adaptive_cache_.ResidentReady(),
                        combined_tokens.data(), nested, nested_fixed256);
                    last = ops::moe_decode_down_to_rank_f32(
                        ctx, combined_hot.data(), combined_ranks.data(), h_base,
                        verify_down_.data_as<float>(), combined_hot_count, top_k, intermediate, hidden_size_, 64,
                        combined_tokens.data(), nullptr, &last, nested, nested_fixed256);
                    adaptive_cache_.RecordResidentConsumer(last);
                    for (int prev_t = 0; prev_t < t; ++prev_t) {
                        batch_device_live[prev_t] = true;
                        batch_device_last[prev_t] = last;
                    }
                    combined_hot_count = 0;
                }
                std::array<ops::MoeExpertSlotDev, kMaxTopK> hot{};
                for (int i = 0; i < hit_count; ++i) {
                    hot[i] = adaptive_cache_.GetResidentSlotDev(hit_slots[i], &nested_meta);
                }
                last = ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, hot.data(), nullptr,
                    x_t, h_base, hit_count,
                    hidden_size_, intermediate, 64, adaptive_cache_.ResidentReady(), nullptr, nested,
                    nested_fixed256);

                last = ops::moe_decode_down_to_rank_f32(
                    ctx, hot.data(), hit_ranks.data(), h_base,
                    row_down, hit_count, top_k, intermediate, hidden_size_, 64,
                    nullptr, nullptr, &last, nested, nested_fixed256);
                adaptive_cache_.RecordResidentConsumer(last);
            }

            // 4b. Execute cold staging (waiting on PCIe DMA completion)
            if (staging.count > 0) {
                batch_any_staging = true;
                std::array<ops::MoeExpertSlotDev, kMaxTopK> cold{};
                for (int i = 0; i < staging.count; ++i) {
                    cold[i] = adaptive_cache_.GetStagingSlotDev(staging.ping_pong_index, staging.staging_slots[i], &nested_meta);
                }
                auto* cold_h = h_base + size_t(hit_count) * intermediate;

                last = ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, cold.data(), nullptr,
                    x_t, cold_h, staging.count,
                    hidden_size_, intermediate, 64, &staging.ready_event, nullptr, nested,
                    nested_fixed256);

                last = ops::moe_decode_down_to_rank_f32(
                    ctx, cold.data(), miss_ranks.data()+cpu_count, cold_h,
                    row_down, staging.count, top_k, intermediate, hidden_size_, 64,
                    nullptr, nullptr, &last, nested, nested_fixed256);

                adaptive_cache_.RecordStagingConsumer(staging.ping_pong_index,last);
            }

            if (batch_split_gpu_join) {
                batch_device_live[t] = hit_count > 0 || staging.count > 0;
                batch_device_last[t] = last;
                batch_row_cpu_count[t] = cpu_count;
                batch_any_device_live = batch_any_device_live || batch_device_live[t];
                continue;
            }

            // P1: release the queued kernels before the host blocks on the CPU
            // join (a condvar wait, not a SYCL wait), so the GPU executes the
            // resident/cold expert kernels while the CPU finishes its rows.
            bool device_event_live = hit_count > 0 || staging.count > 0;
            if (s_join_release) {
                if (!device_event_live) {
                    last = ctx.queue().ext_oneapi_submit_barrier();
                    device_event_live = true;
                }
                try { (void)last.get_info<sycl::info::event::command_execution_status>(); }
                catch (...) {}
            }
            const sycl::event device_last = last;
            static const bool s_defer_cpu_join = [] {
                const int l0_batch = std::max(
                    aila::env::read_int_raw("UR_L0_BATCH_SIZE", 0),
                    aila::env::read_int_raw("SYCL_PI_LEVEL_ZERO_BATCH_SIZE", 0));
                return l0_batch >= 32 &&
                    aila::env::read_flag("AILA_Q4E_DECODE_DEFER_CPU_JOIN", true);
            }();
            ops::MoeCpuHostDownRows deferred_rows{};
            if (s_defer_cpu_join && s_join_release && count == 1 && !verify_rows &&
                !verify_cpu_batch && staging.count == 0 && cpu_count > 0 &&
                !resident_probe_ &&
                expert_executor_.PrepareDeferredHostReadRows(deferred_rows)) {
                deferred_rows.has_host_weights = true;
                for (int k = 0; k < top_k; ++k)
                    deferred_rows.host_weights[static_cast<size_t>(k)] = t_weights[k];
                last = ops::moe_ordered_accumulate_cpu_host_rows(ctx, down, nullptr,
                    y_t, 1, top_k, hidden_size_, deferred_rows, sh_out_t, sh_gate_t,
                    device_event_live ? &device_last : nullptr);
                deferred_cpu_join_pending_ = true;
                deferred_cpu_merge_event_ = last;
                cpu_work.active = false;
                return last;
            }

            const sycl::event* copy_order =
                (aila::q4e::moe_copy_via_copyq() && device_event_live) ? &device_last : nullptr;
            const auto cpu_ready=verify_cpu_batch
                ? expert_executor_.JoinArenaColdRowsRow(down, t, copy_order)
                : expert_executor_.JoinResidentCold(down, copy_order);
            if (cpu_count) last=cpu_ready;
            cpu_work.active=verify_cpu_batch && t + 1 < count;
            // 5. Accumulate ranked down rows into y
            const auto* cpu_rows=expert_executor_.ResidentCpuHostRows();
            if (cpu_rows) {
                ops::MoeCpuHostDownRows rows_with_weights = *cpu_rows;
                rows_with_weights.has_host_weights = true;
                for (int k = 0; k < top_k; ++k)
                    rows_with_weights.host_weights[static_cast<size_t>(k)] = t_weights[k];
                last = ops::moe_ordered_accumulate_cpu_host_rows(ctx,down,nullptr,
                    y_t,1,top_k,hidden_size_,rows_with_weights,sh_out_t,sh_gate_t,
                    device_event_live ? &device_last : nullptr);
            } else {
                if (aila::q4e::moe_copy_via_copyq() && copy_q_) {
                    const sycl::event* weights_order = copy_order;
                    last = aila::q4e::submit_host_copy(ctx, copy_q_.get(), resident_weights_.data(),
                        t_weights, top_k * sizeof(float), weights_order);
                } else {
                    ctx.queue().memcpy(resident_weights_.data(), t_weights, top_k * sizeof(float));
                }
                last = ops::moe_ordered_accumulate_down_rows(ctx,down,resident_weights_.data_as<float>(),
                    y_t,1,top_k,hidden_size_,sh_out_t,sh_gate_t,&last,expert_executor_.ResidentCpuResultPacket());
            }
            expert_executor_.FinishResidentCpuRead(last);
        }

        if (batch_split_gpu_join) {
            if (combined_hot_count > 0) {
                last = ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, combined_hot.data(), nullptr,
                    x_base, h_base, combined_hot_count,
                    hidden_size_, intermediate, 64, adaptive_cache_.ResidentReady(),
                    combined_tokens.data(), nested, nested_fixed256);
                last = ops::moe_decode_down_to_rank_f32(
                    ctx, combined_hot.data(), combined_ranks.data(), h_base,
                    verify_down_.data_as<float>(), combined_hot_count, top_k, intermediate, hidden_size_, 64,
                    combined_tokens.data(), nullptr, &last, nested, nested_fixed256);
                adaptive_cache_.RecordResidentConsumer(last);
                batch_any_device_live = true;
                for (int t = 0; t < count; ++t) {
                    if (batch_device_live[t]) batch_device_last[t] = last;
                }
            }
            if (batch_all_hits) {
                const sycl::event gpu_hot_done = last;
                const bool gpu_hot_live = batch_any_device_live;
                std::array<ops::MoeCpuHostDownRows, 4> deferred_rows_batch{};
                for (int t = 0; t < count; ++t) {
                    const float* t_weights = router_host_weights_ + t * top_k;
                    auto& hr = deferred_rows_batch[static_cast<size_t>(t)];
                    hr.values = expert_executor_.cpu_out_host_ptr();
                    hr.rows = 1;
                    hr.rank_to_row.fill(-1);
                    hr.has_host_weights = true;
                    for (int k = 0; k < top_k; ++k)
                        hr.host_weights[static_cast<size_t>(k)] = t_weights[k];
                }
                last = ops::moe_ordered_accumulate_cpu_host_rows_batch(
                    ctx, verify_down_.data_as<float>(), y_base, count, top_k, hidden_size_,
                    deferred_rows_batch.data(), sh_out_base, sh_gate_base,
                    gpu_hot_live ? &gpu_hot_done : nullptr);
                if (resident_probe_) resident_probe_(resident_probe_data_, *this, layer);
                return last;
            }
            static const bool s_defer_cpu_join_batch = [] {
                const int l0_batch = std::max(
                    aila::env::read_int_raw("UR_L0_BATCH_SIZE", 0),
                    aila::env::read_int_raw("SYCL_PI_LEVEL_ZERO_BATCH_SIZE", 0));
                return l0_batch >= 32 &&
                    aila::env::read_flag("AILA_Q4E_DECODE_DEFER_CPU_JOIN", true);
            }();
            const bool defer_cpu_join_batch_now = s_defer_cpu_join_batch && s_join_release &&
                !batch_any_staging && !resident_probe_ &&
                expert_executor_.CanDeferArenaColdRowsHostRead();
            // Release all queued GPU expert kernels across the entire multi-row
            // window before blocking on the single batched CPU job, so row 1..N
            // GPU hot experts execute concurrently with the CPU worker instead
            // of waiting in the host batch behind row 0's CPU join. When
            // deferring with zero GPU device work, skip an empty barrier flush.
            if (s_join_release && (batch_any_device_live || !defer_cpu_join_batch_now)) {
                if (last == sycl::event{}) {
                    last = ctx.queue().ext_oneapi_submit_barrier();
                }
                try { (void)last.get_info<sycl::info::event::command_execution_status>(); }
                catch (...) {}
            }
            if (defer_cpu_join_batch_now) {
                const sycl::event gpu_hot_done = last;
                const bool gpu_hot_live = batch_any_device_live;
                std::array<ops::MoeCpuHostDownRows, 4> deferred_rows_batch{};
                for (int t = 0; t < count; ++t) {
                    const float* t_weights = router_host_weights_ + t * top_k;
                    (void)expert_executor_.PrepareDeferredHostReadRowsRow(t, deferred_rows_batch[static_cast<size_t>(t)]);
                    deferred_rows_batch[static_cast<size_t>(t)].has_host_weights = true;
                    for (int k = 0; k < top_k; ++k)
                        deferred_rows_batch[static_cast<size_t>(t)].host_weights[static_cast<size_t>(k)] = t_weights[k];
                }
                last = ops::moe_ordered_accumulate_cpu_host_rows_batch(
                    ctx, verify_down_.data_as<float>(), y_base, count, top_k, hidden_size_,
                    deferred_rows_batch.data(), sh_out_base, sh_gate_base,
                    gpu_hot_live ? &gpu_hot_done : nullptr);
                deferred_cpu_join_pending_ = true;
                deferred_cpu_merge_event_ = last;
                cpu_work.active = false;
                return last;
            }
            for (int t = 0; t < count; ++t) {
                float* row_down = verify_down_.data_as<float>() + static_cast<size_t>(t) * top_k * hidden_size_;
                const float* t_weights = router_host_weights_ + t * top_k;
                bf16* y_t = y_base + static_cast<size_t>(t) * hidden_size_;
                const bf16* sh_out_t = sh_out_base ? sh_out_base + static_cast<size_t>(t) * hidden_size_ : nullptr;
                const bf16* sh_gate_t = sh_gate_base ? sh_gate_base + t : nullptr;
                const bool device_event_live = batch_device_live[t];
                const sycl::event device_last = batch_device_last[t];
                const sycl::event* copy_order =
                    (aila::q4e::moe_copy_via_copyq() && device_event_live) ? &device_last : nullptr;
                const auto cpu_ready = expert_executor_.JoinArenaColdRowsRow(row_down, t, copy_order);
                if (batch_row_cpu_count[t]) last = cpu_ready;
                cpu_work.active = (t + 1 < count);
                const auto* cpu_rows = expert_executor_.ResidentCpuHostRows();
                if (cpu_rows) {
                    ops::MoeCpuHostDownRows rows_with_weights = *cpu_rows;
                    rows_with_weights.has_host_weights = true;
                    for (int k = 0; k < top_k; ++k)
                        rows_with_weights.host_weights[static_cast<size_t>(k)] = t_weights[k];
                    last = ops::moe_ordered_accumulate_cpu_host_rows(ctx, row_down, nullptr,
                        y_t, 1, top_k, hidden_size_, rows_with_weights, sh_out_t, sh_gate_t,
                        device_event_live ? &device_last : nullptr);
                } else {
                    float* row_weights_dev = verify_weights_.valid()
                        ? (verify_weights_.data_as<float>() + static_cast<size_t>(t) * top_k)
                        : resident_weights_.data_as<float>();
                    if (aila::q4e::moe_copy_via_copyq() && copy_q_) {
                        last = aila::q4e::submit_host_copy(ctx, copy_q_.get(), row_weights_dev,
                            t_weights, top_k * sizeof(float), copy_order);
                    } else {
                        ctx.queue().memcpy(row_weights_dev, t_weights, top_k * sizeof(float));
                    }
                    last = ops::moe_ordered_accumulate_down_rows(ctx, row_down, row_weights_dev,
                        y_t, 1, top_k, hidden_size_, sh_out_t, sh_gate_t, &last,
                        expert_executor_.ResidentCpuResultPacket());
                }
                expert_executor_.FinishResidentCpuRead(last);
            }
        }

        if (resident_probe_) resident_probe_(resident_probe_data_, *this, layer);
        return last;
    }
    return sycl::event{};
}

sycl::event Qwen4ExpBnb4Backend::RunAdaptivePrefill(Context& ctx,int layer,bool fuse_shared,int count) {
    const int top_k=cfg_.num_experts_per_tok;
    // Synthetic load-time warmup must not steer the adaptive hotset.
    if (cold_epoch_enabled_) {
        const float prefill_weight = std::min(1.0f, 32.0f / static_cast<float>(std::max(1, count)));
        adaptive_cache_.ObserveRouting(layer, router_host_ids_, count * top_k, prefill_weight);
    }
    if (export_enabled_ && cold_epoch_enabled_) export_policy_.Observe(layer, router_host_ids_, count, top_k, true);
    std::fill(prefill_expert_uses_.begin(),prefill_expert_uses_.end(),0);
    prefill_unique_.clear();
    for (int i=0;i<count*top_k;++i) {
        const int e=router_host_ids_[i];
        if (e<0 || e>=cfg_.num_experts) throw std::runtime_error("Invalid Qwen4Exp route");
        ++prefill_expert_uses_[e];
    }
    xmx_plan_cursor_ = 0;
    xmx_route_bucketed_ = false;
    static const bool s_bucket_route = aila::env::read_flag("AILA_Q4E_XMX_BUCKET_ROUTE", true);
    static const int s_prefill_xmx_gate_rows_top =
        std::max(0, aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_GATE_ROWS", 1024));
    if (s_bucket_route && prefill_expert_xmx_ && count >= s_prefill_xmx_gate_rows_top &&
        expert_offsets_host_ && expert_curr_pos_host_ &&
        expert_map_token_host_ && expert_map_rank_host_) {
        int off = 0;
        for (int e = 0; e < cfg_.num_experts; ++e) {
            expert_offsets_host_[e] = off;
            expert_curr_pos_host_[e] = off;
            off += prefill_expert_uses_[e];
        }
        expert_offsets_host_[cfg_.num_experts] = off;
        const int routed_rows = count * top_k;
        for (int i = 0; i < routed_rows; ++i) {
            const int e = router_host_ids_[i];
            const int pos = expert_curr_pos_host_[e]++;
            expert_map_token_host_[pos] = i / top_k;
            expert_map_rank_host_[pos] = i % top_k;
        }
        xmx_route_bucketed_ = true;
    }
    const auto& metadata=host_arena_.GetLayerNestedMetadata(layer);
    for (int e=0;e<cfg_.num_experts;++e) {
        const int slot=adaptive_cache_.GetSlot(layer,e);
        if (prefill_expert_uses_[e] && slot>=0) {
            prefill_unique_.push_back(e);
            adaptive_prefill_slots_[e]=adaptive_cache_.GetResidentSlotDev(slot,&metadata);
        }
    }
    const int hits=int(prefill_unique_.size());
    for (int e=0;e<cfg_.num_experts;++e) {
        if (!prefill_expert_uses_[e] || adaptive_cache_.GetSlot(layer,e)>=0) continue;
        prefill_unique_.push_back(e);
    }
    static const int xmx_rows=aila::env::read_int("AILA_Q4E_GROUPED_XMX_MIN_ROWS",0);
    const int threshold=grouped_xmx_min_rows_override_>=0 ? grouped_xmx_min_rows_override_ : xmx_rows;
    const int min_rows=grouped_xmx_supported_ && threshold>0 ? threshold : std::numeric_limits<int>::max();
    int act_start=0,tile_start=0;
    sycl::event last{};
    if (hits) {
        const auto* ready=adaptive_cache_.ResidentReady();
        last=RunGroupedExperts(ctx,layer,count,0,hits,act_start,tile_start,min_rows,ready ? *ready : sycl::event{});
        adaptive_cache_.RecordResidentConsumer(last);
    }
    const int staging_banks=adaptive_cache_.StagingBanks();
    const int lookahead=std::max(1,staging_banks-1);
    const int slots_per_chunk=adaptive_cache_.StagingSlotsPerBuffer();
    static const bool staging_pump_logged=[&](){
        AILA_LOG_INFO("[Q4E PrefillStage] banks=%d lookahead=%d slots_per_bank=%d",staging_banks,lookahead,slots_per_chunk);
        return true;
    }();
    (void)staging_pump_logged;
    const int total_unique=int(prefill_unique_.size());
    const int bank_first=hits;
    const int num_chunks=total_unique>bank_first ? (total_unique-bank_first+slots_per_chunk-1)/slots_per_chunk : 0;
    std::array<StagingBatch,aila::AdaptiveCacheConfig::kMaxStagingBanks> staged_window{};
    auto chunk_begin=[&](int c){ return bank_first+c*slots_per_chunk; };
    auto chunk_count=[&](int c){ return std::min(slots_per_chunk,total_unique-chunk_begin(c)); };
    auto stage_chunk=[&](int c){
        const int s=chunk_begin(c),n=chunk_count(c);
        auto batch=adaptive_cache_.StageColdExpertsAsync(host_arena_,*copy_q_,layer,
            prefill_unique_.data()+s,n,ping_pong_index_,&expert_executor_, direct_arena_stage_);
        ping_pong_index_=(ping_pong_index_+1)%staging_banks;
        for (int j=0;j<n;++j)
            adaptive_prefill_slots_[prefill_unique_[s+j]]=adaptive_cache_.GetStagingSlotDev(batch.ping_pong_index,j,&metadata);
        staged_window[c%aila::AdaptiveCacheConfig::kMaxStagingBanks]=batch;
    };
    // Round 3 instrumentation (AILA_Q4E_PREFILL_EXPERT_TIMING=1): the expert phase is already
    // pipelined - it stages `lookahead` chunks ahead and hands each chunk's ready_event to
    // RunGroupedExperts as a dependency instead of waiting on the host - so the open question is
    // whether the host still blocks inside the staging call itself. These timers print the host wall
    // time of stage_chunk against RunGroupedExperts per chunk, which separates "the host is blocking"
    // from "the device is genuinely serialised by dependencies".
    static const bool s_prefill_expert_timing =
        aila::env::read_flag("AILA_Q4E_PREFILL_EXPERT_TIMING", false);
    static const bool s_stage_dequant_fence =
        aila::env::read_flag("AILA_Q4E_STAGE_DEQUANT_FENCE", true);
    for (int c=0;c<lookahead&&c<num_chunks;++c) stage_chunk(c);
    for (int c=0;c<num_chunks;++c) {
        const auto& batch=staged_window[c%aila::AdaptiveCacheConfig::kMaxStagingBanks];
        const auto t_gemm0 = std::chrono::steady_clock::now();
        sycl::event staging_consumer{};
        last=RunGroupedExperts(ctx,layer,count,chunk_begin(c),chunk_count(c),act_start,tile_start,min_rows,batch.ready_event,
                               0,0,nullptr,nullptr,nullptr,nullptr,&staging_consumer);
        const double gemm_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gemm0).count();
        adaptive_cache_.RecordStagingConsumer(
            batch.ping_pong_index,
            (s_stage_dequant_fence && staging_consumer != sycl::event{}) ? staging_consumer : last);
        const int next_chunk=c+lookahead;
        double stage_ms = 0.0;
        if (next_chunk<num_chunks) {
            const auto t_stage0 = std::chrono::steady_clock::now();
            stage_chunk(next_chunk);
            stage_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_stage0).count();
        }
        if (s_prefill_expert_timing) {
            AILA_LOG_INFO("[Q4E PrefillExpertTiming] layer=%d chunk=%d/%d gemm_host_ms=%.3f stage_host_ms=%.3f",
                          layer, c + 1, num_chunks, gemm_ms, stage_ms);
        }
    }
    if (act_start!=count*top_k) throw std::logic_error("Incomplete Qwen4Exp prefill routing map");
    ctx.queue().memcpy(batch_weights_dev_,router_host_weights_,size_t(count)*top_k*sizeof(float));
    return ops::moe_ordered_accumulate_down_rows(ctx,buf_.moe_down_acc.data_as<float>(),batch_weights_dev_,
        buf_.y.data_as<bf16>(),count,top_k,hidden_size_,
        fuse_shared ? buf_.shared_out_buf.data_as<bf16>() : nullptr,
        fuse_shared ? buf_.shared_gate_buf.data_as<bf16>() : nullptr,&last,nullptr);
}

sycl::event Qwen4ExpBnb4Backend::RunGroupedExperts(
    Context& ctx, int layer, int count, int firstUnique, int numUnique,
    int& actStart, int& tileStart, int minRows, const sycl::event& ready,
    int route_base, int token_base, const int* use_counts,
    const bf16* x_base, bf16* moe_h_base, float* down_acc_base,
    sycl::event* staging_consumer_out) {
    const int top_k = cfg_.num_experts_per_tok;
    const int* uses = use_counts ? use_counts : prefill_expert_uses_.data();
    const bf16* x_dev = x_base ? x_base : buf_.x.data_as<bf16>();
    bf16* h_dev = moe_h_base ? moe_h_base : buf_.moe_h.data_as<bf16>();
    float* down_dev = down_acc_base ? down_acc_base : buf_.moe_down_acc.data_as<float>();
    // R1 expert-XMX consumer. Only the plain prefill call shape is replaced;
    // every other (test/alternate-base) call stays on the ALU path verbatim.
    //
    // Row-count gate: the systolic units only pay once M is large. Measured on
    // tools/prefill-bench.ps1 at the shipped 8192 chunk, XMX against ALU is
    //   M=2026 13,396 vs 11,838 ms (loses 13%)   M=2995 15,641 vs 15,412 (loses 1%)
    //   M=4952 19,971 vs 22,483 ms (wins 13%)    M=7878 26,159 vs 33,042 (wins 26%)
    // so the crossover is between 2995 and 4952 and 4096 is the gate. Note that
    // AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M is NOT this knob - it is a padded-band
    // parameter, and raising it to 4096 forced every band to pad that far and
    // measured 39,636 ms at M=7878 against 26,159 ms.
    static const int s_prefill_xmx_gate_rows =
        std::max(0, aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_GATE_ROWS", 1024));
    if (prefill_expert_xmx_ && count >= s_prefill_xmx_gate_rows &&
        use_counts == nullptr && x_base == nullptr && moe_h_base == nullptr &&
        down_acc_base == nullptr && route_base == 0 && token_base == 0) {
        return RunGroupedExpertsXmx(ctx, layer, count, firstUnique, numUnique, actStart, ready,
                                    staging_consumer_out);
    }
    static const bool reuse = aila::env::read_flag("AILA_Q4E_WEIGHT_REUSE", true);
    static const bool quads = aila::env::read_flag("AILA_Q4E_4TOKEN_REUSE", true);
    static const bool eights = aila::env::read_flag("AILA_Q4E_8TOKEN_REUSE", false);
    int ne = 0, nq = 0, np = 0, ns = 0, nx = 0;
    for (int u = firstUnique; u < firstUnique + numUnique; ++u) {
        const int n = uses[prefill_unique_[u]];
        if (n >= minRows) nx += n;
        else {
            ne += reuse && eights ? n / 8 : 0;
            const int rem8 = reuse && eights ? n % 8 : n;
            nq += reuse && quads ? rem8 / 4 : 0;
            const int rem4 = reuse && quads ? rem8 % 4 : rem8;
            np += reuse ? rem4 / 2 : 0;
            ns += reuse ? rem4 % 2 : rem4;
        }
    }
    const int alu = ne * 8 + nq * 4 + np * 2 + ns;
    // Eights lead the ALU layout so the descending-width dispatch chain keeps
    // every section behind its own gate/up section. With the default zero
    // eights this reduces to the HEAD quad/pair/single layout verbatim.
    int eo = actStart, qo = eo + ne * 8, po = qo + nq * 4, so = po + np * 2, xo = actStart + alu;
    const int tileBegin = tileStart;
    std::fill_n(expert_offsets_host_, cfg_.num_experts, -1);
    for (int u = firstUnique; u < firstUnique + numUnique; ++u) {
        const int e = prefill_unique_[u], n = uses[e];
        expert_curr_pos_host_[e] = 0;
        expert_offsets_host_[e] = n >= minRows ? xo : -2;
        if (n >= minRows) {
            for (int row = 0; row < n; row += 8)
                expert_tiles_host_[tileStart++] = {xo + row, std::min(8, n - row)};
            xo += n;
        } else {
            const int g = reuse && eights ? n / 8 : 0;
            const int rem = n - g * 8;
            const int q = reuse && quads ? rem / 4 : 0;
            const int rem4 = rem - q * 4, p = reuse ? rem4 / 2 : 0;
            expert_eight_pos_host_[e] = eo; eo += g * 8;
            expert_quad_pos_host_[e] = qo; qo += q * 4;
            expert_pair_pos_host_[e] = po; po += p * 2;
            expert_single_pos_host_[e] = so; so += rem4 - p * 2;
        }
    }
    for (int i = 0; i < count * top_k; ++i) {
        const int e = router_host_ids_[route_base + i], base = expert_offsets_host_[e];
        if (base == -1) continue; // another ready/miss microbatch
        const int seen = expert_curr_pos_host_[e]++, n = uses[e];
        int pos;
        if (base >= 0) pos = base + seen;
        else {
            const int g = reuse && eights ? (n / 8) * 8 : 0;
            const int q = reuse && quads ? ((n - g) / 4) * 4 : 0;
            const int p = reuse ? ((n - g - q) / 2) * 2 : 0;
            pos = seen < g ? expert_eight_pos_host_[e] + seen :
                seen < g + q ? expert_quad_pos_host_[e] + seen - g :
                seen < g + q + p ? expert_pair_pos_host_[e] + seen - g - q :
                expert_single_pos_host_[e] + seen - g - q - p;
        }
        expert_map_token_host_[pos] = token_base + i / top_k;
        expert_map_rank_host_[pos] = i % top_k;
        expert_map_slot_host_[pos] = adaptive_prefill_slots_[e];
    }
    const int acts = alu + nx, tiles = tileStart - tileBegin;
    // Microbatches use disjoint slices: host descriptors remain immutable until
    // their DMA is done. The next layer's router wait drains these consumers.
    ctx.queue().memcpy(expert_map_token_dev_ + actStart, expert_map_token_host_ + actStart, size_t(acts) * sizeof(int));
    ctx.queue().memcpy(expert_map_rank_dev_ + actStart, expert_map_rank_host_ + actStart, size_t(acts) * sizeof(int));
    ctx.queue().memcpy(expert_map_slot_dev_ + actStart, expert_map_slot_host_ + actStart, size_t(acts) * sizeof(ops::MoeExpertSlotDev));
    auto* tilesDev = reinterpret_cast<ops::MoeExpertTile*>(expert_tiles_dev_.data());
    if (tiles) ctx.queue().memcpy(tilesDev + tileBegin, expert_tiles_host_ + tileBegin, size_t(tiles) * sizeof(ops::MoeExpertTile));
    sycl::event last = ready;
    const bool nested = true;
    const bool nested_fixed256 = true;
    if (alu) {
        // P9: one-shot evidence that an AILA_Q4E_PREFILL_ACT_SLM arm really took
        // the SLM-staged expert kernels. The ops layer owns the routing rule; this
        // records the flag and the shapes so a run can prove which path it took
        // (see docs/plans/qwen38-r1-moe-kernel-slmbound-2026-10-16.md).
        static const bool act_slm_reported = [&] {
            static const bool act_slm = aila::env::read_flag("AILA_Q4E_PREFILL_ACT_SLM", false);
            AILA_LOG_INFO(
                "[Q4E PrefillActSlm] enabled=%d in_features=%d intermediate_size=%d "
                "gate_up_quad_tile_bytes=%d",
                act_slm ? 1 : 0, hidden_size_, cfg_.moe_intermediate_size,
                4 * hidden_size_ * static_cast<int>(sizeof(bf16)));
            return true;
        }();
        (void)act_slm_reported;
        last = ops::moe_expert_major_prefill_gate_up(ctx, expert_map_slot_dev_ + actStart,
            expert_map_token_dev_ + actStart, expert_map_rank_dev_ + actStart, nullptr,
            x_dev, h_dev, alu, top_k,
            hidden_size_, cfg_.moe_intermediate_size, 64, &last, nested, np, nq, ne, nested_fixed256);
        last = ops::moe_expert_major_prefill_down(ctx, expert_map_slot_dev_ + actStart,
            expert_map_token_dev_ + actStart, expert_map_rank_dev_ + actStart, nullptr,
            h_dev, down_dev, alu, top_k,
            cfg_.moe_intermediate_size, hidden_size_, 64, &last, nested, np, nq, ne, false, nested_fixed256);
    }
    if (tiles) {
        last = ops::moe_grouped_xmx_gate_up(ctx, expert_map_slot_dev_, tilesDev + tileBegin, tiles,
            expert_map_token_dev_, expert_map_rank_dev_, nullptr, x_dev,
            h_dev, top_k, hidden_size_, cfg_.moe_intermediate_size, nested, &last);
        last = ops::moe_grouped_xmx_down(ctx, expert_map_slot_dev_, tilesDev + tileBegin, tiles,
            expert_map_token_dev_, expert_map_rank_dev_, nullptr, h_dev,
            down_dev, top_k, hidden_size_, cfg_.moe_intermediate_size, nested, &last);
        grouped_xmx_tiles_ += tiles;
        grouped_xmx_acts_ += nx;
        if (grouped_expert_probe_) grouped_expert_probe_(*this, layer, count, actStart + alu, tileBegin, tiles, last);
    }
    if (expert_quality_probe_)
        last = expert_quality_probe_(expert_quality_probe_data_, *this, layer,
                                     count, actStart, acts, last);
    actStart += acts;
    return last;
}

sycl::event Qwen4ExpBnb4Backend::RunGroupedExpertsXmx(Context& ctx, int layer, int count,
    int firstUnique, int numUnique, int& actStart, const sycl::event& ready,
    sycl::event* staging_consumer_out) {
    (void)layer;
    const int top_k = cfg_.num_experts_per_tok;
    const int hidden = hidden_size_;
    const int intermediate = cfg_.moe_intermediate_size;
    const int routed_rows = count * top_k;
    const bf16* x_dev = buf_.x.data_as<bf16>();
    float* down_dev = buf_.moe_down_acc.data_as<float>();
    static const int rows_budget =
        std::max(16, aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_ROWS", 4096));
    static const bool s_single_plan_copy =
        aila::env::read_flag("AILA_Q4E_XMX_SINGLE_PLAN_COPY", true);
    static const int min_band_env =
        std::max(0, aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M", 192));
    const int effective_min_band = (count < 4096) ? std::min(min_band_env, 64) : min_band_env;
    const int min_band = effective_min_band > 0 ? ((effective_min_band + 15) & ~15) : 0;
    const int max_band_rows = xmx_scratch_.max_chunk_rows;
    // AILA_Q4E_PREFILL_EXPERT_XMX_MAX_M caps the band height of a large
    // prefill. The measured loss of a 1560-row-hot expert band comes from the
    // square padding around its real rows; splitting it into <= MAX_M bands
    // keeps every GEMM near its real M and lets the scheduler pack more experts
    // per chunk. Only the bucketed route can address several bands of one
    // expert in the same chunk (it keeps each band's own token/rank slice), so
    // the split stays off for the legacy scan path.
    static const int s_split_band_max =
        std::max(64, aila::env::read_int("AILA_Q4E_PREFILL_EXPERT_XMX_MAX_M", 256));
    const int split_band_rows = (count >= 1024 && xmx_route_bucketed_)
        ? std::min(max_band_rows, s_split_band_max)
        : max_band_rows;

    xmx_members_.clear();
    int total_rows = 0;
    for (int u = firstUnique; u < firstUnique + numUnique; ++u) {
        const int e = prefill_unique_[u];
        const int n = prefill_expert_uses_[e];
        if (n <= 0) continue;
        total_rows += n;
        for (int off = 0; off < n; off += split_band_rows) {
            xmx_members_.push_back({e, off, std::min(split_band_rows, n - off)});
        }
    }
    if (xmx_members_.empty()) return ready;
    std::sort(xmx_members_.begin(), xmx_members_.end(),
        [](const XmxPlanMember& a, const XmxPlanMember& b) {
            if (a.rows != b.rows) return a.rows > b.rows;
            if (a.expert != b.expert) return a.expert < b.expert;
            return a.offset < b.offset;
        });

    sycl::event last = ready;
    sycl::event last_dequant = ready;
    // Pre-warmed oneDNN primitives exist only for M in {64, 128, 192, 256}, so a
    // large prefill rounds every band up to a 64-row multiple instead of
    // JIT-compiling a new matmul per encountered M mid-prefill.
    const bool canonical_band = count >= 1024;
    auto canonical_rows = [canonical_band](int rows) {
        const int rounded = (rows + 15) & ~15;
        return (canonical_band && rounded <= 256) ? ((rounded + 63) & ~63) : rounded;
    };
    size_t mi = 0;
    while (mi < xmx_members_.size()) {
        const size_t chunk_begin = mi;
        int b_count = 0, max_rows = 0, total_chunk_rows = 0;
        if (xmx_members_[mi].rows > rows_budget) {
            b_count = 1;
            max_rows = xmx_members_[mi].rows;
            total_chunk_rows = xmx_members_[mi].rows;
            ++mi;
        } else {
            while (mi < xmx_members_.size() && b_count < 16) {
                const int n = xmx_members_[mi].rows;
                const int m_try = std::max(canonical_rows(std::max(max_rows, n)), min_band);
                if (b_count > 0 && size_t(b_count + 1) * size_t(m_try) > size_t(rows_budget)) break;
                max_rows = std::max(max_rows, n);
                total_chunk_rows += n;
                ++b_count;
                ++mi;
            }
        }
        const int rows_per_band = std::max(canonical_rows(max_rows), min_band);
        if (size_t(b_count) * size_t(rows_per_band) > size_t(max_band_rows)) {
            throw std::runtime_error("Qwen4Exp: XMX chunk exceeds scratch capacity");
        }

        const int bank = xmx_bank_;
        const size_t c_bases_bytes = size_t(17) * sizeof(int);
        const size_t c_rows_bytes = size_t(total_chunk_rows) * sizeof(int);
        const size_t c_ranks_off = c_bases_bytes + c_rows_bytes;
        const size_t c_slots_off = (c_ranks_off + c_rows_bytes + 63u) & ~size_t(63u);
        const size_t c_slots_bytes = size_t(b_count) * sizeof(ops::MoeExpertSlotDev);
        const size_t c_total_bytes = (c_slots_off + c_slots_bytes + 63u) & ~size_t(63u);
        const bool use_compact_pinned = s_single_plan_copy && xmx_plan_pinned_[0] != nullptr &&
            (size_t(xmx_plan_cursor_) + c_total_bytes <= size_t(xmx_plan_bytes_));

        unsigned char* host_base = use_compact_pinned
            ? (xmx_plan_pinned_[0] + xmx_plan_cursor_)
            : (xmx_plan_pinned_[bank] ? xmx_plan_pinned_[bank] : xmx_plan_host_[bank].data());
        unsigned char* dev_base = use_compact_pinned
            ? (xmx_plan_dev_[0] + xmx_plan_cursor_)
            : xmx_plan_dev_[bank];
        const size_t tok_off = use_compact_pinned ? c_bases_bytes : size_t(xmx_tokens_off_);
        const size_t rnk_off = use_compact_pinned ? c_ranks_off : size_t(xmx_ranks_off_);
        const size_t slt_off = use_compact_pinned ? c_slots_off : size_t(xmx_slots_off_);

        auto* bases = reinterpret_cast<int*>(host_base);
        auto* tokens = reinterpret_cast<int*>(host_base + tok_off);
        auto* ranks = reinterpret_cast<int*>(host_base + rnk_off);
        auto* slots = reinterpret_cast<ops::MoeExpertSlotDev*>(host_base + slt_off);

        int packed = 0;
        if (xmx_route_bucketed_) {
            for (int b = 0; b < b_count; ++b) {
                const auto& m = xmx_members_[chunk_begin + b];
                bases[b] = packed;
                slots[b] = adaptive_prefill_slots_[m.expert];
                const int src_off = expert_offsets_host_[m.expert] + m.offset;
                std::memcpy(tokens + packed, expert_map_token_host_ + src_off, size_t(m.rows) * sizeof(int));
                std::memcpy(ranks + packed, expert_map_rank_host_ + src_off, size_t(m.rows) * sizeof(int));
                packed += m.rows;
            }
            bases[b_count] = packed;
        } else {
            std::fill_n(expert_offsets_host_, cfg_.num_experts, -1);
            for (int b = 0; b < b_count; ++b) {
                const auto& m = xmx_members_[chunk_begin + b];
                bases[b] = packed;
                slots[b] = adaptive_prefill_slots_[m.expert];
                expert_offsets_host_[m.expert] = b;
                expert_curr_pos_host_[m.expert] = 0;
                packed += m.rows;
            }
            bases[b_count] = packed;
            for (int i = 0; i < routed_rows; ++i) {
                const int e = router_host_ids_[i];
                const int b = expert_offsets_host_[e];
                if (b < 0) continue;
                const auto& m = xmx_members_[chunk_begin + b];
                const int k = expert_curr_pos_host_[e]++;
                if (k < m.offset || k >= m.offset + m.rows) continue;
                const int pos = bases[b] + (k - m.offset);
                tokens[pos] = i / top_k;
                ranks[pos] = i % top_k;
            }
        }

        if (use_compact_pinned) {
            ctx.queue().memcpy(dev_base, host_base, c_slots_off + c_slots_bytes);
            xmx_plan_cursor_ += int(c_total_bytes);
        } else {
            ctx.queue().memcpy(dev_base, bases, size_t(b_count + 1) * sizeof(int));
            ctx.queue().memcpy(dev_base + tok_off, tokens, size_t(packed) * sizeof(int));
            ctx.queue().memcpy(dev_base + rnk_off, ranks, size_t(packed) * sizeof(int));
            ctx.queue().memcpy(dev_base + slt_off, slots,
                               size_t(b_count) * sizeof(ops::MoeExpertSlotDev));
        }

        ops::MoeExpertXmxChunk chunk{};
        chunk.num_experts = b_count;
        chunk.top_k = top_k;
        chunk.rows_per_band = rows_per_band;
        chunk.member_row_base = reinterpret_cast<const int*>(dev_base);
        chunk.member_token_ids = reinterpret_cast<const int*>(dev_base + tok_off);
        chunk.member_ranks = reinterpret_cast<const int*>(dev_base + rnk_off);
        chunk.slots = reinterpret_cast<const ops::MoeExpertSlotDev*>(dev_base + slt_off);
        last = ops::moe_expert_xmx_prefill_chunk(ctx, xmx_scratch_, chunk, bank, x_dev, down_dev,
                                                 hidden, intermediate, 64, &ready, &last_dequant);
        xmx_bank_ ^= 1;
    }
    if (staging_consumer_out != nullptr) *staging_consumer_out = last_dequant;
    actStart += total_rows;
    return last;
}

void Qwen4ExpBnb4Backend::EmbedTokens(
    Context& ctx, const int* tokenIdsDevice, int count,
    const int* tokenIdsHost) {
    if (!embed_host_.valid()) {
        if (tokenIdsHost) {
            // The resident lookup kernel still reads packed device ids, so a
            // host-id caller gets its ids uploaded asynchronously instead of
            // synchronously; waiting on the previous upload keeps this pinned
            // block safe to rewrite without draining the compute queue.
            if (embed_h2d_event_ != sycl::event{}) embed_h2d_event_.wait_and_throw();
            std::copy_n(tokenIdsHost, count, embed_host_ids_);
            embed_h2d_event_ = ctx.memcpy_h2d_async(
                const_cast<int*>(tokenIdsDevice), embed_host_ids_,
                static_cast<size_t>(count) * sizeof(int));
        }
        ops::embedding_lookup(ctx, *embed_weight_, tokenIdsDevice, count, buf_.x, hidden_size_);
        return;
    }
    if (tokenIdsHost) {
        // Host ids reach the gather directly. The previous forward's row upload
        // may still be reading the pinned rows, and waiting on that copy is a
        // fraction of the queue drain a blocking readback would cost.
        if (embed_h2d_event_ != sycl::event{}) embed_h2d_event_.wait_and_throw();
        std::copy_n(tokenIdsHost, count, embed_host_ids_);
    } else {
        // The blocking readback also drains the previous h2d, so the pinned
        // rows can be reused.
        ctx.memcpy_d2h(
            embed_host_ids_, tokenIdsDevice,
            static_cast<size_t>(count) * sizeof(int));
    }
    const size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(bf16);
    const int64_t rows = embed_host_.shape[0];
    for (int t = 0; t < count; ++t) {
        const int id = embed_host_ids_[t];
        if (id < 0 || id >= rows) {
            throw std::runtime_error("Qwen4ExpBnb4Backend: token id " + std::to_string(id) + " outside the embedding table");
        }
        std::memcpy(embed_host_rows_ + static_cast<size_t>(t) * row_bytes,
                    embed_host_.ptr + static_cast<size_t>(id) * row_bytes, row_bytes);
    }
    embed_h2d_event_ = ctx.memcpy_h2d_async(buf_.x.data(), embed_host_rows_,
                                            static_cast<size_t>(count) * row_bytes);
}

void Qwen4ExpBnb4Backend::RunPendingAdaptiveSwap(Context& ctx) {
    if (!pending_adaptive_swap_) return;
    pending_adaptive_swap_ = false;
    const int max_swaps = pending_adaptive_max_swaps_;
    pending_adaptive_max_swaps_ = 0;
    static const int s_max_per_layer = std::max(1, aila::env::read_int_raw("AILA_Q4E_MAX_SWAPS_PER_LAYER", 2));
    auto swap_plans = adaptive_cache_.PlanSwaps(s_max_per_layer, max_swaps);
    if (!swap_plans.empty()) {
        sycl::queue& dma_q = copy_q_ ? *copy_q_ : ctx.queue();
        if (copy_q_) {
            adaptive_cache_.BeginSwapsAsync(host_arena_, dma_q, swap_plans, direct_arena_stage_);
        } else {
            adaptive_cache_.ExecuteSwapsAsync(host_arena_, dma_q, swap_plans, direct_arena_stage_);
        }
    }
    static const bool s_adaptive_swap_decay =
        aila::env::read_flag("AILA_Q4E_ADAPTIVE_SWAP_DECAY", true);
    static const int s_adaptive_swap_freq = aila::env::read_int_raw("AILA_Q4E_ADAPTIVE_SWAP_FREQ", 8);
    if (s_adaptive_swap_decay) {
        adaptive_cache_.Decay(s_adaptive_swap_freq <= 8 ? 0.92f : 0.85f);
    }
}

sycl::event Qwen4ExpBnb4Backend::ForwardLmHead(
    Context& ctx, Tensor& input, Tensor& output, int rows) {
    if (lmHeadQ8Enabled_) {
        sycl::event ev = ops::bf16_q8_head_forward(ctx, lm_head_q8_weight_, lm_head_q8_scale_,
            input, output, lm_head_q8_input_, lm_head_q8_input_scale_, rows, hidden_size_, cfg_.vocab_size);
        ++lmHeadQ8ForwardCount_;
        lmHeadQ8RowCount_ += static_cast<uint64_t>(rows);
        if (lmHeadQ8ForwardCount_ == 1)
            AILA_LOG_INFO("[Q4E LM Head Q8] actual_enabled=1 rows=%d hidden=%d vocab=%d activation_block=64 input_quantization=1 experimental=1",
                rows, hidden_size_, cfg_.vocab_size);
        return ev;
    }
    if (verify_rows_alignment_ && rows > 1) {
        // Verify-path alignment: one head GEMV per row so the window matches
        // sequential B1 logits exactly.
        for (int r = 0; r < rows; ++r) {
            auto x_row = Tensor::view(ctx, input.data_as<bf16>() + size_t(r) * hidden_size_,
                                      {1, hidden_size_});
            auto y_row = Tensor::view(ctx, output.data_as<bf16>() + size_t(r) * cfg_.vocab_size,
                                      {1, cfg_.vocab_size});
            lm_head_.forward(ctx, x_row, y_row, 1);
        }
        return sycl::event{};
    }
    lm_head_.forward(ctx, input, output, rows);
    return sycl::event{};
}

void Qwen4ExpBnb4Backend::DenseLinear(
    Context& ctx, Linear& linear, Tensor* raw_weight,
    Tensor& input, Tensor& output, int count, int out_features, int in_features) {
    static const bool s_native_gemv = aila::env::read_flag("AILA_Q4E_DENSE_GEMV", true);
    if ((count == 1 || ((verify_decode_kernels_ || verify_rows_alignment_) && count >= 2 && count <= 4)) &&
        s_native_gemv && raw_weight && !component_profile_active_for_test_) {
        ops::bf16_gemv_bf16(ctx, *raw_weight, input.data_as<bf16>(), output.data_as<bf16>(),
                            out_features, in_features, count);
        return;
    }
    const int component = component_profile_active_for_test_ && count == 2 && in_features == hidden_size_
        ? (out_features == cfg_.num_experts ? 3 : out_features == 1 ? 4 : -1) : -1;
    if (component >= 0) ctx.synchronize();
    const auto begin = component >= 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    linear.forward(ctx, input, output, count);
    if (component >= 0) {
        ctx.synchronize();
        component_profile_ms_for_test_[component] += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
}

Qwen4ExpBnb4Backend::DecodePendingMlpResidual::~DecodePendingMlpResidual() noexcept {
    // On a host exception, enqueue the residual while its buffers still exist.
    // A failed device queue cannot promise materialization, just as on the old path.
    try { Materialize(); } catch (...) {}
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::Defer() {
    if (pending) throw std::logic_error("MLP residual already pending");
    pending = true;
    ++deferred;
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::Materialize() {
    if (!pending) return;
    ops::hc_inject_add(ctx, hidden, y, inject, rows, hiddenSize, 4);
    pending = false;
    ++materialized;
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::ConsumeNorm(
    const float* normWeight, float eps, bf16* normalized) {
    if (!pending) throw std::logic_error("No pending MLP residual to normalize");
    if (!ops::try_hc_decode_inject_add_and_grouped_rms_norm(ctx, hidden, y, inject,
        normWeight, eps, normalized, hiddenSize, true, rows)) {
        // Clear residual ownership after its submit, before the norm submit can
        // throw. The non-vec path keeps both original kernels unchanged.
        Materialize();
        ops::grouped_rms_norm(ctx, hidden, normWeight, eps, normalized, rows, 4, hiddenSize, true);
        return;
    }
    // Clear immediately after norm submission, before subsequent HC submissions
    // can throw. Unwinding must never inject a successfully consumed residual twice.
    pending = false;
    ++consumedNorm;
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::MarkConsumed() {
    if (!pending) throw std::logic_error("No pending MLP residual to mark consumed");
    // Same discipline as ConsumeNorm: clear immediately after the fused
    // submission that consumed the residual (the relocated in-place update is
    // part of that submission's dependency chain, not of this host call).
    pending = false;
    ++consumedNorm;
}

std::optional<sycl::event> Qwen4ExpBnb4Backend::RunHyperConnectionCombined(
    Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject,
    const bf16* pending_y, const float* pending_inj, DecodePendingMlpResidual* pending_mlp) {
    const int hc_dim = 4 * hidden_size_;
    const int lowrank = cfg_.hc_lowrank;
    static const bool s_fused_hc = aila::env::read_flag("AILA_Q4E_HC_FUSED", true);

    // F1 (AILA_Q4E_DECODE_FUSED_HC=1, default off): fused (inject-add + norm)
    // + down in one submission with the in-place residual update relocated
    // into the up kernel. Bit-exact mirror of hc_inject_add_and_grouped_rms_norm
    // / grouped_rms_norm, hc_decode_down_inject and hc_decode_up_mix; removes
    // one submission per HC call (97 per count==1 decode token). A shape the
    // fused op rejects falls through to the production dispatch below,
    // byte-identical to the flag-off path.
    static const bool s_decode_fused_hc = aila::env::read_flag("AILA_Q4E_DECODE_FUSED_HC", false);
    if (s_decode_fused_hc && count == 1 && s_fused_hc && hc.mix_down_w && hc.mix_up_w &&
        (!with_inject || hc.inject_w)) {
        const bf16* upd_y = nullptr;
        const float* upd_inj = nullptr;
        bf16* upd_h = nullptr;
        const bf16* h_stack = static_cast<const bf16*>(buf_.hidden4.data());
        if (pending_mlp && pending_mlp->pending) {
            upd_y = pending_mlp->y;
            upd_inj = pending_mlp->inject;
            upd_h = pending_mlp->hidden;
            h_stack = pending_mlp->hidden;
        } else if (pending_y && pending_inj) {
            upd_y = pending_y;
            upd_inj = pending_inj;
            upd_h = static_cast<bf16*>(buf_.hidden4.data());
        }
        const float* up_inj = upd_inj;
        const float* inj_out = with_inject ? buf_.hc_inject_f32.data_as<float>() : nullptr;
        if (with_inject && up_inj != nullptr && up_inj == inj_out &&
            buf_.hc_inject_prev_f32.valid()) {
            ctx.queue().memcpy(buf_.hc_inject_prev_f32.data(), up_inj,
                               size_t(4) * sizeof(float));
            up_inj = buf_.hc_inject_prev_f32.data_as<float>();
        }
        if (q4e_fusion::run_hc_norm_down_fused(
                ctx, h_stack, upd_y, up_inj,
                hc.hc_norm_weight->data_as<float>(), cfg_.rms_norm_eps,
                hc.mix_down_w->data_as<bf16>(),
                with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
                buf_.hc_N.data_as<bf16>(), buf_.hc_mix_down.data_as<bf16>(),
                with_inject ? buf_.hc_inject_f32.data_as<float>() : nullptr,
                lowrank, hidden_size_, 4)) {
            if (pending_mlp && pending_mlp->pending) pending_mlp->MarkConsumed();
            return q4e_fusion::run_hc_up_mix_with_hidden_update(
                ctx, buf_.hc_mix_down.data_as<bf16>(), hc.mix_up_w->data_as<bf16>(),
                buf_.hc_N.data_as<bf16>(), buf_.x.data_as<bf16>(),
                upd_h, upd_y, up_inj, lowrank, hidden_size_, 4);
        }
    }

    if (pending_mlp && pending_mlp->pending) {
        pending_mlp->ConsumeNorm(hc.hc_norm_weight->data_as<float>(), cfg_.rms_norm_eps,
            buf_.hc_N.data_as<bf16>());
    } else if (pending_y && pending_inj) {
        // The vec8 decode form is the same inject-add plus grouped RMSNorm in
        // one submission, and it keeps the updated row in private storage
        // instead of re-reading H for the output pass. A rejected shape falls
        // back to the reference op below, unchanged.
        const bool vec_done = (count >= 1 && count <= 4) &&
            ops::try_hc_decode_inject_add_and_grouped_rms_norm(
                ctx, static_cast<bf16*>(buf_.hidden4.data()),
                pending_y, pending_inj,
                static_cast<const float*>(hc.hc_norm_weight->data()),
                cfg_.rms_norm_eps, static_cast<bf16*>(buf_.hc_N.data()),
                hidden_size_, true, count);
        if (!vec_done) {
            ops::hc_inject_add_and_grouped_rms_norm(
                ctx, static_cast<bf16*>(buf_.hidden4.data()),
                pending_y, pending_inj,
                static_cast<const float*>(hc.hc_norm_weight->data()),
                cfg_.rms_norm_eps, static_cast<bf16*>(buf_.hc_N.data()),
                count, 4, hidden_size_, true);
        }
    } else {
        ops::grouped_rms_norm(
            ctx, static_cast<const bf16*>(buf_.hidden4.data()),
            static_cast<const float*>(hc.hc_norm_weight->data()),
            cfg_.rms_norm_eps, static_cast<bf16*>(buf_.hc_N.data()),
            count, 4, hidden_size_, true);
    }
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kHcNorm);

    static const bool s_small_rows_hc = aila::env::read_flag("AILA_Q4E_SMALL_ROWS_HC", false);
    static const int s_hc_down_wg = [] {
        const int value = aila::env::read_int_raw("AILA_Q4E_HC_DECODE_DOWN_WG", 256);
        return value == 64 || value == 128 ? value : 256;
    }();
    static const bool s_hc_up_streams = aila::env::read_flag("AILA_Q4E_HC_DECODE_UP_STREAMS", false);
    const int hc_down_wg = decodeHcReferenceForTest_ ? 256 : s_hc_down_wg;
    const bool hc_up_streams = !decodeHcReferenceForTest_ && s_hc_up_streams;
    if ((s_small_rows_hc || verify_rows_alignment_ || verify_decode_kernels_) && s_fused_hc &&
        count >= 2 && count <= 4 &&
        hc.mix_down_w && hc.mix_up_w && (!with_inject || hc.inject_w)) {
        ops::hc_decode_down_inject_rows(ctx, buf_.hc_N.data_as<bf16>(),
            hc.mix_down_w->data_as<bf16>(), with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
            buf_.hc_mix_down.data_as<bf16>(), buf_.hc_inject_f32.data_as<float>(),
            lowrank, hidden_size_, 4, count, hc_down_wg);
        ops::hc_decode_up_mix_rows(ctx, buf_.hc_mix_down.data_as<bf16>(), hc.mix_up_w->data_as<bf16>(),
            buf_.hc_N.data_as<bf16>(), buf_.x.data_as<bf16>(), lowrank, hidden_size_, 4, count, hc_up_streams);
        return std::nullopt;
    }
    if (count == 1 && s_fused_hc && hc.mix_down_w && hc.mix_up_w && (!with_inject || hc.inject_w)) {
        // Optional software-pipelined variant of the same two kernels. The
        // decode down GEMM exposes only lowrank + groups = 324 independent dot
        // products, i.e. 21 work-groups, so it is latency-bound on memory-level
        // parallelism rather than bandwidth-bound: each lane ends up with a
        // single 16 B load in flight. Raising prefetch_depth multiplies the
        // in-flight bytes without changing the arithmetic or the reduction
        // order, so the result is bit-identical to the production kernels.
        // 0 keeps the production path.
        static const int s_hc_pl_depth = [] {
            const int value = aila::env::read_int_raw(
                "AILA_Q4E_HC_DECODE_PL_DEPTH", 0);
            return value >= 1 && value <= 4 ? value : 0;
        }();
        if (s_hc_pl_depth > 0) {
            ops::hc_decode_down_inject_pl(
                ctx, static_cast<const bf16*>(buf_.hc_N.data()),
                hc.mix_down_w->data_as<bf16>(),
                with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
                static_cast<bf16*>(buf_.hc_mix_down.data()),
                static_cast<float*>(buf_.hc_inject_f32.data()),
                lowrank, hidden_size_, 4, s_hc_pl_depth, hc_down_wg);
            ops::hc_decode_up_mix_pl(
                ctx, static_cast<const bf16*>(buf_.hc_mix_down.data()),
                hc.mix_up_w->data_as<bf16>(),
                static_cast<const bf16*>(buf_.hc_N.data()),
                static_cast<bf16*>(buf_.x.data()),
                lowrank, hidden_size_, 4, s_hc_pl_depth);
            return std::nullopt;
        }
        ops::hc_decode_down_inject(ctx, static_cast<const bf16*>(buf_.hc_N.data()),
                                   hc.mix_down_w->data_as<bf16>(),
                                   with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
                                   static_cast<bf16*>(buf_.hc_mix_down.data()),
                                   static_cast<float*>(buf_.hc_inject_f32.data()),
                                   lowrank, hidden_size_, 4, hc_down_wg);
        aila::token_graph_stage_prof::mark(
            aila::token_graph_stage_prof::kHcDown);
        // The up-mix is the last submission and writes buf_.x (the activation
        // the decode CPU complement copies out); hand its event back so the
        // copy queue can depend on the kernel directly instead of a barrier.
        auto hc_up_event =
            ops::hc_decode_up_mix(ctx, static_cast<const bf16*>(buf_.hc_mix_down.data()),
                                     hc.mix_up_w->data_as<bf16>(),
                                     static_cast<const bf16*>(buf_.hc_N.data()),
                                     static_cast<bf16*>(buf_.x.data()),
                                     lowrank, hidden_size_, 4, hc_up_streams);
        aila::token_graph_stage_prof::mark(
            aila::token_graph_stage_prof::kHcUp);
        return hc_up_event;
    }

    DenseLinear(ctx, hc.input_mix_weight_down, hc.mix_down_w, buf_.hc_N, buf_.hc_mix_down,
                count, lowrank, hc_dim);
    apply_silu_div4(ctx, buf_.hc_mix_down, static_cast<size_t>(count) * lowrank);

    DenseLinear(ctx, hc.input_mix_weight_up, hc.mix_up_w, buf_.hc_mix_down, buf_.hc_mix_up,
                count, hc_dim, lowrank);
    static const bool s_hc_fused_sigmoid =
        aila::env::read_flag("AILA_Q4E_HC_FUSED_SIGMOID", true);
    if (!s_hc_fused_sigmoid) {
        apply_sigmoid(ctx, buf_.hc_mix_up, static_cast<size_t>(count) * hc_dim);
    }

    ops::hc_mix_reduce(ctx, static_cast<const bf16*>(buf_.hc_N.data()),
                       static_cast<const bf16*>(buf_.hc_mix_up.data()),
                       static_cast<bf16*>(buf_.x.data()),
                       count, hidden_size_, 4, s_hc_fused_sigmoid);

    if (with_inject) {
        DenseLinear(ctx, hc.block_inject_weight, hc.inject_w, buf_.hc_N, buf_.hc_inject_buf,
                    count, 4, hc_dim);
        apply_two_sigmoid_div4(ctx, buf_.hc_inject_buf, buf_.hc_inject_f32, static_cast<size_t>(count) * 4);
    }
    return std::nullopt;
}

std::optional<sycl::event> Qwen4ExpBnb4Backend::RunHyperConnection(
    Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject) {
    return RunHyperConnectionCombined(ctx, hc, count, with_inject, nullptr, nullptr);
}

Tensor& Qwen4ExpBnb4Backend::ForwardImpl(
    Context& ctx, Session& session,
    const int* tokenIdsDevice, int count, bool lastTokenOnly,
    const ForwardOptions& options) {
    ValidateSession(session);
    if (session.token_graph_poisoned()) {
        throw std::logic_error(
            "Qwen4Exp token graph session is poisoned; recreate the session");
    }
    // Direct execution must stop sharing the backend with a retained graph.
    // Empty is the default path: one branch, no allocation, no generation bump.
    // The opt-in L3 graph spans only layer 3's attention tile and is launched
    // synchronously below. Keep its reusable executable alive across ordinary
    // count==1 forwards; every state/layout mutation still aborts it.
    static const bool s_qsa_layer_graph =
        !WholeTokenGraphEnabled() &&
        aila::env::read_flag("AILA_Q4E_QSA_LAYER_GRAPH", false);
    const bool preserve_l3_graph = s_qsa_layer_graph && count == 1;
    if (!preserve_l3_graph) (void)AbortTokenGraph();
    if (count <= 0) return buf_.logits;
    if (!preserve_l3_graph) session.InvalidateTokenGraphState();
    session.EnsureKvHostCapacity(ctx, session.CurrentLength() + count);
    ValidateGdnHeadPath(session, count);
    if (count == 1 && session.graph_gdn_state_enabled_) {
        int linear_rows = 0;
        int device_authority_rows = 0;
        bool any_host_authority_unsynced = false;
        for (const auto& cache : session.layerCaches_) {
            if (cache.linear_conv_head_dev == nullptr) continue;
            ++linear_rows;
            device_authority_rows += cache.linear_conv_head_device_authority ? 1 : 0;
            any_host_authority_unsynced |=
                !cache.linear_conv_head_device_authority &&
                !cache.linear_conv_head_device_synced;
        }
        if (device_authority_rows == 0 && any_host_authority_unsynced) {
            PrepareGdnGraphState(ctx, session);
        }
    }
    static const int s_cold_split_num =
        std::clamp(aila::env::read_int_raw("AILA_Q4E_COLD_SPLIT_NUM", 0), 0, 256);
    static const bool s_direct_arena_stage =
        aila::env::read_flag("AILA_Q4E_STAGE_DIRECT_ARENA", true);
    // The split is a token-level share, not a per-layer floor. Reset it once
    // per forward; prefill and multi-row verify never consume it.
    cold_split_credit_ = 0;
    cold_split_num_ = s_cold_split_num;
    direct_arena_stage_ = s_direct_arena_stage;
    cold_split_this_forward_ = false;
    static const bool profile_components = aila::env::read_flag("AILA_Q4E_PROFILE_COMPONENTS", false);
    component_profile_active_for_test_ = profile_components && options.decode_verification && count == 2;
    component_profile_ms_for_test_.fill(0);
    if (options.decode_verification && options.prefill_remaining >= 0)
        throw std::invalid_argument("Decode verification cannot be a prefill slice");

    int start_pos = session.currentLength_;
    int cached_len = start_pos + count;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: context length " +
                                 std::to_string(cached_len) + " exceeds max_seq_len " +
                                 std::to_string(max_seq_len_));
    }

    const int checkpoint_after = options.snapshot_policy == SnapshotPolicy::ReusablePrefix
        ? options.checkpoint_after : 0;
    if (checkpoint_after < 0 || (checkpoint_after > 0 &&
        !CanCaptureInteriorCheckpoint(session, count, checkpoint_after))) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: interior checkpoint requires short dense prefill, two multi-token sides and two slots");
    }
    ensure_runtime_buffers(ctx, count);
    if (moe_compact_tile_ && count > 4) ctx.queue().memset(buf_.moe_compact_invalid.data(),0,sizeof(int));
    StateSnapshot* interior = checkpoint_after > 0
        ? &ReserveSnapshot(session, start_pos + checkpoint_after) : nullptr;
    Bnb4BitLinear::PrefillDispatchStats prefill_stats_t0{};
    if (count > 1) {
        prefill_stats_t0 = Bnb4BitLinear::get_prefill_dispatch_stats();
    }

    if (count > 1 || start_pos == 0) {
        session.consecutive_no_miss_layers_ = 0;
    }

    // 1. Embedding lookup + optional vision embedding override
    EmbedTokens(ctx, tokenIdsDevice, count, options.host_token_ids);

    // P0: Early PLE prepare pipeline (pure prepare + async DMA to double-buffered staging)
    const int* ple_ids_ptr = embed_host_ids_;
    if (!embed_host_.valid()) {
        // The host ids were already copied into the pinned block by EmbedTokens;
        // only a device-id caller has to read them back.
        if (options.host_token_ids) {
            std::copy_n(options.host_token_ids, count, embed_host_ids_);
        } else {
            ctx.memcpy_d2h(embed_host_ids_, tokenIdsDevice, static_cast<size_t>(count) * sizeof(int));
        }
    }
    static const bool ple_lookahead = aila::env::read_flag("AILA_Q4E_PLE_ASYNC", true);
    const bool reuse_ple = ple_lookahead && options.prefill_remaining >= 0 &&
        PleTicketEpochMatches(session, ple_next_ticket_) &&
        ple_next_ticket_.start_pos == start_pos && ple_next_ticket_.prev_tokens_in == session.ple_state_.prev_tokens &&
        ple_.MatchesInput(ple_next_ticket_, ple_ids_ptr, count);
    if (reuse_ple) ple_ticket_ = ple_next_ticket_;
    else ple_ticket_ = ple_.prepare(ctx, session.ple_state_, ple_ids_ptr, count,
                                    &session, session.generation(), start_pos,
                                    session.ple_ticket_epoch());
    ple_next_ticket_.valid = false;
    if (ple_lookahead && options.prefill_remaining > 0 && options.next_prefill_count > 0) {
        if (!options.next_prefill_tokens || options.next_prefill_count > options.prefill_remaining ||
            options.next_prefill_count > max_seq_len_ - cached_len)
            throw std::invalid_argument("Invalid Qwen4Exp PLE lookahead chunk");
        // N-gram lookup depends only on the known tokens and two-token history.
        // The existing worker and other staging slot prepare it while this
        // chunk runs; recurrent state is committed only by the actual forward.
        Qwen4ExpPleState future;
        future.prev_tokens = ple_ticket_.prev_tokens_next;
        ple_next_ticket_ = ple_.prepare(ctx, future, options.next_prefill_tokens,
            options.next_prefill_count, &session, session.generation(), cached_len,
            session.ple_ticket_epoch());
    }

    if (!session.embeddingOverridePositions_.empty() &&
        session.embeddingOverrideHiddenSize_ == hidden_size_) {
        bf16* hidden_ptr = static_cast<bf16*>(buf_.x.data());
        for (size_t i = 0; i < session.embeddingOverridePositions_.size(); ++i) {
            int pos = session.embeddingOverridePositions_[i];
            if (pos < start_pos || pos >= start_pos + count) continue;
            int local_row = pos - start_pos;
            const bf16* src = session.embeddingOverrideValues_.data() + i * static_cast<size_t>(hidden_size_);
            bf16* dst = hidden_ptr + static_cast<size_t>(local_row) * static_cast<size_t>(hidden_size_);
            ctx.queue().memcpy(dst, src, static_cast<size_t>(hidden_size_) * sizeof(bf16));
        }
    }

    // Repeat embedding 4x into hidden4: [count, 10240]
    const bf16* embed_ptr = static_cast<const bf16*>(buf_.x.data());
    bf16* hidden4_ptr = static_cast<bf16*>(buf_.hidden4.data());
    const int h_size = hidden_size_;

    if (count == 1 && (h_size % 8 == 0)) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = h_size / 8;
        constexpr int wg = 256;
        const size_t num_wgs = (dim8 + wg - 1) / wg;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    const int i = static_cast<int>(item.get_global_id(0));
                    if (i < dim8) {
                        const vec8 val = *reinterpret_cast<const vec8*>(embed_ptr + i * 8);
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            *reinterpret_cast<vec8*>(hidden4_ptr + s * h_size + i * 8) = val;
                        }
                    }
                });
        });
    } else if (h_size % 8 == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = h_size / 8;
        constexpr int wg = 256;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(count, wg), sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t t = item.get_group(0);
                    const int lid = static_cast<int>(item.get_local_id(1));
                    const size_t t_embed_base = t * h_size;
                    const size_t t_hidden_base = t * 10240;
                    for (int i = lid; i < dim8; i += wg) {
                        const vec8 val = *reinterpret_cast<const vec8*>(embed_ptr + t_embed_base + i * 8);
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            *reinterpret_cast<vec8*>(hidden4_ptr + t_hidden_base + s * h_size + i * 8) = val;
                        }
                    }
                });
        });
    } else {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(count) * h_size), [=](sycl::id<1> idx) {
                size_t t = idx[0] / h_size;
                size_t d = idx[0] % h_size;
                bf16 val = embed_ptr[idx[0]];
                for (int s = 0; s < 4; ++s) {
                    hidden4_ptr[t * 10240 + s * h_size + d] = val;
                }
            });
        });
    }

    constexpr int kMaxTopK = 16; // moe_decode_* kernels capture at most 16 slots
    const int top_k = cfg_.num_experts_per_tok;
    if (top_k > kMaxTopK) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: num_experts_per_tok exceeds 16");
    }
    // Experimental tiny-prefill/verification route. Keep it opt-in until full
    // model numerics and state semantics are accepted for each supported shape.
    static const bool s_small_rows_decode = aila::env::read_flag("AILA_Q4E_SMALL_ROWS_DECODE", false);
    static const bool s_verify_small_rows = aila::env::read_flag("AILA_Q4E_VERIFY_SMALL_ROWS", false);
    static const bool s_verify_decode_kernels =
        aila::env::read_flag("AILA_Q4E_VERIFY_DECODE_KERNELS", true);
    const bool verify_rows_alignment = s_verify_small_rows && options.decode_verification &&
        count >= 2 && count <= 4;
    // AILA_Q4E_VERIFY_DECODE_KERNELS keeps verify_rows_alignment_ off on purpose:
    // the window then runs the multi-row decode-row projections and the fused
    // two-launch hyper-connection while the LM head keeps its single-launch
    // multi-row kernel instead of one GEMV per row.
    const bool verify_decode_kernels = options.decode_verification && s_verify_decode_kernels &&
        count >= 2 && count <= 4 && !verify_rows_alignment;
    const bool small_rows_decode =
        (s_small_rows_decode || verify_rows_alignment || verify_decode_kernels) &&
        count >= 2 && count <= 4;
    // Scope the verify-path alignment to this forward; the in-order submission
    // means the flag only affects kernels queued here.
    struct VerifyRowsAlignScope {
        Qwen4ExpBnb4Backend& self;
        bool prev;
        VerifyRowsAlignScope(Qwen4ExpBnb4Backend& s, bool v, bool decode)
            : self(s), prev(s.verify_rows_alignment_) {
            self.verify_rows_alignment_ = v;
            self.verify_decode_kernels_ = decode;
            self.ple_.set_rows_decode(v || decode);
        }
        ~VerifyRowsAlignScope() {
            self.verify_rows_alignment_ = prev;
            self.verify_decode_kernels_ = false;
            self.ple_.set_rows_decode(prev);
        }
    } verifyRowsAlignScope{*this, verify_rows_alignment, verify_decode_kernels};
    auto project_prefill = [&](Bnb4BitLinear& projection, Tensor& input, Tensor& output) {
        if (small_rows_decode) projection.ForwardDecodeRows(ctx, linear_scratch_, input, output, count);
        else projection.forward(ctx, linear_scratch_, input, output, count);
    };
    // Verify-path alignment: a raw-weight BF16 GEMV per row so router/shared
    // gate projections in a 2..4 row window match sequential B1 decode. Inert
    // (single GEMM) unless AILA_Q4E_VERIFY_SMALL_ROWS is on.
    auto dense_rows = [&](Linear& linear, Tensor* raw_weight, Tensor& input, Tensor& output,
                          int out_features, int in_features) {
        if (!verify_rows_alignment_) {
            DenseLinear(ctx, linear, raw_weight, input, output, count, out_features, in_features);
            return;
        }
        for (int r = 0; r < count; ++r) {
            auto in_row = Tensor::view(ctx, input.data_as<bf16>() + size_t(r) * in_features, {1, in_features});
            auto out_row = Tensor::view(ctx, output.data_as<bf16>() + size_t(r) * out_features, {1, out_features});
            DenseLinear(ctx, linear, raw_weight, in_row, out_row, 1, out_features, in_features);
        }
    };
    static const std::string s_dump_dir = aila::env::read_string("AILA_DEBUG_Q4E_DUMP_DIR", "");
    static const bool s_profile = aila::env::read_flag("AILA_PROFILE_Q4E_DECODE", false);
    static const bool s_work_summary = aila::env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
    using prof_clock = std::chrono::steady_clock;
    auto prof_ms = [](prof_clock::time_point a, prof_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto prof_t0 = prof_clock::now();
    static const bool s_trace_markers = aila::env::read_flag("AILA_Q4E_TRACE_MARKERS", false);
    double prof_router_sync_ms = 0.0;
    double prof_acquire_ms = 0.0;
    // P7 route-replay probe: counters only, env-gated, default off. Measures
    // how much of a token's cold set equals the previous token's route for the
    // same layer (input to a speculative CPU precompute design).
    static const bool s_route_replay_count = aila::env::read_flag("AILA_Q4E_ROUTE_REPLAY_COUNT", false);
    static std::array<std::array<int, 16>, 64> s_route_prev{};
    static std::array<uint8_t, 64> s_route_prev_valid{};
    uint64_t rr_slots = 0, rr_overlap = 0, rr_cold = 0, rr_cold_prev = 0;
    const auto adaptive_stats0 = adaptive_cache_.Stats();
    aila::HeteroExecutionStats cold_stats0 = expert_executor_.stats();

    // AILA_PROFILE_Q4E_STAGES serializes after every stage; use for relative cost only.
    static const bool s_stages = aila::env::read_flag("AILA_PROFILE_Q4E_STAGES", false);
    enum { kStagePle, kStageHc, kStageLinearAttn, kStageFullAttn, kStageRouter,
           kStageExperts, kStageShared, kStageHead, kStageCount };
    double stage_ms[kStageCount] = {};
    auto stage_t = prof_clock::now();
    const bool is_prefill = !options.decode_verification && (options.prefill_remaining >= 0 || count > 1);
    if (!is_prefill && expert_executor_.cpu_overlap_enabled() &&
        ops::cpu::TilePoolPinningEnabled() && ops::cpu::PhysicalCoreCount() >= 6) {
        static thread_local bool s_host_pinned = false;
        if (!s_host_pinned) {
            ops::cpu::PinCurrentThreadToNonWorkerCores();
            s_host_pinned = true;
            AILA_LOG_INFO("[Q4E HostThreadPin] pinned_non_worker_cores=1 worker_threads=%d physical_cores=%d",
                          ops::cpu::ConfiguredTilePoolThreads(), ops::cpu::PhysicalCoreCount());
        }
    }
    cold_split_this_forward_ = cold_split_num_ > 0 && direct_arena_stage_ &&
        count == 1 && !is_prefill && copy_q_ != nullptr;
    static const bool s_decode_pending_mlp = aila::env::read_flag("AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL", true);
    const bool decode_pending_mlp = !decodeHcReferenceForTest_ && s_decode_pending_mlp &&
        (count == 1 || (options.decode_verification && count >= 2 && count <= 4)) &&
        !is_prefill && !interior;
    DecodePendingMlpResidual pending_mlp(ctx, buf_.hidden4.data_as<bf16>(),
        buf_.y.data_as<bf16>(), buf_.hc_inject_f32.data_as<float>(), hidden_size_, count);
    static const bool s_moe_front_fused = aila::env::read_flag("AILA_Q4E_MOE_FRONT_FUSED", true);
    static const bool s_front_native_gemv = aila::env::read_flag("AILA_Q4E_DENSE_GEMV", true);
    // D1 layer-boundary decode pipeline defaults.
    static const bool s_cpu_route_early = aila::env::read_flag("AILA_Q4E_CPU_ROUTE_EARLY", true);
    static const bool s_stagein_hoist = aila::env::read_flag("AILA_Q4E_STAGEIN_HOIST", true);
    static const bool s_stagein_after_route =
        aila::env::read_flag("AILA_Q4E_STAGEIN_AFTER_ROUTE", false);
    // Decode-only: read back the route launch's execution status before the
    // shared expert is submitted, so the two submissions do not share the
    // route kernel's in-flight release point.
    static const bool s_join_release = aila::env::read_flag("AILA_Q4E_JOIN_RELEASE", true);
    static const bool s_fuse_shared = aila::env::read_flag("AILA_Q4E_MOE_SHARED_FUSED", true);
    auto stage = [&](int s) {
        if (!s_stages && !s_work_summary) return;
        if (s_stages) ctx.synchronize();
        const auto now = prof_clock::now();
        stage_ms[s] += prof_ms(stage_t, now);
        stage_t = now;
    };

    // --- R1 instrumentation (B6) ------------------------------------------------
    // (a) Host-side submit intervals for the sub-steps that previously only
    //     existed inside the unprinted "MoE-front/shared" residual. Pure
    //     steady_clock deltas around submission blocks; no queue interaction.
    const bool host_split = s_work_summary;
    double host_moe_front_ms = 0.0;
    double host_topk_submit_ms = 0.0;
    double host_shared_submit_ms = 0.0;
    double host_shared_add_ms = 0.0;
    double host_hc_inject_ms = 0.0;
    auto hsplit_begin = [&]() { return host_split ? prof_clock::now() : prof_clock::time_point{}; };
    auto hsplit_add = [&](double& acc, prof_clock::time_point t0) {
        if (host_split) acc += prof_ms(t0, prof_clock::now());
    };
    // (b) Optional queue-span device timing. The gate adds one in-order barrier
    //     per phase boundary and a single wait at the end of Forward, so it must
    //     stay off (default) for any throughput receipt. It is only usable when
    //     the queue was created with sycl::property::queue::enable_profiling
    //     (Context.hpp:32-35, engine gate AILA_PROFILE_Q4E_COLD_DEVICE); the
    //     shared-model worker owns a non-profiling Context (inference/Model.cpp
    //     DeviceState), so there the fields print 0 after a one-time warning.
    static const bool s_decode_stage_events = aila::env::read_flag("AILA_Q4E_DECODE_STAGE_EVENTS", false);
    const bool dev_trace_wanted = s_decode_stage_events && s_work_summary;
    bool dev_trace = false;
    if (dev_trace_wanted) {
        dev_trace = ctx.queue().has_property<sycl::property::queue::enable_profiling>();
        if (!dev_trace) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                AILA_LOG_WARN("[Q4E DecodeStageEvents] gate on but queue profiling is off; "
                              "dev_*_ms stay 0 (queue property fixed in Context.hpp:32-35)");
            }
        }
    }
    // Round 2 (R1 C2): kDevMisc is kept as the "unclassified" slot but no
    // marker credits it anymore; the former misc markers now land in
    // kDevInter (layer boundary), kDevPle (PLE block) and kDevPreRouter
    // (post StageResidentInput flag tail). kDevStageIn splits the
    // MLP-HC -> pre-router window at the resident-input handoff.
    // misc_dev_ms must print ~0: non-zero means a marker lost its bucket.
    enum { kDevMisc, kDevHc, kDevLinear, kDevFull, kDevRouter, kDevShared,
           kDevExperts, kDevResid, kDevHead,
           kDevInter, kDevPle, kDevStageIn, kDevPreRouter, kDevCount };
    double dev_ms[kDevCount] = {};
    struct DevMark { sycl::event ev; int bucket; };
    std::vector<DevMark> dev_marks;
    auto dev_mark = [&](int bucket) {
        if (!dev_trace) return;
        dev_marks.push_back(DevMark{ctx.queue().ext_oneapi_submit_barrier(), bucket});
    };
    dev_mark(kDevMisc);  // pre-loop anchor (mark[0] is never credited)
    // ----------------------------------------------------------------------------

    // D1: a stash from a previous forward call is never valid.
    d1_stash_valid = false;
    d1_stash_layer = -1;
    d1_stash_cpu_count = -1;
    deferred_cpu_join_pending_ = false;
    deferred_cpu_merge_event_ = sycl::event{};
    auto complete_deferred_cpu_join = [&]() {
        if (!deferred_cpu_join_pending_) return;
        const auto t_join0 = (s_stages || s_work_summary) ? prof_clock::now() : prof_clock::time_point{};
        if (expert_executor_.HasPendingArenaColdRows()) {
            expert_executor_.JoinDeferredArenaColdRows(deferred_cpu_merge_event_);
        } else {
            (void)expert_executor_.JoinResidentCold(resident_down_.data_as<float>(), nullptr);
            expert_executor_.FinishResidentCpuRead(deferred_cpu_merge_event_);
        }
        deferred_cpu_join_pending_ = false;
        deferred_cpu_merge_event_ = sycl::event{};
        if (s_stages || s_work_summary) {
            const auto now = prof_clock::now();
            stage_ms[kStageRouter] += prof_ms(stage_t, t_join0);
            stage_ms[kStageExperts] += prof_ms(t_join0, now);
            stage_t = now;
        }
    };
    struct DeferredCpuJoinGuard {
        Qwen4ExpBnb4Backend& self;
        ~DeferredCpuJoinGuard() noexcept {
            if (self.deferred_cpu_join_pending_) {
                self.expert_executor_.AbortResidentCold();
                self.deferred_cpu_join_pending_ = false;
                self.deferred_cpu_merge_event_ = sycl::event{};
            }
        }
    } deferredJoinGuard{*this};

    // Block-native QSA decode attends 512 selected blocks plus the 4-token
    // tail, so a single token past the 2,051-token dense budget no longer needs
    // the deferred CPU join drained at every full-attention layer boundary.
    static const bool s_qsa_block_native = aila::env::read_flag("AILA_QSA_BLOCK_NATIVE", true);

    // 2. 48 Transformer Layers
    for (int L = 0; L < cfg_.num_hidden_layers; ++L) {
        auto& layer = layers_[L];
        auto& cache = session.layerCaches_[L];
        if (!s_dump_dir.empty() || s_stages || dev_trace ||
            (!layer.is_linear && (start_pos + count) > 2051 &&
             (!s_qsa_block_native || count > 4 || verify_rows_alignment_))) {
            complete_deferred_cpu_join();
        }
        // Round 2: layer boundary = queue idle from the previous layer's last
        // marker through this loop header (dump checks, host bookkeeping).
        dev_mark(kDevInter);

        // 2.0. PLE on Layer 1 (Epilogue Residual Fused via P0 PleTicket)
        if (L == 1) {
            // PLE reads the residual stack before this layer's HC norm.
            pending_mlp.Materialize();
            if (!PleTicketEpochMatches(session, ple_ticket_) ||
                ple_ticket_.start_pos != start_pos) {
                ple_ticket_ = ple_.prepare(ctx, session.ple_state_, ple_ids_ptr, count,
                                           &session, session.generation(), start_pos,
                                           session.ple_ticket_epoch());
            }
            ple_.consume(ctx, ple_ticket_, session.ple_state_,
                         hidden4_ptr, nullptr, hidden4_ptr, checkpoint_after,
                         interior ? &interior->ple_snapshot : nullptr,
                         buf_.phase_scratch.valid() ? buf_.phase_scratch.data() : nullptr,
                         buf_.phase_scratch.valid() ? buf_.phase_scratch.size_bytes() : 0);
            ple_.commit(ple_ticket_, session.ple_state_);
        }

        stage(kStagePle);
        // Round 2: PLE-only window (pending_mlp.Materialize + ple prepare/
        // consume at L==1; ~empty on the other 47 layers).
        dev_mark(kDevPle);

        // 2.1. Attn Hyper-Connection
        if (verification_hidden_in_for_test_ &&
            count <= verification_stage_rows_for_test_) {
            pending_mlp.Materialize();
            ctx.queue().memcpy(
                verification_hidden_in_for_test_ + size_t(L) * verification_stage_rows_for_test_ * 10240,
                buf_.hidden4.data(), size_t(count) * 10240 * sizeof(bf16));
        }
        if (pending_mlp.pending)
            RunHyperConnectionCombined(ctx, layer.attn_hc, count, true, nullptr, nullptr, &pending_mlp);
        else
            RunHyperConnection(ctx, layer.attn_hc, count, true);
        if (verification_attn_in_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_attn_in_for_test_ + size_t(L) * verification_stage_rows_for_test_ * hidden_size_,
                buf_.x.data(), size_t(count) * hidden_size_ * sizeof(bf16));
        }
        stage(kStageHc);
        dev_mark(kDevHc);

        // Attention tiles advance this layer's state before routing the whole window.
        RunAttentionWindow(ctx, session, L, count, start_pos, interior, checkpoint_after, !is_prefill);
        if (verification_attn_raw_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_attn_raw_for_test_ + size_t(L) * verification_stage_rows_for_test_ * 6144,
                buf_.attn_out.data(), size_t(count) * 6144 * sizeof(bf16));
        }
        if (verification_attn_out_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_attn_out_for_test_ + size_t(L) * verification_stage_rows_for_test_ * hidden_size_,
                buf_.y.data(), size_t(count) * hidden_size_ * sizeof(bf16));
        }

        stage(layer.is_linear ? kStageLinearAttn : kStageFullAttn);
        dev_mark(layer.is_linear ? kDevLinear : kDevFull);

        // 2.3. Attn Residual Add & 2.4. MLP Hyper-Connection (P1 Fused)
        static const bool s_fuse_attn_mlp_gr = aila::env::read_flag("AILA_Q4E_FUSE_ATTN_MLP_GR", true);
        std::optional<sycl::event> mlp_hc_ev;
        if (s_fuse_attn_mlp_gr) {
            mlp_hc_ev = RunHyperConnectionCombined(ctx, layer.mlp_hc, count, true,
                                                   static_cast<const bf16*>(buf_.y.data()),
                                                   static_cast<const float*>(buf_.hc_inject_f32.data()));
        } else {
            ops::hc_inject_add(ctx, static_cast<bf16*>(buf_.hidden4.data()),
                               static_cast<const bf16*>(buf_.y.data()),
                               static_cast<const float*>(buf_.hc_inject_f32.data()),
                               count, hidden_size_, 4);
            mlp_hc_ev = RunHyperConnection(ctx, layer.mlp_hc, count, true);
        }
        stage(kStageHc);
        dev_mark(kDevHc);

        const bool moe_front_fused = s_moe_front_fused && s_front_native_gemv && moeFrontFusedGpu_ &&
            !decodeHcReferenceForTest_ && !gpuNf4Q8ReferenceForTest_ &&
            (count == 1 || (options.decode_verification && count >= 2 && count <= 4)) && !is_prefill &&
            hidden_size_ == 2560 && cfg_.num_experts == 512 && layer.router_w && layer.shared_expert_gate_w &&
            layer.router_w->dtype() == dnnl::memory::data_type::bf16 &&
            layer.shared_expert_gate_w->dtype() == dnnl::memory::data_type::bf16 &&
            layer.router_w->numel() == int64_t(512) * hidden_size_ && layer.shared_expert_gate_w->numel() == hidden_size_;
        const bool fused_x_stagein = moe_front_fused && expert_executor_.cpu_overlap_enabled() &&
            expert_executor_.x_host_ptr() != nullptr && count <= expert_executor_.max_cpu_tokens();
        const bool stagein_hoisted = !fused_x_stagein && s_stagein_hoist &&
            (count == 1 || (options.decode_verification && count >= 2 && count <= 4)) &&
            count <= expert_executor_.max_cpu_tokens() &&
            expert_executor_.cpu_overlap_enabled();
        verify_x_already_staged_ = fused_x_stagein || stagein_hoisted;
        const bool stagein_after_route = !fused_x_stagein && s_stagein_after_route && !s_stagein_hoist &&
            count == 1 && !is_prefill &&
            expert_executor_.cpu_overlap_enabled() && copy_q_ &&
            aila::q4e::moe_copy_via_copyq();
        static bool stagein_after_route_logged = false;
        if (stagein_after_route && !stagein_after_route_logged) {
            AILA_LOG_INFO("[Q4E StageInAfterRoute] enabled=1 count=1 copy_q_gate=1");
            stagein_after_route_logged = true;
        }
        if (stagein_hoisted)
            expert_executor_.StageResidentInput(buf_.x.data_as<bf16>(), count,
                mlp_hc_ev ? &*mlp_hc_ev : nullptr);

        // 2.5. MoE Computation
        if (!fused_x_stagein && !stagein_after_route && !stagein_hoisted && count == 1 && expert_executor_.cpu_overlap_enabled())
            expert_executor_.StageResidentInput(buf_.x.data_as<bf16>(), count,
                mlp_hc_ev ? &*mlp_hc_ev : nullptr);
        dev_mark(kDevStageIn);
        dev_mark(kDevPreRouter);
        const auto t_moe_front = hsplit_begin();
        if (moe_front_fused) {
            ops::bf16_moe_front_fused(ctx, layer.router_w->data_as<bf16>(), layer.shared_expert_gate_w->data_as<bf16>(),
                buf_.x.data_as<bf16>(), buf_.router_scores.data_as<bf16>(), buf_.shared_gate_buf.data_as<bf16>(),
                nullptr, nullptr, hidden_size_, cfg_.num_experts,
                fused_x_stagein ? expert_executor_.x_host_ptr() : nullptr,
                count);
            if (!moeFrontFusedLogged_) {
                AILA_LOG_INFO("[Q4E MoE Front Fused] actual_enabled=1 hidden=%d experts=%d shared_gate=1 q8_input=0 launches_saved_per_layer=1",
                    hidden_size_, cfg_.num_experts);
                moeFrontFusedLogged_ = true;
            }
        } else {
            dense_rows(layer.router_gate, layer.router_w, buf_.x, buf_.router_scores,
                       cfg_.num_experts, hidden_size_);
        }
        hsplit_add(host_moe_front_ms, t_moe_front);
        if (verification_router_inputs_for_test_ && options.decode_verification && count == 2)
            ctx.queue().memcpy(verification_router_inputs_for_test_ + size_t(L) * 2 * hidden_size_,
                buf_.x.data(), size_t(2) * hidden_size_ * sizeof(bf16));
        if (verification_router_x_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_router_x_for_test_ + size_t(L) * verification_stage_rows_for_test_ * hidden_size_,
                buf_.x.data(), size_t(count) * hidden_size_ * sizeof(bf16));
        }
        // Top-k writes straight into pinned host memory; the event wait is the only readback.
        const auto t_topk = hsplit_begin();
        std::chrono::steady_clock::time_point t_route_split{};
        if (s_profile && L == 0) t_route_split = std::chrono::steady_clock::now();
        sycl::event actual_route_event = ops::router_softmax_topk_device(
            ctx, static_cast<const bf16*>(buf_.router_scores.data()), nullptr,
            router_host_ids_, router_host_weights_,
            count, cfg_.num_experts, top_k, true);
        if (s_profile && L == 0) {
            const auto t_after_topk = std::chrono::steady_clock::now();
            AILA_LOG_INFO(
                "[Q4E RouteSplit] count=%d topk_submit_us=%.1f classify_submit_us=0 (no epoch)",
                count,
                std::chrono::duration<double, std::micro>(t_after_topk - t_route_split).count());
        }
        hsplit_add(host_topk_submit_ms, t_topk);
        dev_mark(kDevRouter);
        auto bind_actual_route = [&] {
            if (s_route_replay_count && count == 1 && L >= 0 && L < 64 &&
                adaptive_cache_.IsAllocated()) {
                int cold = 0, cold_prev = 0, overlap = 0;
                for (int k = 0; k < top_k; ++k) {
                    const int e = router_host_ids_[k];
                    const bool is_cold = adaptive_cache_.GetSlot(L, e) < 0;
                    bool in_prev = false;
                    if (s_route_prev_valid[L]) {
                        for (int j = 0; j < top_k; ++j) {
                            if (s_route_prev[L][j] == e) { in_prev = true; break; }
                        }
                    }
                    if (is_cold) ++cold;
                    if (in_prev) { ++overlap; if (is_cold) ++cold_prev; }
                }
                rr_slots += static_cast<uint64_t>(top_k);
                rr_cold += static_cast<uint64_t>(cold);
                rr_cold_prev += static_cast<uint64_t>(cold_prev);
                rr_overlap += static_cast<uint64_t>(overlap);
                for (int k = 0; k < top_k; ++k) s_route_prev[L][k] = router_host_ids_[k];
                s_route_prev_valid[L] = 1;
            }
        };
        auto wait_actual_route = [&] {
            const auto begin = prof_clock::now();
            if (L == 0 && adaptive_cache_.HasPendingSwaps()) {
                adaptive_cache_.CompletePendingSwaps();
            }
            actual_route_event.wait_and_throw();
            if (fused_x_stagein) {
                expert_executor_.MarkComputeQueueSyncedAfterRoute();
            }
            bind_actual_route();
            const auto end = prof_clock::now();
            prof_router_sync_ms += prof_ms(begin, end);
        };

        // Shared expert only depends on x: submit it before waiting on route_event so the GPU runs it
        // concurrently while the host fills missed routed experts and sets up dispatch.
        static const bool s_route_before_shared = aila::env::read_flag("AILA_Q4E_ROUTE_BEFORE_SHARED", false);
        const bool route_early_cold =
            s_route_before_shared && s_cpu_route_early && count == 1 && !is_prefill &&
            !options.decode_verification && !decodeHcReferenceForTest_ &&
            cfg_.num_experts_per_tok <= 16 && expert_executor_.cpu_overlap_enabled() &&
            adaptive_cache_.IsAllocated() && host_arena_.NumLayers() > 0;
        auto after_route = [&] {
            wait_actual_route();
            if (stagein_after_route)
                expert_executor_.StageResidentInput(buf_.x.data_as<bf16>(), count,
                    mlp_hc_ev ? &*mlp_hc_ev : nullptr);
            if (!(s_cpu_route_early && count == 1 && !is_prefill &&
                  !options.decode_verification && !decodeHcReferenceForTest_ &&
                  cfg_.num_experts_per_tok <= 16 && expert_executor_.cpu_overlap_enabled() &&
                  adaptive_cache_.IsAllocated() && host_arena_.NumLayers() > 0)) return;
            int miss_n = 0;
            std::array<int, 16> miss_ids{}, miss_ranks{};
            for (int k = 0; k < top_k; ++k) {
                const int e = router_host_ids_[k];
                if (adaptive_cache_.GetSlot(L, e) < 0) {
                    miss_ids[miss_n] = e;
                    miss_ranks[miss_n] = k;
                    ++miss_n;
                }
            }
            int cpu_n = 0;
            if (miss_n > 0)
                cpu_n = expert_executor_.StartArenaCold(host_arena_, L, miss_ids.data(),
                    miss_ranks.data(), miss_n, fused_x_stagein ? nullptr : buf_.x.data_as<bf16>());
            d1_stash_layer = L;
            d1_stash_cpu_count = cpu_n;
            for (int c = 0; c < cpu_n; ++c) {
                d1_stash_ids[c] = miss_ids[c];
                d1_stash_ranks[c] = miss_ranks[c];
            }
            d1_stash_valid = true;
        };
        if (route_early_cold) {
            complete_deferred_cpu_join();
            after_route();
        }

        const auto t_shared = hsplit_begin();
        if (!moe_front_fused)
            dense_rows(layer.shared_expert_gate, layer.shared_expert_gate_w, buf_.x, buf_.shared_gate_buf,
                       1, hidden_size_);
        if (count == 1 || ((verify_decode_kernels_ || verify_rows_alignment_) && count >= 2 && count <= 4)) {
            bool fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                ctx, layer.shared_gate_up_proj, buf_.x, buf_.shared_inter_buf, cfg_.shared_expert_intermediate_size, count);
            if (!fused_ok) {
                layer.shared_gate_up_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.x, buf_.shared_gate_up_buf, count);
                ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(buf_.shared_gate_up_buf.data()),
                                               static_cast<bf16*>(buf_.shared_inter_buf.data()), count, cfg_.shared_expert_intermediate_size);
            }
            layer.shared_down_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.shared_inter_buf, buf_.shared_out_buf, count);
        } else if (verify_rows_alignment_) {
            for (int r = 0; r < count; ++r) {
                auto x_row = Tensor::view(ctx, buf_.x.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                auto gu_row = Tensor::view(ctx, buf_.shared_gate_up_buf.data_as<bf16>() + size_t(r) * cfg_.shared_expert_intermediate_size,
                                           {1, cfg_.shared_expert_intermediate_size});
                auto si_row = Tensor::view(ctx, buf_.shared_inter_buf.data_as<bf16>() + size_t(r) * cfg_.shared_expert_intermediate_size,
                                           {1, cfg_.shared_expert_intermediate_size});
                auto so_row = Tensor::view(ctx, buf_.shared_out_buf.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                bool fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                    ctx, layer.shared_gate_up_proj, x_row, si_row, cfg_.shared_expert_intermediate_size);
                if (!fused_ok) {
                    layer.shared_gate_up_proj.ForwardDecodeRows(ctx, linear_scratch_, x_row, gu_row, 1);
                    ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(gu_row.data()),
                                                   static_cast<bf16*>(si_row.data()), 1, cfg_.shared_expert_intermediate_size);
                }
                layer.shared_down_proj.ForwardDecodeRows(ctx, linear_scratch_, si_row, so_row, 1);
            }
        } else {
            project_prefill(layer.shared_gate_up_proj, buf_.x, buf_.shared_gate_up_buf);
            ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(buf_.shared_gate_up_buf.data()),
                                           static_cast<bf16*>(buf_.shared_inter_buf.data()), count, cfg_.shared_expert_intermediate_size);
            project_prefill(layer.shared_down_proj, buf_.shared_inter_buf, buf_.shared_out_buf);
        }
        hsplit_add(host_shared_submit_ms, t_shared);
        dev_mark(kDevShared);

        if (!route_early_cold) {
            complete_deferred_cpu_join();
            after_route();
        }
        stage(kStageRouter);
        if (adaptive_cache_.IsAllocated()) {
            const uint64_t misses_before_moe = static_cast<uint64_t>(adaptive_cache_.Stats().cache_misses);
            if (count > 4) {
                RunAdaptivePrefill(ctx, L, s_fuse_shared, count);
            } else {
                RunResidentDecode(ctx, L, s_fuse_shared, count, options.decode_verification);
            }
            const uint64_t misses_after_moe = static_cast<uint64_t>(adaptive_cache_.Stats().cache_misses);
            if (misses_after_moe > misses_before_moe) {
                session.consecutive_no_miss_layers_ = 0;
            } else {
                session.consecutive_no_miss_layers_++;
            }
        } else {
            throw std::runtime_error("Qwen4Exp: adaptive expert cache is not allocated.");
        }
        stage(kStageExperts);
        dev_mark(kDevExperts);

        const auto t_shared_add = hsplit_begin();
        if (!s_fuse_shared) {
            add_shared_expert_to_y(ctx, buf_.y, buf_.shared_out_buf, buf_.shared_gate_buf,
                                   count, hidden_size_);
        }
        hsplit_add(host_shared_add_ms, t_shared_add);

        // 2.6. MLP Residual Add
        const auto t_hc_inject = hsplit_begin();
        if (decode_pending_mlp) {
            pending_mlp.Defer();
            // Per-layer observers and serial stage profiling require a complete
            // hidden stack here. Ordinary decode consumes it in the next norm.
            if (!s_dump_dir.empty() || s_stages) pending_mlp.Materialize();
        } else {
            ops::hc_inject_add(ctx, static_cast<bf16*>(buf_.hidden4.data()),
                               static_cast<const bf16*>(buf_.y.data()),
                               static_cast<const float*>(buf_.hc_inject_f32.data()),
                               count, hidden_size_, 4);
        }
        hsplit_add(host_hc_inject_ms, t_hc_inject);
        stage(kStageShared);
        dev_mark(kDevResid);

        if (!s_dump_dir.empty()) {
            std::string dump_path = s_dump_dir + "/layer_" + std::to_string(L) + ".bin";
            dump_hidden_row(ctx, dump_path, buf_.hidden4, count - 1, 10240);
        }
    }

    if (s_route_replay_count && count > 1) s_route_prev_valid.fill(0);
    if (s_route_replay_count && count == 1 && rr_slots > 0) {
        AILA_LOG_INFO("[Q4E RouteReplay] pos=%d slots=%llu overlap=%llu cold=%llu cold_from_prev=%llu overlap_rate=%.3f cold_cover=%.3f",
            start_pos, (unsigned long long)rr_slots, (unsigned long long)rr_overlap,
            (unsigned long long)rr_cold, (unsigned long long)rr_cold_prev,
            rr_slots ? double(rr_overlap) / double(rr_slots) : 0.0,
            rr_cold ? double(rr_cold_prev) / double(rr_cold) : 0.0);
    }

    bool run_adaptive_swap_now = false;
    int adaptive_max_swaps = 4;
    const bool end_of_prefill = is_prefill && options.need_logits && count > 4;
    if (adaptive_cache_.IsAllocated() && cold_epoch_enabled_) {
        static const int s_adaptive_swap_freq = aila::env::read_int_raw("AILA_Q4E_ADAPTIVE_SWAP_FREQ", 8);
        static const int s_max_swaps = aila::env::read_int_raw("AILA_Q4E_MAX_SWAPS", 4);
        static const int s_swap_min_pos = aila::env::read_int_raw("AILA_Q4E_ADAPTIVE_SWAP_MIN_POS", 48);
        adaptive_max_swaps = s_max_swaps;
        if (end_of_prefill) {
            run_adaptive_swap_now = true;
            adaptive_max_swaps = std::min(16, std::max(s_max_swaps, 16));
        } else if (!interior && (count == 1 || options.decode_verification) &&
                   s_adaptive_swap_freq > 0 && cached_len >= s_swap_min_pos) {
            if (count == 1) {
                if (cached_len % s_adaptive_swap_freq == 0) {
                    run_adaptive_swap_now = true;
                }
            } else {
                static int s_last_swap_bucket = -1;
                const int prev_bucket = start_pos / s_adaptive_swap_freq;
                const int curr_bucket = cached_len / s_adaptive_swap_freq;
                if (curr_bucket < s_last_swap_bucket) {
                    s_last_swap_bucket = -1;
                }
                if (curr_bucket > prev_bucket && curr_bucket != s_last_swap_bucket) {
                    s_last_swap_bucket = curr_bucket;
                    run_adaptive_swap_now = true;
                }
            }
        }
    }
    if ((run_adaptive_swap_now && !defer_adaptive_swap_to_caller_) || !options.need_logits || !s_dump_dir.empty()) {
        complete_deferred_cpu_join();
    }
    if (run_adaptive_swap_now) {
        if (defer_adaptive_swap_to_caller_) {
            pending_adaptive_swap_ = true;
            pending_adaptive_max_swaps_ = adaptive_max_swaps;
        } else {
            static const int s_max_per_layer = std::max(1, aila::env::read_int_raw("AILA_Q4E_MAX_SWAPS_PER_LAYER", 2));
            auto swap_plans = adaptive_cache_.PlanSwaps(s_max_per_layer, adaptive_max_swaps);
            if (!swap_plans.empty()) {
                sycl::queue& dma_q = copy_q_ ? *copy_q_ : ctx.queue();
                if (copy_q_) {
                    adaptive_cache_.BeginSwapsAsync(host_arena_, dma_q, swap_plans, direct_arena_stage_);
                    if (!s_dump_dir.empty()) adaptive_cache_.CompletePendingSwaps();
                } else {
                    adaptive_cache_.ExecuteSwapsAsync(host_arena_, dma_q, swap_plans, direct_arena_stage_);
                }
            }
            static const bool s_adaptive_swap_decay =
                aila::env::read_flag("AILA_Q4E_ADAPTIVE_SWAP_DECAY", true);
            if (s_adaptive_swap_decay) {
                adaptive_cache_.Decay(end_of_prefill ? 0.50f : 0.85f);
            }
        }
    }

    if (options.need_logits) {
        // 3. Final Hyper-Connection Mixer
        if (pending_mlp.pending)
            RunHyperConnectionCombined(ctx, mixer_, count, false, nullptr, nullptr, &pending_mlp);
        else
            RunHyperConnection(ctx, mixer_, count, false);

        if (!s_dump_dir.empty()) {
            std::string mixer_path = s_dump_dir + "/mixer.bin";
            dump_hidden_row(ctx, mixer_path, buf_.x, count - 1, 2560);
        }

        // 4. LM Head (Compute for last token)
        if (lastTokenOnly) {
            const bf16* last_x = static_cast<const bf16*>(buf_.x.data()) + static_cast<size_t>(count - 1) * hidden_size_;
            Tensor x_last_view = Tensor::view(ctx, const_cast<bf16*>(last_x), {1, hidden_size_});
            (void)ForwardLmHead(ctx, x_last_view, buf_.logits, 1);
            lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(), {1, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
        } else {
            if (buf_.logits.shape(0) < count) {
                buf_.logits = Tensor::allocate(ctx, {count, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
            }
            (void)ForwardLmHead(ctx, buf_.x, buf_.logits, count);
            lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(), {count, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
        }

        complete_deferred_cpu_join();
        stage(kStageHead);
        dev_mark(kDevHead);

        if (!s_dump_dir.empty()) {
            std::string logits_path = s_dump_dir + "/logits.bin";
            dump_hidden_row(ctx, logits_path, buf_.logits, 0, cfg_.vocab_size);
        }
    }
    complete_deferred_cpu_join();

    // No-logits forwards must also leave a materialized stack before snapshot,
    // state publication or return. This is a no-op after the final mixer.
    pending_mlp.Materialize();
    if (!options.decode_verification && !is_prefill && count > 1 && count <= 4) {
        CommitStreamedIndexerTail(ctx, session, start_pos, start_pos + count);
    }
    if (moe_compact_tile_ && count > 4) {
        int invalid=0;ctx.memcpy_d2h(&invalid,buf_.moe_compact_invalid.data(),sizeof(invalid));
        if(invalid)throw std::runtime_error("Compact MoE Q36 contribution is nonfinite or exceeds 2^20; window not committed");
    }
    if (interior) {
        const int position = start_pos + checkpoint_after;
        const int ratio = std::max(1, cfg_.indexer_compress_ratio);
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) {
                interior->layers[i].num_compressed_blocks =
                    std::min(session.layerCaches_[i].num_compressed_blocks, position / ratio);
            }
        }
        ctx.synchronize();
        interior->length = position;
    }
    if (options.snapshot_policy == SnapshotPolicy::ReusablePrefix) {
        static const bool s_skip_redundant_end_snapshot =
            aila::env::read_flag("AILA_Q4E_SKIP_REDUNDANT_END_SNAPSHOT", true);
        static const bool s_decode_stride_snapshot =
            aila::env::read_flag("AILA_Q4E_DECODE_STRIDE_SNAPSHOT", false);
        const bool is_unforced_decode_step =
            (count == 1 && !options.capture_snapshot && options.prefill_remaining < 0);
        if ((!is_unforced_decode_step || s_decode_stride_snapshot) &&
            !(s_skip_redundant_end_snapshot && interior != nullptr && count > 64 && !options.capture_snapshot)) {
            CaptureSnapshot(ctx, session, cached_len, count, options.capture_snapshot);
        }
    }
    session.currentLength_ = cached_len;
    if (options.prefill_remaining == 0 && !options.defer_decode_trim) TrimDecodeRuntimeBuffers(ctx);

    if (expert_histogram_enabled_ && count > 1) {
        for (int b = 0; b < 6; ++b) {
            const auto& bin = expert_histogram_[b];
            AILA_LOG_INFO("[Q4E ExpertHistogram] cumulative bucket=%d requests=%llu acts=%llu hits=%llu misses=%llu",
                b, static_cast<unsigned long long>(bin[0]), static_cast<unsigned long long>(bin[1]),
                static_cast<unsigned long long>(bin[2]), static_cast<unsigned long long>(bin[3]));
        }
        AILA_LOG_INFO("[Q4E GroupedXMX] cumulative tiles=%llu acts=%llu",
            static_cast<unsigned long long>(grouped_xmx_tiles_), static_cast<unsigned long long>(grouped_xmx_acts_));
    }
    if (dev_trace && !dev_marks.empty()) {
        dev_marks.back().ev.wait_and_throw();
        for (size_t i = 1; i < dev_marks.size(); ++i) {
            const auto t_prev = dev_marks[i - 1].ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            const auto t_cur = dev_marks[i].ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            if (t_cur > t_prev) dev_ms[dev_marks[i].bucket] += static_cast<double>(t_cur - t_prev) * 1e-6;
        }
    }
    if (s_work_summary) {
        const auto& a = adaptive_cache_.Stats();
        const auto& c = expert_executor_.stats();
        // These are coordinator/CPU intervals, never an additive GPU profile.
        // This summary itself adds no queue wait; the dev_events arm above is
        // the only gated synchronization.
        const double submitted_ms_now = prof_ms(prof_t0, prof_clock::now());
        const double stage_sum_ms = stage_ms[kStagePle] + stage_ms[kStageHc] + stage_ms[kStageLinearAttn] +
            stage_ms[kStageFullAttn] + stage_ms[kStageRouter] + stage_ms[kStageExperts] +
            stage_ms[kStageShared] + stage_ms[kStageHead];
        const double router_other_ms = std::max(0.0, stage_ms[kStageRouter] - prof_router_sync_ms -
            host_moe_front_ms - host_topk_submit_ms - host_shared_submit_ms);
        const double shared_other_ms = std::max(0.0, stage_ms[kStageShared] - host_shared_add_ms - host_hc_inject_ms);
        // HC submit split (AILA_Q4E_HC_SUBMIT_TIMING): steady_clock deltas
        // around the HC decode submit calls, windowed to this forward. All
        // zeros when the gate is off (the ops never read a clock then).
        const ops::HcSubmitTiming hc_submit = ops::HcSubmitTimingTakeWindow();
        auto hc_sub_us = [](const ops::HcSubmitTimingStage& s) {
            return static_cast<double>(s.total_ns) * 1e-3;
        };
        auto hc_sub_min_us = [](const ops::HcSubmitTimingStage& s) {
            return static_cast<double>(s.min_ns) * 1e-3;
        };
        auto hc_sub_max_us = [](const ops::HcSubmitTimingStage& s) {
            return static_cast<double>(s.max_ns) * 1e-3;
        };
        // Per-gap-bucket mean HC submit cost: the mean of the submits whose
        // preceding inter-submit gap fell into the bucket (0 with no samples).
        auto hc_gap_mean_us = [](const ops::HcGapBucket& b) {
            return b.count ? static_cast<double>(b.submit_total_ns) /
                                 static_cast<double>(b.count) * 1e-3
                           : 0.0;
        };
        // Staging DMA shape for this forward. `copies` is the number of H2D calls
        // the direct-arena path actually issued after collapsing runs of adjacent
        // arena slots, so staged/copies is the average experts per transfer.
        if (aila::env::read_flag("AILA_Q4E_PREFILL_EXPERT_TIMING", false)) {
            const uint64_t staged_now = a.cold_experts_staged - adaptive_stats0.cold_experts_staged;
            const uint64_t copies_now = a.staging_copies_merged - adaptive_stats0.staging_copies_merged;
            AILA_LOG_INFO("[Q4E StageShape] pos=%d count=%d staged=%llu copies=%llu "
                          "experts_per_copy=%.2f mb=%.1f banks=%d slots=%d",
                          start_pos, count, (unsigned long long)staged_now,
                          (unsigned long long)copies_now,
                          copies_now ? static_cast<double>(staged_now) /
                                           static_cast<double>(copies_now)
                                     : 0.0,
                          (a.cold_bytes_transferred - adaptive_stats0.cold_bytes_transferred) /
                              1048576.0,
                          adaptive_cache_.StagingBanks(),
                          adaptive_cache_.StagingSlotsPerBuffer());
        }
        // The HC submit/gap block leads the message: the line is truncated at
        // ~1025 chars, and hc_submit_down/up n/min/max must survive the cut.
        AILA_LOG_INFO("[Q4E AdaptiveWork] pos=%d count=%d ple_reused=%d "
            "hc_submit_norm_us=%.3f hc_submit_norm_n=%llu hc_submit_norm_min_us=%.3f hc_submit_norm_max_us=%.3f "
            "hc_submit_normfb_us=%.3f hc_submit_normfb_n=%llu hc_submit_normfb_min_us=%.3f hc_submit_normfb_max_us=%.3f "
            "hc_submit_down_us=%.3f hc_submit_down_n=%llu hc_submit_down_min_us=%.3f hc_submit_down_max_us=%.3f "
            "hc_submit_up_us=%.3f hc_submit_up_n=%llu hc_submit_up_min_us=%.3f hc_submit_up_max_us=%.3f "
            "hc_gap_b0_n=%llu hc_gap_b0_us=%.3f hc_gap_b1_n=%llu hc_gap_b1_us=%.3f "
            "hc_gap_b2_n=%llu hc_gap_b2_us=%.3f hc_gap_b3_n=%llu hc_gap_b3_us=%.3f hc_gap_max_us=%.3f "
            "submitted_ms=%.3f router_wait_ms=%.3f "
            "ple_host_ms=%.3f hc_host_ms=%.3f linear_host_ms=%.3f full_host_ms=%.3f "
            "experts_host_ms=%.3f head_host_ms=%.3f fill_ms=%.3f bank_wait_ms=%.3f swap_ms=%.3f "
            "cpu_compute_ms=%.3f cpu_join_ms=%.3f cpu_tasks=%llu hits=%llu misses=%llu staged=%llu h2d_mb=%.3f "
            "moe_front_ms=%.3f topk_submit_ms=%.3f shared_submit_ms=%.3f shared_add_ms=%.3f hc_inject_ms=%.3f "
            "router_other_ms=%.3f shared_other_ms=%.3f tail_ms=%.3f dev_events=%d "
            "misc_dev_ms=%.3f hc_dev_ms=%.3f linear_dev_ms=%.3f full_dev_ms=%.3f router_dev_ms=%.3f "
            "shared_dev_ms=%.3f experts_dev_ms=%.3f resid_dev_ms=%.3f head_dev_ms=%.3f "
            "inter_dev_ms=%.3f ple_dev_ms=%.3f stagein_dev_ms=%.3f prerouter_dev_ms=%.3f",
            start_pos,count,int(reuse_ple),
            hc_sub_us(hc_submit.norm_pending), (unsigned long long)hc_submit.norm_pending.count,
            hc_sub_min_us(hc_submit.norm_pending), hc_sub_max_us(hc_submit.norm_pending),
            hc_sub_us(hc_submit.norm_fallback), (unsigned long long)hc_submit.norm_fallback.count,
            hc_sub_min_us(hc_submit.norm_fallback), hc_sub_max_us(hc_submit.norm_fallback),
            hc_sub_us(hc_submit.down), (unsigned long long)hc_submit.down.count,
            hc_sub_min_us(hc_submit.down), hc_sub_max_us(hc_submit.down),
            hc_sub_us(hc_submit.up), (unsigned long long)hc_submit.up.count,
            hc_sub_min_us(hc_submit.up), hc_sub_max_us(hc_submit.up),
            (unsigned long long)hc_submit.gap_b0.count, hc_gap_mean_us(hc_submit.gap_b0),
            (unsigned long long)hc_submit.gap_b1.count, hc_gap_mean_us(hc_submit.gap_b1),
            (unsigned long long)hc_submit.gap_b2.count, hc_gap_mean_us(hc_submit.gap_b2),
            (unsigned long long)hc_submit.gap_b3.count, hc_gap_mean_us(hc_submit.gap_b3),
            static_cast<double>(hc_submit.gap_max_ns) * 1e-3,
            submitted_ms_now,prof_router_sync_ms,
            stage_ms[kStagePle],stage_ms[kStageHc],stage_ms[kStageLinearAttn],stage_ms[kStageFullAttn],
            stage_ms[kStageExperts],stage_ms[kStageHead],
            (a.staging_fill_ns-adaptive_stats0.staging_fill_ns)*1e-6,
            (a.staging_wait_ns-adaptive_stats0.staging_wait_ns)*1e-6,
            (a.swap_host_ns-adaptive_stats0.swap_host_ns)*1e-6,
            (c.cpu_compute_ns-cold_stats0.cpu_compute_ns)*1e-6,
            (c.cpu_join_wait_ns-cold_stats0.cpu_join_wait_ns)*1e-6,
            (unsigned long long)(c.cpu_tasks_completed-cold_stats0.cpu_tasks_completed),
            (unsigned long long)(a.cache_hits-adaptive_stats0.cache_hits),
            (unsigned long long)(a.cache_misses-adaptive_stats0.cache_misses),
            (unsigned long long)(a.cold_experts_staged-adaptive_stats0.cold_experts_staged),
            (a.cold_bytes_transferred-adaptive_stats0.cold_bytes_transferred)/1048576.0,
            host_moe_front_ms, host_topk_submit_ms, host_shared_submit_ms, host_shared_add_ms, host_hc_inject_ms,
            router_other_ms, shared_other_ms, submitted_ms_now - stage_sum_ms, int(dev_trace),
            dev_ms[kDevMisc], dev_ms[kDevHc], dev_ms[kDevLinear], dev_ms[kDevFull], dev_ms[kDevRouter],
            dev_ms[kDevShared], dev_ms[kDevExperts], dev_ms[kDevResid], dev_ms[kDevHead],
            dev_ms[kDevInter], dev_ms[kDevPle], dev_ms[kDevStageIn], dev_ms[kDevPreRouter]);
        // Dedicated HC submit/gap line, printed only with
        // AILA_Q4E_HC_SUBMIT_TIMING=1. It is short enough that none of its
        // fields can be cut by the logger's ~1025 char cap; the same values
        // also lead the [Q4E AdaptiveWork] line above.
        static const bool s_hc_submit_line =
            aila::env::read_int_raw("AILA_Q4E_HC_SUBMIT_TIMING", 0) != 0;
        if (s_hc_submit_line) {
            AILA_LOG_INFO("[Q4E HcSubmit] pos=%d count=%d hc_host_ms=%.3f "
                "hc_submit_norm_us=%.3f hc_submit_norm_n=%llu hc_submit_norm_min_us=%.3f hc_submit_norm_max_us=%.3f "
                "hc_submit_normfb_us=%.3f hc_submit_normfb_n=%llu hc_submit_normfb_min_us=%.3f hc_submit_normfb_max_us=%.3f "
                "hc_submit_down_us=%.3f hc_submit_down_n=%llu hc_submit_down_min_us=%.3f hc_submit_down_max_us=%.3f "
                "hc_submit_up_us=%.3f hc_submit_up_n=%llu hc_submit_up_min_us=%.3f hc_submit_up_max_us=%.3f "
                "hc_gap_b0_n=%llu hc_gap_b0_us=%.3f hc_gap_b1_n=%llu hc_gap_b1_us=%.3f "
                "hc_gap_b2_n=%llu hc_gap_b2_us=%.3f hc_gap_b3_n=%llu hc_gap_b3_us=%.3f hc_gap_max_us=%.3f",
                start_pos, count, stage_ms[kStageHc],
                hc_sub_us(hc_submit.norm_pending), (unsigned long long)hc_submit.norm_pending.count,
                hc_sub_min_us(hc_submit.norm_pending), hc_sub_max_us(hc_submit.norm_pending),
                hc_sub_us(hc_submit.norm_fallback), (unsigned long long)hc_submit.norm_fallback.count,
                hc_sub_min_us(hc_submit.norm_fallback), hc_sub_max_us(hc_submit.norm_fallback),
                hc_sub_us(hc_submit.down), (unsigned long long)hc_submit.down.count,
                hc_sub_min_us(hc_submit.down), hc_sub_max_us(hc_submit.down),
                hc_sub_us(hc_submit.up), (unsigned long long)hc_submit.up.count,
                hc_sub_min_us(hc_submit.up), hc_sub_max_us(hc_submit.up),
                (unsigned long long)hc_submit.gap_b0.count, hc_gap_mean_us(hc_submit.gap_b0),
                (unsigned long long)hc_submit.gap_b1.count, hc_gap_mean_us(hc_submit.gap_b1),
                (unsigned long long)hc_submit.gap_b2.count, hc_gap_mean_us(hc_submit.gap_b2),
                (unsigned long long)hc_submit.gap_b3.count, hc_gap_mean_us(hc_submit.gap_b3),
                static_cast<double>(hc_submit.gap_max_ns) * 1e-3);
        }
    }
    if (component_profile_active_for_test_) {
        const auto& p = component_profile_ms_for_test_;
        AILA_LOG_INFO("[Q4E Components] pos=%d count=%d linear_in=%.4f gdn=%.4f linear_out=%.4f router=%.4f shared_gate=%.4f ms serialized=1",
            start_pos,count,p[0],p[1],p[2],p[3],p[4]);
        component_profile_active_for_test_=false;
    }
    if (s_profile) {
        ctx.synchronize();
        const auto& as = adaptive_cache_.Stats();
        AILA_LOG_INFO("[Q4E Adaptive Profile] pos=%d count=%d total=%.2fms router_sync=%.2fms hits=%llu misses=%llu staged=%llu DMA=%.1fMB logits=%d",
                      start_pos, count, prof_ms(prof_t0, prof_clock::now()),
                      prof_router_sync_ms,
                      (unsigned long long)as.cache_hits,
                      (unsigned long long)as.cache_misses,
                      (unsigned long long)as.cold_experts_staged,
                      (as.warm_bytes_transferred + as.cold_bytes_transferred + as.swap_bytes_transferred) / (1024.0 * 1024.0),
                      options.need_logits ? 1 : 0);
        if (count > 1) {
            auto dstats = Bnb4BitLinear::get_prefill_dispatch_stats();
            AILA_LOG_INFO("[Q4E Prefill Dispatch] linear calls: fused_gemm=%llu cached=%llu on_demand=%llu sync_waits=%llu direct_io=%llu",
                          static_cast<unsigned long long>(dstats.fused_gemm),
                          static_cast<unsigned long long>(dstats.cached_weight),
                          static_cast<unsigned long long>(dstats.on_demand_dequant),
                          static_cast<unsigned long long>(dstats.sync_wait),
                          static_cast<unsigned long long>(dstats.direct_io));
        }
        if (s_stages) {
            AILA_LOG_INFO("[Q4E Stages] ple=%.2f hc=%.2f linear_attn=%.2f full_attn=%.2f router=%.2f "
                          "experts=%.2f (acquire=%.2f) shared=%.2f head=%.2f ms",
                          stage_ms[kStagePle], stage_ms[kStageHc], stage_ms[kStageLinearAttn],
                          stage_ms[kStageFullAttn], stage_ms[kStageRouter], stage_ms[kStageExperts],
                          prof_acquire_ms, stage_ms[kStageShared], stage_ms[kStageHead]);
        }
    }
    static const bool s_log_dispatch = aila::env::read_flag("AILA_Q4E_LOG_DISPATCH", false);
    if (s_log_dispatch && count > 1) {
        const auto prefill_stats_t1 = Bnb4BitLinear::get_prefill_dispatch_stats();
        const uint64_t d_fused = prefill_stats_t1.fused_gemm - prefill_stats_t0.fused_gemm;
        const uint64_t d_dequant = prefill_stats_t1.on_demand_dequant - prefill_stats_t0.on_demand_dequant;
        const uint64_t d_sync = prefill_stats_t1.sync_wait - prefill_stats_t0.sync_wait;
        AILA_LOG_INFO("[Q4E Prefill] dispatch delta: fused_gemm=+%llu, on_demand_dequant=+%llu, sync_wait=+%llu",
                      static_cast<unsigned long long>(d_fused),
                      static_cast<unsigned long long>(d_dequant),
                      static_cast<unsigned long long>(d_sync));
    }
    return options.need_logits ? lastTokenLogits_ : unusedLogits_;
}

// ============================================================================
// Snapshot & Rollback
// ============================================================================

Qwen4ExpBnb4Backend::StateSnapshot& Qwen4ExpBnb4Backend::ReserveSnapshot(Session& session, int nextLength) {
    auto& snapshots = session.snapshots_;
    auto slot = std::find_if(snapshots.begin(), snapshots.end(),
        [nextLength](const auto& snapshot) { return snapshot.length == nextLength; });
    if (slot == snapshots.end()) {
        slot = std::min_element(snapshots.begin(), snapshots.end(),
            [](const auto& left, const auto& right) { return left.length < right.length; });
    }
    auto& snapshot = *slot;
    snapshot.length = -1;
    return snapshot;
}

void Qwen4ExpBnb4Backend::CaptureSnapshot(
    Context& ctx, Session& session, int nextLength, int tokenCount,
    bool force, bool preserve_token_graph) {
    if (session.token_graph_poisoned()) {
        throw std::logic_error(
            "Qwen4Exp cannot checkpoint a poisoned token-graph session");
    }
    if (session.snapshots_.empty()) return;
    int step = aila::env::read_int("AILA_Q4E_PREFILL_STEP", 64);
    if (step <= 0) step = 64;
    if (!force && tokenCount <= 1 && nextLength % step != 0) return;

    ctx.synchronize();
    SynchronizeGdnDeviceHeadsToHost(
        ctx, session, preserve_token_graph);
    auto& snapshot = ReserveSnapshot(session, nextLength);
    try {
        sycl::event last_d2h_event{};
        bool have_last_d2h_event = false;
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) continue;
            auto& cache = session.layerCaches_[i];
            if (cache.device_state_dirty) {
                if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
                    last_d2h_event = ctx.memcpy_d2h_async(cache.host_linear_state.data(), cache.linear_state.data(),
                                                          cache.host_linear_state.size() * sizeof(float));
                    have_last_d2h_event = true;
                }
                if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
                    last_d2h_event = ctx.memcpy_d2h_async(cache.host_linear_conv_state.data(), cache.linear_conv_state.data(),
                                                          cache.host_linear_conv_state.size() * sizeof(float));
                    have_last_d2h_event = true;
                }
            }
        }
        if (have_last_d2h_event) {
            last_d2h_event.wait_and_throw();
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            auto& cache = session.layerCaches_[i];
            if (layers_[i].is_linear) {
                std::copy(cache.host_linear_state.begin(), cache.host_linear_state.end(),
                          snapshot.layers[i].linear_state_host.begin());
                std::copy(cache.host_linear_conv_state.begin(), cache.host_linear_conv_state.end(),
                          snapshot.layers[i].linear_conv_state_host.begin());
                cache.device_state_dirty = false;
                snapshot.layers[i].linear_conv_head = cache.linear_conv_head;
            } else {
                snapshot.layers[i].num_compressed_blocks = cache.num_compressed_blocks;
                const int rem = nextLength % 4;
                if (cache.indexer_k_tail_only && rem > 0 && cache.indexer_k.valid()) {
                    snapshot.layers[i].indexer_k_tail.resize(static_cast<size_t>(rem) * 128u);
                    ctx.memcpy_d2h(snapshot.layers[i].indexer_k_tail.data(),
                                   cache.indexer_k.data(),
                                   snapshot.layers[i].indexer_k_tail.size() * sizeof(bf16));
                } else {
                    snapshot.layers[i].indexer_k_tail.clear();
                }
            }
        }
        ple_.capture_snapshot(ctx, session.ple_state_, snapshot.ple_snapshot);
        snapshot.length = nextLength;
    } catch (...) {
        ctx.synchronize();
        throw;
    }
}

int Qwen4ExpBnb4Backend::Truncate(Session& session, int newLength) {
    ValidateSession(session);
    // Abort before touching graph-visible state. The synchronize below is the
    // caller-side drain; AbortTokenGraph() itself never waits.
    (void)AbortTokenGraph();
    if (newLength < 0 || newLength > session.currentLength_) {
        return session.currentLength_;
    }
    session.context_->synchronize();
    if (newLength == 0) {
        Reset(session);
        return 0;
    }
    int target = -1;
    const StateSnapshot* selected = nullptr;
    for (const auto& snapshot : session.snapshots_) {
        if (snapshot.length <= newLength && snapshot.length > target) {
            target = snapshot.length;
            selected = &snapshot;
        }
    }
    if (target < 0) {
        Reset(session);
        return 0;
    }
    const auto& snapshot = *selected;
    sycl::event last_h2d_event{};
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        const auto& snap_layer = snapshot.layers[i];
        if (layers_[i].is_linear) {
            std::copy(snap_layer.linear_state_host.begin(), snap_layer.linear_state_host.end(),
                      cache.host_linear_state.begin());
            std::copy(snap_layer.linear_conv_state_host.begin(), snap_layer.linear_conv_state_host.end(),
                      cache.host_linear_conv_state.begin());
            cache.linear_conv_head = snap_layer.linear_conv_head;
            if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
                last_h2d_event = session.context_->memcpy_h2d_async(cache.linear_state.data(), cache.host_linear_state.data(),
                                                                    cache.host_linear_state.size() * sizeof(float));
            }
            if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
                last_h2d_event = session.context_->memcpy_h2d_async(cache.linear_conv_state.data(), cache.host_linear_conv_state.data(),
                                                                    cache.host_linear_conv_state.size() * sizeof(float));
            }
            cache.device_state_dirty = false;
        } else {
            cache.num_compressed_blocks = snap_layer.num_compressed_blocks;
            if (cache.indexer_k_tail_only && !snap_layer.indexer_k_tail.empty() && cache.indexer_k.valid()) {
                session.context_->memcpy_h2d(
                    cache.indexer_k.data(),
                    snap_layer.indexer_k_tail.data(),
                    snap_layer.indexer_k_tail.size() * sizeof(bf16));
            }
            if (cache.kv_stream_enabled) {
                ops::qsa_kv_stream_reset(*session.context_, cache.kv_stream_map);
            }
        }
    }
    if (last_h2d_event != sycl::event{}) {
        last_h2d_event.wait_and_throw();
    }
    PublishGdnHostHeadsToDevice(*session.context_, session);
    ple_.restore_snapshot(*session.context_, session.ple_state_, snapshot.ple_snapshot);
    session.bump_generation();
    ple_ticket_.valid = false;
    ple_next_ticket_.valid = false;

    for (auto& entry : session.snapshots_) {
        if (entry.length > target) entry.length = -1;
    }
    session.currentLength_ = target;
    session.consecutive_no_miss_layers_ = 0;
    return target;
}

void Qwen4ExpBnb4Backend::Reset(Session& session) {
    ValidateSession(session);
    // Abort first, then drain before mutating model-visible cache state.
    (void)AbortTokenGraph();
    session.context_->synchronize();
    session.consecutive_no_miss_layers_ = 0;
    for (auto& snapshot : session.snapshots_) snapshot.length = -1;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        if (layers_[i].is_linear) {
            if (cache.linear_state.valid()) {
                session.context_->queue().memset(
                    cache.linear_state.data(), 0, cache.linear_state.size_bytes());
            }
            if (cache.linear_conv_state.valid()) {
                session.context_->queue().memset(
                    cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
            }
            std::fill(cache.host_linear_state.begin(), cache.host_linear_state.end(), 0.0f);
            std::fill(cache.host_linear_conv_state.begin(), cache.host_linear_conv_state.end(), 0.0f);
            cache.device_state_dirty = false;
            cache.linear_conv_head = 0;
        } else {
            cache.num_compressed_blocks = 0;
            if (cache.kv_stream_enabled) {
                ops::qsa_kv_stream_reset(*session.context_, cache.kv_stream_map);
            }
        }
    }
    PublishGdnHostHeadsToDevice(*session.context_, session);
    ple_.init_state(*session.context_, session.ple_state_);
    session.bump_generation();
    ple_ticket_.valid = false;
    ple_next_ticket_.valid = false;
    session.currentLength_ = 0;
    session.context_->synchronize();
}

bool Qwen4ExpBnb4Backend::truncate_kv_cache(int new_len) {
    if (new_len >= get_current_context_len()) return true;
    if (new_len <= 0) {
        reset();
        return false;
    }
    return Truncate(DefaultSession(*context_), new_len) > 0;
}

void Qwen4ExpBnb4Backend::reset() {
    if (defaultSession_) Reset(*defaultSession_);
}

std::unique_ptr<Qwen4ExpBnb4Backend::Session> Qwen4ExpBnb4Backend::CreateSession(
    Context& ctx, std::size_t maxSnapshots) {
    if (next_session_instance_id_ == 0) {
        throw std::overflow_error("Qwen4Exp session instance id overflow");
    }
    const uint64_t instanceId = next_session_instance_id_++;
    auto session = std::unique_ptr<Session>(
        new Session(ctx, max_seq_len_, sessionOwner_, maxSnapshots, instanceId,
                    &token_graph_runtime_, resolved_kv_resident_tokens_));
    ple_.init_state(ctx, session->ple_state_);
    // Exact device payload of the tensors this Session owns, matching
    // qwen4exp_session_memory's device reservation. Model weights, expert
    // cache and shared execution scratch are budgeted elsewhere.
    size_t cacheBytes = 0;
    for (const auto& cache : session->layerCaches_) {
        for (const Tensor* tensor : {&cache.k, &cache.v,
                                     &cache.indexer_compressed_k, &cache.linear_state,
                                     &cache.linear_conv_state, &cache.k_scale,
                                     &cache.v_scale,
                                     &cache.kv_stream_map.page_table,
                                     &cache.kv_stream_map.slot_block,
                                     &cache.kv_stream_map.slot_stamp,
                                     &cache.kv_stream_map.slot_ref,
                                     &cache.kv_stream_map.miss_block,
                                     &cache.kv_stream_map.miss_slot,
                                     &cache.kv_stream_map.ctl}) {
            cacheBytes = aila::memory::checked_memory_sum(cacheBytes, tensor->size_bytes());
        }
        if (!cache.kv_stream_enabled || cache.indexer_k_tail_only) {
            cacheBytes = aila::memory::checked_memory_sum(cacheBytes, cache.indexer_k.size_bytes());
        }
    }
    cacheBytes = aila::memory::checked_memory_sum(
        cacheBytes, session->gdn_conv_heads_dev_.size_bytes());
    session->cacheBytes_ = aila::memory::checked_memory_sum(
        cacheBytes, session->ple_state_.conv_state.size_bytes());
    session->cacheBytes_ = aila::memory::checked_memory_sum(
        session->cacheBytes_, session->ple_state_.conv_head_dev.size_bytes());
    for (auto& snapshot : session->snapshots_) {
        snapshot.layers.resize(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) continue;
            const auto& cache = session->layerCaches_[i];
            snapshot.layers[i].linear_state_host.resize(cache.host_linear_state.size());
            snapshot.layers[i].linear_conv_state_host.resize(cache.host_linear_conv_state.size());
        }
    }
    return session;
}

void Qwen4ExpBnb4Backend::PrepareKvStagingBuffers(
    Context& ctx, int stageLen, bool int8Kv, bool sparsePrefill,
    Tensor& stageK, Tensor& stageV, Tensor& stageKScale, Tensor& stageVScale) {
    const auto dtype = int8Kv ? dnnl::memory::data_type::s8 : dnnl::memory::data_type::bf16;
    const size_t elemBytes = int8Kv ? 1u : sizeof(bf16);
    const size_t kvBytes = static_cast<size_t>(full_kv_heads_) * stageLen * full_head_dim_ * elemBytes;
    const size_t scaleBytes = int8Kv ? (static_cast<size_t>(full_kv_heads_) * stageLen * sizeof(float)) : 0u;
    if (buf_.kv_stream_stage_in_phase_scratch && buf_.phase_scratch.valid()) {
        uint8_t* base = buf_.phase_scratch.data_as<uint8_t>();
        const size_t scratchBytes = buf_.phase_scratch.size_bytes();
        const size_t kvAligned = (kvBytes + 255u) & ~size_t(255u);
        const size_t scaleAligned = (scaleBytes + 255u) & ~size_t(255u);
        if (sparsePrefill && stageLen <= max_seq_len_ &&
            buf_.kv_stream_stage_v_offset + kvAligned + 2u * scaleAligned <= scratchBytes &&
            kvBytes <= buf_.kv_stream_stage_k_max_bytes) {
            stageK = Tensor::view(ctx, base, {full_kv_heads_, stageLen, full_head_dim_}, dtype);
            stageV = Tensor::view(ctx, base + buf_.kv_stream_stage_v_offset, {full_kv_heads_, stageLen, full_head_dim_}, dtype);
            if (int8Kv) {
                stageKScale = Tensor::view(ctx, base + buf_.kv_stream_stage_k_scale_offset, {full_kv_heads_, stageLen}, dnnl::memory::data_type::f32);
                stageVScale = Tensor::view(ctx, base + buf_.kv_stream_stage_v_scale_offset, {full_kv_heads_, stageLen}, dnnl::memory::data_type::f32);
            }
            return;
        }
        if (!sparsePrefill &&
            buf_.kv_stream_stage_v_offset + 2u * kvAligned + 2u * scaleAligned <= scratchBytes) {
            uint8_t* ptr = base + buf_.kv_stream_stage_v_offset;
            stageK = Tensor::view(ctx, ptr, {full_kv_heads_, stageLen, full_head_dim_}, dtype);
            stageV = Tensor::view(ctx, ptr + kvAligned, {full_kv_heads_, stageLen, full_head_dim_}, dtype);
            if (int8Kv) {
                stageKScale = Tensor::view(ctx, ptr + 2u * kvAligned, {full_kv_heads_, stageLen}, dnnl::memory::data_type::f32);
                stageVScale = Tensor::view(ctx, ptr + 2u * kvAligned + scaleAligned, {full_kv_heads_, stageLen}, dnnl::memory::data_type::f32);
            }
            return;
        }
    }
    if (!buf_.kv_stream_stage_k.valid() || buf_.kv_stream_stage_k.dtype() != dtype ||
        buf_.kv_stream_stage_k.shape(1) < stageLen) {
        buf_.kv_stream_stage_k = Tensor::allocate(ctx, {full_kv_heads_, max_seq_len_, full_head_dim_}, dtype);
        buf_.kv_stream_stage_v = Tensor::allocate(ctx, {full_kv_heads_, max_seq_len_, full_head_dim_}, dtype);
        if (int8Kv) {
            buf_.kv_stream_stage_k_scale = Tensor::allocate(ctx, {full_kv_heads_, max_seq_len_}, dnnl::memory::data_type::f32);
            buf_.kv_stream_stage_v_scale = Tensor::allocate(ctx, {full_kv_heads_, max_seq_len_}, dnnl::memory::data_type::f32);
        }
    }
    stageK = Tensor::view(ctx, buf_.kv_stream_stage_k.data(), {full_kv_heads_, stageLen, full_head_dim_}, dtype);
    stageV = Tensor::view(ctx, buf_.kv_stream_stage_v.data(), {full_kv_heads_, stageLen, full_head_dim_}, dtype);
    if (int8Kv) {
        stageKScale = Tensor::view(ctx, buf_.kv_stream_stage_k_scale.data(), {full_kv_heads_, stageLen}, dnnl::memory::data_type::f32);
        stageVScale = Tensor::view(ctx, buf_.kv_stream_stage_v_scale.data(), {full_kv_heads_, stageLen}, dnnl::memory::data_type::f32);
    }
}

Tensor& Qwen4ExpBnb4Backend::Forward(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count,
    const ForwardOptions& options) {
    if (WholeTokenGraphEnabled() &&
        WholeTokenGraphEligible(ctx, session, count, options)) {
        return ForwardWholeTokenGraph(
            ctx, session, tokenIdsDevice, count, options);
    }
    if (whole_token_graph_.executable &&
        !ReleaseWholeTokenGraph(&ctx, &session, false)) {
        throw std::runtime_error(
            "Qwen4Exp whole-token graph could not drain before direct path");
    }
    return ForwardImpl(ctx, session, tokenIdsDevice, count, true, options);
}

Tensor& Qwen4ExpBnb4Backend::ForwardLegacyLogits(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count) {
    if (whole_token_graph_.executable &&
        !ReleaseWholeTokenGraph(&ctx, &session, false)) {
        throw std::runtime_error(
            "Qwen4Exp whole-token graph could not drain before legacy path");
    }
    return ForwardImpl(ctx, session, tokenIdsDevice, count, false, {});
}

Tensor& Qwen4ExpBnb4Backend::forward(
    Context& ctx, const int* token_ids_device, int seq_len) {
    return Forward(ctx, DefaultSession(ctx), token_ids_device, seq_len);
}

bool Qwen4ExpBnb4Backend::CanCaptureInteriorCheckpoint(
    const Session& session, int count, int checkpointAfter) const {
    const bool allow_long = aila::env::read_flag("AILA_Q4E_LONG_INLINE_CHECKPOINT", false) &&
                            prefill_attention_tile_ == 0;
    const int max_count = allow_long ? session.Capacity() : 64;
    if (count < 4 || count > max_count || checkpointAfter < 2 || checkpointAfter > count - 2 ||
        session.snapshots_.size() < 2 || count > session.Capacity() - session.CurrentLength() ||
        !session.embeddingOverridePositions_.empty() || session.mropePosT_.valid()) return false;
    if (allow_long) return true;
    // Match ForwardImpl's sparse-prefill predicate exactly. At the default ratio/budget,
    // 2052 is already sparse (513 complete blocks); the final dense position is 2051.
    const int ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
    const int blockTopK = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / ratio;
    return (session.CurrentLength() + count) / ratio <= blockTopK;
}

bool Qwen4ExpBnb4Backend::supports_interior_checkpoint(int count, int checkpoint_after) const {
    bool explicit_inline = false;
    const bool has_env = aila::env::read_flag_if_present("AILA_Q4E_INLINE_CHECKPOINT", &explicit_inline);
    if (has_env && !explicit_inline) return false;
    auto* self = const_cast<Qwen4ExpBnb4Backend*>(this);
    if (!defaultSession_ && context_) {
        self->DefaultSession(*context_);
    }
    if (!defaultSession_ || !CanCaptureInteriorCheckpoint(*defaultSession_, count, checkpoint_after)) return false;
    // X2 measured default: initial 30-token text prefill with a 26-token reusable prefix.
    // Explicit opt-in enables other structurally valid anchors, without a bitwise whitelist.
    return has_env || (defaultSession_->CurrentLength() == 0 && count == 30 && checkpoint_after == 26);
}

Tensor& Qwen4ExpBnb4Backend::forward(
    Context& ctx, const int* token_ids_device, int seq_len, const ForwardOptions& options) {
    return Forward(ctx, DefaultSession(ctx), token_ids_device, seq_len, options);
}

Qwen4ExpBnb4Backend::Session& Qwen4ExpBnb4Backend::DefaultSession(Context& ctx) {
    if (!defaultSession_) {
        defaultSession_ = CreateSession(ctx);
    }
    return *defaultSession_;
}

void Qwen4ExpBnb4Backend::ValidateSession(const Session& session) const {
    if (session.owner_ != sessionOwner_) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: session belongs to another backend");
    }
}

bool Qwen4ExpBnb4Backend::PleTicketEpochMatches(
    const Session& session, const PleTicket& ticket) const noexcept {
    return ticket.valid && ticket.session_ptr == &session &&
        ticket.generation == session.generation() &&
        ticket.ticket_epoch == session.ple_ticket_epoch();
}

void Qwen4ExpBnb4Backend::ValidateTokenGraphLaunch(Session& session) {
    ValidateSession(session);
    const auto& key = token_graph_runtime_.key();
    if (key.body != aila::TokenGraphBody::WholeToken) {
        ValidateGdnHeadPath(session, 1);
        if (!session.graph_gdn_state_enabled_) return;
    } else if (!session.graph_gdn_state_enabled_) {
        throw std::invalid_argument(
            "Qwen4Exp whole-token graph requires device GDN state");
    }
    for (const auto& cache : session.layerCaches_) {
        if (cache.linear_state.valid() && !GdnDeviceHeadReadable(cache)) {
            throw std::invalid_argument(
                "Qwen4Exp token graph cannot launch before the host "
                "GDN heads are published to the device");
        }
    }
}

bool Qwen4ExpBnb4Backend::ReleaseQsaLayerGraph(
    Context* ctx, bool idle_confirmed) noexcept {
    if (!qsa_layer_graph_.executable) {
        qsa_layer_graph_.node_count = 0;
        return true;
    }

    bool safe = idle_confirmed ||
        token_graph_runtime_.event_status() ==
            aila::TokenGraphEventStatus::Done;
    if (!safe && ctx) {
        safe = ctx->drain_noexcept();
    }
    if (!safe) return false;

    qsa_layer_graph_.executable.reset();
    qsa_layer_graph_.node_count = 0;
    return true;
}

bool Qwen4ExpBnb4Backend::BeginCapture(
    Session& session, aila::TokenGraphBody body,
    aila::TokenGraphAttention attention, uint64_t topology_flags) {
    const auto key =
        MakeTokenGraphKey(session, body, attention, topology_flags);
    return token_graph_runtime_.BeginCapture(key);
}

bool Qwen4ExpBnb4Backend::MarkReady() noexcept {
    return token_graph_runtime_.MarkExecutableReady();
}

bool Qwen4ExpBnb4Backend::Launch(
    Session& session, uint32_t sequence, sycl::event event) {
    if (token_graph_runtime_.phase() != aila::TokenGraphPhase::Ready) {
        return false;
    }
    const auto& key = token_graph_runtime_.key();
    if (key.session != &session ||
        key.session_instance_id != session.session_instance_id()) {
        return false;
    }
    if (key.body == aila::TokenGraphBody::QsaLayerProof) {
        ValidateSession(session);
    } else {
        ValidateTokenGraphLaunch(session);
    }
    if (!token_graph_runtime_.Launch(sequence, std::move(event))) {
        return false;
    }
    if (key.body != aila::TokenGraphBody::QsaLayerProof) {
        MarkGdnGraphStateDirty(session);
    }
    return true;
}

bool Qwen4ExpBnb4Backend::Complete(
    uint32_t sequence, bool gate_valid) noexcept {
    return token_graph_runtime_.Complete(sequence, gate_valid);
}

bool Qwen4ExpBnb4Backend::AbortTokenGraph() noexcept {
    // Empty is the default direct-only state. Never turn it into Aborting.
    if (token_graph_runtime_.phase() == aila::TokenGraphPhase::Empty) {
        return false;
    }
    return token_graph_runtime_.Abort();
}

bool Qwen4ExpBnb4Backend::AbortAndPoisonTokenGraph(
    Session& session) noexcept {
    if (session.owner_ != sessionOwner_) return false;
    const bool changed = token_graph_runtime_.Poison();
    session.PoisonTokenGraph();
    return changed;
}

aila::TokenGraphKey Qwen4ExpBnb4Backend::MakeTokenGraphKey(
    const Session& session, aila::TokenGraphBody body,
    aila::TokenGraphAttention attention, uint64_t topology_flags) const {
    ValidateSession(session);
    if (session.token_graph_poisoned()) {
        throw std::logic_error(
            "Qwen4Exp token graph is poisoned; recreate the session");
    }
    if (body != aila::TokenGraphBody::WholeToken) {
        ValidateGdnHeadPath(session, 1);
    }
    if (session.mropePosT_.valid() || session.mropePosH_.valid() ||
        session.mropePosW_.valid() || session.mropePromptLength_ > 0 ||
        session.mropeTextPositionDelta_ != 0) {
        throw std::logic_error(
            "Qwen4Exp token graph key rejects mRoPE position state");
    }
    if (!session.embeddingOverridePositions_.empty() ||
        !session.embeddingOverrideValues_.empty() ||
        session.embeddingOverrideHiddenSize_ != 0) {
        throw std::logic_error(
            "Qwen4Exp token graph key rejects embedding overrides");
    }
    aila::TokenGraphKey key{};
    key.backend = this;
    key.session = &session;
    key.session_instance_id = session.session_instance_id();
    key.session_generation = session.generation();
    key.session_state_generation = session.token_graph_generation();
    key.runtime_generation = runtime_generation_;
    key.runtime_capacity = runtime_seq_capacity_;
    key.residency_generation = adaptive_cache_.ResidencyGeneration();
    key.descriptor_generation = adaptive_cache_.DescriptorGeneration();
    key.ple_workspace_generation = ple_.workspace_generation();
    key.ple_graph_view_generation = session.ple_state_.graph_view_generation;
    key.topology_generation = topology_generation_;
    key.topology_flags = topology_flags;
    key.body = body;
    key.attention = attention;
    return key;
}

void Qwen4ExpBnb4Backend::set_embedding_overrides(
    const std::vector<int>& positions,
    const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
    int hidden_size) {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    (void)AbortTokenGraph();
    session.InvalidateTokenGraphState();
    session.embeddingOverridePositions_ = positions;
    session.embeddingOverrideValues_ = embeddings;
    session.embeddingOverrideHiddenSize_ = hidden_size;
}

void Qwen4ExpBnb4Backend::clear_embedding_overrides() {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    (void)AbortTokenGraph();
    session.InvalidateTokenGraphState();
    session.embeddingOverridePositions_.clear();
    session.embeddingOverrideValues_.clear();
    session.embeddingOverrideHiddenSize_ = 0;
}

void Qwen4ExpBnb4Backend::set_mrope_positions(
    Context& ctx,
    const std::vector<int>& pos_t,
    const std::vector<int>& pos_h,
    const std::vector<int>& pos_w,
    int text_pos_delta) {
    auto& session = DefaultSession(ctx);
    if (pos_t.empty() || pos_h.empty() || pos_w.empty() ||
        pos_t.size() != pos_h.size() || pos_t.size() != pos_w.size()) {
        clear_mrope_positions();
        return;
    }
    (void)AbortTokenGraph();
    session.InvalidateTokenGraphState();
    session.mropePosT_ = Tensor::allocate(ctx, {static_cast<int64_t>(pos_t.size())}, dnnl::memory::data_type::s32);
    session.mropePosH_ = Tensor::allocate(ctx, {static_cast<int64_t>(pos_h.size())}, dnnl::memory::data_type::s32);
    session.mropePosW_ = Tensor::allocate(ctx, {static_cast<int64_t>(pos_w.size())}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(session.mropePosT_.data(), pos_t.data(), pos_t.size() * sizeof(int));
    ctx.memcpy_h2d(session.mropePosH_.data(), pos_h.data(), pos_h.size() * sizeof(int));
    ctx.memcpy_h2d(session.mropePosW_.data(), pos_w.data(), pos_w.size() * sizeof(int));
    session.mropePromptLength_ = static_cast<int>(pos_t.size());
    session.mropeTextPositionDelta_ = text_pos_delta;
}

void Qwen4ExpBnb4Backend::clear_mrope_positions() {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    (void)AbortTokenGraph();
    session.InvalidateTokenGraphState();
    session.mropePosT_ = Tensor();
    session.mropePosH_ = Tensor();
    session.mropePosW_ = Tensor();
    session.mropePromptLength_ = 0;
    session.mropeTextPositionDelta_ = 0;
}
