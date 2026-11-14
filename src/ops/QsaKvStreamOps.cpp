#include "ops/QsaKvStreamOps.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ops {
namespace {

constexpr int kKvCtlInts = 16;
constexpr int kWarpSize = 32;

struct StreamMapDeviceView {
    int32_t* page_table = nullptr;
    int32_t* slot_block = nullptr;
    int32_t* slot_stamp = nullptr;
    int32_t* slot_ref = nullptr;
    int32_t* miss_block = nullptr;
    int32_t* miss_slot = nullptr;
    int32_t* ctl = nullptr;
    int n_blocks = 0;
    int n_slots = 0;
};

StreamMapDeviceView make_device_view(QsaKvStreamMap& map) {
    StreamMapDeviceView v;
    v.page_table = map.page_table.data_as<int32_t>();
    v.slot_block = map.slot_block.data_as<int32_t>();
    v.slot_stamp = map.slot_stamp.data_as<int32_t>();
    v.slot_ref = map.slot_ref.data_as<int32_t>();
    v.miss_block = map.miss_block.data_as<int32_t>();
    v.miss_slot = map.miss_slot.data_as<int32_t>();
    v.ctl = map.ctl.data_as<int32_t>();
    v.n_blocks = map.n_blocks;
    v.n_slots = map.n_slots;
    return v;
}

template <typename T>
using global_atomic = sycl::atomic_ref<
    T, sycl::memory_order::relaxed, sycl::memory_scope::device,
    sycl::access::address_space::global_space>;

template <typename T>
using local_atomic = sycl::atomic_ref<
    T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

template <int NW>
inline int block_scan(
    const sycl::nd_item<1>& it,
    int v,
    const sycl::local_accessor<int, 1>& warp_sums,
    int& total) {
    const sycl::sub_group sg = it.get_sub_group();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int w = static_cast<int>(sg.get_group_linear_id());
    const int x = sycl::inclusive_scan_over_group(sg, v, sycl::plus<int>());
    if (lane == kWarpSize - 1) warp_sums[w] = x;
    it.barrier(sycl::access::fence_space::local_space);
    if (w == 0) {
        const int y = sycl::inclusive_scan_over_group(
            sg, lane < NW ? static_cast<int>(warp_sums[lane]) : 0, sycl::plus<int>());
        if (lane < NW) warp_sums[lane] = y;
    }
    it.barrier(sycl::access::fence_space::local_space);
    total = warp_sums[NW - 1];
    const int excl = x - v + (w > 0 ? static_cast<int>(warp_sums[w - 1]) : 0);
    it.barrier(sycl::access::fence_space::local_space);
    return excl;
}

template <int RT>
void resolve_launch(
    sycl::queue& q,
    const StreamMapDeviceView mm,
    const int32_t* logical_blocks,
    int num_rows,
    int start_pos,
    int block_topk,
    int compress_ratio,
    int32_t* resolved_blocks_out,
    int32_t* resolved_tail_indices_out) {
    const int stride = block_topk + 4;
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> sh(sycl::range<1>(3), cgh);
        sycl::local_accessor<int, 1> warp_sums(sycl::range<1>(32), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(RT, RT),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarpSize)]] {
                const auto grp = it.get_group();
                const int tid = static_cast<int>(it.get_local_id(0));
                local_atomic<int> s_nmiss(sh[0]);
                local_atomic<int> s_lookups(sh[1]);
                const int epoch = mm.ctl[0] + 1;
                if (tid == 0) {
                    sh[0] = 0;
                    sh[1] = 0;
                }
                sycl::group_barrier(grp);

                int lookups = 0;
                for (int r = 0; r < num_rows; ++r) {
                    const int visible_len = start_pos + r + 1;
                    const int num_comp = visible_len / compress_ratio;
                    const int tail_len = visible_len - num_comp * compress_ratio;
                    const int n_act = (num_comp > block_topk) ? block_topk : num_comp;
                    const int32_t* row_logical =
                        logical_blocks ? (logical_blocks + static_cast<size_t>(r) * stride) : nullptr;
                    for (int i = tid; i < n_act; i += RT) {
                        const int b = row_logical ? row_logical[i] : i;
                        ++lookups;
                        const int sl = mm.page_table[b];
                        if (sl >= 0) {
                            mm.slot_stamp[sl] = epoch;
                            mm.slot_ref[sl] = 1;
                        } else if (sl == -1) {
                            int expected = -1;
                            if (global_atomic<int32_t>(mm.page_table[b])
                                    .compare_exchange_strong(expected, -2)) {
                                mm.miss_block[s_nmiss.fetch_add(1)] = b;
                            }
                        }
                    }
                    if (tail_len > 0 && tid == 0) {
                        const int b_tail = num_comp;
                        ++lookups;
                        const int sl = mm.page_table[b_tail];
                        if (sl >= 0) {
                            mm.slot_stamp[sl] = epoch;
                            mm.slot_ref[sl] = 1;
                        } else if (sl == -1) {
                            int expected = -1;
                            if (global_atomic<int32_t>(mm.page_table[b_tail])
                                    .compare_exchange_strong(expected, -2)) {
                                mm.miss_block[s_nmiss.fetch_add(1)] = b_tail;
                            }
                        }
                    }
                }
                s_lookups.fetch_add(lookups);
                sycl::group_barrier(grp);

                const int need = sh[0];
                const int n = mm.n_slots;
                int hand = mm.ctl[1];
                int got = 0;
                for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
                    const int j = static_cast<int>((static_cast<long long>(hand) + tid) % n);
                    const bool mine = (mm.slot_stamp[j] == epoch);
                    const bool cand = !mine && (mm.slot_block[j] < 0 || mm.slot_ref[j] == 0);
                    int total = 0;
                    const int rank = block_scan<RT / kWarpSize>(it, cand ? 1 : 0, warp_sums, total);
                    const int want = need - got;
                    if (tid == 0) sh[2] = RT;
                    it.barrier(sycl::access::fence_space::local_space);
                    if (cand && rank == want - 1) sh[2] = tid + 1;
                    it.barrier(sycl::access::fence_space::local_space);
                    const int cut = sh[2];
                    if (cand && rank < want) {
                        mm.miss_slot[got + rank] = j;
                        mm.slot_stamp[j] = epoch;
                    } else if (tid < cut && !mine) {
                        mm.slot_ref[j] = 0;
                    }
                    got += (total < want) ? total : want;
                    hand = static_cast<int>((static_cast<long long>(hand) + cut) % n);
                    sycl::group_barrier(grp);
                }

                const int placed = (got < need) ? got : need;
                for (int k = tid; k < need; k += RT) {
                    const int b = mm.miss_block[k];
                    if (k >= placed) {
                        mm.page_table[b] = -1;
                        continue;
                    }
                    const int sl = mm.miss_slot[k];
                    const int old = mm.slot_block[sl];
                    if (old >= 0) mm.page_table[old] = -1;
                    mm.slot_block[sl] = b;
                    mm.slot_stamp[sl] = epoch;
                    mm.slot_ref[sl] = 1;
                    mm.page_table[b] = sl;
                }
                sycl::group_barrier(grp);

                for (int r = 0; r < num_rows; ++r) {
                    const int visible_len = start_pos + r + 1;
                    const int num_comp = visible_len / compress_ratio;
                    const int tail_len = visible_len - num_comp * compress_ratio;
                    const int n_act = (num_comp > block_topk) ? block_topk : num_comp;
                    const int32_t* row_logical =
                        logical_blocks ? (logical_blocks + static_cast<size_t>(r) * stride) : nullptr;
                    int32_t* row_out = resolved_blocks_out + static_cast<size_t>(r) * stride;
                    for (int i = tid; i < n_act; i += RT) {
                        const int b = row_logical ? row_logical[i] : i;
                        row_out[i] = mm.page_table[b];
                    }
                    if (tail_len > 0 && tid < tail_len) {
                        const int tail_sl = mm.page_table[num_comp];
                        const int phys_tok = tail_sl * compress_ratio + tid;
                        if (n_act > 0) {
                            row_out[n_act + tid] = phys_tok;
                        }
                        if (resolved_tail_indices_out != nullptr) {
                            resolved_tail_indices_out[r * compress_ratio + tid] = phys_tok;
                        }
                    }
                }

                if (tid == 0) {
                    mm.ctl[0] = epoch;
                    mm.ctl[1] = hand;
                    mm.ctl[2] = placed;
                    if (placed < need) mm.ctl[3] = 1;
                    auto* counters = reinterpret_cast<unsigned long long*>(mm.ctl + 4);
                    counters[0] += static_cast<unsigned long long>(placed);
                    counters[1] += static_cast<unsigned long long>(sh[1]);
                    counters[2] += 1ull;
                }
            });
    });
}

} // namespace

void qsa_kv_stream_init_map(Context& ctx, QsaKvStreamMap& map, int capacity, int n_slots) {
    if (capacity <= 0 || n_slots <= 0) {
        throw std::invalid_argument("qsa_kv_stream_init_map requires positive capacity and slot count");
    }
    const int n_blocks = (capacity + 3) / 4;
    map.n_blocks = n_blocks;
    map.n_slots = n_slots;
    map.page_table = Tensor::allocate(ctx, {n_blocks}, dnnl::memory::data_type::s32);
    map.slot_block = Tensor::allocate(ctx, {n_slots}, dnnl::memory::data_type::s32);
    map.slot_stamp = Tensor::allocate(ctx, {n_slots}, dnnl::memory::data_type::s32);
    map.slot_ref = Tensor::allocate(ctx, {n_slots}, dnnl::memory::data_type::s32);
    map.miss_block = Tensor::allocate(ctx, {n_slots}, dnnl::memory::data_type::s32);
    map.miss_slot = Tensor::allocate(ctx, {n_slots}, dnnl::memory::data_type::s32);
    map.ctl = Tensor::allocate(ctx, {kKvCtlInts}, dnnl::memory::data_type::s32);
    qsa_kv_stream_reset(ctx, map);
}

void qsa_kv_stream_reset(Context& ctx, QsaKvStreamMap& map) {
    if (!map.valid()) return;
    const StreamMapDeviceView mm = make_device_view(map);
    constexpr size_t kTotalThreads = 128 * 256;
    ctx.queue().parallel_for(sycl::nd_range<1>(kTotalThreads, 256), [=](sycl::nd_item<1> it) {
        const long long i0 = static_cast<long long>(it.get_global_id(0));
        const long long st = static_cast<long long>(kTotalThreads);
        for (long long i = i0; i < mm.n_blocks; i += st) {
            mm.page_table[i] = -1;
        }
        for (long long i = i0; i < mm.n_slots; i += st) {
            mm.slot_block[i] = -1;
            mm.slot_stamp[i] = -1;
            mm.slot_ref[i] = 0;
        }
        if (i0 < kKvCtlInts) {
            mm.ctl[i0] = 0;
        }
    });
}

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
    Tensor* resolved_tail_indices_out) {
    if (num_rows <= 0) return;
    if (!map.valid()) {
        throw std::invalid_argument("qsa_kv_stream_resolve: invalid QsaKvStreamMap");
    }
    if (compress_ratio != 4) {
        throw std::invalid_argument("qsa_kv_stream_resolve: requires compress_ratio == 4");
    }
    if (map.n_slots < 256) {
        throw std::invalid_argument("qsa_kv_stream_resolve: n_slots must be >= 256");
    }
    if (!slots_k.valid() || !slots_v.valid() || !host_k.valid() || !host_v.valid() ||
        slots_k.dtype() != slots_v.dtype() || slots_k.dtype() != host_k.dtype() ||
        slots_k.dtype() != host_v.dtype() ||
        slots_k.ndim() != 3 || slots_v.ndim() != 3 ||
        host_k.ndim() != 3 || host_v.ndim() != 3 ||
        slots_k.shape(0) != host_k.shape(0) || slots_k.shape(2) != host_k.shape(2)) {
        throw std::invalid_argument("qsa_kv_stream_resolve: KV pool shape or dtype mismatch");
    }
    const int kv_heads = static_cast<int>(slots_k.shape(0));
    const int slot_tokens = static_cast<int>(slots_k.shape(1));
    const int host_capacity = static_cast<int>(host_k.shape(1));
    const int head_dim = static_cast<int>(slots_k.shape(2));
    if (slot_tokens < map.n_slots * 4 || head_dim % 16 != 0) {
        throw std::invalid_argument("qsa_kv_stream_resolve: invalid slot_tokens or head_dim alignment");
    }
    const bool is_int8 = (slots_k.dtype() == dnnl::memory::data_type::s8);
    if (is_int8 && (!slots_k_scale || !slots_v_scale || !slots_k_scale->valid() ||
                    !slots_v_scale->valid() || !host_k_scale || !host_v_scale ||
                    !host_k_scale->valid() || !host_v_scale->valid())) {
        throw std::invalid_argument("qsa_kv_stream_resolve: int8 KV requires scale tensors");
    }
    const int stride = block_topk + 4;
    if (!resolved_blocks_out.valid() ||
        resolved_blocks_out.numel() < static_cast<int64_t>(num_rows) * stride) {
        throw std::invalid_argument("qsa_kv_stream_resolve: resolved_blocks_out too small");
    }

    auto& queue = ctx.queue();
    const int max_wg = static_cast<int>(
        queue.get_device().get_info<sycl::info::device::max_work_group_size>());
    int rt = 256;
    if (map.n_slots >= 1024 && max_wg >= 1024) rt = 1024;
    else if (map.n_slots >= 512 && max_wg >= 512) rt = 512;

    const StreamMapDeviceView mm = make_device_view(map);
    int32_t* resolved_ptr = resolved_blocks_out.data_as<int32_t>();
    int32_t* resolved_tail_ptr =
        (resolved_tail_indices_out && resolved_tail_indices_out->valid())
            ? resolved_tail_indices_out->data_as<int32_t>()
            : nullptr;

    if (rt == 1024) {
        resolve_launch<1024>(queue, mm, logical_blocks, num_rows, start_pos, block_topk,
                             compress_ratio, resolved_ptr, resolved_tail_ptr);
    } else if (rt == 512) {
        resolve_launch<512>(queue, mm, logical_blocks, num_rows, start_pos, block_topk,
                            compress_ratio, resolved_ptr, resolved_tail_ptr);
    } else {
        resolve_launch<256>(queue, mm, logical_blocks, num_rows, start_pos, block_topk,
                            compress_ratio, resolved_ptr, resolved_tail_ptr);
    }

    const uint8_t* hk_ptr = static_cast<const uint8_t*>(host_k.data());
    const uint8_t* hv_ptr = static_cast<const uint8_t*>(host_v.data());
    uint8_t* sk_ptr = static_cast<uint8_t*>(slots_k.data());
    uint8_t* sv_ptr = static_cast<uint8_t*>(slots_v.data());
    const float* hks_ptr = (is_int8 && host_k_scale) ? host_k_scale->data_as<float>() : nullptr;
    const float* hvs_ptr = (is_int8 && host_v_scale) ? host_v_scale->data_as<float>() : nullptr;
    float* sks_ptr = (is_int8 && slots_k_scale) ? slots_k_scale->data_as<float>() : nullptr;
    float* svs_ptr = (is_int8 && slots_v_scale) ? slots_v_scale->data_as<float>() : nullptr;

    const size_t elem_bytes = is_int8 ? 1u : 2u;
    const size_t token_bytes = static_cast<size_t>(head_dim) * elem_bytes;
    const int token_uint4s = static_cast<int>(token_bytes / 16u);
    const size_t host_head_bytes = static_cast<size_t>(host_capacity) * token_bytes;
    const size_t slot_head_bytes = static_cast<size_t>(slot_tokens) * token_bytes;

    constexpr int kCopyGroups = 96;
    constexpr int kCopyWg = 128;
    queue.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(kCopyGroups) * kCopyWg, kCopyWg),
        [=](sycl::nd_item<1> it) {
            const int need = mm.ctl[2];
            if (need <= 0) return;
            const int lid = static_cast<int>(it.get_local_id(0));
            for (int k = static_cast<int>(it.get_group(0)); k < need; k += kCopyGroups) {
                const int b = mm.miss_block[k];
                const int sl = mm.miss_slot[k];
                const int tok0 = b * 4;
                const int valid_tokens = sycl::min(4, host_capacity - tok0);
                const int copy_uint4s = valid_tokens * token_uint4s;
                const size_t src_tok_off = static_cast<size_t>(tok0) * token_bytes;
                const size_t dst_tok_off = static_cast<size_t>(sl * 4) * token_bytes;
                for (int h = 0; h < kv_heads; ++h) {
                    const auto* src_k = reinterpret_cast<const sycl::uint4*>(
                        hk_ptr + static_cast<size_t>(h) * host_head_bytes + src_tok_off);
                    const auto* src_v = reinterpret_cast<const sycl::uint4*>(
                        hv_ptr + static_cast<size_t>(h) * host_head_bytes + src_tok_off);
                    auto* dst_k = reinterpret_cast<sycl::uint4*>(
                        sk_ptr + static_cast<size_t>(h) * slot_head_bytes + dst_tok_off);
                    auto* dst_v = reinterpret_cast<sycl::uint4*>(
                        sv_ptr + static_cast<size_t>(h) * slot_head_bytes + dst_tok_off);
                    for (int i = lid; i < copy_uint4s; i += kCopyWg) {
                        dst_k[i] = src_k[i];
                        dst_v[i] = src_v[i];
                    }
                    if (is_int8 && lid < valid_tokens) {
                        const size_t h_off = static_cast<size_t>(h) * host_capacity + tok0;
                        const size_t s_off = static_cast<size_t>(h) * slot_tokens + (sl * 4);
                        sks_ptr[s_off + lid] = hks_ptr[h_off + lid];
                        svs_ptr[s_off + lid] = hvs_ptr[h_off + lid];
                    }
                }
            }
        });
}

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
    int stage_len) {
    if (stage_len <= 0) return;
    if (!stage_k.valid() || !stage_v.valid() || !host_k.valid() || !host_v.valid() ||
        stage_k.dtype() != host_k.dtype() || stage_v.dtype() != host_v.dtype() ||
        stage_k.shape(0) != host_k.shape(0) || stage_k.shape(2) != host_k.shape(2) ||
        stage_k.shape(1) < stage_len || host_k.shape(1) < stage_len) {
        throw std::invalid_argument("qsa_kv_stream_stage_from_host: invalid tensor shapes");
    }
    const int kv_heads = static_cast<int>(stage_k.shape(0));
    const int stage_stride = static_cast<int>(stage_k.shape(1));
    const int host_capacity = static_cast<int>(host_k.shape(1));
    const int head_dim = static_cast<int>(stage_k.shape(2));
    const bool is_int8 = (stage_k.dtype() == dnnl::memory::data_type::s8);
    if (is_int8 && (!stage_k_scale || !stage_v_scale || !stage_k_scale->valid() ||
                    !stage_v_scale->valid() || !host_k_scale || !host_v_scale ||
                    !host_k_scale->valid() || !host_v_scale->valid())) {
        throw std::invalid_argument("qsa_kv_stream_stage_from_host: int8 KV requires scale tensors");
    }

    const size_t elem_bytes = is_int8 ? 1u : 2u;
    const size_t token_bytes = static_cast<size_t>(head_dim) * elem_bytes;
    const size_t total_uint4s = static_cast<size_t>(stage_len) * (token_bytes / 16u);
    const size_t host_head_bytes = static_cast<size_t>(host_capacity) * token_bytes;
    const size_t stage_head_bytes = static_cast<size_t>(stage_stride) * token_bytes;

    const uint8_t* hk_ptr = static_cast<const uint8_t*>(host_k.data());
    const uint8_t* hv_ptr = static_cast<const uint8_t*>(host_v.data());
    uint8_t* dk_ptr = static_cast<uint8_t*>(stage_k.data());
    uint8_t* dv_ptr = static_cast<uint8_t*>(stage_v.data());
    const float* hks_ptr = (is_int8 && host_k_scale) ? host_k_scale->data_as<float>() : nullptr;
    const float* hvs_ptr = (is_int8 && host_v_scale) ? host_v_scale->data_as<float>() : nullptr;
    float* dks_ptr = (is_int8 && stage_k_scale) ? stage_k_scale->data_as<float>() : nullptr;
    float* dvs_ptr = (is_int8 && stage_v_scale) ? stage_v_scale->data_as<float>() : nullptr;

    constexpr int kWg = 256;
    const int n_blocks = std::min(512, std::max(1, static_cast<int>((total_uint4s + kWg - 1) / kWg)));
    ctx.queue().parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(n_blocks) * kWg, kWg),
        [=](sycl::nd_item<1> it) {
            const size_t gid = it.get_global_id(0);
            const size_t stride = it.get_global_range(0);
            for (int h = 0; h < kv_heads; ++h) {
                const auto* src_k = reinterpret_cast<const sycl::uint4*>(
                    hk_ptr + static_cast<size_t>(h) * host_head_bytes);
                const auto* src_v = reinterpret_cast<const sycl::uint4*>(
                    hv_ptr + static_cast<size_t>(h) * host_head_bytes);
                auto* dst_k = reinterpret_cast<sycl::uint4*>(
                    dk_ptr + static_cast<size_t>(h) * stage_head_bytes);
                auto* dst_v = reinterpret_cast<sycl::uint4*>(
                    dv_ptr + static_cast<size_t>(h) * stage_head_bytes);
                for (size_t i = gid; i < total_uint4s; i += stride) {
                    dst_k[i] = src_k[i];
                    dst_v[i] = src_v[i];
                }
                if (is_int8) {
                    const size_t h_off = static_cast<size_t>(h) * host_capacity;
                    const size_t s_off = static_cast<size_t>(h) * stage_stride;
                    for (size_t t = gid; t < static_cast<size_t>(stage_len); t += stride) {
                        dks_ptr[s_off + t] = hks_ptr[h_off + t];
                        dvs_ptr[s_off + t] = hvs_ptr[h_off + t];
                    }
                }
            }
        });
}

QsaKvStreamCounters qsa_kv_stream_counters(Context& ctx, const QsaKvStreamMap& map) {
    QsaKvStreamCounters out;
    if (!map.valid()) return out;
    int32_t c[kKvCtlInts] = {};
    ctx.memcpy_d2h(c, map.ctl.data(), sizeof(c));
    unsigned long long u[3] = {};
    std::memcpy(u, c + 4, sizeof(u));
    out.misses = u[0];
    out.lookups = u[1];
    out.calls = u[2];
    out.overflow = (c[3] != 0);
    return out;
}

} // namespace ops
