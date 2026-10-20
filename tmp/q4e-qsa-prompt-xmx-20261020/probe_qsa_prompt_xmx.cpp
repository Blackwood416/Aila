// Standalone A/B probe for Qwen4Exp sparse-QSA prompt attention.
//
// A: production attention_prefill_qsa_batched with
//    AILA_QSA_HEADS_PER_WG=4 (three 4-head work-groups per KV head).
// B: an independent 12+4-row Intel XMX kernel. One work-group owns one
//    query row and one KV head, keeps the 12 real query heads in two 8-row
//    matrix tiles, and streams each active K/V token once.
//
// The selector and candidate sets are prepared outside timing with the
// production host selector. Timing is a single warm-up per arm followed by
// one device-event A/B round at each of {2052, 4096}. The probe also computes
// a full FP64 causal reference and applies the two numerical bounds from
// qwen38-r1-qsa-prompt-xmx-bound-2026-10-20.md.
//
// This file is intentionally standalone. It does not integrate a production
// dispatcher, dynamic worklist, graph, or environment sweep. Gate fusion is
// omitted so both arms exercise the same core attention operation.

#include "core/Context.hpp"
#include "ops/Ops.hpp"
#include "ops/QsaIndexerOps.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;
namespace intel_matrix = sycl::ext::intel::experimental::matrix;

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadsPerKv = kHeads / kKvHeads;
constexpr int kHeadDim = 256;
constexpr int kBlockTopK = 512;
constexpr int kCompressRatio = 4;
constexpr int kActiveStride = kBlockTopK + 4;
constexpr int kQueryRows = 64;
constexpr int kCacheCapacity = 4096;
constexpr int kXmxRows = 16;
constexpr int kRealRows = kHeadsPerKv;
constexpr int kXmxK = 16;
constexpr int kXmxN = 8;
// DG2/A770 exposes BF16 XMX8: M<=8, N=8, K=16 with an 8-lane subgroup.
// Keep these tied together so IGC never sees an XMX8 tile in an SG16 shape.
constexpr int kSubgroup = 8;
constexpr int kSubgroups = 4;
constexpr int kWorkgroup = kSubgroup * kSubgroups;
constexpr int kOutputWidth = kHeads * kHeadDim;
constexpr float kScale = 1.0f / 16.0f;

struct alignas(64) QsaXmxShared {
    // K is transposed into [D][active token] for a row-major B tile.
    bf16 k_chunk[kHeadDim][kXmxK];
    // V stays [active token][D].
    bf16 v_chunk[kXmxK][kHeadDim];
    // Twelve query heads plus four zero rows.
    bf16 q_pad[kXmxRows][kHeadDim];
    float o[kRealRows][kHeadDim];
    float score[kXmxRows][kXmxK];
    bf16 p_hi[kXmxRows][kXmxK];
    bf16 p_lo[kXmxRows][kXmxK];
    float running_max[kXmxRows];
    float running_sum[kXmxRows];
};

static_assert(sizeof(QsaXmxShared) <= 48 * 1024,
              "12+4 QSA XMX shared state should fit comfortably below 64 KiB");
static_assert(kHeadDim % kXmxK == 0);
static_assert(kHeadDim % kXmxN == 0);
static_assert(kHeadsPerKv == 12);
static_assert(kXmxRows > 0 && kXmxK > 0 && kXmxN > 0);
static_assert(kRealRows > 0 && kRealRows <= kXmxRows);
static_assert(kSubgroup == 8, "A770 BF16 DPAS requires an 8-lane subgroup");
static_assert(kSubgroups > 0);
static_assert(kWorkgroup > 0 && kWorkgroup % kSubgroup == 0);
static_assert(kWorkgroup / kSubgroup == kSubgroups);
static_assert(kXmxRows == 2 * 8);

template <typename T>
auto local_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space,
                                    sycl::access::decorated::no>(p);
}

struct TimingBegin {
    void operator()() const {}
};

struct TimingEnd {
    void operator()() const {}
};

struct HostCase {
    int length = 0;
    int start_pos = 0;
    std::vector<bf16> q;
    std::vector<bf16> k;
    std::vector<bf16> v;
    std::vector<int> active_blocks;
    std::vector<int> valid_counts;
    std::vector<std::vector<int>> active_tokens;
    std::size_t total_active_tokens = 0;
    int sparse_rows = 0;
};

struct TimedArmA {
    sycl::event begin;
    sycl::event end;
};

struct NumericReport {
    bool finite = true;
    int exact = 0;
    int within_one_ulp = 0;
    int elements = 0;
    double max_abs_a = 0.0;
    double max_abs_b_minus_a = 0.0;
    double max_abs_a_minus_ref = 0.0;
    double max_abs_b_minus_ref = 0.0;
    double max_abs_ref = 0.0;
    double nrmse = 0.0;
    double cosine = 0.0;
    bool bound_ab = false;
    bool bound_b_ref = false;
};

std::uint32_t next_random(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_unit(std::uint32_t& state) {
    const float unit =
        static_cast<float>(next_random(state) >> 8) * (1.0f / 16777216.0f);
    return unit * 2.0f - 1.0f;
}

int active_token_for(const int* row_blocks, int num_blocks, int tail_len,
                     int visible_len, int index) {
    const int block_tokens = num_blocks * kCompressRatio;
    if (index < block_tokens) {
        return row_blocks[index / kCompressRatio] * kCompressRatio +
               index % kCompressRatio;
    }
    return visible_len - tail_len + (index - block_tokens);
}

HostCase make_host_case(int length) {
    if (length < 2052 || length > 4096) {
        throw std::invalid_argument("case length must be in [2052, 4096]");
    }

    HostCase c;
    c.length = length;
    c.start_pos = length - kQueryRows;
    c.q.resize(static_cast<std::size_t>(kQueryRows) * kOutputWidth);
    c.k.resize(static_cast<std::size_t>(kKvHeads) * kCacheCapacity * kHeadDim);
    c.v.resize(static_cast<std::size_t>(kKvHeads) * kCacheCapacity * kHeadDim);
    c.active_blocks.assign(
        static_cast<std::size_t>(kQueryRows) * kActiveStride, -1);
    c.valid_counts.assign(static_cast<std::size_t>(kQueryRows) * 2, 0);
    c.active_tokens.resize(kQueryRows);

    std::uint32_t state = 0x6d2b79f5u ^ static_cast<std::uint32_t>(length);
    for (auto& value : c.q) {
        value = bf16(random_unit(state) * 0.5f);
    }
    for (auto& value : c.k) {
        value = bf16(random_unit(state) * 0.5f);
    }
    for (auto& value : c.v) {
        value = bf16(random_unit(state) * 0.5f);
    }

    for (int row = 0; row < kQueryRows; ++row) {
        const int visible_len = c.start_pos + row + 1;
        const int complete_blocks = visible_len / kCompressRatio;
        const int tail_len = visible_len - complete_blocks * kCompressRatio;
        std::vector<float> scores(static_cast<std::size_t>(complete_blocks));
        std::uint32_t score_state =
            0x9e3779b9u ^
            (static_cast<std::uint32_t>(length) * 0x85ebca6bu) ^
            (static_cast<std::uint32_t>(row + 1) * 0xc2b2ae35u);
        for (float& score : scores) {
            score = static_cast<float>(next_random(score_state) >> 8) *
                    (1.0f / 16777216.0f);
        }

        int* row_blocks =
            c.active_blocks.data() + static_cast<std::size_t>(row) * kActiveStride;
        const int selected_blocks = ops::qsa_select_topk_blocks_host(
            scores.data(), complete_blocks, kBlockTopK, visible_len,
            kCompressRatio, row_blocks);
        if (selected_blocks <= 0 || selected_blocks > kBlockTopK) {
            throw std::runtime_error("production QSA selector returned an invalid count");
        }
        c.valid_counts[static_cast<std::size_t>(row) * 2] = selected_blocks;
        c.valid_counts[static_cast<std::size_t>(row) * 2 + 1] = tail_len;
        if (selected_blocks == kBlockTopK && complete_blocks > kBlockTopK) {
            ++c.sparse_rows;
        }

        auto& tokens = c.active_tokens[static_cast<std::size_t>(row)];
        tokens.reserve(static_cast<std::size_t>(selected_blocks) *
                           kCompressRatio +
                       tail_len);
        for (int b = 0; b < selected_blocks; ++b) {
            for (int part = 0; part < kCompressRatio; ++part) {
                tokens.push_back(row_blocks[b] * kCompressRatio + part);
            }
        }
        const int tail_start = visible_len - tail_len;
        for (int t = 0; t < tail_len; ++t) {
            tokens.push_back(tail_start + t);
        }
        c.total_active_tokens += tokens.size();
    }
    return c;
}

sycl::event run_xmx_12plus4(
    sycl::queue& q,
    const bf16* q_data,
    const bf16* k_cache,
    const bf16* v_cache,
    const int* active_blocks,
    const int* valid_counts,
    bf16* output,
    int count,
    int start_pos,
    int max_seq_len) {
    if (count <= 0) {
        throw std::invalid_argument("XMX attention requires a positive row count");
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument("XMX attention requires a positive cache capacity");
    }
    const std::size_t groups =
        static_cast<std::size_t>(count) * kKvHeads;
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<QsaXmxShared, 1> shared(1, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(groups * kWorkgroup, kWorkgroup),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
                using xmx_a =
                    joint_matrix<sycl::sub_group, bf16, use::a, 8, kXmxK,
                                 layout::row_major>;
                using xmx_b =
                    joint_matrix<sycl::sub_group, bf16, use::b, kXmxK, kXmxN,
                                 layout::row_major>;
                using xmx_acc =
                    joint_matrix<sycl::sub_group, float, use::accumulator, 8, 8>;

                auto& s = shared[0];
                auto sg = item.get_sub_group();
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id =
                    static_cast<int>(sg.get_group_linear_id());
                const int group = static_cast<int>(item.get_group(0));
                const int row = group / kKvHeads;
                const int kv_head = group % kKvHeads;
                const int num_blocks = sycl::max(
                    0, valid_counts[static_cast<std::size_t>(row) * 2]);
                const int tail_len = sycl::max(
                    0, valid_counts[static_cast<std::size_t>(row) * 2 + 1]);
                const int visible_len = start_pos + row + 1;
                const int active_len = num_blocks * kCompressRatio + tail_len;
                const int* row_blocks =
                    active_blocks + static_cast<std::size_t>(row) * kActiveStride;

                for (int i = lid; i < kXmxRows * kHeadDim; i += kWorkgroup) {
                    const int head = i / kHeadDim;
                    const int d = i % kHeadDim;
                    if (head < kRealRows) {
                        const std::size_t src =
                            (static_cast<std::size_t>(row) * kHeads +
                             kv_head * kHeadsPerKv + head) *
                                kHeadDim +
                            d;
                        s.q_pad[head][d] = q_data[src];
                    } else {
                        s.q_pad[head][d] = bf16(0.0f);
                    }
                }
                for (int i = lid; i < kRealRows * kHeadDim; i += kWorkgroup) {
                    s.o[i / kHeadDim][i % kHeadDim] = 0.0f;
                }
                if (lid < kXmxRows) {
                    s.running_max[lid] = -1.0e30f;
                    s.running_sum[lid] = 0.0f;
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int seq_base = 0; seq_base < active_len;
                     seq_base += kXmxK) {
                    const int valid_tokens =
                        sycl::min(kXmxK, active_len - seq_base);

                    // Gather one 16-token K/V chunk. K is transposed for QK,
                    // while V remains row-major for PV.
                    for (int i = lid; i < kXmxK * kHeadDim; i += kWorkgroup) {
                        const int t = i / kHeadDim;
                        const int d = i % kHeadDim;
                        bf16 k_value = bf16(0.0f);
                        bf16 v_value = bf16(0.0f);
                        if (t < valid_tokens) {
                            const int token = active_token_for(
                                row_blocks, num_blocks, tail_len, visible_len,
                                seq_base + t);
                            const std::size_t offset =
                                (static_cast<std::size_t>(kv_head) * max_seq_len +
                                 token) *
                                    kHeadDim +
                                d;
                            k_value = k_cache[offset];
                            v_value = v_cache[offset];
                        }
                        s.k_chunk[d][t] = k_value;
                        s.v_chunk[t][d] = v_value;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Four subgroups cover M={0..7,8..15} x N={0..7,8..15}.
                    const int m_half = sg_id / 2;
                    const int n_half = sg_id % 2;
                    xmx_acc qk;
                    joint_matrix_fill(sg, qk, 0.0f);
                    for (int d0 = 0; d0 < kHeadDim; d0 += kXmxK) {
                        xmx_a a;
                        xmx_b b;
                        joint_matrix_load(
                            sg, a,
                            local_ptr(&s.q_pad[m_half * 8][d0]), kHeadDim);
                        joint_matrix_load(
                            sg, b,
                            local_ptr(&s.k_chunk[d0][n_half * 8]), kXmxK);
                        joint_matrix_mad(sg, qk, a, b, qk);
                    }
                    intel_matrix::joint_matrix_apply(
                        sg, qk,
                        [&](float& value, std::size_t r, std::size_t col) {
                            s.score[m_half * 8 + static_cast<int>(r)]
                                   [n_half * 8 + static_cast<int>(col)] = value;
                        });
                    item.barrier(sycl::access::fence_space::local_space);

                    // Each of 16 lanes owns one matrix row. Rows 12..15 are
                    // zero padding; only rows 0..11 update softmax and O.
                    if (lid < kXmxRows) {
                        const int head = lid;
                        if (head < kRealRows) {
                            float chunk_max = -1.0e30f;
                            for (int t = 0; t < valid_tokens; ++t) {
                                chunk_max = sycl::fmax(
                                    chunk_max, s.score[head][t] * kScale);
                            }
                            const float old_max = s.running_max[head];
                            const float new_max = sycl::fmax(old_max, chunk_max);
                            const float rescale =
                                old_max <= -1.0e20f
                                    ? 0.0f
                                    : sycl::native::exp(old_max - new_max);
                            float new_sum = s.running_sum[head] * rescale;
                            for (int t = 0; t < kXmxK; ++t) {
                                float p = 0.0f;
                                if (t < valid_tokens) {
                                    const float score =
                                        s.score[head][t] * kScale;
                                    p = sycl::native::exp(
                                        sycl::fmax(-88.0f, score - new_max));
                                    new_sum += p;
                                }
                                const float hi = static_cast<float>(bf16(p));
                                s.p_hi[head][t] = bf16(hi);
                                s.p_lo[head][t] = bf16(p - hi);
                            }
                            for (int d = 0; d < kHeadDim; ++d) {
                                s.o[head][d] *= rescale;
                            }
                            s.running_max[head] = new_max;
                            s.running_sum[head] = new_sum;
                        } else {
                            for (int t = 0; t < kXmxK; ++t) {
                                s.p_hi[head][t] = bf16(0.0f);
                                s.p_lo[head][t] = bf16(0.0f);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Each subgroup owns eight disjoint output-column tiles.
                    // P is represented as BF16 hi+lo, so each PV tile needs
                    // two matrix MADs but still consumes the gathered V once.
                    for (int tile = sg_id; tile < kHeadDim / kXmxN;
                         tile += kSubgroups) {
                        const int d0 = tile * kXmxN;
                        xmx_a phi0, plo0, phi1, plo1;
                        xmx_b vv;
                        xmx_acc acc0, acc1;
                        joint_matrix_fill(sg, acc0, 0.0f);
                        joint_matrix_fill(sg, acc1, 0.0f);
                        joint_matrix_load(
                            sg, phi0, local_ptr(&s.p_hi[0][0]), kXmxK);
                        joint_matrix_load(
                            sg, plo0, local_ptr(&s.p_lo[0][0]), kXmxK);
                        joint_matrix_load(
                            sg, phi1, local_ptr(&s.p_hi[8][0]), kXmxK);
                        joint_matrix_load(
                            sg, plo1, local_ptr(&s.p_lo[8][0]), kXmxK);
                        joint_matrix_load(
                            sg, vv, local_ptr(&s.v_chunk[0][d0]), kHeadDim);
                        joint_matrix_mad(sg, acc0, phi0, vv, acc0);
                        joint_matrix_mad(sg, acc0, plo0, vv, acc0);
                        joint_matrix_mad(sg, acc1, phi1, vv, acc1);
                        joint_matrix_mad(sg, acc1, plo1, vv, acc1);
                        intel_matrix::joint_matrix_apply(
                            sg, acc0,
                            [&](float& value, std::size_t r, std::size_t col) {
                                s.o[static_cast<int>(r)][d0 +
                                                         static_cast<int>(col)] +=
                                    value;
                            });
                        intel_matrix::joint_matrix_apply(
                            sg, acc1,
                            [&](float& value, std::size_t r, std::size_t col) {
                                const int head = 8 + static_cast<int>(r);
                                if (head < kRealRows) {
                                    s.o[head][d0 + static_cast<int>(col)] +=
                                        value;
                                }
                            });
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid < kRealRows) {
                    const int head = lid;
                    const float inv_sum =
                        s.running_sum[head] > 0.0f
                            ? 1.0f / s.running_sum[head]
                            : 0.0f;
                    const std::size_t dst_base =
                        (static_cast<std::size_t>(row) * kHeads +
                         kv_head * kHeadsPerKv + head) *
                        kHeadDim;
                    for (int d = 0; d < kHeadDim; ++d) {
                        output[dst_base + d] = bf16(s.o[head][d] * inv_sum);
                    }
                }
            });
    });
}

TimedArmA run_arm_a(
    Context& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& active_blocks,
    const Tensor& valid_counts,
    Tensor& output,
    int start_pos,
    int length) {
    TimedArmA timed;
    timed.begin = ctx.queue().single_task(TimingBegin{});
    ops::attention_prefill_qsa_batched(
        ctx, q, k, v, active_blocks, output, kQueryRows, start_pos, kHeads,
        kKvHeads, kHeadDim, kBlockTopK, kCompressRatio, nullptr, &valid_counts);
    timed.end = ctx.queue().single_task(TimingEnd{});
    (void)length;
    return timed;
}

double event_span_ms(const TimedArmA& arm) {
    const auto begin_ns =
        arm.begin
            .get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto end_ns =
        arm.end
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    return static_cast<double>(end_ns - begin_ns) * 1.0e-6;
}

double event_span_ms(const sycl::event& event) {
    const auto begin_ns =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end_ns =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(end_ns - begin_ns) * 1.0e-6;
}

std::uint16_t bf16_bits(bf16 value) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<double> run_fp64_reference(
    const HostCase& c,
    int capacity,
    int thread_count) {
    std::vector<double> reference(
        static_cast<std::size_t>(kQueryRows) * kOutputWidth, 0.0);
    const int tasks = kQueryRows * kHeads;
    std::atomic<int> next_task{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));

    auto worker = [&]() {
        std::vector<double> scores;
        scores.reserve(kBlockTopK * kCompressRatio + kCompressRatio);
        for (;;) {
            const int task = next_task.fetch_add(1, std::memory_order_relaxed);
            if (task >= tasks) {
                return;
            }
            const int row = task / kHeads;
            const int head = task % kHeads;
            const int kv_head = head / kHeadsPerKv;
            const auto& tokens = c.active_tokens[static_cast<std::size_t>(row)];
            scores.resize(tokens.size());

            const std::size_t q_base =
                (static_cast<std::size_t>(row) * kHeads + head) * kHeadDim;
            const std::size_t kv_base =
                static_cast<std::size_t>(kv_head) * capacity * kHeadDim;

            double max_score = -1.0e300;
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                const std::size_t k_base =
                    kv_base + static_cast<std::size_t>(tokens[i]) * kHeadDim;
                double dot = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(c.q[q_base + d]) *
                           static_cast<double>(c.k[k_base + d]);
                }
                const double score = dot * static_cast<double>(kScale);
                scores[i] = score;
                max_score = std::max(max_score, score);
            }

            double sum = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                sum += score;
            }
            const double inv_sum = sum > 0.0 ? 1.0 / sum : 0.0;
            const std::size_t out_base =
                (static_cast<std::size_t>(row) * kHeads + head) * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (std::size_t i = 0; i < tokens.size(); ++i) {
                    const std::size_t v_index =
                        kv_base +
                        static_cast<std::size_t>(tokens[i]) * kHeadDim + d;
                    acc += scores[i] * static_cast<double>(c.v[v_index]);
                }
                reference[out_base + d] = acc * inv_sum;
            }
        }
    };

    for (int i = 0; i < thread_count; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& thread : workers) {
        thread.join();
    }
    return reference;
}

NumericReport compare_outputs(
    const std::vector<bf16>& a,
    const std::vector<bf16>& b,
    const std::vector<double>& reference) {
    NumericReport report;
    report.elements = static_cast<int>(a.size());
    double sum_square_error = 0.0;
    double sum_square_reference = 0.0;
    double dot_ab = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const double av = static_cast<double>(a[i]);
        const double bv = static_cast<double>(b[i]);
        const double rv = reference[i];
        if (!std::isfinite(av) || !std::isfinite(bv) ||
            !std::isfinite(rv)) {
            report.finite = false;
        }
        const std::uint16_t abits = bf16_bits(a[i]);
        const std::uint16_t bbits = bf16_bits(b[i]);
        if (abits == bbits) {
            ++report.exact;
        }
        if (std::abs(static_cast<int>(abits) - static_cast<int>(bbits)) <= 1) {
            ++report.within_one_ulp;
        }
        report.max_abs_a = std::max(report.max_abs_a, std::abs(av));
        report.max_abs_a_minus_ref =
            std::max(report.max_abs_a_minus_ref, std::abs(av - rv));
        report.max_abs_b_minus_ref =
            std::max(report.max_abs_b_minus_ref, std::abs(bv - rv));
        report.max_abs_b_minus_a =
            std::max(report.max_abs_b_minus_a, std::abs(bv - av));
        report.max_abs_ref = std::max(report.max_abs_ref, std::abs(rv));
        const double error = bv - rv;
        sum_square_error += error * error;
        sum_square_reference += rv * rv;
        dot_ab += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    report.nrmse = sum_square_reference > 0.0
                       ? std::sqrt(sum_square_error / sum_square_reference)
                       : (sum_square_error == 0.0
                              ? 0.0
                              : 1.0e300);
    report.cosine =
        norm_a > 0.0 && norm_b > 0.0 ? dot_ab / std::sqrt(norm_a * norm_b)
                                     : 0.0;
    report.bound_ab =
        report.finite &&
        report.max_abs_b_minus_a <= 1.0e-4 * report.max_abs_a;
    const double ref_bound =
        std::max(4.0 * report.max_abs_a_minus_ref,
                 1.0e-6 * report.max_abs_ref);
    report.bound_b_ref = report.finite && report.max_abs_b_minus_ref <= ref_bound;
    return report;
}

void print_case_result(
    int length,
    const HostCase& host,
    double a_ms,
    double b_ms,
    const NumericReport& numeric) {
    const double mean_active =
        static_cast<double>(host.total_active_tokens) / kQueryRows;
    const double one_pass_bytes =
        mean_active * kKvHeads * (2.0 * kHeadDim * sizeof(bf16)) *
        kQueryRows;
    const double b_gibs =
        b_ms > 0.0 ? one_pass_bytes / (b_ms * 1.0e6) : 0.0;
    const bool speed_gate = b_ms <= 0.60 * a_ms;
    std::cout << std::fixed << std::setprecision(6)
              << "case=" << length
              << " rows=" << kQueryRows
              << " start_pos=" << host.start_pos
              << " sparse_rows=" << host.sparse_rows
              << " mean_active_tokens=" << mean_active
              << " a_ms=" << a_ms
              << " b_ms=" << b_ms
              << " speedup=" << (b_ms > 0.0 ? a_ms / b_ms : 0.0)
              << " one_pass_GiB_s=" << b_gibs
              << " speed_gate_40pct=" << (speed_gate ? 1 : 0) << '\n'
              << "  numeric"
              << " finite=" << (numeric.finite ? 1 : 0)
              << " A_vs_B_max_abs=" << numeric.max_abs_b_minus_a
              << " bound1=" << (numeric.bound_ab ? 1 : 0)
              << " B_vs_FP64_max_abs=" << numeric.max_abs_b_minus_ref
              << " A_vs_FP64_max_abs=" << numeric.max_abs_a_minus_ref
              << " bound2=" << (numeric.bound_b_ref ? 1 : 0)
              << " exact="
              << (numeric.elements
                      ? 100.0 * numeric.exact / numeric.elements
                      : 0.0)
              << "% one_ulp="
              << (numeric.elements
                      ? 100.0 * numeric.within_one_ulp / numeric.elements
                      : 0.0)
              << "% nrmse=" << numeric.nrmse
              << " cosine=" << numeric.cosine << '\n';
}

}  // namespace

int main() {
    try {
        if (_putenv_s("AILA_QSA_HEADS_PER_WG", "4") != 0) {
            throw std::runtime_error("failed to force AILA_QSA_HEADS_PER_WG=4");
        }

        Context ctx(nullptr, true);
        auto& queue = ctx.queue();
        if (!queue.has_property<sycl::property::queue::enable_profiling>()) {
            throw std::runtime_error("probe requires a profiling-enabled queue");
        }
        if (!queue.get_device().has(sycl::aspect::ext_intel_matrix)) {
            throw std::runtime_error("device does not expose Intel matrix XMX");
        }
        std::cout << "device="
                  << queue.get_device().get_info<sycl::info::device::name>()
                  << " profiling=1 A=production_4head_wg B=12plus4_xmx\n";

        int fp64_threads =
            std::max(1u, std::thread::hardware_concurrency());
        fp64_threads = std::min(fp64_threads, 16);
        bool all_numeric_pass = true;
        bool all_speed_gate_pass = true;

        for (int length : {2052, 4096}) {
            HostCase host = make_host_case(length);
            const int capacity = kCacheCapacity;
            Tensor q_dev = Tensor::allocate(
                ctx, {kQueryRows, kOutputWidth}, dnnl::memory::data_type::bf16);
            Tensor k_dev = Tensor::allocate(
                ctx, {kKvHeads, capacity, kHeadDim},
                dnnl::memory::data_type::bf16);
            Tensor v_dev = Tensor::allocate(
                ctx, {kKvHeads, capacity, kHeadDim},
                dnnl::memory::data_type::bf16);
            Tensor blocks_dev = Tensor::allocate(
                ctx, {kQueryRows, kActiveStride},
                dnnl::memory::data_type::s32);
            Tensor counts_dev = Tensor::allocate(
                ctx, {kQueryRows, 2}, dnnl::memory::data_type::s32);
            Tensor out_a_dev = Tensor::allocate(
                ctx, {kQueryRows, kOutputWidth}, dnnl::memory::data_type::bf16);
            Tensor out_b_dev = Tensor::allocate(
                ctx, {kQueryRows, kOutputWidth}, dnnl::memory::data_type::bf16);

            ctx.memcpy_h2d(q_dev.data(), host.q.data(),
                           host.q.size() * sizeof(bf16));
            ctx.memcpy_h2d(k_dev.data(), host.k.data(),
                           host.k.size() * sizeof(bf16));
            ctx.memcpy_h2d(v_dev.data(), host.v.data(),
                           host.v.size() * sizeof(bf16));
            ctx.memcpy_h2d(blocks_dev.data(), host.active_blocks.data(),
                           host.active_blocks.size() * sizeof(int));
            ctx.memcpy_h2d(counts_dev.data(), host.valid_counts.data(),
                           host.valid_counts.size() * sizeof(int));

            // One warm-up per arm, no timing.
            ops::attention_prefill_qsa_batched(
                ctx, q_dev, k_dev, v_dev, blocks_dev, out_a_dev, kQueryRows,
                host.start_pos, kHeads, kKvHeads, kHeadDim, kBlockTopK,
                kCompressRatio, nullptr, &counts_dev);
            ctx.synchronize();
            sycl::event b_warm = run_xmx_12plus4(
                queue, q_dev.data_as<bf16>(), k_dev.data_as<bf16>(),
                v_dev.data_as<bf16>(), blocks_dev.data_as<int>(),
                counts_dev.data_as<int>(), out_b_dev.data_as<bf16>(), kQueryRows,
                host.start_pos, capacity);
            b_warm.wait_and_throw();

            // One A/B round. No A/B/B/A noise protocol in this routine probe.
            TimedArmA a_timed = run_arm_a(
                ctx, q_dev, k_dev, v_dev, blocks_dev, counts_dev, out_a_dev,
                host.start_pos, length);
            ctx.synchronize();
            const double a_ms = event_span_ms(a_timed);
            sycl::event b_timed = run_xmx_12plus4(
                queue, q_dev.data_as<bf16>(), k_dev.data_as<bf16>(),
                v_dev.data_as<bf16>(), blocks_dev.data_as<int>(),
                counts_dev.data_as<int>(), out_b_dev.data_as<bf16>(), kQueryRows,
                host.start_pos, capacity);
            b_timed.wait_and_throw();
            const double b_ms = event_span_ms(b_timed);

            std::vector<bf16> out_a(host.q.size());
            std::vector<bf16> out_b(host.q.size());
            ctx.memcpy_d2h(out_a.data(), out_a_dev.data(),
                           out_a.size() * sizeof(bf16));
            ctx.memcpy_d2h(out_b.data(), out_b_dev.data(),
                           out_b.size() * sizeof(bf16));
            std::vector<double> reference =
                run_fp64_reference(host, capacity, fp64_threads);
            const NumericReport numeric =
                compare_outputs(out_a, out_b, reference);
            print_case_result(length, host, a_ms, b_ms, numeric);

            all_numeric_pass =
                all_numeric_pass && numeric.bound_ab && numeric.bound_b_ref;
            all_speed_gate_pass =
                all_speed_gate_pass && b_ms <= 0.60 * a_ms;
        }

        std::cout << "summary numeric_gate=" << (all_numeric_pass ? "pass" : "fail")
                  << " speed_gate_40pct_both="
                  << (all_speed_gate_pass ? "pass" : "fail")
                  << " probe_go="
                  << (all_numeric_pass && all_speed_gate_pass ? "yes" : "no")
                  << '\n';
        return all_numeric_pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "probe failed: " << error.what() << '\n';
        return 2;
    }
}
