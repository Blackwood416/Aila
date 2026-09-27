#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include <vector>

namespace ops {

using bf16 = sycl::ext::oneapi::bfloat16;

// Prepares Q (RMSNorm with 1+w + RoPE) and writes raw K into cache for QSA indexer (decode: count == 1).
// indexer_qk: [1, 640] (first 512 for q [4, 128], last 128 for token_k [1, 128])
// q_norm_w:   [128] (preprocessed with 1.0 + w)
// k_cache:    [capacity, 128] (raw keys cached before norm and RoPE)
// q_out:      [4, 128] (normed and RoPE-rotated query heads)
// pos:        current token position
void qsa_prepare_q_and_cache_k(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    Tensor& k_cache,
    Tensor& q_out,
    int pos,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

// Batched write of raw K into k_cache (prefill: count > 1).
// indexer_qk: [count, 640]
// k_cache:    [capacity, 128]
void qsa_cache_k_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    Tensor& k_cache,
    int start_pos,
    int count);

// Computes attention scores for complete key blocks (each block is 4 tokens).
// For block b = 0 .. num_complete_blocks - 1:
// 1. Mean-pools 4 raw keys: k_pooled = mean(k_cache[4*b .. 4*b+3])
// 2. RMSNorm with k_norm_w (1+w)
// 3. RoPE at group_starts = 4 * b
// 4. Dot product with 4 query heads in q [4, 128], relu, sum over 4 heads, divided by sqrt(128).
// Writes to scores_out [num_complete_blocks].
void qsa_compute_block_scores(
    Context& ctx,
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& k_norm_w,
    Tensor& scores_out,
    int num_complete_blocks,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

// Selects the exact top-k active blocks (ties prefer lower block ids) and expands them into
// individual token indices, appending any trailing tokens
// [num_complete_blocks * compress_ratio .. cached_len - 1]. NaN scores rank below numeric scores.
// Returns the total number of active tokens (<= 2051).
int qsa_select_topk_and_expand_tokens(
    Context& ctx,
    const Tensor& scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    Tensor& active_tokens_out);

// Prepares one causal prefill query at absolute position pos, writes its raw K,
// and selects sparse active tokens from exactly the visible prefix [0, pos].
// Returns 0 when the prefix fits the dense budget; otherwise returns the active
// token count and writes indices to active_tokens_out.
int qsa_prepare_causal_query_and_select_tokens(
    Context& ctx,
    const Tensor& indexer_qk_row,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    Tensor& k_cache,
    Tensor& q_out_row,
    Tensor& scores_out,
    Tensor& active_tokens_out,
    int pos,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0,
    Tensor* compressed_k = nullptr,
    int* num_compressed_blocks = nullptr);

// CPU host reference functions for verification and testing
void qsa_prepare_q_and_cache_k_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    bf16* k_cache,
    bf16* q_out,
    int pos,
    int capacity,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

void qsa_compute_block_scores_host(
    const bf16* q,
    const bf16* k_cache,
    const bf16* k_norm_w,
    float* scores_out,
    int num_complete_blocks,
    int capacity,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

// Upstream-faithful reference adhering strictly to HuggingFace modeling_qwen4_exp.py
// precision boundaries: RMSNorm evaluates in f32, scales by (1.0 + weight_raw), casts
// to bf16, RoPE rotates in bf16, and matmul is performed in f32 followed by relu / sqrt(128).
void qsa_compute_block_scores_upstream_ref(
    const bf16* q,
    const bf16* k_cache,
    const bf16* k_norm_w_raw,
    float* scores_out,
    int num_complete_blocks,
    float theta,
    float eps = 1e-6f);

// Incremental QSA (A1): Compress complete key blocks into normalized + rotated vectors.
// For block b in [start_block, start_block + num_blocks):
// 1. Mean-pools 4 raw keys from k_cache.
// 2. Applies RMSNorm with k_norm_w.
// 3. Applies RoPE at group start position 4 * b.
// 4. Writes normalized + rotated key to compressed_k[b, 128].
void qsa_compress_key_blocks(
    Context& ctx,
    const Tensor& k_cache,
    const Tensor& k_norm_w,
    Tensor& compressed_k,
    int start_block,
    int num_blocks,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

void qsa_compress_key_blocks_host(
    const bf16* k_cache,
    const bf16* k_norm_w,
    bf16* compressed_k,
    int start_block,
    int num_blocks,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

// Incremental QSA (A1): Compute block scores using pre-compressed keys.
// Evaluates Q dot compressed_K, relu, head sum, divided by sqrt(128).
void qsa_compute_block_scores_compressed(
    Context& ctx,
    const Tensor& q,
    const Tensor& compressed_k,
    Tensor& scores_out,
    int num_complete_blocks);

void qsa_compute_block_scores_compressed_host(
    const bf16* q,
    const bf16* compressed_k,
    float* scores_out,
    int num_complete_blocks);

int qsa_select_topk_and_expand_tokens_host(
    const float* scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    int* active_tokens_out);

int qsa_prepare_causal_query_and_select_tokens_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    const bf16* k_norm_w,
    bf16* k_cache,
    bf16* q_out,
    float* scores_out,
    int* active_tokens_out,
    int pos,
    int capacity,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0,
    bf16* compressed_k = nullptr,
    int* num_compressed_blocks = nullptr);

// Phase A3: Block-native active block selection (device).
// Writes selected block IDs (ascending) followed by trailing token IDs to active_blocks_out.
// Returns the number of active blocks selected (<= block_topk).
int qsa_select_topk_blocks(
    Context& ctx,
    const Tensor& scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    Tensor& active_blocks_out);

int qsa_select_topk_blocks_host(
    const float* scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    int* active_blocks_out);

int qsa_prepare_causal_query_and_select_blocks(
    Context& ctx,
    const Tensor& indexer_qk_row,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    Tensor& k_cache,
    Tensor& q_out_row,
    Tensor& scores_out,
    Tensor& active_blocks_out,
    int pos,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0,
    Tensor* compressed_k = nullptr,
    int* num_compressed_blocks = nullptr);

int qsa_prepare_causal_query_and_select_blocks_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    const bf16* k_norm_w,
    bf16* k_cache,
    bf16* q_out,
    float* scores_out,
    int* active_blocks_out,
    int pos,
    int capacity,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0,
    bf16* compressed_k = nullptr,
    int* num_compressed_blocks = nullptr);

// Phase A4: Batched QSA Query Preparation
void qsa_prepare_q_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    Tensor& q_out,
    int start_pos,
    int count,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

void qsa_prepare_q_batched_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    bf16* q_out,
    int start_pos,
    int count,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

// Phase A4: Batched Scoring of complete blocks using compressed K
void qsa_compute_block_scores_compressed_batched(
    Context& ctx,
    const Tensor& q,
    const Tensor& compressed_k,
    Tensor& scores_out,
    int query_count,
    int max_complete_blocks,
    int compress_ratio,
    int start_pos);

void qsa_compute_block_scores_compressed_batched_host(
    const bf16* q,
    const bf16* compressed_k,
    float* scores_out,
    int query_count,
    int max_complete_blocks,
    int compress_ratio,
    int start_pos);

// Phase A4 & P2: Batched Radix Top-K block selection with R4c Complement Selection and valid counts
void qsa_select_topk_blocks_batched(
    Context& ctx,
    const Tensor& scores,
    Tensor& active_blocks_out,
    int query_count,
    int max_complete_blocks,
    int start_pos,
    int block_topk,
    int compress_ratio,
    Tensor* valid_counts_out = nullptr);

void qsa_select_topk_blocks_batched_host(
    const float* scores,
    int* active_blocks_out,
    int query_count,
    int max_complete_blocks,
    int start_pos,
    int block_topk,
    int compress_ratio,
    int* valid_counts_out = nullptr);

// Phase P2 (R4d): Reentrant Batched Pre-Indexer
// Fuses raw K caching, query preparation (norm + RoPE), and block compression
// for newly completed blocks in a single, safe, reentrant entry point.
void qsa_pre_indexer_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    Tensor& k_cache,
    Tensor& q_out,
    Tensor* compressed_k,
    int* num_compressed_blocks,
    int start_pos,
    int count,
    int compress_ratio,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

void qsa_pre_indexer_batched_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    const bf16* k_norm_w,
    bf16* k_cache,
    bf16* q_out,
    bf16* compressed_k,
    int* num_compressed_blocks,
    int start_pos,
    int count,
    int compress_ratio,
    int capacity,
    float theta,
    bool interleaved = false,
    const int* pos_t = nullptr,
    const int* pos_h = nullptr,
    const int* pos_w = nullptr,
    int prompt_pos_len = 0,
    int text_pos_delta = 0,
    int mrope_section_t = 0,
    int mrope_section_h = 0,
    int mrope_section_w = 0);

} // namespace ops

