#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace aila::alia {

enum class TtsChunkDecisionReason {
    NoText,
    NoBoundary,
    BelowHardMin,
    BelowSoftMin,
    HardBoundaryFlush,
    SoftBoundaryFlush,
    SoftMax,
    Force,
    EarlyFirstChunk,
};

const char* tts_chunk_decision_reason_name(TtsChunkDecisionReason reason);

struct TtsTextChunkPolicy {
    int first_soft_min_chars = 8;
    int steady_soft_min_chars = 24;
    int first_soft_max_chars = 48;
    int steady_soft_max_chars = 120;
    int first_hard_min_chars = 8;
    int steady_hard_min_chars = 96;
    bool first_soft_boundary_flush = true;
    bool coalesce_steady_text_chunks = false;
};

struct TtsTextChunkRequest {
    bool force = false;
    bool low_latency_first_chunk = false;
    bool early_first_chunk = false;
};

struct TtsTextChunkResult {
    std::vector<std::string> chunks;
    TtsChunkDecisionReason reason = TtsChunkDecisionReason::NoText;
    size_t pending_chars_before = 0;
    size_t cutoff_chars = 0;
};

struct TtsFirstChunkEarlyFlushConfig {
    bool enabled = true;
    int token_delay = 0;
    int delay_ms = 0;
};

struct TtsFirstAudioPriorityConfig {
    bool enabled = true;
    bool queue_aware_tiny_first_text = true;
    int tiny_following_min_bytes = 16;
    int tiny_max_deferred_steps = 2;
    int base_timeout_ms = 250;
    int active_extra_ms = 120;
};

struct TtsFirstAudioPriorityWaitHint {
    bool tiny_first_text = false;
    bool has_following_text = false;
    int first_text_bytes = 0;
    int following_text_bytes = 0;
    int total_text_bytes = 0;
};

struct TtsStreamActionTagGuardConfig {
    bool enabled = false;
};

struct TtsPauseSegmentConfig {
    bool enabled = true;
    int pause_ms = 160;
    int max_pause_ms = 240;
    std::string continuation_suffix;
};

struct TtsSilenceLookaheadPrefetchConfig {
    bool enabled = false;
    int min_text_bytes = 12;
};

struct TtsSilenceLookaheadPrefetchRequest {
    bool first_audio_callback_emitted = false;
    int silence_ms = 0;
    bool next_queue_item_is_text = false;
    int next_text_bytes = 0;
};

enum class TtsPreparedSegmentKind {
    Text,
    Silence,
};

struct TtsPreparedSegment {
    TtsPreparedSegmentKind kind = TtsPreparedSegmentKind::Text;
    std::string text;
    int silence_ms = 0;
};

std::vector<std::string> split_spoken_text_for_tts(
    const std::string& text,
    bool split_sentence_boundaries = true,
    size_t min_first_chunk_chars = 0);

TtsFirstChunkEarlyFlushConfig read_tts_first_chunk_early_flush_config();
TtsFirstAudioPriorityConfig read_tts_first_audio_priority_config();
TtsStreamActionTagGuardConfig read_tts_stream_action_tag_guard_config();
TtsPauseSegmentConfig read_tts_pause_segment_config();
TtsSilenceLookaheadPrefetchConfig read_tts_silence_lookahead_prefetch_config();

std::vector<TtsPreparedSegment> split_tts_text_pause_segments(
    const std::string& text,
    const TtsPauseSegmentConfig& config);

bool should_prefetch_tts_silence_lookahead(
    const TtsSilenceLookaheadPrefetchConfig& config,
    const TtsSilenceLookaheadPrefetchRequest& request);

TtsFirstAudioPriorityWaitHint analyze_tts_first_audio_priority_wait_hint(
    const std::string& text,
    const TtsPauseSegmentConfig& config,
    size_t tiny_text_max_bytes);

TtsTextChunkResult take_ready_tts_text_chunks(
    std::string& buffer,
    const TtsTextChunkPolicy& policy,
    const TtsTextChunkRequest& request);

}  // namespace aila::alia
