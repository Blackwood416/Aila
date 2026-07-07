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
    int token_delay = 2;
    int delay_ms = 80;
};

struct TtsFirstAudioPriorityConfig {
    bool enabled = true;
    int base_timeout_ms = 250;
    int active_extra_ms = 120;
};

struct TtsStreamActionTagGuardConfig {
    bool enabled = false;
};

std::vector<std::string> split_spoken_text_for_tts(
    const std::string& text,
    bool split_sentence_boundaries = true,
    size_t min_first_chunk_chars = 0);

TtsFirstChunkEarlyFlushConfig read_tts_first_chunk_early_flush_config();
TtsFirstAudioPriorityConfig read_tts_first_audio_priority_config();
TtsStreamActionTagGuardConfig read_tts_stream_action_tag_guard_config();

TtsTextChunkResult take_ready_tts_text_chunks(
    std::string& buffer,
    const TtsTextChunkPolicy& policy,
    const TtsTextChunkRequest& request);

}  // namespace aila::alia
