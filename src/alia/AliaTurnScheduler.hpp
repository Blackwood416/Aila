#pragma once

#include <string>

namespace aila::alia {

struct AliaTurnSchedulerConfig {
    bool enabled = true;
    int min_prefill_text_chars = 12;
    int min_incremental_text_chars = 8;
    int max_cached_final_suffix_tokens = 16;
    int min_hidden_asr_decode_audio_ms = 450;
    int min_hidden_prefill_audio_ms = 400;
    int speculative_min_chars = 24;
    int speculative_required_stable_ticks = 1;
    int speculative_min_ascii_words = 3;
};

struct AliaAsrSchedulerEvent {
    double chunk_end_ms = 0.0;
    double remaining_audio_ms = -1.0;
    double prefill_audio_budget_ms = -1.0;
    bool final_chunk = false;
    bool text_changed = false;
    int stable_chars = 0;
    int partial_chars = 0;
    int combined_chars = 0;
    int ascii_words = 0;
};

struct AliaPrefillSchedulerState {
    bool speculative_enabled = false;
    bool speculative_started = false;
    int last_prefill_text_chars = 0;
    int candidate_stable_ticks = 0;
};

struct AliaPrefillDecision {
    bool prefill = false;
    bool start_speculative = false;
    std::string phase;
    std::string action;
    std::string lane;
    std::string reason;
};

struct AliaAsrDecodeDecision {
    bool decode = true;
    bool force_final = false;
    std::string phase;
    std::string action;
    std::string lane;
    std::string reason;
};

struct AliaFinalPrefixDecision {
    bool use_cached_prefix = true;
    std::string reason;
};

AliaTurnSchedulerConfig read_alia_turn_scheduler_config();

AliaAsrDecodeDecision decide_asr_decode(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event);

AliaPrefillDecision decide_asr_prefill(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event,
    const AliaPrefillSchedulerState& state);

AliaFinalPrefixDecision decide_final_cached_prefix(
    const AliaTurnSchedulerConfig& config,
    int prefilled_prompt_tokens,
    int prompt_suffix_tokens);

std::string final_prefix_path_name(const AliaFinalPrefixDecision& decision,
                                   int prefilled_prompt_tokens,
                                   int prompt_suffix_tokens,
                                   int max_decode_suffix_tokens);

}  // namespace aila::alia
