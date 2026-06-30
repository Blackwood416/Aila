#include "AliaTurnScheduler.hpp"

#include "../utils/EnvUtils.hpp"

#include <algorithm>

namespace aila::alia {

AliaTurnSchedulerConfig read_alia_turn_scheduler_config() {
    AliaTurnSchedulerConfig config;
    config.enabled = aila::env::read_flag("AILA_TURN_SCHEDULER", true);
    config.min_prefill_text_chars = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MIN_PREFILL_TEXT_CHARS", 12));
    config.min_incremental_text_chars = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MIN_INCREMENTAL_TEXT_CHARS", 8));
    config.max_cached_final_suffix_tokens = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MAX_CACHED_FINAL_SUFFIX_TOKENS", 16));
    config.min_hidden_asr_decode_audio_ms = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MIN_HIDDEN_ASR_DECODE_AUDIO_MS", 450));
    config.min_hidden_prefill_audio_ms = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MIN_HIDDEN_PREFILL_AUDIO_MS", 400));
    config.speculative_min_chars = std::max(
        1, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_MIN_CHARS", 24));
    config.speculative_required_stable_ticks = std::max(
        1, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_STABLE_TICKS", 1));
    config.speculative_min_ascii_words = std::max(
        0, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_MIN_ASCII_WORDS", 3));
    return config;
}

AliaAsrDecodeDecision decide_asr_decode(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event) {
    AliaAsrDecodeDecision decision;
    decision.phase = event.final_chunk ? "final_commit" : "listening_partial";
    decision.action = event.final_chunk ? "force_final_asr_decode" : "decode_partial_asr";
    decision.lane = "asr_decode";
    decision.reason = event.final_chunk ? "final ASR text required" : "partial ASR decode";
    decision.force_final = event.final_chunk;
    if (!config.enabled) {
        decision.phase = "scheduler_disabled";
        decision.reason = event.final_chunk ? "scheduler disabled: final ASR text required"
                                            : "scheduler disabled: partial ASR decode";
        return decision;
    }
    if (event.final_chunk) {
        return decision;
    }
    if (event.remaining_audio_ms >= 0.0 &&
        event.remaining_audio_ms < static_cast<double>(config.min_hidden_asr_decode_audio_ms)) {
        decision.decode = false;
        decision.phase = "near_final_partial";
        decision.action = "skip_partial_asr_decode";
        decision.lane = "none";
        decision.reason = "not enough remaining audio budget";
    }
    return decision;
}

AliaPrefillDecision decide_asr_prefill(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event,
    const AliaPrefillSchedulerState& state) {
    AliaPrefillDecision decision;
    decision.phase = event.final_chunk ? "final_commit" : "listening_partial";
    decision.action = "skip";
    decision.lane = "none";
    if (!config.enabled) {
        decision.prefill = event.text_changed;
        decision.phase = "scheduler_disabled";
        decision.action = decision.prefill ? "prefill_asr_text" : "skip";
        decision.lane = decision.prefill ? "foreground_prefill" : "none";
        decision.reason = decision.prefill ? "scheduler disabled: changed text"
                                           : "scheduler disabled: unchanged text";
        return decision;
    }
    if (!event.text_changed) {
        decision.phase = event.final_chunk ? "final_commit" : "unchanged_partial";
        decision.reason = "unchanged text";
        return decision;
    }
    if (event.final_chunk) {
        decision.action = "commit_foreground_turn";
        decision.lane = "foreground_turn";
        decision.reason = "final text handled by foreground turn";
        return decision;
    }
    if (event.combined_chars < config.min_prefill_text_chars) {
        decision.phase = "short_partial";
        decision.reason = "text shorter than prefill minimum";
        return decision;
    }
    const double prefill_audio_budget_ms = event.prefill_audio_budget_ms >= 0.0
        ? event.prefill_audio_budget_ms
        : event.remaining_audio_ms;
    if (prefill_audio_budget_ms >= 0.0 &&
        prefill_audio_budget_ms < static_cast<double>(config.min_hidden_prefill_audio_ms)) {
        decision.phase = "near_final_partial";
        decision.reason = "not enough remaining prefill budget";
        return decision;
    }
    const int delta_chars = event.combined_chars - state.last_prefill_text_chars;
    if (!event.final_chunk &&
        state.last_prefill_text_chars > 0 &&
        delta_chars >= 0 &&
        delta_chars < config.min_incremental_text_chars) {
        decision.phase = "incremental_partial";
        decision.reason = "incremental text delta too small";
        return decision;
    }
    if (state.speculative_started) {
        decision.phase = "speculative_running";
        decision.reason = "speculative foreground already running";
        return decision;
    }

    decision.prefill = true;
    decision.action = "prefill_asr_text";
    decision.lane = "foreground_prefill";
    decision.reason = "partial text prefill";

    if (state.speculative_enabled &&
        !state.speculative_started &&
        event.combined_chars >= config.speculative_min_chars &&
        state.candidate_stable_ticks >= config.speculative_required_stable_ticks &&
        (event.ascii_words == 0 || event.ascii_words >= config.speculative_min_ascii_words)) {
        decision.start_speculative = true;
        decision.prefill = false;
        decision.phase = "speculative_candidate";
        decision.action = "start_speculative_foreground";
        decision.lane = "foreground_speculative";
        decision.reason = "start speculative foreground";
    }
    return decision;
}

AliaFinalPrefixDecision decide_final_cached_prefix(
    const AliaTurnSchedulerConfig& config,
    int prefilled_prompt_tokens,
    int prompt_suffix_tokens) {
    AliaFinalPrefixDecision decision;
    if (!config.enabled) {
        decision.reason = "scheduler disabled";
        return decision;
    }
    if (prefilled_prompt_tokens <= 0) {
        decision.reason = "no cached prefix";
        return decision;
    }
    if (prompt_suffix_tokens <= config.max_cached_final_suffix_tokens) {
        decision.reason = "cached suffix within fast threshold";
        return decision;
    }
    decision.use_cached_prefix = false;
    decision.reason = "cached suffix exceeds fast threshold";
    return decision;
}

}  // namespace aila::alia
