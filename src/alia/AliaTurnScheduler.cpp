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
    config.speculative_min_chars = std::max(
        1, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_MIN_CHARS", 24));
    config.speculative_required_stable_ticks = std::max(
        1, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_STABLE_TICKS", 1));
    config.speculative_min_ascii_words = std::max(
        0, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_MIN_ASCII_WORDS", 3));
    return config;
}

AliaPrefillDecision decide_asr_prefill(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event,
    const AliaPrefillSchedulerState& state) {
    AliaPrefillDecision decision;
    if (!config.enabled) {
        decision.prefill = event.text_changed;
        decision.reason = decision.prefill ? "scheduler disabled: changed text"
                                           : "scheduler disabled: unchanged text";
        return decision;
    }
    if (!event.text_changed) {
        decision.reason = "unchanged text";
        return decision;
    }
    if (event.combined_chars < config.min_prefill_text_chars) {
        decision.reason = "text shorter than prefill minimum";
        return decision;
    }
    const int delta_chars = event.combined_chars - state.last_prefill_text_chars;
    if (!event.final_chunk &&
        state.last_prefill_text_chars > 0 &&
        delta_chars >= 0 &&
        delta_chars < config.min_incremental_text_chars) {
        decision.reason = "incremental text delta too small";
        return decision;
    }

    decision.prefill = !state.speculative_started;
    decision.reason = event.final_chunk ? "final text prefill" : "partial text prefill";

    if (!event.final_chunk &&
        !state.speculative_started &&
        event.combined_chars >= config.speculative_min_chars &&
        state.candidate_stable_ticks >= config.speculative_required_stable_ticks &&
        (event.ascii_words == 0 || event.ascii_words >= config.speculative_min_ascii_words)) {
        decision.start_speculative = true;
        decision.prefill = false;
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
