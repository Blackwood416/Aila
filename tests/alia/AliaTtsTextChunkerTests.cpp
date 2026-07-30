#include "alia/AliaTtsTextChunker.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(
    TestResults& results,
    bool value,
    const char* expression,
    const char* file,
    int line) {
    if (value) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression << '\n';
}

void expect_eq_string(
    TestResults& results,
    const std::string& left,
    const std::string& right,
    const char* left_expression,
    const char* right_expression,
    const char* file,
    int line) {
    if (left == right) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << left_expression
              << " == " << right_expression << ", got \"" << left
              << "\" vs \"" << right << "\"\n";
}

void expect_eq_size(
    TestResults& results,
    size_t left,
    size_t right,
    const char* left_expression,
    const char* right_expression,
    const char* file,
    int line) {
    if (left == right) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << left_expression
              << " == " << right_expression << ", got " << left
              << " vs " << right << '\n';
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_STRING(results, left, right) \
    expect_eq_string((results), (left), (right), #left, #right, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_SIZE(results, left, right) \
    expect_eq_size((results), (left), (right), #left, #right, __FILE__, __LINE__)

void expect_eq_int(
    TestResults& results,
    int left,
    int right,
    const char* left_expression,
    const char* right_expression,
    const char* file,
    int line) {
    if (left == right) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << left_expression
              << " == " << right_expression << ", got " << left
              << " vs " << right << '\n';
}

#define AILA_EXPECT_EQ_INT(results, left, right) \
    expect_eq_int((results), (left), (right), #left, #right, __FILE__, __LINE__)

void set_env_var(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env_var(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

aila::alia::TtsTextChunkPolicy test_policy() {
    aila::alia::TtsTextChunkPolicy policy;
    policy.first_soft_min_chars = 5;
    policy.steady_soft_min_chars = 24;
    policy.first_soft_max_chars = 48;
    policy.steady_soft_max_chars = 120;
    policy.first_hard_min_chars = 8;
    policy.steady_hard_min_chars = 96;
    policy.first_soft_boundary_flush = true;
    policy.coalesce_steady_text_chunks = false;
    return policy;
}

void hard_boundary_below_first_min_waits(TestResults& results) {
    std::string buffer = "hi.";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        false,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_TRUE(results, result.chunks.empty());
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "below_hard_min");
    AILA_EXPECT_EQ_STRING(results, buffer, "hi.");
}

void first_soft_boundary_flushes_after_soft_min(TestResults& results) {
    std::string buffer = "hello, next";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        false,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "hello,");
    AILA_EXPECT_EQ_STRING(results, buffer, "next");
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "soft_boundary_flush");
}

void early_first_chunk_flushes_without_boundary(TestResults& results) {
    std::string buffer = "abcdefgh";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        true,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "abcdefgh");
    AILA_EXPECT_TRUE(results, buffer.empty());
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "early_first_chunk");
}

void tiny_first_pause_prefix_waits_for_following_text(TestResults& results) {
    std::string buffer = "啊……";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        true,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_TRUE(results, result.chunks.empty());
    AILA_EXPECT_EQ_STRING(results, buffer, "啊……");
}

void tiny_first_pause_prefix_flushes_with_following_text(TestResults& results) {
    std::string buffer = "啊……那";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        true,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "啊……那");
    AILA_EXPECT_TRUE(results, buffer.empty());
}

void first_audio_wait_hint_detects_tiny_pause_without_following_text(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const aila::alia::TtsFirstAudioPriorityWaitHint hint =
        aila::alia::analyze_tts_first_audio_priority_wait_hint("啊……", config, 8);

    AILA_EXPECT_TRUE(results, hint.tiny_first_text);
    AILA_EXPECT_TRUE(results, !hint.has_following_text);
    AILA_EXPECT_EQ_INT(results, hint.first_text_bytes, 3);
    AILA_EXPECT_EQ_INT(results, hint.following_text_bytes, 0);
    AILA_EXPECT_EQ_INT(results, hint.total_text_bytes, 3);
}

void first_audio_wait_hint_detects_tiny_pause_with_following_text(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const aila::alia::TtsFirstAudioPriorityWaitHint hint =
        aila::alia::analyze_tts_first_audio_priority_wait_hint(
            "啊……肩膀好酸呢。",
            config,
            8);

    AILA_EXPECT_TRUE(results, hint.tiny_first_text);
    AILA_EXPECT_TRUE(results, hint.has_following_text);
    AILA_EXPECT_EQ_INT(results, hint.first_text_bytes, 3);
    AILA_EXPECT_EQ_INT(results, hint.following_text_bytes, 18);
    AILA_EXPECT_EQ_INT(results, hint.total_text_bytes, 21);
}

void first_audio_wait_hint_ignores_normal_first_text(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const aila::alia::TtsFirstAudioPriorityWaitHint hint =
        aila::alia::analyze_tts_first_audio_priority_wait_hint(
            "父亲大人说过，",
            config,
            8);

    AILA_EXPECT_TRUE(results, !hint.tiny_first_text);
    AILA_EXPECT_TRUE(results, !hint.has_following_text);
    AILA_EXPECT_EQ_INT(results, hint.first_text_bytes, 21);
    AILA_EXPECT_EQ_INT(results, hint.following_text_bytes, 0);
    AILA_EXPECT_EQ_INT(results, hint.total_text_bytes, 21);
}

void first_audio_wait_hint_uses_queue_aware_tiny_threshold(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const aila::alia::TtsFirstAudioPriorityWaitHint hint =
        aila::alia::analyze_tts_first_audio_priority_wait_hint(
            "啊，这个",
            config,
            16);

    AILA_EXPECT_TRUE(results, hint.tiny_first_text);
    AILA_EXPECT_EQ_INT(results, hint.first_text_bytes, 12);
}

void steady_chunk_ignores_first_chunk_early_flush(TestResults& results) {
    std::string buffer = "abcdefgh";
    const aila::alia::TtsTextChunkRequest request{
        false,
        false,
        true,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_TRUE(results, result.chunks.empty());
    AILA_EXPECT_EQ_STRING(results, buffer, "abcdefgh");
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "below_hard_min");
}

void force_flushes_all_text(TestResults& results) {
    std::string buffer = "abc";
    const aila::alia::TtsTextChunkRequest request{
        true,
        true,
        false,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "abc");
    AILA_EXPECT_TRUE(results, buffer.empty());
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "force");
}

void first_chunk_early_flush_defaults_enabled(TestResults& results) {
    unset_env_var("AILA_TTS_FIRST_CHUNK_EARLY_FLUSH");
    unset_env_var("AILA_TTS_FIRST_CHUNK_EARLY_TOKEN_DELAY");
    unset_env_var("AILA_TTS_FIRST_CHUNK_EARLY_MS");

    const aila::alia::TtsFirstChunkEarlyFlushConfig config =
        aila::alia::read_tts_first_chunk_early_flush_config();

    AILA_EXPECT_TRUE(results, config.enabled);
    AILA_EXPECT_EQ_INT(results, config.token_delay, 0);
    AILA_EXPECT_EQ_INT(results, config.delay_ms, 0);
}

void first_audio_priority_defaults_to_adaptive_window(TestResults& results) {
    unset_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY");
    unset_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TIMEOUT_MS");
    unset_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_ACTIVE_EXTRA_MS");
    unset_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_QUEUE_AWARE_TINY");
    unset_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_FOLLOWING_MIN_BYTES");
    unset_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_MAX_DEFERRED_STEPS");

    const aila::alia::TtsFirstAudioPriorityConfig config =
        aila::alia::read_tts_first_audio_priority_config();

    AILA_EXPECT_TRUE(results, config.enabled);
    AILA_EXPECT_EQ_INT(results, config.base_timeout_ms, 250);
    AILA_EXPECT_EQ_INT(results, config.active_extra_ms, 120);
    AILA_EXPECT_TRUE(results, config.queue_aware_tiny_first_text);
    AILA_EXPECT_EQ_INT(results, config.tiny_following_min_bytes, 16);
    AILA_EXPECT_EQ_INT(results, config.tiny_max_deferred_steps, 6);
}

void first_audio_priority_env_clamps_negative_windows(TestResults& results) {
    set_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY", "0");
    set_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TIMEOUT_MS", "-5");
    set_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_ACTIVE_EXTRA_MS", "-7");
    set_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_QUEUE_AWARE_TINY", "0");
    set_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_FOLLOWING_MIN_BYTES", "-9");
    set_env_var("AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_MAX_DEFERRED_STEPS", "-3");

    const aila::alia::TtsFirstAudioPriorityConfig config =
        aila::alia::read_tts_first_audio_priority_config();

    AILA_EXPECT_TRUE(results, !config.enabled);
    AILA_EXPECT_EQ_INT(results, config.base_timeout_ms, 0);
    AILA_EXPECT_EQ_INT(results, config.active_extra_ms, 0);
    AILA_EXPECT_TRUE(results, !config.queue_aware_tiny_first_text);
    AILA_EXPECT_EQ_INT(results, config.tiny_following_min_bytes, 0);
    AILA_EXPECT_EQ_INT(results, config.tiny_max_deferred_steps, 0);
}

void stream_action_tag_guard_defaults_disabled(TestResults& results) {
    unset_env_var("AILA_TTS_STREAM_ACTION_TAG_GUARD");

    const aila::alia::TtsStreamActionTagGuardConfig config =
        aila::alia::read_tts_stream_action_tag_guard_config();

    AILA_EXPECT_TRUE(results, !config.enabled);
}

void stream_action_tag_guard_can_be_enabled(TestResults& results) {
    set_env_var("AILA_TTS_STREAM_ACTION_TAG_GUARD", "1");

    const aila::alia::TtsStreamActionTagGuardConfig config =
        aila::alia::read_tts_stream_action_tag_guard_config();

    AILA_EXPECT_TRUE(results, config.enabled);
}

void ellipsis_pause_segments_split_text_and_silence(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const std::vector<aila::alia::TtsPreparedSegment> segments =
        aila::alia::split_tts_text_pause_segments("我……我有点紧张", config);

    AILA_EXPECT_EQ_SIZE(results, segments.size(), 3);
    AILA_EXPECT_TRUE(
        results,
        segments[0].kind == aila::alia::TtsPreparedSegmentKind::Text);
    AILA_EXPECT_EQ_STRING(results, segments[0].text, "我");
    AILA_EXPECT_TRUE(
        results,
        segments[1].kind == aila::alia::TtsPreparedSegmentKind::Silence);
    AILA_EXPECT_EQ_INT(results, segments[1].silence_ms, 240);
    AILA_EXPECT_TRUE(
        results,
        segments[2].kind == aila::alia::TtsPreparedSegmentKind::Text);
    AILA_EXPECT_EQ_STRING(results, segments[2].text, "我有点紧张");
}

void ascii_ellipsis_pause_segment_uses_single_pause(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const std::vector<aila::alia::TtsPreparedSegment> segments =
        aila::alia::split_tts_text_pause_segments("I...think", config);

    AILA_EXPECT_EQ_SIZE(results, segments.size(), 3);
    AILA_EXPECT_EQ_STRING(results, segments[0].text, "I");
    AILA_EXPECT_TRUE(
        results,
        segments[1].kind == aila::alia::TtsPreparedSegmentKind::Silence);
    AILA_EXPECT_EQ_INT(results, segments[1].silence_ms, 160);
    AILA_EXPECT_EQ_STRING(results, segments[2].text, "think");
}

void ellipsis_pause_segment_can_close_continuing_text(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;
    config.continuation_suffix = "，";

    const std::vector<aila::alia::TtsPreparedSegment> segments =
        aila::alia::split_tts_text_pause_segments("艾莉亚……父亲", config);

    AILA_EXPECT_EQ_SIZE(results, segments.size(), 3);
    AILA_EXPECT_EQ_STRING(results, segments[0].text, "艾莉亚，");
    AILA_EXPECT_EQ_INT(results, segments[1].silence_ms, 240);
    AILA_EXPECT_EQ_STRING(results, segments[2].text, "父亲");
}

void ellipsis_pause_segment_auto_suffix_matches_script(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = true;
    config.pause_ms = 160;
    config.max_pause_ms = 240;
    config.continuation_suffix = "auto";

    const auto chinese =
        aila::alia::split_tts_text_pause_segments("艾莉亚……父亲", config);
    const auto ascii =
        aila::alia::split_tts_text_pause_segments("I...think", config);

    AILA_EXPECT_EQ_STRING(results, chinese[0].text, "艾莉亚，");
    AILA_EXPECT_EQ_STRING(results, ascii[0].text, "I,");
}

void ellipsis_pause_segments_can_be_disabled(TestResults& results) {
    aila::alia::TtsPauseSegmentConfig config;
    config.enabled = false;
    config.pause_ms = 160;
    config.max_pause_ms = 240;

    const std::vector<aila::alia::TtsPreparedSegment> segments =
        aila::alia::split_tts_text_pause_segments("我……我有点紧张", config);

    AILA_EXPECT_EQ_SIZE(results, segments.size(), 1);
    AILA_EXPECT_TRUE(
        results,
        segments[0].kind == aila::alia::TtsPreparedSegmentKind::Text);
    AILA_EXPECT_EQ_STRING(results, segments[0].text, "我……我有点紧张");
}

void silence_lookahead_prefetch_defaults_disabled(TestResults& results) {
    unset_env_var("AILA_TTS_SILENCE_LOOKAHEAD_PREFETCH");
    unset_env_var("AILA_TTS_SILENCE_LOOKAHEAD_PREFETCH_MIN_TEXT_BYTES");

    const aila::alia::TtsSilenceLookaheadPrefetchConfig config =
        aila::alia::read_tts_silence_lookahead_prefetch_config();

    AILA_EXPECT_TRUE(results, !config.enabled);
    AILA_EXPECT_EQ_INT(results, config.min_text_bytes, 12);
}

void silence_lookahead_prefetch_can_be_enabled(TestResults& results) {
    set_env_var("AILA_TTS_SILENCE_LOOKAHEAD_PREFETCH", "1");
    set_env_var("AILA_TTS_SILENCE_LOOKAHEAD_PREFETCH_MIN_TEXT_BYTES", "0");

    const aila::alia::TtsSilenceLookaheadPrefetchConfig config =
        aila::alia::read_tts_silence_lookahead_prefetch_config();

    AILA_EXPECT_TRUE(results, config.enabled);
    AILA_EXPECT_EQ_INT(results, config.min_text_bytes, 0);
}

void silence_lookahead_prefetch_requires_ready_following_text(TestResults& results) {
    aila::alia::TtsSilenceLookaheadPrefetchConfig config;
    config.enabled = true;
    config.min_text_bytes = 12;

    AILA_EXPECT_TRUE(
        results,
        aila::alia::should_prefetch_tts_silence_lookahead(
            config,
            aila::alia::TtsSilenceLookaheadPrefetchRequest{
                true,
                160,
                true,
                15,
            }));

    AILA_EXPECT_TRUE(
        results,
        !aila::alia::should_prefetch_tts_silence_lookahead(
            config,
            aila::alia::TtsSilenceLookaheadPrefetchRequest{
                false,
                160,
                true,
                15,
            }));

    AILA_EXPECT_TRUE(
        results,
        !aila::alia::should_prefetch_tts_silence_lookahead(
            config,
            aila::alia::TtsSilenceLookaheadPrefetchRequest{
                true,
                0,
                true,
                15,
            }));

    AILA_EXPECT_TRUE(
        results,
        !aila::alia::should_prefetch_tts_silence_lookahead(
            config,
            aila::alia::TtsSilenceLookaheadPrefetchRequest{
                true,
                160,
                false,
                15,
            }));

    AILA_EXPECT_TRUE(
        results,
        !aila::alia::should_prefetch_tts_silence_lookahead(
            config,
            aila::alia::TtsSilenceLookaheadPrefetchRequest{
                true,
                160,
                true,
                3,
            }));
}

}  // namespace

int main() {
    TestResults results;
    hard_boundary_below_first_min_waits(results);
    first_soft_boundary_flushes_after_soft_min(results);
    early_first_chunk_flushes_without_boundary(results);
    tiny_first_pause_prefix_waits_for_following_text(results);
    tiny_first_pause_prefix_flushes_with_following_text(results);
    first_audio_wait_hint_detects_tiny_pause_without_following_text(results);
    first_audio_wait_hint_detects_tiny_pause_with_following_text(results);
    first_audio_wait_hint_ignores_normal_first_text(results);
    first_audio_wait_hint_uses_queue_aware_tiny_threshold(results);
    steady_chunk_ignores_first_chunk_early_flush(results);
    force_flushes_all_text(results);
    first_chunk_early_flush_defaults_enabled(results);
    first_audio_priority_defaults_to_adaptive_window(results);
    first_audio_priority_env_clamps_negative_windows(results);
    stream_action_tag_guard_defaults_disabled(results);
    stream_action_tag_guard_can_be_enabled(results);
    ellipsis_pause_segments_split_text_and_silence(results);
    ascii_ellipsis_pause_segment_uses_single_pause(results);
    ellipsis_pause_segment_can_close_continuing_text(results);
    ellipsis_pause_segment_auto_suffix_matches_script(results);
    ellipsis_pause_segments_can_be_disabled(results);
    silence_lookahead_prefetch_defaults_disabled(results);
    silence_lookahead_prefetch_can_be_enabled(results);
    silence_lookahead_prefetch_requires_ready_following_text(results);

    std::cout << "AilaAliaTtsTextChunkerTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
