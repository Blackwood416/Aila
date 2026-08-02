#include "alia_api.h"
#include "alia/AliaContext.hpp"
#include "alia/AliaTtsPipeline.hpp"
#include "alia/AliaTtsTextChunker.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string tts_model = "models/Qwen3-TTS-12Hz-0.6B-Base";
    std::string ref_audio = "alia_ref.wav";
    std::string text = "艾莉亚，请用一句话打个招呼。";
    std::string output_dir = "tmp/alia-real-smoke/tts_stream_probe";
    int max_seq_len = 2048;
    int max_tokens = 512;
    int stream_batch = 4;
    int chunk_delay_ms = 1000;
};

struct AudioCapture {
    std::mutex mutex;
    std::vector<float> samples;
    std::vector<std::chrono::steady_clock::time_point> callback_times;
    std::vector<int> callback_sample_counts;
    int callback_count = 0;
};

#ifdef _WIN32
std::string wide_to_utf8(const wchar_t* value) {
    if (!value || value[0] == L'\0') {
        return {};
    }
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) {
        return {};
    }
    std::string result(static_cast<size_t>(required - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), required,
                        nullptr, nullptr);
    return result;
}

std::vector<std::string> windows_command_line_args_utf8() {
    int wide_argc = 0;
    LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (!wide_argv || wide_argc <= 0) {
        return {};
    }
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(wide_argc));
    for (int i = 0; i < wide_argc; ++i) {
        args.push_back(wide_to_utf8(wide_argv[i]));
    }
    LocalFree(wide_argv);
    return args;
}
#endif

void print_usage() {
    std::cout
        << "AilaAliaTtsStreamProbe - TTS-only stream session A/B probe\n\n"
        << "Options:\n"
        << "  --tts-model <dir>\n"
        << "  --ref <path>\n"
        << "  --text <utf8 text>\n"
        << "  --output-dir <dir>\n"
        << "  --max-tokens <N>\n"
        << "  --stream-batch <N>\n"
        << "  --chunk-delay-ms <N>\n";
}

bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](std::string& target) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            target = argv[++i];
            return true;
        };
        auto require_int = [&](int& target) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            target = std::atoi(argv[++i]);
            return target >= 0;
        };
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--tts-model") {
            if (!require_value(opts.tts_model)) return false;
        } else if (arg == "--ref") {
            if (!require_value(opts.ref_audio)) return false;
        } else if (arg == "--text") {
            if (!require_value(opts.text)) return false;
        } else if (arg == "--output-dir") {
            if (!require_value(opts.output_dir)) return false;
        } else if (arg == "--max-tokens") {
            if (!require_int(opts.max_tokens)) return false;
        } else if (arg == "--stream-batch") {
            if (!require_int(opts.stream_batch)) return false;
        } else if (arg == "--chunk-delay-ms") {
            if (!require_int(opts.chunk_delay_ms)) return false;
        }
    }
    return true;
}

void audio_callback(const float* samples, int count, void* user_data) {
    if (!samples || count <= 0 || !user_data) {
        return;
    }
    auto* capture = static_cast<AudioCapture*>(user_data);
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->samples.insert(capture->samples.end(), samples, samples + count);
    capture->callback_times.push_back(std::chrono::steady_clock::now());
    capture->callback_sample_counts.push_back(count);
    ++capture->callback_count;
}

long long max_callback_gap_ms(AudioCapture& capture) {
    std::lock_guard<std::mutex> lock(capture.mutex);
    if (capture.callback_times.size() < 2) {
        return 0;
    }
    long long best = 0;
    for (size_t i = 1; i < capture.callback_times.size(); ++i) {
        const long long gap = std::chrono::duration_cast<
            std::chrono::milliseconds>(
            capture.callback_times[i] - capture.callback_times[i - 1])
            .count();
        best = std::max(best, gap);
    }
    return best;
}

long long max_virtual_underrun_ms(AudioCapture& capture) {
    std::lock_guard<std::mutex> lock(capture.mutex);
    if (capture.callback_times.size() < 2 ||
        capture.callback_times.size() != capture.callback_sample_counts.size()) {
        return 0;
    }
    long long best = 0;
    long long buffer_end_ms = 0;
    bool has_buffer = false;
    const auto first_at = capture.callback_times.front();
    for (size_t i = 0; i < capture.callback_times.size(); ++i) {
        const long long at_ms = std::chrono::duration_cast<
            std::chrono::milliseconds>(
            capture.callback_times[i] - first_at)
            .count();
        const long long duration_ms =
            (static_cast<long long>(capture.callback_sample_counts[i]) * 1000) /
            24000;
        if (!has_buffer) {
            buffer_end_ms = at_ms + duration_ms;
            has_buffer = true;
            continue;
        }
        if (at_ms > buffer_end_ms) {
            best = std::max(best, at_ms - buffer_end_ms);
            buffer_end_ms = at_ms;
        }
        buffer_end_ms += duration_ms;
    }
    return best;
}

bool save_wav(const std::string& path, const std::vector<float>& samples,
              int sample_rate) {
    const std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    const uint32_t data_bytes = static_cast<uint32_t>(samples.size()) * 2;
    const uint32_t file_size = 36 + data_bytes;
    auto write_u32 = [&](uint32_t value) {
        out.put(static_cast<char>(value & 0xff));
        out.put(static_cast<char>((value >> 8) & 0xff));
        out.put(static_cast<char>((value >> 16) & 0xff));
        out.put(static_cast<char>((value >> 24) & 0xff));
    };
    auto write_u16 = [&](uint16_t value) {
        out.put(static_cast<char>(value & 0xff));
        out.put(static_cast<char>((value >> 8) & 0xff));
    };

    out.write("RIFF", 4);
    write_u32(file_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(16);
    write_u16(1);
    write_u16(1);
    write_u32(static_cast<uint32_t>(sample_rate));
    write_u32(static_cast<uint32_t>(sample_rate * 2));
    write_u16(2);
    write_u16(16);
    out.write("data", 4);
    write_u32(data_bytes);
    for (float sample : samples) {
        const float clamped = std::clamp(sample, -1.0f, 1.0f);
        const int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        out.put(static_cast<char>(pcm & 0xff));
        out.put(static_cast<char>((pcm >> 8) & 0xff));
    }
    return true;
}

size_t max_consecutive_low_energy_samples(const std::vector<float>& samples) {
    size_t best = 0;
    size_t current = 0;
    for (float sample : samples) {
        if (std::fabs(sample) < 0.001f) {
            ++current;
            best = std::max(best, current);
        } else {
            current = 0;
        }
    }
    return best;
}

int expected_text_appends(const std::vector<std::string>& chunks) {
    const aila::alia::TtsPauseSegmentConfig pause_config =
        aila::alia::read_tts_pause_segment_config();
    int text_items = 0;
    for (const std::string& chunk : chunks) {
        const std::vector<aila::alia::TtsPreparedSegment> segments =
            aila::alia::split_tts_text_pause_segments(chunk, pause_config);
        for (const auto& segment : segments) {
            if (segment.kind == aila::alia::TtsPreparedSegmentKind::Text &&
                !segment.text.empty()) {
                ++text_items;
            }
        }
    }
    return std::max(0, text_items - 1);
}

int expected_queue_items(const std::vector<std::string>& chunks) {
    const aila::alia::TtsPauseSegmentConfig pause_config =
        aila::alia::read_tts_pause_segment_config();
    int items = 0;
    for (const std::string& chunk : chunks) {
        const std::vector<aila::alia::TtsPreparedSegment> segments =
            aila::alia::split_tts_text_pause_segments(chunk, pause_config);
        for (const auto& segment : segments) {
            if (segment.kind == aila::alia::TtsPreparedSegmentKind::Text) {
                if (!segment.text.empty()) {
                    ++items;
                }
            } else if (segment.silence_ms > 0) {
                ++items;
            }
        }
    }
    return items;
}

bool load_tts_slot(AliaContext& ctx, const Options& opts) {
    ctx.tts_model_dir = opts.tts_model;
    ctx.configure_model_slots();
    if (!ctx.tts.load_metadata()) {
        std::cerr << "tts_metadata_error=" << ctx.tts.last_error() << "\n";
        return false;
    }
    if (!ctx.tts.load_model(opts.max_seq_len)) {
        std::cerr << "tts_model_load_error=" << ctx.tts.last_error() << "\n";
        return false;
    }
    return true;
}

AliaGenConfig make_config(const Options& opts) {
    AliaGenConfig config{};
    config.max_tokens = opts.max_tokens;
    config.temperature = 0.6f;
    config.top_p = 0.9f;
    return config;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    std::vector<std::string> utf8_args = windows_command_line_args_utf8();
    std::vector<char*> utf8_argv;
    if (!utf8_args.empty()) {
        utf8_argv.reserve(utf8_args.size());
        for (std::string& arg : utf8_args) {
            utf8_argv.push_back(arg.data());
        }
        argc = static_cast<int>(utf8_argv.size());
        argv = utf8_argv.data();
    }
#endif

    Options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage();
        return 2;
    }

    std::filesystem::path ref_path = std::filesystem::absolute(opts.ref_audio);
    if (!std::filesystem::exists(ref_path)) {
        std::cerr << "ref_audio_missing=" << ref_path.string() << "\n";
        return 1;
    }
#ifdef _WIN32
    _putenv_s("AILA_TTS_REF_AUDIO", ref_path.string().c_str());
    _putenv_s("AILA_TTS_STREAM_SESSION", "1");
#else
    setenv("AILA_TTS_REF_AUDIO", ref_path.string().c_str(), 1);
    setenv("AILA_TTS_STREAM_SESSION", "1", 1);
#endif

    AliaContext ctx(opts.max_seq_len);
    if (!load_tts_slot(ctx, opts)) {
        return 1;
    }
    std::string reference_error;
    if (!ctx.tts_pipeline->preload_reference_voice(&reference_error)) {
        std::cerr << "reference_voice_error=" << reference_error << "\n";
        return 1;
    }

    const AliaGenConfig config = make_config(opts);
    const std::vector<std::string> chunks =
        aila::alia::split_spoken_text_for_tts(opts.text, true, 8);
    if (chunks.empty()) {
        std::cerr << "split_chunks_empty=true\n";
        return 1;
    }
    const std::filesystem::path output_root =
        std::filesystem::absolute(opts.output_dir);
    std::filesystem::create_directories(output_root);

    // Baseline: single complete utterance through the legacy synchronous path.
    AudioCapture baseline_capture;
    const auto baseline_started_at = std::chrono::steady_clock::now();
    ctx.tts_pipeline->reset();
    if (!ctx.tts_pipeline->enqueue_text(opts.text) ||
        !ctx.tts_pipeline->synthesize_pending(
            config, audio_callback, &baseline_capture, []() { return false; })) {
        std::cerr << "baseline_synthesis_failed=true\n";
        return 1;
    }
    ctx.tts_pipeline->reset();
    const std::string baseline_wav =
        (output_root / "baseline.wav").string();
    if (!save_wav(baseline_wav, baseline_capture.samples, 24000)) {
        std::cerr << "baseline_wav_save_failed=true\n";
        return 1;
    }
    long long baseline_first_audio_ms = -1;
    {
        std::lock_guard<std::mutex> lock(baseline_capture.mutex);
        if (!baseline_capture.callback_times.empty()) {
            baseline_first_audio_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    baseline_capture.callback_times.front() - baseline_started_at)
                    .count();
        }
    }

    // Session: async producer feeds sentence chunks with a small delay.
    AudioCapture session_capture;
    const auto session_started_at = std::chrono::steady_clock::now();
    ctx.tts_pipeline->begin_turn_metrics();
    if (!ctx.tts_pipeline->start_async_turn(
            config, audio_callback, &session_capture,
            []() { return false; })) {
        std::cerr << "session_start_failed=true\n";
        return 1;
    }

    std::thread producer([&]() {
        for (const std::string& chunk : chunks) {
            ctx.tts_pipeline->enqueue_text(chunk);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(opts.chunk_delay_ms));
        }
    });
    producer.join();
    if (!ctx.tts_pipeline->finish_async_turn()) {
        std::cerr << "session_finish_failed=true\n";
        return 1;
    }

    const aila::alia::AliaTtsMetrics metrics =
        ctx.tts_pipeline->last_metrics();
    const std::string session_wav =
        (output_root / "session.wav").string();
    if (!save_wav(session_wav, session_capture.samples, 24000)) {
        std::cerr << "session_wav_save_failed=true\n";
        return 1;
    }
    long long session_first_audio_ms = -1;
    {
        std::lock_guard<std::mutex> lock(session_capture.mutex);
        if (!session_capture.callback_times.empty()) {
            session_first_audio_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    session_capture.callback_times.front() - session_started_at)
                    .count();
        }
    }

    bool pass = true;
    std::vector<std::string> failures;
    const size_t min_session_samples = static_cast<size_t>(
        static_cast<double>(baseline_capture.samples.size()) * 0.9);
    if (session_capture.samples.size() < min_session_samples) {
        pass = false;
        failures.push_back("session_truncated");
    }
    const int min_session_callbacks = static_cast<int>(
        static_cast<double>(baseline_capture.callback_count) * 0.8);
    if (session_capture.callback_count < min_session_callbacks) {
        pass = false;
        failures.push_back("session_callbacks_low");
    }
    constexpr size_t kMaxSilenceSamples = 24000 * 2;  // 2s at 24 kHz
    const size_t max_silence =
        max_consecutive_low_energy_samples(session_capture.samples);
    if (max_silence > kMaxSilenceSamples) {
        pass = false;
        failures.push_back("session_long_silence");
    }
    const long long baseline_callback_gap_ms =
        max_callback_gap_ms(baseline_capture);
    const long long session_callback_gap_ms =
        max_callback_gap_ms(session_capture);
    constexpr long long kMaxVirtualUnderrunMs = 500;
    const long long baseline_underrun_ms =
        max_virtual_underrun_ms(baseline_capture);
    const long long session_underrun_ms =
        max_virtual_underrun_ms(session_capture);
    if (baseline_underrun_ms > kMaxVirtualUnderrunMs) {
        pass = false;
        failures.push_back("baseline_underrun");
    }
    if (session_underrun_ms > kMaxVirtualUnderrunMs) {
        pass = false;
        failures.push_back("session_underrun");
    }
    const int expected_appends = expected_text_appends(chunks);
    if (metrics.session_text_appends != expected_appends) {
        pass = false;
        failures.push_back("session_appends_missing");
    }
    const std::vector<aila::alia::AliaTtsPlaybackSegment> segments =
        ctx.tts_pipeline->playback_segments_snapshot();
    const int expected_segments = expected_queue_items(chunks);
    if (static_cast<int>(segments.size()) != expected_segments) {
        pass = false;
        failures.push_back("session_segment_count");
    }
    for (const auto& segment : segments) {
        if (!segment.complete || segment.end_sample < segment.start_sample) {
            pass = false;
            failures.push_back("session_segment_incomplete");
            break;
        }
    }
    if (metrics.chunks_synthesized != 1) {
        pass = false;
        failures.push_back("session_not_single_synthesis");
    }

    std::cout << (pass ? "ALIA_TTS_STREAM_PROBE_PASS\n"
                       : "ALIA_TTS_STREAM_PROBE_FAIL\n")
              << "text_chunks=" << chunks.size() << "\n"
              << "baseline_samples=" << baseline_capture.samples.size() << "\n"
              << "baseline_callbacks=" << baseline_capture.callback_count << "\n"
              << "baseline_first_audio_ms=" << baseline_first_audio_ms << "\n"
              << "session_samples=" << session_capture.samples.size() << "\n"
              << "session_callbacks=" << session_capture.callback_count << "\n"
              << "session_first_audio_ms=" << session_first_audio_ms << "\n"
              << "session_max_silence_samples=" << max_silence << "\n"
              << "baseline_max_callback_gap_ms=" << baseline_callback_gap_ms << "\n"
              << "session_max_callback_gap_ms=" << session_callback_gap_ms << "\n"
              << "baseline_max_underrun_ms=" << baseline_underrun_ms << "\n"
              << "session_max_underrun_ms=" << session_underrun_ms << "\n"
              << "expected_text_appends=" << expected_appends << "\n"
              << "session_segments=" << segments.size() << "\n"
              << "expected_segments=" << expected_segments << "\n"
              << "session_segments_complete="
              << std::count_if(segments.begin(), segments.end(),
                               [](const aila::alia::AliaTtsPlaybackSegment& segment) {
                                   return segment.complete;
                               })
              << "\n"
              << "stream_session_enabled=" << metrics.stream_session_enabled << "\n"
              << "tts_chunks_synthesized=" << metrics.chunks_synthesized << "\n"
              << "session_text_appends=" << metrics.session_text_appends << "\n"
              << "session_eos_suppressed=" << metrics.session_eos_suppressed << "\n"
              << "session_resets=" << metrics.session_resets << "\n"
              << "probe_failures=";
    for (size_t i = 0; i < failures.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << failures[i];
    }
    std::cout << "\n"
              << "baseline_wav=" << baseline_wav << "\n"
              << "session_wav=" << session_wav << "\n";
    return pass ? 0 : 1;
}
