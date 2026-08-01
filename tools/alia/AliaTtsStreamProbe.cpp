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
    int chunk_delay_ms = 120;
};

struct AudioCapture {
    std::mutex mutex;
    std::vector<float> samples;
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
    ++capture->callback_count;
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
#else
    setenv("AILA_TTS_REF_AUDIO", ref_path.string().c_str(), 1);
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

    // Session: async producer feeds sentence chunks with a small delay.
    AudioCapture session_capture;
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

    std::cout << "ALIA_TTS_STREAM_PROBE_PASS\n"
              << "text_chunks=" << chunks.size() << "\n"
              << "baseline_samples=" << baseline_capture.samples.size() << "\n"
              << "baseline_callbacks=" << baseline_capture.callback_count << "\n"
              << "session_samples=" << session_capture.samples.size() << "\n"
              << "session_callbacks=" << session_capture.callback_count << "\n"
              << "stream_session_enabled=" << metrics.stream_session_enabled << "\n"
              << "tts_chunks_synthesized=" << metrics.chunks_synthesized << "\n"
              << "session_text_appends=" << metrics.session_text_appends << "\n"
              << "session_eos_suppressed=" << metrics.session_eos_suppressed << "\n"
              << "session_resets=" << metrics.session_resets << "\n"
              << "baseline_wav=" << baseline_wav << "\n"
              << "session_wav=" << session_wav << "\n";
    return 0;
}
