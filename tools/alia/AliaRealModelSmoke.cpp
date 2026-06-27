#include "alia_api.h"
#include "alia/AliaBackgroundPipeline.hpp"
#include "alia/AliaContext.hpp"
#include "alia/AliaForegroundPipeline.hpp"
#include "alia/AliaTtsPipeline.hpp"
#include "audio/AudioPreprocessor.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string asr_model = "models/Qwen3-ASR-1.7B-BNB-NF4";
    std::string foreground_model = "models/qwen3.5-4B-bnb-nf4-offline-visiondense";
    std::string foreground_lora =
        "F:/unsloth/qwen35_4b_alia_identity_r16_lr1e5/checkpoint-1400";
    std::string background_model = "models/qwen3.5-0.8B-bnb-nf4-offline";
    std::string tts_model = "models/Qwen3-TTS-12Hz-0.6B-Base";
    std::string audio_path = "tmp/alia-real-smoke/alia_request.wav";
    std::string output_wav = "tmp/alia-real-smoke/alia_full_pipeline_target_models.wav";
    std::string request_text = "Alia, please say hello in one short sentence.";
    std::string tool_probe_text =
        "Call the host tool inspect_window with parameter id equal to 42 now. "
        "Return only the tool call.";
    int max_seq_len = 2048;
    int max_tokens = 48;
    int timeout_sec = 1500;
    int stream_chunk_ms = 1000;
    int stream_prefill_interval_ms = 0;
    bool generate_audio_if_missing = true;
    bool run_tool_probe = true;
    bool stream_asr_prefill = false;
};

struct AudioCapture {
    std::mutex mutex;
    Clock::time_point turn_start{};
    Clock::time_point first_audio{};
    std::vector<float> samples;
    std::vector<int> chunk_sizes;
    int callback_count = 0;
    int nonzero_samples = 0;
};

std::mutex g_background_mutex;
std::string g_background_json;

std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    out << '"';
    return out.str();
}

bool env_flag_enabled(const char* name, bool default_value = false) {
#ifdef _WIN32
    char* raw = nullptr;
    size_t len = 0;
    if (_dupenv_s(&raw, &len, name) != 0 || !raw) {
        if (raw) {
            std::free(raw);
        }
        return default_value;
    }
    const bool enabled = *raw ? std::atoi(raw) != 0 : default_value;
    std::free(raw);
    return enabled;
#else
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    return std::atoi(raw) != 0;
#endif
}

aila::alia::AliaAsrMetrics subtract_asr_metrics(
    const aila::alia::AliaAsrMetrics& after,
    const aila::alia::AliaAsrMetrics& before) {
    aila::alia::AliaAsrMetrics delta;
    delta.transcribe_calls = after.transcribe_calls - before.transcribe_calls;
    delta.generated_tokens = after.generated_tokens - before.generated_tokens;
    delta.prefix_reuse_attempts = after.prefix_reuse_attempts - before.prefix_reuse_attempts;
    delta.prefix_reuse_hits = after.prefix_reuse_hits - before.prefix_reuse_hits;
    delta.prefix_reused_tokens = after.prefix_reused_tokens - before.prefix_reused_tokens;
    delta.prefix_appended_tokens = after.prefix_appended_tokens - before.prefix_appended_tokens;
    delta.input_audio_ms = after.input_audio_ms - before.input_audio_ms;
    delta.mel_ms = after.mel_ms - before.mel_ms;
    delta.upload_ms = after.upload_ms - before.upload_ms;
    delta.encoder_ms = after.encoder_ms - before.encoder_ms;
    delta.readback_ms = after.readback_ms - before.readback_ms;
    delta.prompt_ms = after.prompt_ms - before.prompt_ms;
    delta.prefill_ms = after.prefill_ms - before.prefill_ms;
    delta.decode_ms = after.decode_ms - before.decode_ms;
    delta.total_ms = after.total_ms - before.total_ms;
    return delta;
}

void print_asr_profile_call(const char* prefix,
                            int index,
                            bool force_decode,
                            double chunk_end_ms,
                            double call_ms,
                            const aila::alia::AliaAsrMetrics& delta,
                            const std::string& stable_text,
                            const std::string& partial_text) {
    std::cout << prefix
              << "=index:" << index
              << ",force:" << (force_decode ? "true" : "false")
              << ",chunk_end_ms:" << chunk_end_ms
              << ",call_ms:" << call_ms
              << ",transcribes:" << delta.transcribe_calls
              << ",generated_tokens:" << delta.generated_tokens
              << ",prefix_reuse_attempts:" << delta.prefix_reuse_attempts
              << ",prefix_reuse_hits:" << delta.prefix_reuse_hits
              << ",prefix_reused_tokens:" << delta.prefix_reused_tokens
              << ",prefix_appended_tokens:" << delta.prefix_appended_tokens
              << ",input_audio_ms:" << delta.input_audio_ms
              << ",mel_ms:" << delta.mel_ms
              << ",upload_ms:" << delta.upload_ms
              << ",encoder_ms:" << delta.encoder_ms
              << ",readback_ms:" << delta.readback_ms
              << ",prompt_ms:" << delta.prompt_ms
              << ",prefill_ms:" << delta.prefill_ms
              << ",decode_ms:" << delta.decode_ms
              << ",total_ms:" << delta.total_ms
              << ",stable_chars:" << stable_text.size()
              << ",partial_chars:" << partial_text.size()
              << "\n";
}

void print_usage() {
    std::cout
        << "AilaAliaRealSmoke - real-model Alia full pipeline smoke\n\n"
        << "Options:\n"
        << "  --asr-model <dir>\n"
        << "  --foreground-model <dir>\n"
        << "  --foreground-lora <dir>\n"
        << "  --background-model <dir>\n"
        << "  --tts-model <dir>\n"
        << "  --audio <path>\n"
        << "  --output-wav <path>\n"
        << "  --request-text <text>  default \"Alia, please say hello in one short sentence.\"\n"
        << "  --tool-probe-text <text>\n"
        << "  --max-seq <N>          default 2048\n"
        << "  --max-tokens <N>       default 48\n"
        << "  --timeout-sec <N>      default 1500\n"
        << "  --stream-chunk-ms <N>  ASR stream chunk duration for --stream-asr-prefill, default 1000\n"
        << "  --stream-prefill-interval-ms <N>\n"
        << "                         ASR partial/prefill cadence, default matches --stream-chunk-ms\n"
        << "  --no-generate-audio    fail if --audio is missing instead of using target TTS\n"
        << "  --stream-asr-prefill   feed ASR in chunks and prefill foreground VLM from stable/partial text\n"
        << "  --skip-tool-probe      skip the dedicated LoRA tool-call probe\n";
}

bool parse_int_arg(const char* text, int& out) {
    if (!text) {
        return false;
    }
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value < 0 || value > 1000000) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
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
            if (!parse_int_arg(argv[++i], target)) {
                std::cerr << "Invalid integer for " << arg << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--asr-model") {
            if (!require_value(opts.asr_model)) return false;
        } else if (arg == "--foreground-model") {
            if (!require_value(opts.foreground_model)) return false;
        } else if (arg == "--foreground-lora") {
            if (!require_value(opts.foreground_lora)) return false;
        } else if (arg == "--background-model") {
            if (!require_value(opts.background_model)) return false;
        } else if (arg == "--tts-model") {
            if (!require_value(opts.tts_model)) return false;
        } else if (arg == "--audio") {
            if (!require_value(opts.audio_path)) return false;
        } else if (arg == "--output-wav") {
            if (!require_value(opts.output_wav)) return false;
        } else if (arg == "--request-text") {
            if (!require_value(opts.request_text)) return false;
        } else if (arg == "--tool-probe-text") {
            if (!require_value(opts.tool_probe_text)) return false;
        } else if (arg == "--max-seq") {
            if (!require_int(opts.max_seq_len)) return false;
        } else if (arg == "--max-tokens") {
            if (!require_int(opts.max_tokens)) return false;
        } else if (arg == "--timeout-sec") {
            if (!require_int(opts.timeout_sec)) return false;
        } else if (arg == "--stream-chunk-ms") {
            if (!require_int(opts.stream_chunk_ms)) return false;
        } else if (arg == "--stream-prefill-interval-ms") {
            if (!require_int(opts.stream_prefill_interval_ms)) return false;
        } else if (arg == "--no-generate-audio") {
            opts.generate_audio_if_missing = false;
        } else if (arg == "--stream-asr-prefill") {
            opts.stream_asr_prefill = true;
        } else if (arg == "--skip-tool-probe") {
            opts.run_tool_probe = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    if (opts.max_seq_len <= 0 || opts.max_tokens <= 0 || opts.timeout_sec <= 0 ||
        opts.stream_chunk_ms <= 0 || opts.stream_prefill_interval_ms < 0) {
        std::cerr << "max-seq, max-tokens, timeout-sec, and stream-chunk-ms must be positive; "
                  << "stream-prefill-interval-ms must be non-negative.\n";
        return false;
    }
    if (opts.stream_prefill_interval_ms == 0) {
        opts.stream_prefill_interval_ms = opts.stream_chunk_ms;
    }
    if (opts.request_text.empty()) {
        std::cerr << "request-text must not be empty.\n";
        return false;
    }
    if (opts.foreground_lora.empty()) {
        std::cerr << "foreground-lora must not be empty.\n";
        return false;
    }
    if (opts.tool_probe_text.empty()) {
        std::cerr << "tool-probe-text must not be empty.\n";
        return false;
    }
    return true;
}

std::string leaf_name(const std::string& path) {
    std::filesystem::path normalized = std::filesystem::path(path).lexically_normal();
    if (normalized.filename().empty() || normalized.filename() == ".") {
        normalized = normalized.parent_path();
    }
    return normalized.filename().string();
}

bool validate_target_model_dir(
    const char* label,
    const std::string& path,
    const char* expected_leaf) {
    const std::string leaf = leaf_name(path);
    if (leaf != expected_leaf) {
        std::cerr << "target_model_mismatch=" << quote(label)
                  << " path=" << quote(path)
                  << " expected_leaf=" << quote(expected_leaf)
                  << " actual_leaf=" << quote(leaf) << "\n";
        return false;
    }
    if (!std::filesystem::exists(path)) {
        std::cerr << "target_model_missing=" << quote(label)
                  << " path=" << quote(path) << "\n";
        return false;
    }
    return true;
}

bool validate_target_models(const Options& opts) {
    bool ok = true;
    ok = validate_target_model_dir(
             "asr", opts.asr_model, "Qwen3-ASR-1.7B-BNB-NF4") && ok;
    ok = validate_target_model_dir(
             "foreground", opts.foreground_model,
             "qwen3.5-4B-bnb-nf4-offline-visiondense") && ok;
    ok = validate_target_model_dir(
             "background", opts.background_model,
             "qwen3.5-0.8B-bnb-nf4-offline") && ok;
    ok = validate_target_model_dir(
             "tts", opts.tts_model, "Qwen3-TTS-12Hz-0.6B-Base") && ok;
    if (!opts.foreground_lora.empty() && !std::filesystem::exists(opts.foreground_lora)) {
        std::cerr << "foreground_lora_missing=" << quote(opts.foreground_lora) << "\n";
        ok = false;
    }
    return ok;
}

std::string foreground_state_name(aila::alia::ForegroundTurnState state) {
    switch (state) {
        case aila::alia::ForegroundTurnState::Idle: return "Idle";
        case aila::alia::ForegroundTurnState::Running: return "Running";
        case aila::alia::ForegroundTurnState::Aborting: return "Aborting";
        case aila::alia::ForegroundTurnState::Aborted: return "Aborted";
        case aila::alia::ForegroundTurnState::Completed: return "Completed";
        case aila::alia::ForegroundTurnState::Failed: return "Failed";
    }
    return "Unknown";
}

std::string foreground_mode_name(aila::alia::ForegroundDecodeMode mode) {
    switch (mode) {
        case aila::alia::ForegroundDecodeMode::None: return "None";
        case aila::alia::ForegroundDecodeMode::LoadedVlm: return "LoadedVlm";
    }
    return "Unknown";
}

std::string background_state_name(aila::alia::BackgroundJobState state) {
    switch (state) {
        case aila::alia::BackgroundJobState::Idle: return "Idle";
        case aila::alia::BackgroundJobState::Running: return "Running";
        case aila::alia::BackgroundJobState::Aborting: return "Aborting";
        case aila::alia::BackgroundJobState::Completed: return "Completed";
        case aila::alia::BackgroundJobState::Failed: return "Failed";
    }
    return "Unknown";
}

std::string background_mode_name(aila::alia::BackgroundDecodeMode mode) {
    switch (mode) {
        case aila::alia::BackgroundDecodeMode::None: return "None";
        case aila::alia::BackgroundDecodeMode::LoadedVlm: return "LoadedVlm";
    }
    return "Unknown";
}

bool load_mono_16k_audio(const std::string& path, std::vector<float>& mono_16k) {
    AudioBuffer audio;
    std::string error;
    if (!load_audio(path, audio, &error)) {
        std::cerr << "audio_load_error=" << quote(error) << "\n";
        return false;
    }

    std::vector<float> mono;
    if (audio.channels > 1) {
        mono.resize(audio.samples.size() / static_cast<size_t>(audio.channels));
        for (size_t i = 0; i < mono.size(); ++i) {
            float sum = 0.0f;
            for (int c = 0; c < audio.channels; ++c) {
                sum += audio.samples[i * static_cast<size_t>(audio.channels) + c];
            }
            mono[i] = sum / static_cast<float>(audio.channels);
        }
    } else {
        mono = std::move(audio.samples);
    }

    resample_to_16k(mono, audio.sample_rate, mono_16k);
    std::cout << "audio_input_path=" << quote(path)
              << " input_rate=" << audio.sample_rate
              << " input_channels=" << audio.channels
              << " mono16_samples=" << mono_16k.size()
              << " mono16_seconds=" << (static_cast<double>(mono_16k.size()) / 16000.0)
              << "\n";
    return !mono_16k.empty();
}

void audio_callback(const float* samples, int sample_count, void* user_data) {
    if (!samples || sample_count <= 0 || !user_data) {
        return;
    }

    auto* capture = static_cast<AudioCapture*>(user_data);
    std::lock_guard<std::mutex> lock(capture->mutex);
    if (capture->callback_count == 0) {
        capture->first_audio = Clock::now();
    }
    ++capture->callback_count;
    capture->chunk_sizes.push_back(sample_count);
    for (int i = 0; i < sample_count; ++i) {
        if (std::fabs(samples[i]) > 1.0e-6f) {
            ++capture->nonzero_samples;
        }
        capture->samples.push_back(samples[i]);
    }
}

int tool_callback(const char* tool_json, char* out_result_buf, int max_result_len, void* user_data) {
    (void)tool_json;
    (void)user_data;

    const std::string result = "{\"ok\":true,\"result\":\"real smoke tool callback executed\"}";
    if (out_result_buf && max_result_len > 0) {
        const size_t copy_len = std::min(result.size(), static_cast<size_t>(max_result_len - 1));
        std::copy_n(result.data(), copy_len, out_result_buf);
        out_result_buf[copy_len] = '\0';
    }
    return 0;
}

void background_callback(const char* extracted_json, void*) {
    std::lock_guard<std::mutex> lock(g_background_mutex);
    g_background_json = extracted_json ? extracted_json : "";
}

bool save_capture_wav(const std::string& path, const std::vector<float>& samples) {
    if (samples.empty()) {
        return false;
    }
    const std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    return save_wav(path, samples, 24000);
}

bool generate_request_audio_with_target_tts(AliaContext& ctx, const Options& opts) {
    if (!ctx.tts_pipeline || !ctx.tts_pipeline->ready()) {
        std::cerr << "request_audio_generation_error=\"target TTS pipeline is not ready\"\n";
        return false;
    }

    AliaGenConfig gen{};
    gen.temperature = 0.6f;
    gen.top_p = 0.9f;
    gen.max_tokens = std::max(opts.max_tokens, 48);

    AudioCapture capture;
    capture.turn_start = Clock::now();
    ctx.tts_pipeline->reset();
    if (!ctx.tts_pipeline->enqueue_text(opts.request_text)) {
        std::cerr << "request_audio_generation_error=\"request text enqueue failed\"\n";
        return false;
    }
    if (!ctx.tts_pipeline->synthesize_pending(gen, audio_callback, &capture)) {
        std::cerr << "request_audio_generation_error=\"target TTS synthesis failed\"\n";
        return false;
    }
    ctx.tts_pipeline->reset();

    std::lock_guard<std::mutex> lock(capture.mutex);
    if (capture.callback_count <= 0 || capture.nonzero_samples <= 0) {
        std::cerr << "request_audio_generation_error=\"target TTS produced no usable audio\""
                  << " callbacks=" << capture.callback_count
                  << " nonzero_samples=" << capture.nonzero_samples << "\n";
        return false;
    }
    if (!save_capture_wav(opts.audio_path, capture.samples)) {
        std::cerr << "request_audio_save_failed=" << quote(opts.audio_path) << "\n";
        return false;
    }

    std::cout << "request_audio_generated=true"
              << " request_text=" << quote(opts.request_text)
              << " output_path=" << quote(opts.audio_path)
              << " callback_count=" << capture.callback_count
              << " samples=" << capture.samples.size()
              << "\n";
    return true;
}

bool run_foreground_tool_probe(AliaContext& ctx, const Options& opts) {
    if (!ctx.asr_pipeline || !ctx.foreground_pipeline) {
        std::cerr << "tool_probe_error=\"Alia foreground pipelines are not ready\"\n";
        return false;
    }

    ctx.asr_pipeline->reset();
    ctx.asr_pipeline->append_stable_text(opts.tool_probe_text);

    AliaGenConfig gen{};
    gen.temperature = 0.0f;
    gen.top_p = 1.0f;
    gen.max_tokens = std::max(opts.max_tokens, 96);

    const auto probe_start = Clock::now();
    int rc = alia_start_conversation_turn(
        &ctx, &gen, tool_callback, nullptr, nullptr);
    if (rc != ALIA_OK) {
        std::cerr << "tool_probe_start_rc=" << rc << "\n";
        return false;
    }
    if (!ctx.foreground_pipeline->wait_until_idle_for(std::chrono::seconds(opts.timeout_sec))) {
        std::cerr << "tool_probe_timeout=true\n";
        alia_abort_inference(&ctx, ALIA_PIPELINE_VLM_FOREGROUND);
        ctx.foreground_pipeline->join();
        return false;
    }
    ctx.foreground_pipeline->join();
    const auto probe_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - probe_start).count();

    const auto state = ctx.foreground_pipeline->state();
    const auto mode = ctx.foreground_pipeline->last_decode_mode();
    const std::string assistant_text = ctx.foreground_pipeline->last_assistant_text();
    const std::string tool_json = ctx.foreground_pipeline->last_tool_call_json();
    const std::string tool_result = ctx.foreground_pipeline->last_tool_result_text();
    std::cout << "tool_probe_ms=" << probe_ms << "\n"
              << "tool_probe_state=" << foreground_state_name(state) << "\n"
              << "tool_probe_decode_mode=" << foreground_mode_name(mode) << "\n"
              << "tool_probe_user_text=" << quote(ctx.foreground_pipeline->last_user_text()) << "\n"
              << "tool_probe_assistant_text=" << quote(assistant_text) << "\n"
              << "tool_probe_tool_call_json=" << quote(tool_json) << "\n"
              << "tool_probe_tool_result_text=" << quote(tool_result) << "\n"
              << "tool_probe_error=" << quote(ctx.foreground_pipeline->last_error()) << "\n";

    if (state != aila::alia::ForegroundTurnState::Completed ||
        mode != aila::alia::ForegroundDecodeMode::LoadedVlm ||
        tool_json.empty() ||
        tool_json.find("inspect_window") == std::string::npos ||
        tool_result.empty()) {
        std::cerr << "tool_probe_validation_failed=true\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage();
        return 2;
    }
    if (!validate_target_models(opts)) {
        std::cerr << "target_model_enforcement_failed=true\n";
        return 2;
    }

    std::cout << "ALIA_REAL_MODEL_SMOKE_BEGIN\n"
              << "target_model_enforcement=true\n"
              << "asr_model=" << quote(opts.asr_model) << "\n"
              << "foreground_model=" << quote(opts.foreground_model) << "\n"
              << "foreground_lora=" << quote(opts.foreground_lora) << "\n"
              << "background_model=" << quote(opts.background_model) << "\n"
              << "tts_model=" << quote(opts.tts_model) << "\n"
              << "request_text=" << quote(opts.request_text) << "\n"
              << "tool_probe=" << (opts.run_tool_probe ? "true" : "false") << "\n"
              << "max_seq_len=" << opts.max_seq_len << "\n"
              << "max_tokens=" << opts.max_tokens << "\n";

    const bool audio_exists_before_load = std::filesystem::exists(opts.audio_path);
    if (!audio_exists_before_load && !opts.generate_audio_if_missing) {
        std::cerr << "audio_missing=" << quote(opts.audio_path) << "\n";
        return 1;
    }

    auto ctx = std::make_unique<AliaContext>(opts.max_seq_len);
    ctx->asr_model_dir = opts.asr_model;
    ctx->vlm_4b_model_dir = opts.foreground_model;
    ctx->vlm_4b_lora_dir = opts.foreground_lora;
    ctx->vlm_0_8b_model_dir = opts.background_model;
    ctx->tts_model_dir = opts.tts_model;
    ctx->configure_model_slots();

    const auto load_start = Clock::now();
    if (!ctx->load_model_slots()) {
        std::cerr << "model_load_error=" << quote(ctx->last_error) << "\n"
                  << "asr_slot_error=" << quote(ctx->asr.last_error()) << "\n"
                  << "foreground_slot_error=" << quote(ctx->foreground_vlm.last_error()) << "\n"
                  << "background_slot_error=" << quote(ctx->background_vlm.last_error()) << "\n"
                  << "tts_slot_error=" << quote(ctx->tts.last_error()) << "\n";
        return 1;
    }
    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - load_start).count();
    std::cout << "model_load_ms=" << load_ms << "\n"
              << "foreground_lora_applied="
              << (ctx->foreground_vlm.lora_applied() ? "true" : "false") << "\n"
              << "foreground_lora_pair_count="
              << ctx->foreground_vlm.lora_pair_count() << "\n";

    if (!audio_exists_before_load) {
        std::cout << "request_audio_missing=true path=" << quote(opts.audio_path) << "\n";
        if (!generate_request_audio_with_target_tts(*ctx, opts)) {
            return 1;
        }
    }

    std::vector<float> mono_16k;
    if (!load_mono_16k_audio(opts.audio_path, mono_16k)) {
        return 1;
    }

    const auto asr_start = Clock::now();
    int rc = ALIA_OK;
    int asr_prefill_calls = 0;
    int asr_text_calls = 0;
    int asr_prefill_skipped_unchanged = 0;
    double asr_stream_get_text_total_ms = 0.0;
    double asr_stream_get_text_max_ms = 0.0;
    double asr_stream_vlm_prefill_total_ms = 0.0;
    double asr_stream_vlm_prefill_max_ms = 0.0;
    double asr_stream_tick_total_ms = 0.0;
    double asr_stream_tick_max_ms = 0.0;
    const bool asr_profile_calls = env_flag_enabled("AILA_ASR_PROFILE_CALLS", false);
    int asr_profile_call_index = 0;
    constexpr double kAsrSampleRate = 16000.0;
    const double asr_audio_duration_ms =
        static_cast<double>(mono_16k.size()) * 1000.0 / kAsrSampleRate;
    double asr_stream_simulated_clock_ms = 0.0;
    double asr_stream_simulated_tail_ms = -1.0;
    auto get_asr_text = [&](std::string& stable_text, std::string& partial_text, bool force_decode) -> bool {
        char* stable_raw = nullptr;
        char* partial_raw = nullptr;
        const int get_rc = force_decode
            ? alia_asr_get_text(ctx.get(), &stable_raw, &partial_raw)
            : alia_asr_get_partial_text(ctx.get(), &stable_raw, &partial_raw);
        stable_text = stable_raw ? stable_raw : "";
        partial_text = partial_raw ? partial_raw : "";
        alia_free_string(stable_raw);
        alia_free_string(partial_raw);
        if (get_rc != ALIA_OK) {
            std::cerr << "alia_asr_get_text_rc=" << get_rc << "\n";
            return false;
        }
        return true;
    };

    std::string stable_text;
    std::string partial_text;
    if (opts.stream_asr_prefill) {
        const int stream_chunk_samples =
            std::max(1, static_cast<int>(std::llround(
                static_cast<double>(opts.stream_chunk_ms) * kAsrSampleRate / 1000.0)));
        double next_prefill_ms = static_cast<double>(opts.stream_prefill_interval_ms);
        std::string last_prefill_stable_text;
        std::string last_prefill_partial_text;
        for (size_t offset = 0; offset < mono_16k.size();
             offset += static_cast<size_t>(stream_chunk_samples)) {
            const size_t count = std::min<size_t>(
                static_cast<size_t>(stream_chunk_samples),
                mono_16k.size() - offset);
            const double chunk_end_ms =
                static_cast<double>(offset + count) * 1000.0 / kAsrSampleRate;
            const bool final_chunk = offset + count >= mono_16k.size();
            rc = alia_asr_feed_audio(
                ctx.get(),
                mono_16k.data() + offset,
                static_cast<int>(count));
            if (rc != ALIA_OK) {
                std::cerr << "alia_asr_feed_audio_rc=" << rc << "\n";
                return 1;
            }
            const bool should_prefill =
                final_chunk || chunk_end_ms + 0.001 >= next_prefill_ms;
            if (!should_prefill) {
                continue;
            }

            const auto chunk_op_start = Clock::now();
            const aila::alia::AliaAsrMetrics metrics_before =
                ctx->asr_pipeline->last_metrics();
            const auto get_text_start = Clock::now();
            if (!get_asr_text(stable_text, partial_text, final_chunk)) {
                return 1;
            }
            const double get_text_ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - get_text_start).count());
            if (asr_profile_calls) {
                const aila::alia::AliaAsrMetrics metrics_after =
                    ctx->asr_pipeline->last_metrics();
                print_asr_profile_call("asr_profile_call",
                                       asr_profile_call_index++,
                                       final_chunk,
                                       chunk_end_ms,
                                       get_text_ms,
                                       subtract_asr_metrics(metrics_after, metrics_before),
                                       stable_text,
                                       partial_text);
            }
            asr_stream_get_text_total_ms += get_text_ms;
            asr_stream_get_text_max_ms =
                std::max(asr_stream_get_text_max_ms, get_text_ms);
            ++asr_text_calls;
            if (!stable_text.empty() || !partial_text.empty()) {
                if (stable_text == last_prefill_stable_text &&
                    partial_text == last_prefill_partial_text) {
                    ++asr_prefill_skipped_unchanged;
                } else {
                    const auto vlm_prefill_start = Clock::now();
                    rc = alia_vlm_prefill_asr_text(
                        ctx.get(),
                        stable_text.c_str(),
                        partial_text.c_str());
                    if (rc != ALIA_OK) {
                        std::cerr << "alia_vlm_prefill_asr_text_rc=" << rc << "\n";
                        return 1;
                    }
                    last_prefill_stable_text = stable_text;
                    last_prefill_partial_text = partial_text;
                    ++asr_prefill_calls;
                    const double vlm_prefill_ms = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            Clock::now() - vlm_prefill_start).count());
                    asr_stream_vlm_prefill_total_ms += vlm_prefill_ms;
                    asr_stream_vlm_prefill_max_ms =
                        std::max(asr_stream_vlm_prefill_max_ms, vlm_prefill_ms);
                }
            }
            const double chunk_op_ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - chunk_op_start).count());
            asr_stream_tick_total_ms += chunk_op_ms;
            asr_stream_tick_max_ms =
                std::max(asr_stream_tick_max_ms, chunk_op_ms);
            asr_stream_simulated_clock_ms =
                std::max(asr_stream_simulated_clock_ms, chunk_end_ms) + chunk_op_ms;
            while (next_prefill_ms <= chunk_end_ms + 0.001) {
                next_prefill_ms += static_cast<double>(opts.stream_prefill_interval_ms);
            }
        }
        asr_stream_simulated_tail_ms =
            std::max(0.0, asr_stream_simulated_clock_ms - asr_audio_duration_ms);
    } else {
        rc = alia_asr_feed_audio(ctx.get(), mono_16k.data(), static_cast<int>(mono_16k.size()));
        if (rc != ALIA_OK) {
            std::cerr << "alia_asr_feed_audio_rc=" << rc << "\n";
            return 1;
        }
        const aila::alia::AliaAsrMetrics metrics_before =
            ctx->asr_pipeline->last_metrics();
        const auto get_text_start = Clock::now();
        if (!get_asr_text(stable_text, partial_text, true)) {
            return 1;
        }
        const double get_text_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - get_text_start).count());
        if (asr_profile_calls) {
            const aila::alia::AliaAsrMetrics metrics_after =
                ctx->asr_pipeline->last_metrics();
            print_asr_profile_call("asr_profile_call",
                                   asr_profile_call_index++,
                                   true,
                                   asr_audio_duration_ms,
                                   get_text_ms,
                                   subtract_asr_metrics(metrics_after, metrics_before),
                                   stable_text,
                                   partial_text);
        }
        ++asr_text_calls;
    }
    const auto asr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - asr_start).count();
    const aila::alia::AliaAsrMetrics asr_metrics = ctx->asr_pipeline->last_metrics();
    std::string user_text = stable_text;
    if (user_text.empty()) {
        user_text = partial_text;
    } else if (!partial_text.empty()) {
        if (!std::isspace(static_cast<unsigned char>(user_text.back())) &&
            !std::isspace(static_cast<unsigned char>(partial_text.front())) &&
            !std::ispunct(static_cast<unsigned char>(partial_text.front()))) {
            user_text += " ";
        }
        user_text += partial_text;
    }
    std::cout << "asr_ms=" << asr_ms << "\n"
              << "asr_stream_prefill_enabled=" << (opts.stream_asr_prefill ? "true" : "false") << "\n"
              << "asr_stream_chunk_ms=" << (opts.stream_asr_prefill ? opts.stream_chunk_ms : 0) << "\n"
              << "asr_stream_prefill_interval_ms="
              << (opts.stream_asr_prefill ? opts.stream_prefill_interval_ms : 0) << "\n"
              << "asr_stream_text_calls=" << asr_text_calls << "\n"
              << "asr_stream_prefill_calls=" << asr_prefill_calls << "\n"
              << "asr_stream_prefill_skipped_unchanged="
              << asr_prefill_skipped_unchanged << "\n"
              << "asr_stream_get_text_total_ms=" << asr_stream_get_text_total_ms << "\n"
              << "asr_stream_get_text_max_ms=" << asr_stream_get_text_max_ms << "\n"
              << "asr_stream_vlm_prefill_total_ms=" << asr_stream_vlm_prefill_total_ms << "\n"
              << "asr_stream_vlm_prefill_max_ms=" << asr_stream_vlm_prefill_max_ms << "\n"
              << "asr_stream_tick_total_ms=" << asr_stream_tick_total_ms << "\n"
              << "asr_stream_tick_max_ms=" << asr_stream_tick_max_ms << "\n"
              << "asr_partial_full_decode_count="
              << ctx->asr_pipeline->partial_full_decode_count() << "\n"
              << "asr_partial_tail_decode_count="
              << ctx->asr_pipeline->partial_tail_decode_count() << "\n"
              << "asr_partial_throttled_count="
              << ctx->asr_pipeline->partial_throttled_count() << "\n"
              << "asr_profile_transcribe_calls=" << asr_metrics.transcribe_calls << "\n"
              << "asr_profile_generated_tokens=" << asr_metrics.generated_tokens << "\n"
              << "asr_profile_prefix_reuse_attempts="
              << asr_metrics.prefix_reuse_attempts << "\n"
              << "asr_profile_prefix_reuse_hits="
              << asr_metrics.prefix_reuse_hits << "\n"
              << "asr_profile_prefix_reused_tokens="
              << asr_metrics.prefix_reused_tokens << "\n"
              << "asr_profile_prefix_appended_tokens="
              << asr_metrics.prefix_appended_tokens << "\n"
              << "asr_profile_input_audio_ms=" << asr_metrics.input_audio_ms << "\n"
              << "asr_profile_mel_ms=" << asr_metrics.mel_ms << "\n"
              << "asr_profile_upload_ms=" << asr_metrics.upload_ms << "\n"
              << "asr_profile_encoder_ms=" << asr_metrics.encoder_ms << "\n"
              << "asr_profile_readback_ms=" << asr_metrics.readback_ms << "\n"
              << "asr_profile_prompt_ms=" << asr_metrics.prompt_ms << "\n"
              << "asr_profile_prefill_ms=" << asr_metrics.prefill_ms << "\n"
              << "asr_profile_decode_ms=" << asr_metrics.decode_ms << "\n"
              << "asr_profile_total_ms=" << asr_metrics.total_ms << "\n"
              << "asr_audio_duration_ms=" << asr_audio_duration_ms << "\n"
              << "asr_stream_simulated_tail_ms=" << asr_stream_simulated_tail_ms << "\n"
              << "asr_stable_text=" << quote(stable_text) << "\n"
              << "asr_partial_text=" << quote(partial_text) << "\n";
    if (user_text.empty()) {
        std::cerr << "asr_empty_text=true\n";
        return 1;
    }

    AliaGenConfig gen{};
    gen.temperature = 0.6f;
    gen.top_p = 0.9f;
    gen.max_tokens = opts.max_tokens;

    AudioCapture audio_capture;
    audio_capture.turn_start = Clock::now();
    const auto fg_start = Clock::now();
    rc = alia_start_conversation_turn(
        ctx.get(), &gen, tool_callback, audio_callback, &audio_capture);
    if (rc != ALIA_OK) {
        std::cerr << "alia_start_conversation_turn_rc=" << rc << "\n";
        return 1;
    }

    if (!ctx->foreground_pipeline->wait_until_idle_for(std::chrono::seconds(opts.timeout_sec))) {
        std::cerr << "foreground_timeout=true\n";
        alia_abort_inference(ctx.get(), ALIA_PIPELINE_ALL);
        ctx->foreground_pipeline->join();
        return 1;
    }
    ctx->foreground_pipeline->join();
    const auto fg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - fg_start).count();

    const auto fg_state = ctx->foreground_pipeline->state();
    const auto fg_mode = ctx->foreground_pipeline->last_decode_mode();
    const std::string assistant_text = ctx->foreground_pipeline->last_assistant_text();
    const double simulated_vad_asr_tail_ms = opts.stream_asr_prefill
        ? asr_stream_simulated_tail_ms
        : static_cast<double>(asr_ms);
    const long long foreground_first_content_delta_ms =
        ctx->foreground_pipeline->last_first_content_delta_ms();
    const long long foreground_first_tts_enqueue_ms =
        ctx->foreground_pipeline->last_first_tts_enqueue_ms();
    const aila::alia::AliaForegroundMetrics foreground_metrics =
        ctx->foreground_pipeline->last_metrics();
    const double simulated_vad_to_first_content_ms =
        foreground_first_content_delta_ms >= 0
            ? simulated_vad_asr_tail_ms + static_cast<double>(foreground_first_content_delta_ms)
            : -1.0;
    const double simulated_vad_to_first_tts_enqueue_ms =
        foreground_first_tts_enqueue_ms >= 0
            ? simulated_vad_asr_tail_ms + static_cast<double>(foreground_first_tts_enqueue_ms)
            : -1.0;
    std::cout << "foreground_ms=" << fg_ms << "\n"
              << "foreground_state=" << foreground_state_name(fg_state) << "\n"
              << "foreground_decode_mode=" << foreground_mode_name(fg_mode) << "\n"
              << "foreground_prompt_tokens=" << ctx->foreground_pipeline->last_prompt_token_count() << "\n"
              << "foreground_generated_tokens=" << ctx->foreground_pipeline->last_generated_token_count() << "\n"
              << "foreground_asr_prefill_tokens="
              << ctx->foreground_pipeline->last_asr_prefill_token_count() << "\n"
              << "foreground_asr_prefill_reused_tokens="
              << ctx->foreground_pipeline->last_asr_prefill_reused_token_count() << "\n"
              << "foreground_asr_prefill_suffix_tokens="
              << ctx->foreground_pipeline->last_asr_prefill_suffix_token_count() << "\n"
              << "foreground_asr_prefill_skipped_small_suffix="
              << ctx->foreground_pipeline->last_asr_prefill_skipped_small_suffix_count() << "\n"
              << "foreground_asr_prefill_ms="
              << ctx->foreground_pipeline->last_asr_prefill_ms() << "\n"
              << "foreground_first_content_delta_ms="
              << foreground_first_content_delta_ms << "\n"
              << "foreground_first_tts_enqueue_ms="
              << foreground_first_tts_enqueue_ms << "\n"
              << "foreground_profile_prompt_tokens="
              << foreground_metrics.prompt_tokens << "\n"
              << "foreground_profile_prefilled_prompt_tokens="
              << foreground_metrics.prefilled_prompt_tokens << "\n"
              << "foreground_profile_prompt_suffix_tokens="
              << foreground_metrics.prompt_suffix_tokens << "\n"
              << "foreground_profile_generated_tokens="
              << foreground_metrics.generated_tokens << "\n"
              << "foreground_profile_prompt_build_ms="
              << foreground_metrics.prompt_build_ms << "\n"
              << "foreground_profile_prompt_prefill_ms="
              << foreground_metrics.prompt_prefill_ms << "\n"
              << "foreground_profile_first_token_delta_ms="
              << foreground_metrics.first_token_delta_ms << "\n"
              << "foreground_profile_first_content_delta_ms="
              << foreground_metrics.first_content_delta_ms << "\n"
              << "foreground_profile_first_tts_enqueue_ms="
              << foreground_metrics.first_tts_enqueue_ms << "\n"
              << "foreground_profile_decode_ms="
              << foreground_metrics.decode_ms << "\n"
              << "foreground_profile_model_ms="
              << foreground_metrics.model_ms << "\n"
              << "simulated_vad_asr_tail_ms=" << simulated_vad_asr_tail_ms << "\n"
              << "simulated_vad_to_first_content_ms="
              << simulated_vad_to_first_content_ms << "\n"
              << "simulated_vad_to_first_tts_enqueue_ms="
              << simulated_vad_to_first_tts_enqueue_ms << "\n"
              << "foreground_user_text=" << quote(ctx->foreground_pipeline->last_user_text()) << "\n"
              << "foreground_assistant_text=" << quote(assistant_text) << "\n"
              << "foreground_tool_call_json=" << quote(ctx->foreground_pipeline->last_tool_call_json()) << "\n"
              << "foreground_tool_result_text=" << quote(ctx->foreground_pipeline->last_tool_result_text()) << "\n"
              << "foreground_error=" << quote(ctx->foreground_pipeline->last_error()) << "\n";

    {
        std::lock_guard<std::mutex> lock(audio_capture.mutex);
        const aila::alia::AliaTtsMetrics tts_metrics =
            ctx->tts_pipeline->last_metrics();
        const double first_audio_ms = audio_capture.callback_count > 0
            ? static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                  audio_capture.first_audio - audio_capture.turn_start).count())
            : -1.0;
        const double simulated_vad_to_first_audio_ms = first_audio_ms >= 0.0
            ? simulated_vad_asr_tail_ms + first_audio_ms
            : -1.0;
        std::cout << "tts_callback_count=" << audio_capture.callback_count << "\n"
                  << "tts_first_audio_ms=" << first_audio_ms << "\n"
                  << "simulated_vad_to_first_audio_ms="
                  << simulated_vad_to_first_audio_ms << "\n"
                  << "tts_chunks_synthesized=" << tts_metrics.chunks_synthesized << "\n"
                  << "tts_reference_audio_enabled="
                  << tts_metrics.reference_audio_enabled << "\n"
                  << "tts_reference_embedding_dim="
                  << tts_metrics.reference_embedding_dim << "\n"
                  << "tts_reference_embedding_ms="
                  << tts_metrics.reference_embedding_ms << "\n"
                  << "tts_reference_audio_path="
                  << quote(tts_metrics.reference_audio_path) << "\n"
                  << "tts_reference_audio_error="
                  << quote(tts_metrics.reference_audio_error) << "\n"
                  << "tts_first_text_chars=" << tts_metrics.first_text_chars << "\n"
                  << "tts_first_text_tokens=" << tts_metrics.first_text_tokens << "\n"
                  << "tts_first_backend_frames=" << tts_metrics.first_backend_frames << "\n"
                  << "tts_first_backend_callbacks=" << tts_metrics.first_backend_callbacks << "\n"
                  << "tts_first_backend_audio_samples="
                  << tts_metrics.first_backend_audio_samples << "\n"
                  << "tts_first_backend_codes_ms="
                  << tts_metrics.first_backend_codes_ms << "\n"
                  << "tts_first_backend_mimi_init_ms="
                  << tts_metrics.first_backend_mimi_init_ms << "\n"
                  << "tts_first_backend_audio_ms="
                  << tts_metrics.first_backend_audio_ms << "\n"
                  << "tts_first_backend_total_ms="
                  << tts_metrics.first_backend_total_ms << "\n"
                  << "tts_backend_total_ms=" << tts_metrics.backend_total_ms << "\n"
                  << "tts_total_samples=" << audio_capture.samples.size() << "\n"
                  << "tts_nonzero_samples=" << audio_capture.nonzero_samples << "\n"
                  << "tts_chunk_sizes=";
        for (size_t i = 0; i < audio_capture.chunk_sizes.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << audio_capture.chunk_sizes[i];
        }
        std::cout << "\n";

        if (!save_capture_wav(opts.output_wav, audio_capture.samples)) {
            std::cerr << "tts_save_wav_failed=" << quote(opts.output_wav) << "\n";
            return 1;
        }
        std::cout << "tts_output_wav=" << quote(opts.output_wav) << "\n";
    }

    if (fg_state != aila::alia::ForegroundTurnState::Completed ||
        fg_mode != aila::alia::ForegroundDecodeMode::LoadedVlm ||
        assistant_text.empty() ||
        audio_capture.callback_count <= 0 ||
        audio_capture.nonzero_samples <= 0) {
        std::cerr << "foreground_or_tts_validation_failed=true\n";
        return 1;
    }

    {
        std::lock_guard<std::mutex> lock(g_background_mutex);
        g_background_json.clear();
    }
    alia_register_background_callback(ctx.get(), background_callback);
    const std::string turn_text = "User: " + user_text + "\nAssistant: " + assistant_text;
    const auto bg_start = Clock::now();
    rc = alia_trigger_background_processing(ctx.get(), turn_text.c_str());
    if (rc != ALIA_OK) {
        std::cerr << "alia_trigger_background_processing_rc=" << rc << "\n";
        return 1;
    }
    if (!ctx->background_pipeline->wait_until_idle_for(std::chrono::seconds(opts.timeout_sec))) {
        std::cerr << "background_timeout=true\n";
        alia_abort_inference(ctx.get(), ALIA_PIPELINE_VLM_BACKGROUND);
        ctx->background_pipeline->join();
        return 1;
    }
    ctx->background_pipeline->join();
    const auto bg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - bg_start).count();

    std::string callback_json;
    {
        std::lock_guard<std::mutex> lock(g_background_mutex);
        callback_json = g_background_json;
    }
    const auto bg_state = ctx->background_pipeline->state();
    const auto bg_mode = ctx->background_pipeline->last_decode_mode();
    const std::string bg_json = ctx->background_pipeline->last_result_json();
    const bool bg_schema_valid = aila::alia::AliaBackgroundPipeline::has_required_schema_keys(bg_json);
    std::cout << "background_ms=" << bg_ms << "\n"
              << "background_state=" << background_state_name(bg_state) << "\n"
              << "background_decode_mode=" << background_mode_name(bg_mode) << "\n"
              << "background_schema_valid=" << (bg_schema_valid ? "true" : "false") << "\n"
              << "background_retry_count=" << ctx->background_pipeline->last_schema_retry_count() << "\n"
              << "background_schema_repair_applied="
              << (ctx->background_pipeline->last_schema_repair_applied() ? "true" : "false") << "\n"
              << "background_schema_diagnostic="
              << quote(ctx->background_pipeline->last_schema_diagnostic()) << "\n"
              << "background_result_json=" << quote(bg_json) << "\n"
              << "background_callback_json=" << quote(callback_json) << "\n"
              << "background_error=" << quote(ctx->background_pipeline->last_error()) << "\n";

    if (bg_state != aila::alia::BackgroundJobState::Completed ||
        bg_mode != aila::alia::BackgroundDecodeMode::LoadedVlm ||
        !bg_schema_valid ||
        callback_json.empty()) {
        std::cerr << "background_validation_failed=true\n";
        return 1;
    }

    if (opts.run_tool_probe) {
        if (!run_foreground_tool_probe(*ctx, opts)) {
            return 1;
        }
    } else {
        std::cout << "tool_probe_skipped=true\n";
    }

    std::cout << "ALIA_REAL_MODEL_SMOKE_PASS\n";
    return 0;
}
