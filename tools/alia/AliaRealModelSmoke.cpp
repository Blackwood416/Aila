#include "alia_api.h"
#include "alia/AliaBackgroundPipeline.hpp"
#include "alia/AliaContext.hpp"
#include "alia/AliaForegroundPipeline.hpp"
#include "audio/AudioPreprocessor.hpp"

#include <algorithm>
#include <chrono>
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
    std::string foreground_lora;
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
    int rollback_tokens = 0;
    bool enforce_target_models = true;
    bool generate_audio_if_missing = true;
    bool run_tool_probe = false;
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
        << "  --tool-probe         run a real foreground VLM tool-call probe\n"
        << "  --max-seq <N>          default 2048\n"
        << "  --max-tokens <N>       default 48\n"
        << "  --timeout-sec <N>      default 1500\n"
        << "  --rollback-tokens <N>  default 0, set >0 for optional rollback probe\n"
        << "  --no-generate-audio    fail if --audio is missing instead of using target TTS\n"
        << "  --allow-non-target-models\n";
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
        } else if (arg == "--tool-probe") {
            opts.run_tool_probe = true;
        } else if (arg == "--max-seq") {
            if (!require_int(opts.max_seq_len)) return false;
        } else if (arg == "--max-tokens") {
            if (!require_int(opts.max_tokens)) return false;
        } else if (arg == "--timeout-sec") {
            if (!require_int(opts.timeout_sec)) return false;
        } else if (arg == "--rollback-tokens") {
            if (!require_int(opts.rollback_tokens)) return false;
        } else if (arg == "--no-generate-audio") {
            opts.generate_audio_if_missing = false;
        } else if (arg == "--allow-non-target-models") {
            opts.enforce_target_models = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    if (opts.max_seq_len <= 0 || opts.max_tokens <= 0 || opts.timeout_sec <= 0) {
        std::cerr << "max-seq, max-tokens, and timeout-sec must be positive.\n";
        return false;
    }
    if (opts.request_text.empty()) {
        std::cerr << "request-text must not be empty.\n";
        return false;
    }
    if (opts.run_tool_probe && opts.tool_probe_text.empty()) {
        std::cerr << "tool-probe-text must not be empty when --tool-probe is set.\n";
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
        case aila::alia::ForegroundDecodeMode::NoModelFallback: return "NoModelFallback";
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
        case aila::alia::BackgroundDecodeMode::NoModelFallback: return "NoModelFallback";
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
    if (opts.enforce_target_models && !validate_target_models(opts)) {
        std::cerr << "target_model_enforcement_failed=true\n";
        return 2;
    }

    std::cout << "ALIA_REAL_MODEL_SMOKE_BEGIN\n"
              << "target_model_enforcement="
              << (opts.enforce_target_models ? "true" : "false") << "\n"
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
    int rc = alia_asr_feed_audio(ctx.get(), mono_16k.data(), static_cast<int>(mono_16k.size()));
    if (rc != ALIA_OK) {
        std::cerr << "alia_asr_feed_audio_rc=" << rc << "\n";
        return 1;
    }
    char* stable_raw = nullptr;
    char* partial_raw = nullptr;
    rc = alia_asr_get_text(ctx.get(), &stable_raw, &partial_raw);
    std::string stable_text = stable_raw ? stable_raw : "";
    std::string partial_text = partial_raw ? partial_raw : "";
    std::free(stable_raw);
    std::free(partial_raw);
    const auto asr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - asr_start).count();
    const std::string user_text = !stable_text.empty() ? stable_text : partial_text;
    std::cout << "asr_ms=" << asr_ms << "\n"
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
    std::cout << "foreground_ms=" << fg_ms << "\n"
              << "foreground_state=" << foreground_state_name(fg_state) << "\n"
              << "foreground_decode_mode=" << foreground_mode_name(fg_mode) << "\n"
              << "foreground_user_text=" << quote(ctx->foreground_pipeline->last_user_text()) << "\n"
              << "foreground_assistant_text=" << quote(assistant_text) << "\n"
              << "foreground_tool_call_json=" << quote(ctx->foreground_pipeline->last_tool_call_json()) << "\n"
              << "foreground_tool_result_text=" << quote(ctx->foreground_pipeline->last_tool_result_text()) << "\n"
              << "foreground_error=" << quote(ctx->foreground_pipeline->last_error()) << "\n";

    {
        std::lock_guard<std::mutex> lock(audio_capture.mutex);
        const double first_audio_ms = audio_capture.callback_count > 0
            ? static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                  audio_capture.first_audio - audio_capture.turn_start).count())
            : -1.0;
        std::cout << "tts_callback_count=" << audio_capture.callback_count << "\n"
                  << "tts_first_audio_ms=" << first_audio_ms << "\n"
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

    if (opts.rollback_tokens > 0) {
        const auto rollback_start = Clock::now();
        rc = alia_vlm_rollback_kv_cache(ctx.get(), opts.rollback_tokens);
        const auto rollback_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - rollback_start).count();
        std::cout << "rollback_tokens=" << opts.rollback_tokens << "\n"
                  << "rollback_rc=" << rc << "\n"
                  << "rollback_ms=" << rollback_ms << "\n";
        if (rc != ALIA_OK) {
            return 1;
        }
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

    if (opts.run_tool_probe && !run_foreground_tool_probe(*ctx, opts)) {
        return 1;
    }

    std::cout << "ALIA_REAL_MODEL_SMOKE_PASS\n";
    return 0;
}
