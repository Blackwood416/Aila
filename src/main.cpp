#include "engine/Engine.hpp"
#include "cli/CLI.hpp"
#include "chat/ChatJson.hpp"
#include "chat/ChatStreamJson.hpp"
#include "bench/Benchmark.hpp"
#include "AudioPreprocessor.hpp"
#include "profile/Device.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "vision/DetectionRenderer.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <cstdio>

namespace {

std::string json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string output = "\"";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character < 0x20) {
                    output += "\\u00";
                    output += hex[character >> 4]; output += hex[character & 15];
                } else output += static_cast<char>(character);
        }
    }
    output += '"';
    return output;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = (std::min)(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

} // namespace
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <clocale>
#include <windows.h>
#include <shellapi.h>
#include <vector>
#include <string>
#pragma comment(lib, "shell32.lib")

static std::vector<std::string> g_utf8_args;
static std::vector<char*> g_utf8_argv;

void convert_args_to_utf8(int& argc, char**& argv) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr) {
        return;
    }

    g_utf8_args.clear();
    g_utf8_args.resize(wargc);
    g_utf8_argv.clear();
    g_utf8_argv.resize(wargc + 1);

    for (int i = 0; i < wargc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (len > 0) {
            g_utf8_args[i].resize(len - 1);
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &g_utf8_args[i][0], len, nullptr, nullptr);
        }
        g_utf8_argv[i] = &g_utf8_args[i][0];
    }
    g_utf8_argv[wargc] = nullptr;
    LocalFree(wargv);

    argc = wargc;
    argv = g_utf8_argv.data();
}
#endif

int main(int argc, char** argv) {
    if (aila::env::disable_persistent_sycl_cache()) {
        std::fprintf(
            stderr,
            "[Aila] SYCL_CACHE_PERSISTENT was inherited enabled; forcing 0 "
            "(known runtime crash; set AILA_KEEP_SYCL_CACHE_PERSISTENT=1 to keep)\n");
    }
#ifdef _WIN32
    // 如果 stdin/stdout/stderr 被重定向（非交互模式），设置为二进制模式，
    // 从而防止 Windows CRT 做本地化字符集转换和换行符翻译（如 \n -> \r\n），确保管道中传输的是原始 UTF-8 字节。
    if (_isatty(_fileno(stdin)) == 0) {
        _setmode(_fileno(stdin), _O_BINARY);
    }
    if (_isatty(_fileno(stdout)) == 0) {
        _setmode(_fileno(stdout), _O_BINARY);
    }
    if (_isatty(_fileno(stderr)) == 0) {
        _setmode(_fileno(stderr), _O_BINARY);
    }
    // 配合强制当前 CRT 使用 UTF-8 区域设置
    std::setlocale(LC_ALL, ".UTF-8");

    // 将 Windows 宽字符命令行参数转为 UTF-8 覆盖当前的 argv，
    // 从而使 CLI 所有命令行参数在读取中文或其它非 ASCII 字符时天然是合法的 UTF-8，彻底解决命令行中文参数乱码问题。
    convert_args_to_utf8(argc, argv);
#endif

    // Parse command line arguments
    CLIOptions opts;
    if (!parse_cli_args(argc, argv, opts)) {
        return 1;
    }

    if (opts.show_help) {
        print_help();
        return 0;
    }

    if (opts.show_version) {
        print_version();
        return 0;
    }

    // Validate model directory
    if (opts.model_dir.empty()) {
        std::cerr << "Error: Model directory is required." << std::endl;
        std::cerr << "Use -m <path> or set AILA_MODEL_DIR environment variable." << std::endl;
        std::cerr << "Use --help for usage information." << std::endl;
        return 1;
    }

    // Apply log level from CLI / environment
    aila::set_log_level(opts.log_level);

    // Check GPU device
    CheckDevice();
    AILA_LOG_INFO("[Config] log_level=%s max_seq_len=%d",
                  aila::log_level_name(opts.log_level), opts.max_seq_len);

    // Initialize engine
    InferenceEngine engine;
    if (!engine.init(opts.model_dir, opts.max_seq_len, opts.lora_dir)) {
        AILA_LOG_ERROR("Failed to initialize inference engine");
        return 1;
    }

    // Configure speaker embedding cache
    {
        std::string cache_dir = opts.tts_ref_cache_dir;
        if (cache_dir.empty()) {
            cache_dir = aila::env::read_string("AILA_REF_CACHE_DIR", "");
        }
        if (!cache_dir.empty()) {
            engine.setRefCacheDir(cache_dir);
        }
    }

    // YOLO26 single-image detection. With --bench, the same image is used for
    // warmup and measured full-pipeline iterations.
    if (!opts.detect_path.empty()) {
        DetectionConfig detection_config;
        detection_config.confidence_threshold = opts.detection_confidence;
        detection_config.max_detections = opts.detection_max;
        std::vector<Detection> detections;
        std::vector<double> full_times, preprocess_times, device_times, postprocess_times;
        const int warmups = opts.bench_mode ? (std::max)(0, opts.bench_warmup) : 0;
        const int iterations = opts.bench_mode ? (std::max)(1, opts.bench_iters) : 1;
        for (int iteration = -warmups; iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            detections = engine.detect_file(opts.detect_path, detection_config);
            const auto end = std::chrono::steady_clock::now();
            if (engine.last_error_code() != EngineErrorCode::Ok) {
                AILA_LOG_ERROR("Object detection failed: %s", engine.last_error_message().c_str());
                return 2;
            }
            if (iteration >= 0) {
                full_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                preprocess_times.push_back(engine.last_detect_preprocess_ms());
                device_times.push_back(engine.last_detect_device_wall_ms());
                postprocess_times.push_back(engine.last_detect_postprocess_ms());
            }
        }
        if (!opts.detection_output.empty()) {
            std::string render_error;
            if (!aila::vision::save_detection_png(
                    opts.detect_path, opts.detection_output, detections, &render_error)) {
                AILA_LOG_ERROR("Failed to save detection image: %s", render_error.c_str());
                return 2;
            }
        }
        const Yolo26Config& model = engine.model_spec().yolo26;
        std::ostringstream json;
        json << std::setprecision(9)
             << "{\"width\":" << engine.last_detect_image_width()
             << ",\"height\":" << engine.last_detect_image_height()
             << ",\"model_scale\":" << json_string(model.scale)
             << ",\"num_classes\":" << model.num_classes
             << ",\"detections\":[";
        for (size_t index = 0; index < detections.size(); ++index) {
            const Detection& value = detections[index];
            if (index) json << ',';
            json << "{\"x1\":" << value.x1 << ",\"y1\":" << value.y1
                 << ",\"x2\":" << value.x2 << ",\"y2\":" << value.y2
                 << ",\"confidence\":" << value.confidence
                 << ",\"class_id\":" << value.class_id
                 << ",\"class_name\":" << json_string(value.class_name) << '}';
        }
        json << ']';
        if (opts.bench_mode) {
            json << ",\"benchmark\":{\"iterations\":" << iterations
                 << ",\"warmup_iterations\":" << warmups
                 << ",\"full_pipeline_ms\":{\"median\":" << percentile(full_times, 0.5)
                 << ",\"p10\":" << percentile(full_times, 0.1)
                 << ",\"p90\":" << percentile(full_times, 0.9) << "}"
                 << ",\"preprocess_ms_median\":" << percentile(preprocess_times, 0.5)
                 << ",\"device_wall_ms_median\":" << percentile(device_times, 0.5)
                 << ",\"postprocess_ms_median\":" << percentile(postprocess_times, 0.5)
                 << ",\"peak_device_bytes\":" << engine.last_detect_peak_device_bytes()
                 << '}';
        }
        json << '}';
        std::cout << json.str() << '\n';
        return 0;
    }

    // Benchmark mode
    if (opts.bench_mode) {
        BenchmarkConfig bench_cfg;
        bench_cfg.prompt_length = opts.bench_pp;
        bench_cfg.gen_length    = opts.bench_tg;
        bench_cfg.bench_iters   = opts.bench_iters;
        bench_cfg.warmup_iters  = opts.bench_warmup;
        bench_cfg.decode_do_sample = opts.bench_sample;
        bench_cfg.decode_gen_config.max_new_tokens = opts.bench_tg;
        bench_cfg.decode_gen_config.temperature = opts.temperature;
        bench_cfg.decode_gen_config.top_k = opts.top_k;
        bench_cfg.decode_gen_config.top_p = opts.top_p;
        bench_cfg.decode_gen_config.do_sample = opts.bench_sample;
        bench_cfg.decode_gen_config.repetition_penalty = opts.repetition_penalty;
        bench_cfg.decode_gen_config.presence_penalty = opts.presence_penalty;
        bench_cfg.decode_gen_config.frequency_penalty = opts.frequency_penalty;
        bench_cfg.decode_gen_config.sampling_seed = opts.sampling_seed;
        bench_cfg.decode_gen_config.use_fixed_seed = opts.use_fixed_seed || opts.bench_sample;
        bench_cfg.decode_gen_config.decode_chunk_size = opts.decode_chunk_size;
        bench_cfg.decode_gen_config.stream_chunk_size = opts.stream_chunk_size;

        auto result = run_benchmark(engine, bench_cfg);
        print_benchmark_results(result);
        return 0;
    }

    // Build generation config from CLI options
    GenerationConfig gen_config;
    gen_config.max_new_tokens    = opts.max_new_tokens;
    gen_config.temperature       = opts.temperature;
    gen_config.top_k             = opts.top_k;
    gen_config.top_p             = opts.top_p;
    gen_config.do_sample         = opts.do_sample;
    gen_config.sampling_seed     = opts.sampling_seed;
    gen_config.use_fixed_seed    = opts.use_fixed_seed;
    gen_config.decode_chunk_size = opts.decode_chunk_size;
    gen_config.stream_chunk_size = opts.stream_chunk_size;
    gen_config.thinking_budget_tokens = opts.thinking_budget_tokens;
    gen_config.repetition_penalty = opts.repetition_penalty;
    gen_config.presence_penalty   = opts.presence_penalty;
    gen_config.frequency_penalty  = opts.frequency_penalty;

    // Single-shot messages JSON mode
    if (!opts.messages_json_path.empty()) {
        std::string messages_json;
        if (opts.messages_json_path == "-") {
            messages_json.assign(std::istreambuf_iterator<char>(std::cin),
                                 std::istreambuf_iterator<char>());
        } else {
            std::ifstream in(opts.messages_json_path, std::ios::binary);
            if (!in.is_open()) {
                AILA_LOG_ERROR("Failed to open messages JSON file: %s", opts.messages_json_path.c_str());
                return 1;
            }
            messages_json.assign(std::istreambuf_iterator<char>(in),
                                 std::istreambuf_iterator<char>());
        }
        if (messages_json.empty()) {
            AILA_LOG_ERROR("Messages JSON input is empty");
            return 1;
        }

        if (opts.chat_output_json && opts.chat_stream_jsonl) {
            AILA_LOG_ERROR("--chat-output-json and --chat-stream-jsonl cannot be used together");
            return 1;
        }

        // 剔除可能存在的 UTF-8 BOM (\xEF\xBB\xBF)
        if (messages_json.size() >= 3 &&
            static_cast<unsigned char>(messages_json[0]) == 0xEF &&
            static_cast<unsigned char>(messages_json[1]) == 0xBB &&
            static_cast<unsigned char>(messages_json[2]) == 0xBF) {
            messages_json = messages_json.substr(3);
        }

        if (opts.chat_stream_jsonl) {
            aila::chat::ChatRequest request;
            std::string parse_error;
            if (!aila::chat::parse_chat_request_json(messages_json, gen_config, request, &parse_error)) {
                AILA_LOG_ERROR("messages-json parse failed: %s", parse_error.c_str());
                return 1;
            }
            int rc = engine.generate_chat_request_stream(
                request,
                [](const aila::chat::StructuredStreamEvent& event) {
                    std::cout << aila::chat::stream_event_to_json(event) << std::endl;
                    return true;
                });
            if (rc != 0) {
                AILA_LOG_ERROR("messages-json structured stream failed: %s",
                               engine.last_error_message().c_str());
                return 1;
            }
        } else if (opts.chat_output_json) {
            if (opts.stream_output) {
                AILA_LOG_WARN("--chat-output-json disables token streaming so stdout remains valid JSON");
            }
            std::string out = engine.generate_chat_json(messages_json, gen_config, nullptr);
            if (engine.last_error_code() != EngineErrorCode::Ok) {
                AILA_LOG_ERROR("chat JSON generation failed: %s",
                               engine.last_error_message().c_str());
                return 2;
            }
            std::cout << out << std::endl;
        } else if (opts.stream_output) {
            std::cout << "\nAila: ";
            engine.generate_messages_json(messages_json, gen_config,
                [](const std::string& token_text) {
                    std::cout << token_text << std::flush;
                });
            if (engine.last_error_code() != EngineErrorCode::Ok) {
                AILA_LOG_ERROR("messages-json generation failed: %s",
                               engine.last_error_message().c_str());
                return 2;
            }
            std::cout << std::endl;
        } else {
            std::string out = engine.generate_messages_json(messages_json, gen_config, nullptr);
            if (engine.last_error_code() != EngineErrorCode::Ok) {
                AILA_LOG_ERROR("messages-json generation failed: %s",
                               engine.last_error_message().c_str());
                return 2;
            }
            std::cout << out << std::endl;
        }
        return 0;
    }

    // ASR transcription mode
    if (!opts.transcribe_path.empty()) {
        gen_config.do_sample = false;  // greedy for ASR
        std::string lang;
        std::string transcript;

        std::function<void(const std::string&)> token_cb = nullptr;
        if (opts.stream_output) {
            token_cb = [](const std::string& token_text) {
                std::cout << token_text << std::flush;
            };
        }

        transcript = engine.transcribe(
            opts.transcribe_path,
            gen_config,
            &lang,
            opts.forced_language,
            opts.system_prompt,
            opts.segment_sec,
            opts.past_text_conditioning,
            token_cb
        );

        if (engine.last_error_code() != EngineErrorCode::Ok) {
            AILA_LOG_ERROR("Transcription failed: %s", engine.last_error_message().c_str());
            return 2;
        }

        if (opts.stream_output) {
            std::cout << std::endl;
        } else {
            std::cout << transcript << std::endl;
        }

        if (!lang.empty()) {
            std::cout << "[Language] " << lang << std::endl;
        }

        double audio_s = engine.last_transcribe_duration_s();
        double latency_ms = engine.last_transcribe_latency_ms();
        int tokens = engine.last_transcribe_tokens();
        double speed = latency_ms > 0 ? (audio_s / (latency_ms / 1000.0)) : 0.0;
        double tok_s = latency_ms > 0 ? (static_cast<double>(tokens) / (latency_ms / 1000.0)) : 0.0;

        std::cout << "[Audio: " << std::fixed << std::setprecision(1) << audio_s << "s, "
                  << "Latency: " << std::fixed << std::setprecision(0) << latency_ms << "ms, "
                  << "Speed: " << std::fixed << std::setprecision(1) << speed << "x, "
                  << tok_s << " tok/s]" << std::endl;

        return 0;
    }

    // TTS synthesis mode
    if (!opts.tts_text.empty()) {
        if (engine.model_spec().family != ModelFamily::Qwen3TTS) {
            AILA_LOG_ERROR("The loaded model does not support TTS! Please load a Qwen3-TTS model.");
            return 2;
        }

        std::string input_text = trim(normalize_input_for_model(opts.tts_text));
        if (input_text.empty()) {
            AILA_LOG_ERROR("TTS synthesis failed: input text is empty after normalization");
            return 2;
        }

        GenerationConfig tts_gen = gen_config;
        if (tts_gen.repetition_penalty <= 1.0f) tts_gen.repetition_penalty = 1.1f;
        tts_gen.max_new_tokens = opts.max_new_tokens > 0 ? opts.max_new_tokens : tts_gen.max_new_tokens;

        if (opts.tts_stream) {
            // Streaming mode: output raw PCM float chunks to stdout
            // Redirect logs to stderr so PCM stays clean on stdout
            aila::set_log_callback([](int /*level*/, const char* msg, void*) {
                fputs(msg, stderr);
                fputc('\n', stderr);
                fflush(stderr);
            }, nullptr);
            AILA_LOG_INFO("Streaming TTS: \"%s\" (raw PCM to stdout)", input_text.c_str());
#ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            // Batch size: CLI > env AILA_TTS_STREAM_BATCH > default 4
            int batch = opts.tts_stream_batch;
            if (batch <= 1) {
                batch = aila::env::read_int_raw("AILA_TTS_STREAM_BATCH", 4);
                if (batch <= 1) batch = 4;
            }
            // Accumulate samples for optional WAV output
            std::mutex sample_mutex;
            std::vector<float> all_samples;
            VoiceCloneMode clone_mode = VoiceCloneMode::Auto;
            if (opts.tts_voice_clone_mode == "icl") clone_mode = VoiceCloneMode::Icl;
            else if (opts.tts_voice_clone_mode == "xvector-only" || opts.tts_voice_clone_mode == "xvector") clone_mode = VoiceCloneMode::XVectorOnly;

            auto worker = engine.synthesizeSpeechStream(
                input_text, opts.tts_reference_path, opts.tts_speaker_name,
                opts.tts_instruct_text, opts.tts_language, tts_gen,
                [&](const float* samples, int count) {
                    if (count > 0) {
                        fwrite(samples, sizeof(float), static_cast<size_t>(count), stdout);
                        fflush(stdout);
                        std::lock_guard<std::mutex> lock(sample_mutex);
                        all_samples.insert(all_samples.end(), samples, samples + count);
                    }
                }, batch, opts.tts_ref_text, clone_mode);
            worker.join();

            // Save WAV if requested
            std::string output_path = opts.tts_output_path;
            if (!output_path.empty() && !all_samples.empty()) {
                if (!save_wav(output_path, all_samples, 24000)) {
                    AILA_LOG_ERROR("Failed to write WAV file to: %s", output_path.c_str());
                    return 2;
                }
                AILA_LOG_INFO("Streaming audio saved to: %s (%.2fs)",
                              output_path.c_str(),
                              static_cast<double>(all_samples.size()) / 24000.0);
            }
            AILA_LOG_INFO("TTS streaming complete");
            return 0;
        }

        // Blocking mode: write to WAV file
        std::string output_path = opts.tts_output_path.empty() ? "output.wav" : opts.tts_output_path;
        AILA_LOG_INFO("Synthesizing text: \"%s\"", input_text.c_str());

        std::vector<float> samples;
        VoiceCloneMode clone_mode = VoiceCloneMode::Auto;
        if (opts.tts_voice_clone_mode == "icl") clone_mode = VoiceCloneMode::Icl;
        else if (opts.tts_voice_clone_mode == "xvector-only" || opts.tts_voice_clone_mode == "xvector") clone_mode = VoiceCloneMode::XVectorOnly;

        bool ok = engine.synthesizeSpeech(
            input_text, opts.tts_reference_path, opts.tts_speaker_name,
            opts.tts_instruct_text, opts.tts_language, tts_gen, samples,
            opts.tts_ref_text, clone_mode);

        if (!ok) {
            AILA_LOG_ERROR("TTS synthesis failed: %s", engine.last_error_message().c_str());
            return 2;
        }

        if (!save_wav(output_path, samples, 24000)) {
            AILA_LOG_ERROR("Failed to write WAV file to: %s", output_path.c_str());
            return 2;
        }

        AILA_LOG_INFO("TTS synthesis succeeded! Output saved to: %s (Samples: %zu, Duration: %.2fs)",
                      output_path.c_str(), samples.size(), static_cast<double>(samples.size()) / 24000.0);
        return 0;
    }

    // ForceAligner mode
    if (!opts.align_audio.empty()) {
        if (engine.model_spec().family != ModelFamily::Qwen3ForceAligner) {
            AILA_LOG_ERROR("The loaded model does not support forced alignment! Please load a Qwen3-ForceAligner model.");
            return 2;
        }
        if (opts.align_text.empty()) {
            AILA_LOG_ERROR("--align-text is required for forced alignment mode");
            return 2;
        }

        // Load audio file
        AudioBuffer audio;
        std::string load_err;
        if (!load_audio(opts.align_audio, audio, &load_err)) {
            AILA_LOG_ERROR("Failed to load alignment audio: %s", load_err.c_str());
            return 2;
        }

        // Convert to mono
        std::vector<float> mono(audio.samples.size() / audio.channels);
        if (audio.channels > 1) {
            for (size_t i = 0; i < mono.size(); ++i) {
                float sum = 0;
                for (int c = 0; c < audio.channels; ++c)
                    sum += audio.samples[i * audio.channels + c];
                mono[i] = sum / audio.channels;
            }
        } else {
            mono = std::move(audio.samples);
        }

        auto result = engine.align(mono, audio.sample_rate,
                                   opts.align_text, opts.align_language);

        if (engine.last_error_code() != EngineErrorCode::Ok) {
            AILA_LOG_ERROR("Alignment failed: %s", engine.last_error_message().c_str());
            return 2;
        }

        for (const auto& w : result) {
            std::cout << "  \"" << w.text << "\"  " << w.start_ms << "ms - " << w.end_ms << "ms" << std::endl;
        }
        return 0;
    }

    // Run interactive loop
    return run_interactive(engine, gen_config, opts.stream_output, opts.tts_reference_path);
}
