#include "engine/Engine.hpp"
#include "cli/CLI.hpp"
#include "bench/Benchmark.hpp"
#include "AudioPreprocessor.hpp"
#include "profile/Device.hpp"
#include "profile/Profiling.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
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
        std::string cache_dir = opts.tts_spk_cache_dir;
        if (cache_dir.empty()) {
            cache_dir = aila::env::read_string("AILA_SPK_CACHE_DIR", "");
        }
        if (!cache_dir.empty()) {
            engine.setSpeakerCacheDir(cache_dir);
        }
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

        if (opts.stream_output) {
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

        std::string output_path = opts.tts_output_path.empty() ? "output.wav" : opts.tts_output_path;
        AILA_LOG_INFO("Synthesizing text: \"%s\"", input_text.c_str());

        GenerationConfig tts_gen = gen_config;
        // TTS needs higher repetition_penalty to avoid loops
        if (tts_gen.repetition_penalty <= 1.0f) tts_gen.repetition_penalty = 1.1f;
        tts_gen.max_new_tokens = opts.max_new_tokens > 0 ? opts.max_new_tokens : tts_gen.max_new_tokens;

        std::vector<float> samples;
        bool ok = engine.synthesizeSpeech(
            input_text,
            opts.tts_speaker_path,    // Base: reference audio
            opts.tts_speaker_name,    // CustomVoice: speaker name
            opts.tts_instruct_text,   // VoiceDesign: instruct
            opts.tts_language,        // language
            tts_gen,
            samples
        );

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

    // Run interactive loop
    return run_interactive(engine, gen_config, opts.stream_output, opts.tts_speaker_path);
}
