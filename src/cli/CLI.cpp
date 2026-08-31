#include "CLI.hpp"
#include "engine/Engine.hpp"
#include "profile/Profiling.hpp"
#include "AudioPreprocessor.hpp"
#include "SpeakerEncoder.hpp"
#include "utils/EnvUtils.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <stdio.h>
#include <clocale>
#else
#include <unistd.h>
#endif

// ============================================================
// Console utilities (platform-specific)
// ============================================================
namespace {

bool detect_interactive_terminal() {
#ifdef _WIN32
    return (_isatty(_fileno(stdin)) != 0) && (_isatty(_fileno(stdout)) != 0);
#else
    return isatty(fileno(stdin)) && isatty(fileno(stdout));
#endif
}

#ifdef _WIN32
bool is_valid_utf8(const std::string& s) {
    int expected = 0;
    for (unsigned char c : s) {
        if (expected == 0) {
            if ((c >> 7) == 0) continue;
            if ((c >> 5) == 0x6) expected = 1;
            else if ((c >> 4) == 0xE) expected = 2;
            else if ((c >> 3) == 0x1E) expected = 3;
            else return false;
        } else {
            if ((c >> 6) != 0x2) return false;
            --expected;
        }
    }
    return expected == 0;
}

std::string codepage_to_utf8(const std::string& text, UINT codepage) {
    if (text.empty()) return text;
    int wlen = MultiByteToWideChar(codepage, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (wlen <= 0) return text;

    std::wstring wide(static_cast<size_t>(wlen), L'\0');
    if (MultiByteToWideChar(codepage, 0, text.data(), static_cast<int>(text.size()),
                            wide.data(), wlen) <= 0) {
        return text;
    }

    int u8len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wlen, nullptr, 0, nullptr, nullptr);
    if (u8len <= 0) return text;

    std::string out(static_cast<size_t>(u8len), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wide.data(), wlen, out.data(), u8len, nullptr, nullptr) <= 0) {
        return text;
    }
    return out;
}
#endif

#ifdef _WIN32
void setup_console_utf8(bool interactive_terminal) {
    if (!interactive_terminal) return;
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    std::setlocale(LC_ALL, ".UTF-8");
}
#else
void setup_console_utf8(bool /*interactive_terminal*/) {}
#endif

} // namespace

#ifdef _WIN32
std::string normalize_input_for_model(const std::string& text) {
    if (text.empty() || is_valid_utf8(text)) return text;

    UINT cp = GetConsoleCP();
    if (cp == 0 || cp == CP_UTF8) cp = GetACP();
    std::string converted = codepage_to_utf8(text, cp);
    if (is_valid_utf8(converted) && !converted.empty() && converted != text) return converted;

    converted = codepage_to_utf8(text, GetACP());
    return converted;
}
#else
std::string normalize_input_for_model(const std::string& text) {
    return text;
}
#endif

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::vector<float> load_or_extract_speaker_embedding(InferenceEngine* engine, const std::string& path) {
    std::vector<float> embedding;
    if (!engine || path.empty()) return embedding;

    // Check if the path is an audio file (extension-based heuristic)
    bool is_audio = false;
    if (path.size() >= 4) {
        std::string ext = path.substr(path.size() - 4);
        for (char& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".wav" || ext == ".mp3" || ext == "flac") {
            is_audio = true;
        }
    }

    if (is_audio) {
        // Determine the model's embedding dimension first (fast mmap load)
        std::string safetensors_path = engine->model_dir() + "/model.safetensors";
        aila::audio::SpeakerEncoder encoder;
        std::string error;
        if (!encoder.loadWeights(safetensors_path, &error)) {
            std::cerr << "[TTS] Error: Failed to load speaker encoder weights: " << error << std::endl;
            return embedding;
        }
        int spk_dim = encoder.embeddingDim();

        // Check cache with correct dimension
        if (engine->lookupRefCache(path, spk_dim, embedding)) {
            std::cout << "[TTS] Speaker embedding loaded from cache (Dimension: "
                      << embedding.size() << ")" << std::endl;
            return embedding;
        }

        // Cache miss — extract with CPU ECAPA-TDNN (f32 precision)
        std::cout << "[TTS] Extracting speaker embedding (CPU, dim=" << spk_dim << ")..." << std::endl;
        if (encoder.extractEmbeddingFromFile(path, embedding, &error)) {
            engine->cacheRefEmbedding(path, embedding);
            std::cout << "[TTS] Speaker embedding extracted and cached (Dimension: "
                      << embedding.size() << ")" << std::endl;
            return embedding;
        }
        std::cerr << "[TTS] Error: Speaker embedding extraction failed: " << error << std::endl;
        return embedding;
    }

    // Otherwise, treat path as a .bin file containing pre-extracted embedding
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[TTS] Error: Failed to open speaker embedding file: " << path << std::endl;
        return embedding;
    }

    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size <= 0 || size % sizeof(float) != 0) {
        std::cerr << "[TTS] Error: Invalid speaker embedding file size: " << size << " bytes" << std::endl;
        return embedding;
    }

    size_t count = static_cast<size_t>(size) / sizeof(float);
    embedding.resize(count);
    if (!in.read(reinterpret_cast<char*>(embedding.data()), size)) {
        std::cerr << "[TTS] Error: Failed to read speaker embedding data" << std::endl;
        embedding.clear();
    } else {
        std::cout << "[TTS] Loaded speaker embedding from: " << path
                  << " (Dimension: " << count << ")" << std::endl;
    }

    return embedding;
}

// ============================================================
// CLI Argument Parsing
// ============================================================

void print_help() {
    std::cout <<
R"(Aila - SYCL + oneDNN LLM Inference Engine

Usage: aila [options]

Options:
  -m, --model <path>       Model directory (required, or set AILA_MODEL_DIR)
  -s, --max-seq <N>        Maximum sequence length (default: 4096, or AILA_MAX_SEQ_LEN)
  -t, --temperature <F>    Sampling temperature (default: 0.7)
  -k, --top-k <N>          Top-K sampling (default: 15)
  -p, --top-p <F>          Top-P (nucleus) sampling (default: 0.95)
  --seed <N>               Sampling RNG seed (enables fixed-seed mode)
  --greedy                 Use greedy decoding
  --sample                 Use sampling (default)
  --stream                 Force streaming output
  --no-stream              Force non-streaming output
  --max-tokens <N>         Maximum new tokens (default: 1024)
  --thinking-budget <N>    Thinking token budget: -1=off, 0=no-think, >0 cap
  --decode-chunk <N>       Decode chunk size (default: 12)
  --stream-chunk <N>       Stream chunk size (default: 4)
  --rep-penalty <F>        Repetition penalty (default: 1.0, >1.0 to penalize)
  --pres-penalty <F>       Presence penalty (default: 0.0)
  --freq-penalty <F>       Frequency penalty (default: 0.0)
  --detect <image>         Run YOLO26 object detection on one image
  --conf <F>               Detection confidence threshold in [0,1] (default: 0.25)
  --max-det <N>            Maximum detections in [1,300] (default: 300)
  --save-detect <png>      Save an annotated PNG at the original image size
  --bench                  Run benchmark mode
  --bench-pp <N>           Benchmark prompt length (default: 512)
  --bench-tg <N>           Benchmark generation length (default: 128)
  --bench-iters <N>        Benchmark iterations (default: 5)
  --bench-warmup <N>       Benchmark warmup iterations (default: 1)
  --bench-sample           Benchmark decode in sampling mode
  --bench-greedy           Benchmark decode in greedy mode (default)
  --log-level <level>      Minimum log level (debug/info/warning/error, default: info)
  --messages-json <path>   Single-shot generation from OpenAI-style messages JSON file ('-' = stdin)
  --chat-output-json       With --messages-json, print structured assistant JSON instead of raw text
  --chat-stream-jsonl      With --messages-json, print structured stream events as JSONL
  --lora <path>              LoRA adapter directory (or set AILA_LORA_DIR)
  --forced-lang <lang>     Force ASR language (e.g. Chinese, English)
  --asr-system <prompt>    ASR system prompt text bias
  --asr-segment <sec>      ASR segment split duration in seconds (default: 0.0, disabled)
  --asr-past               Enable past-text conditioning history for ASR segments
  --no-asr-past            Disable past-text conditioning for ASR segments (default)
  --transcribe <path>      Offline audio transcription (ASR) file path
  --synthesize <prompt>    TTS voice synthesis from text prompt (Qwen3-TTS only)
  --output-wav <path>      Output path for synthesized WAV file (default: output.wav)
  --ref <path>             Reference audio for TTS voice cloning (Base model)
  --ref-text <text>        Reference transcript for ICL voice clone (Base model)
  --voice-clone-mode <m>   Voice clone mode: auto, icl, xvector-only (default: auto)
  --instruct <text>        Voice design / style description text (VoiceDesign model)
  --speaker <name>         CustomVoice speaker name (vivian, ryan, serena, ...)
  --language <lang>        Language code: chinese, english, japanese, korean (default: auto)
  --ref-cache-dir <dir>    Reference embedding cache directory (default: alongside audio)
  --align-text <text>      ForceAligner: transcript text to align
  --align-audio <path>     ForceAligner: audio file path for alignment
  --align-lang <lang>      ForceAligner: language (default: Chinese)
  --q35-prefill-step <N>   Qwen3.5 prefill checkpoint step (default: 64)
  --kv-quant               Enable KV cache quantization (FP8, E4M3)
  -h, --help               Show this help message
  -v, --version            Show version

Environment Variables:
  AILA_MODEL_DIR           Default model directory
  AILA_MAX_SEQ_LEN         Default max sequence length
  AILA_LOG_LEVEL           Default log level (debug/info/warning/error, default: info)
  AILA_STREAM_OUTPUT       Force stream mode (0/1)
  AILA_DECODE_CHUNK_SIZE   Default decode chunk size
  AILA_STREAM_CHUNK_SIZE   Default stream chunk size
  AILA_THINKING_BUDGET     Default thinking budget (-1=off, 0=no-think)
  AILA_Q35_PREFILL_STEP    Default Qwen3.5 prefill checkpoint step (default: 64)
  AILA_KV_QUANT            Enable global KV cache quantization (0/1, default: 0)
  AILA_ASR_KV_QUANT        ASR KV quant override, inherits AILA_KV_QUANT
  AILA_TTS_KV_QUANT        TTS KV quant override, inherits AILA_KV_QUANT
  AILA_VLM_KV_QUANT        Qwen3.5 VLM KV quant override, inherits AILA_KV_QUANT

Interactive Commands:
  /help                    Show available commands
  /quit, /exit             Exit the program
  /clear                   Clear conversation history
  /context                 Show context usage
  /greedy                  Switch to greedy decoding
  /sample                  Switch to sampling
  /seed <N>                Set sampling seed (fixed-seed mode)
  /stream_on               Enable streaming output
  /stream_off              Disable streaming output
  /decode_chunk <N>        Set decode chunk size
  /stream_chunk <N>        Set stream chunk size
  /thinking_budget <N|off> Set thinking budget (-1/off disables, 0=no-think)
  /log_level <level>       Set log level (debug/info/warning/error)
  /config                  Show current configuration
)" << std::flush;
}

void print_version() {
    std::cout << "Aila v0.2.0" << std::endl;
}

bool parse_cli_args(int argc, char** argv, CLIOptions& opts) {
    // Load defaults from environment
    opts.model_dir = aila::env::read_string("AILA_MODEL_DIR", "");
    opts.lora_dir = aila::env::read_string("AILA_LORA_DIR", "");
    opts.max_seq_len = aila::env::read_int("AILA_MAX_SEQ_LEN", 4096);
    opts.decode_chunk_size = aila::env::read_int("AILA_DECODE_CHUNK_SIZE", 12);
    opts.stream_chunk_size = aila::env::read_int("AILA_STREAM_CHUNK_SIZE", 4);
    opts.thinking_budget_tokens = aila::env::read_int("AILA_THINKING_BUDGET", -1);
    if (opts.thinking_budget_tokens < -1) {
        opts.thinking_budget_tokens = -1;
    }
    opts.q35_prefill_step = aila::env::read_int("AILA_Q35_PREFILL_STEP", 64);
    {
        std::string log_level_env = aila::env::read_string("AILA_LOG_LEVEL", "");
        if (!log_level_env.empty()) {
            opts.log_level = aila::log_level_from_string(log_level_env);
        }
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
            return true;
        }
        if (arg == "-v" || arg == "--version") {
            opts.show_version = true;
            return true;
        }
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            opts.model_dir = argv[++i];
            continue;
        }
        if ((arg == "-s" || arg == "--max-seq") && i + 1 < argc) {
            opts.max_seq_len = std::atoi(argv[++i]);
            if (opts.max_seq_len <= 0) {
                std::cerr << "Error: --max-seq must be a positive integer" << std::endl;
                return false;
            }
            continue;
        }
        if ((arg == "-t" || arg == "--temperature") && i + 1 < argc) {
            opts.temperature = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if ((arg == "-k" || arg == "--top-k") && i + 1 < argc) {
            opts.top_k = std::atoi(argv[++i]);
            continue;
        }
        if ((arg == "-p" || arg == "--top-p") && i + 1 < argc) {
            opts.top_p = static_cast<float>(std::atof(argv[++i]));
            if (opts.top_p <= 0.0f) opts.top_p = 1e-6f;
            if (opts.top_p > 1.0f) opts.top_p = 1.0f;
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            try {
                opts.sampling_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                opts.use_fixed_seed = true;
            } catch (...) {
                std::cerr << "Error: --seed must be an unsigned integer" << std::endl;
                return false;
            }
            continue;
        }
        if (arg == "--greedy") {
            opts.do_sample = false;
            opts.bench_sample = false;
            continue;
        }
        if (arg == "--sample") {
            opts.do_sample = true;
            opts.bench_sample = true;
            continue;
        }
        if (arg == "--stream") {
            opts.stream_output = true;
            opts.explicit_stream = true;
            continue;
        }
        if (arg == "--no-stream") {
            opts.stream_output = false;
            opts.explicit_stream = true;
            continue;
        }
        if (arg == "--max-tokens" && i + 1 < argc) {
            opts.max_new_tokens = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--thinking-budget" && i + 1 < argc) {
            opts.thinking_budget_tokens = std::atoi(argv[++i]);
            if (opts.thinking_budget_tokens < -1) {
                std::cerr << "Error: --thinking-budget must be -1, 0, or a positive integer" << std::endl;
                return false;
            }
            continue;
        }
        if (arg == "--decode-chunk" && i + 1 < argc) {
            opts.decode_chunk_size = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--stream-chunk" && i + 1 < argc) {
            opts.stream_chunk_size = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--q35-prefill-step" && i + 1 < argc) {
            opts.q35_prefill_step = std::atoi(argv[++i]);
            if (opts.q35_prefill_step <= 0) {
                std::cerr << "Error: --q35-prefill-step must be a positive integer" << std::endl;
                return false;
            }
            continue;
        }
        if (arg == "--kv-quant") {
            opts.kv_quant = true;
            continue;
        }
        if (arg == "--rep-penalty" && i + 1 < argc) {
            opts.repetition_penalty = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--pres-penalty" && i + 1 < argc) {
            opts.presence_penalty = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--freq-penalty" && i + 1 < argc) {
            opts.frequency_penalty = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--detect" && i + 1 < argc) {
            opts.detect_path = argv[++i];
            continue;
        }
        if (arg == "--conf" && i + 1 < argc) {
            char* end = nullptr;
            opts.detection_confidence = std::strtof(argv[++i], &end);
            if (!end || *end != '\0' || !std::isfinite(opts.detection_confidence) ||
                opts.detection_confidence < 0.0f || opts.detection_confidence > 1.0f) {
                std::cerr << "Error: --conf must be a finite number in [0,1]" << std::endl;
                return false;
            }
            continue;
        }
        if (arg == "--max-det" && i + 1 < argc) {
            opts.detection_max = std::atoi(argv[++i]);
            if (opts.detection_max < 1 || opts.detection_max > 300) {
                std::cerr << "Error: --max-det must be in [1,300]" << std::endl;
                return false;
            }
            continue;
        }
        if (arg == "--save-detect" && i + 1 < argc) {
            opts.detection_output = argv[++i];
            continue;
        }
        if (arg == "--bench") {
            opts.bench_mode = true;
            continue;
        }
        if (arg == "--bench-pp" && i + 1 < argc) {
            opts.bench_pp = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--bench-tg" && i + 1 < argc) {
            opts.bench_tg = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--bench-iters" && i + 1 < argc) {
            opts.bench_iters = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--bench-warmup" && i + 1 < argc) {
            opts.bench_warmup = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--bench-sample") {
            opts.bench_sample = true;
            continue;
        }
        if (arg == "--bench-greedy") {
            opts.bench_sample = false;
            continue;
        }
        if (arg == "--log-level" && i + 1 < argc) {
            opts.log_level = aila::log_level_from_string(argv[++i]);
            continue;
        }
        if (arg == "--messages-json" && i + 1 < argc) {
            opts.messages_json_path = argv[++i];
            continue;
        }
        if (arg == "--chat-output-json") {
            opts.chat_output_json = true;
            continue;
        }
        if (arg == "--chat-stream-jsonl") {
            opts.chat_stream_jsonl = true;
            continue;
        }
        if (arg == "--lora" && i + 1 < argc) {
            opts.lora_dir = argv[++i];
            continue;
        }
        if (arg == "--transcribe" && i + 1 < argc) {
            opts.transcribe_path = argv[++i];
            continue;
        }
        if (arg == "--forced-lang" && i + 1 < argc) {
            opts.forced_language = argv[++i];
            continue;
        }
        if (arg == "--asr-system" && i + 1 < argc) {
            opts.system_prompt = argv[++i];
            continue;
        }
        if (arg == "--asr-segment" && i + 1 < argc) {
            opts.segment_sec = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (arg == "--synthesize" && i + 1 < argc) {
            opts.tts_text = argv[++i];
            continue;
        }
        if (arg == "--output-wav" && i + 1 < argc) {
            opts.tts_output_path = argv[++i];
            continue;
        }
        if (arg == "--ref" && i + 1 < argc) {
            opts.tts_reference_path = argv[++i];
            continue;
        }
        if (arg == "--ref-text" && i + 1 < argc) {
            opts.tts_ref_text = argv[++i];
            continue;
        }
        if (arg == "--voice-clone-mode" && i + 1 < argc) {
            opts.tts_voice_clone_mode = argv[++i];
            continue;
        }
        if (arg == "--speaker" && i + 1 < argc) {
            opts.tts_speaker_name = argv[++i];
            continue;
        }
        if (arg == "--instruct" && i + 1 < argc) {
            opts.tts_instruct_text = argv[++i];
            continue;
        }
        if (arg == "--language" && i + 1 < argc) {
            opts.tts_language = argv[++i];
            continue;
        }
        if (arg == "--ref-cache-dir" && i + 1 < argc) {
            opts.tts_ref_cache_dir = argv[++i];
            continue;
        }
        if (arg == "--stream-tts") {
            opts.tts_stream = true;
            continue;
        }
        if (arg == "--stream-batch" && i + 1 < argc) {
            opts.tts_stream_batch = std::max(1, std::atoi(argv[++i]));
            continue;
        }

        // ForceAligner CLI args
        if (arg == "--align-text" && i + 1 < argc) {
            opts.align_text = argv[++i];
            continue;
        }
        if (arg == "--align-audio" && i + 1 < argc) {
            opts.align_audio = argv[++i];
            continue;
        }
        if (arg == "--align-lang" && i + 1 < argc) {
            opts.align_language = argv[++i];
            continue;
        }

        if (arg == "--asr-past") {
            opts.past_text_conditioning = true;
            continue;
        }
        if (arg == "--no-asr-past") {
            opts.past_text_conditioning = false;
            continue;
        }
        // Positional: treat first positional as model dir
        if (arg[0] != '-' && opts.model_dir.empty()) {
            opts.model_dir = arg;
            continue;
        }

        std::cerr << "Error: Unknown option '" << arg << "'" << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return false;
    }

    aila::env::g_q35_prefill_step_override = opts.q35_prefill_step;
    aila::env::g_kv_quant_override = opts.kv_quant;
    return true;
}

// ============================================================
// Command Registry
// ============================================================

void CommandRegistry::register_command(const std::string& name, const std::string& help, Handler handler) {
    commands_.push_back({name, help, handler});
}

bool CommandRegistry::try_handle(const std::string& input) {
    if (input.empty() || input[0] != '/') return false;

    for (auto& cmd : commands_) {
        if (input == cmd.name || input.rfind(cmd.name + " ", 0) == 0) {
            std::string args;
            if (input.size() > cmd.name.size() + 1) {
                args = input.substr(cmd.name.size() + 1);
            }
            cmd.handler(args);
            return true;
        }
    }

    std::cout << "Unknown command: " << input << std::endl;
    std::cout << "Type /help for available commands" << std::endl;
    return true;
}

void CommandRegistry::print_help() const {
    std::cout << "\nAvailable commands:" << std::endl;
    for (auto& cmd : commands_) {
        std::cout << "  " << cmd.name;
        if (!cmd.help.empty()) {
            // Pad to column
            int pad = 22 - static_cast<int>(cmd.name.size());
            if (pad > 0) std::cout << std::string(pad, ' ');
            std::cout << cmd.help;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

CommandRegistry build_default_commands(GenerationConfig& gen_config, bool& stream_output,
                                       bool& should_quit, InferenceEngine* engine,
                                       const std::string& default_reference_path) {
    CommandRegistry registry;

    registry.register_command("/quit", "Exit the program", [&](const std::string&) {
        should_quit = true;
        return true;
    });

    registry.register_command("/exit", "Exit the program", [&](const std::string&) {
        should_quit = true;
        return true;
    });

    registry.register_command("/clear", "Clear conversation history", [&, engine](const std::string&) {
        if (engine) engine->reset_context();
        std::cout << "[Context] Conversation cleared" << std::endl;
        return true;
    });

    registry.register_command("/context", "Show context usage", [&, engine](const std::string&) {
        if (engine) {
            std::cout << "\n[Context]" << std::endl;
            std::cout << "  Context tokens:   " << engine->conversation_context_length()
                       << " / " << engine->max_context_length() << std::endl;
            std::cout << "  KV cache tokens:  " << engine->context_length()
                       << " / " << engine->max_context_length() << std::endl;
            std::cout << "  History messages: " << engine->active_history_message_count() << std::endl;
            std::cout << "  History turns:    " << engine->active_history_turn_count() << std::endl;
            std::cout << std::endl;
        }
        return true;
    });

    registry.register_command("/transcribe", "Transcribe audio file (ASR)", [&, engine](const std::string& args) {
        if (args.empty()) {
            std::cout << "[ASR] Usage: /transcribe <wav_path>" << std::endl;
            return true;
        }
        if (!engine) {
            std::cout << "[ASR] Engine is not initialized" << std::endl;
            return true;
        }
        if (engine->model_spec().family != ModelFamily::Qwen3ASR) {
            std::cout << "[ASR] Current loaded model is not an ASR model!" << std::endl;
            return true;
        }
        std::cout << "[ASR] Transcribing: " << args << std::endl;

        std::string lang;
        std::string transcript = engine->transcribe(
            args,
            gen_config,
            &lang,
            "",    // forced_language (auto-detect)
            "",    // system_prompt (default none)
            20.0f, // segment_sec = 20.0s
            true,  // past_text_conditioning = true
            [](const std::string& token_text) {
                std::cout << token_text << std::flush;
            }
        );

        if (engine->last_error_code() != EngineErrorCode::Ok) {
            std::cout << "\n[ASR] Error: " << engine->last_error_message() << std::endl;
        } else {
            std::cout << "\n[ASR] Finished." << std::endl;
            if (!lang.empty()) {
                std::cout << "[ASR] Detected Language: " << lang << std::endl;
            }
            double audio_s = engine->last_transcribe_duration_s();
            double latency_ms = engine->last_transcribe_latency_ms();
            int tokens = engine->last_transcribe_tokens();
            double speed = latency_ms > 0 ? (audio_s / (latency_ms / 1000.0)) : 0.0;
            double tok_s = latency_ms > 0 ? (static_cast<double>(tokens) / (latency_ms / 1000.0)) : 0.0;

            std::cout << "[Audio: " << std::fixed << std::setprecision(1) << audio_s << "s, "
                      << "Latency: " << std::fixed << std::setprecision(0) << latency_ms << "ms, "
                      << "Speed: " << std::fixed << std::setprecision(1) << speed << "x, "
                      << tok_s << " tok/s]" << std::endl;
        }
        return true;
    });

    registry.register_command("/align", "Forced alignment: audio + text -> word timestamps", [&, engine](const std::string& args) {
        // Parse: text="..." audio=file.wav [language="Chinese"]
        std::string text, audio_path, language = "Chinese";

        auto parse_kv = [](const std::string& s, const std::string& key, std::string& out) -> bool {
            size_t pos = s.find(key + "=\"");
            if (pos == std::string::npos) {
                pos = s.find(key + "=");
                if (pos == std::string::npos) return false;
                // Unquoted value
                size_t val_start = pos + key.size() + 1;
                size_t val_end = s.find(' ', val_start);
                if (val_end == std::string::npos) val_end = s.size();
                out = s.substr(val_start, val_end - val_start);
                return true;
            }
            // Quoted value
            size_t val_start = pos + key.size() + 2;
            size_t val_end = s.find('"', val_start);
            if (val_end == std::string::npos) return false;
            out = s.substr(val_start, val_end - val_start);
            return true;
        };

        if (!parse_kv(args, "text", text) || !parse_kv(args, "audio", audio_path)) {
            std::cout << "[Align] Usage: /align text=\"...\" audio=file.wav [language=\"Chinese\"]" << std::endl;
            return true;
        }
        parse_kv(args, "language", language);

        if (!engine) {
            std::cout << "[Align] Engine is not initialized" << std::endl;
            return true;
        }
        if (engine->model_spec().family != ModelFamily::Qwen3ForceAligner) {
            std::cout << "[Align] Current model is not a ForceAligner model!" << std::endl;
            return true;
        }

        std::cout << "[Align] Text: \"" << text << "\"" << std::endl;
        std::cout << "[Align] Audio: " << audio_path << std::endl;
        std::cout << "[Align] Language: " << language << std::endl;

        AudioBuffer audio;
        std::string load_error;
        if (!load_audio(audio_path, audio, &load_error)) {
            std::cout << "[Align] Failed to open audio file: " << audio_path
                      << " (" << load_error << ")" << std::endl;
            return true;
        }

        std::vector<float> mono;
        if (audio.channels > 1) {
            mono.resize(audio.samples.size() / audio.channels);
            for (size_t i = 0; i < mono.size(); ++i) {
                float sum = 0;
                for (int c = 0; c < audio.channels; ++c)
                    sum += audio.samples[i * audio.channels + c];
                mono[i] = sum / audio.channels;
            }
        } else {
            mono = std::move(audio.samples);
        }

        auto result = engine->align(mono, audio.sample_rate, text, language);

        if (engine->last_error_code() != EngineErrorCode::Ok) {
            std::cout << "[Align] Error: " << engine->last_error_message() << std::endl;
            return true;
        }

        std::cout << "\n[Align] Results (" << result.size() << " words):" << std::endl;
        for (const auto& w : result) {
            std::cout << "  \"" << w.text << "\"  " << w.start_ms << "ms - " << w.end_ms << "ms" << std::endl;
        }

        return true;
    });

    auto tts_handler = [&, engine, default_reference_path](const std::string& args) {
        std::string trimmed_args = trim(args);
        if (trimmed_args.empty()) {
            if (default_reference_path.empty()) {
                std::cout << "[TTS] Usage: /tts <text_prompt> [--ref <path>] [--ref-text <text>]" << std::endl;
            } else {
                std::cout << "[TTS] Usage: /tts <text_prompt> [--ref <path>]  (session reference: "
                          << default_reference_path << ")" << std::endl;
            }
            return true;
        }
        if (!engine) {
            std::cout << "[TTS] Engine is not initialized" << std::endl;
            return true;
        }
        if (engine->model_spec().family != ModelFamily::Qwen3TTS) {
            std::cout << "[TTS] Current loaded model does not support TTS! Please specify a Qwen3-TTS model." << std::endl;
            return true;
        }

        // Parse --ref and --ref-text arguments if any
        std::string text_prompt = trimmed_args;
        std::string ref_path = "";
        std::string ref_text = "";
        size_t ref_text_pos = trimmed_args.find("--ref-text ");
        if (ref_text_pos != std::string::npos) {
            ref_text = trim(trimmed_args.substr(ref_text_pos + 11));
            trimmed_args = trim(trimmed_args.substr(0, ref_text_pos));
        }
        size_t ref_pos = trimmed_args.find("--ref ");
        if (ref_pos != std::string::npos) {
            text_prompt = trim(trimmed_args.substr(0, ref_pos));
            ref_path = trim(trimmed_args.substr(ref_pos + 6));
        } else {
            ref_pos = trimmed_args.find("--ref=");
            if (ref_pos != std::string::npos) {
                text_prompt = trim(trimmed_args.substr(0, ref_pos));
                ref_path = trim(trimmed_args.substr(ref_pos + 6));
            } else {
                text_prompt = trimmed_args;
            }
        }

        // Fall back to session default reference if no --ref in command
        if (ref_path.empty() && !default_reference_path.empty()) {
            ref_path = default_reference_path;
        }

        std::string norm_args = normalize_input_for_model(text_prompt);
        std::cout << "[TTS] Synthesizing: \"" << norm_args << "\"";
        if (!ref_path.empty()) {
            std::cout << " (Reference: " << ref_path << ")";
        }
        std::cout << std::endl;

        std::vector<float> samples;
        std::vector<float> speaker_embedding;
        if (!ref_path.empty()) {
            speaker_embedding = load_or_extract_speaker_embedding(engine, ref_path);
        }

        bool ok = engine->synthesizeSpeech(
            norm_args, ref_path, "", "", "", gen_config, samples, ref_text, VoiceCloneMode::Auto);
        if (!ok) {
            std::cout << "[TTS] Synthesis failed: " << engine->last_error_message() << std::endl;
            return true;
        }

        std::string output_path = "tts_output.wav";
        if (save_wav(output_path, samples, 24000)) {
            std::cout << "[TTS] Synthesis completed successfully! Audio saved to: " << output_path 
                      << " (Samples: " << samples.size() << ", Duration: " 
                      << std::fixed << std::setprecision(2) << (static_cast<double>(samples.size()) / 24000.0) << "s)" << std::endl;
        } else {
            std::cout << "[TTS] Synthesis succeeded, but failed to write WAV file to " << output_path << std::endl;
        }
        return true;
    };

    registry.register_command("/synthesize", "Synthesize text to speech (TTS)", tts_handler);
    registry.register_command("/tts", "Synthesize text to speech (TTS)", tts_handler);

    registry.register_command("/greedy", "Switch to greedy decoding", [&](const std::string&) {
        gen_config.do_sample = false;
        std::cout << "[Config] Switched to greedy decoding" << std::endl;
        return true;
    });

    registry.register_command("/sample", "Switch to sampling", [&](const std::string&) {
        gen_config.do_sample = true;
        std::cout << "[Config] Switched to sampling (temp=" << gen_config.temperature
                  << ", top_k=" << gen_config.top_k
                  << ", top_p=" << gen_config.top_p << ")" << std::endl;
        return true;
    });

    registry.register_command("/seed", "Set sampling seed (fixed mode)", [&](const std::string& args) {
        uint64_t v = 0;
        std::istringstream iss(args);
        if (iss >> v) {
            gen_config.sampling_seed = v;
            gen_config.use_fixed_seed = true;
            std::cout << "[Config] sampling_seed=" << gen_config.sampling_seed
                      << " (fixed-seed mode enabled)" << std::endl;
        } else {
            std::cout << "[Config] Usage: /seed <unsigned_int>" << std::endl;
        }
        return true;
    });

    registry.register_command("/stream_on", "Enable streaming output", [&](const std::string&) {
        stream_output = true;
        std::cout << "[Config] Stream output enabled" << std::endl;
        return true;
    });

    registry.register_command("/stream_off", "Disable streaming output", [&](const std::string&) {
        stream_output = false;
        std::cout << "[Config] Stream output disabled" << std::endl;
        return true;
    });

    registry.register_command("/decode_chunk", "Set decode chunk size", [&](const std::string& args) {
        int v = 0;
        std::istringstream iss(args);
        if ((iss >> v) && v > 0) {
            gen_config.decode_chunk_size = v;
            std::cout << "[Config] decode_chunk_size=" << gen_config.decode_chunk_size << std::endl;
        } else {
            std::cout << "[Config] Usage: /decode_chunk <positive_int>" << std::endl;
        }
        return true;
    });

    registry.register_command("/stream_chunk", "Set stream chunk size", [&](const std::string& args) {
        int v = 0;
        std::istringstream iss(args);
        if ((iss >> v) && v > 0) {
            gen_config.stream_chunk_size = v;
            std::cout << "[Config] stream_chunk_size=" << gen_config.stream_chunk_size << std::endl;
        } else {
            std::cout << "[Config] Usage: /stream_chunk <positive_int>" << std::endl;
        }
        return true;
    });

    registry.register_command("/thinking_budget", "Set thinking budget (-1/off disables, 0=no-think)", [&](const std::string& args) {
        std::string v = trim(args);
        if (v == "off" || v == "disable" || v == "disabled") {
            gen_config.thinking_budget_tokens = -1;
        } else {
            int n = std::atoi(v.c_str());
            if (n < -1) {
                std::cout << "[Config] thinking_budget must be -1, 0, or a positive integer" << std::endl;
                return true;
            }
            gen_config.thinking_budget_tokens = n;
        }
        std::cout << "[Config] thinking_budget_tokens=" << gen_config.thinking_budget_tokens << std::endl;
        return true;
    });

    registry.register_command("/log_level", "Set log level (debug/info/warning/error)", [&](const std::string& args) {
        if (!args.empty()) {
            aila::LogLevel lv = aila::log_level_from_string(args);
            aila::set_log_level(lv);
            std::cout << "[Config] log_level=" << aila::log_level_name(lv) << std::endl;
        } else {
            std::cout << "[Config] log_level=" << aila::log_level_name(aila::get_log_level()) << std::endl;
            std::cout << "[Config] Usage: /log_level <debug|info|warning|error>" << std::endl;
        }
        return true;
    });

    registry.register_command("/config", "Show current configuration", [&, engine](const std::string&) {
        std::cout << "\n[Configuration]" << std::endl;
        std::cout << "  do_sample:          " << (gen_config.do_sample ? "true" : "false") << std::endl;
        std::cout << "  temperature:        " << gen_config.temperature << std::endl;
        std::cout << "  top_k:              " << gen_config.top_k << std::endl;
        std::cout << "  top_p:              " << gen_config.top_p << std::endl;
        std::cout << "  fixed_seed:         " << (gen_config.use_fixed_seed ? "true" : "false") << std::endl;
        std::cout << "  sampling_seed:      " << gen_config.sampling_seed << std::endl;
        std::cout << "  max_new_tokens:     " << gen_config.max_new_tokens << std::endl;
        std::cout << "  decode_chunk_size:  " << gen_config.decode_chunk_size << std::endl;
        std::cout << "  stream_chunk_size:  " << gen_config.stream_chunk_size << std::endl;
        std::cout << "  thinking_budget:    " << gen_config.thinking_budget_tokens << std::endl;
        std::cout << "  stream_output:      " << (stream_output ? "true" : "false") << std::endl;
        std::cout << "  log_level:          " << aila::log_level_name(aila::get_log_level()) << std::endl;
        std::cout << "  rep_penalty:        " << gen_config.repetition_penalty << std::endl;
        std::cout << "  pres_penalty:       " << gen_config.presence_penalty << std::endl;
        std::cout << "  freq_penalty:       " << gen_config.frequency_penalty << std::endl;
        if (engine) {
            std::cout << "  context_tokens:     " << engine->conversation_context_length()
                      << " / " << engine->max_context_length() << std::endl;
            std::cout << "  kv_cache_tokens:    " << engine->context_length()
                      << " / " << engine->max_context_length() << std::endl;
            std::cout << "  history_messages:   " << engine->active_history_message_count() << std::endl;
            std::cout << "  history_turns:      " << engine->active_history_turn_count() << std::endl;
        }
        std::cout << std::endl;
        return true;
    });

    registry.register_command("/help", "Show available commands", [&](const std::string&) {
        registry.print_help();
        return true;
    });

    return registry;
}

// ============================================================
// Interactive Loop
// ============================================================

int run_interactive(InferenceEngine& engine, GenerationConfig& gen_config, bool stream_output,
                    const std::string& default_reference_path /* = "" in header */) {
    bool interactive_terminal = detect_interactive_terminal();
    setup_console_utf8(interactive_terminal);

    // Auto-detect stream mode if not explicitly set
    if (!stream_output && interactive_terminal) {
        stream_output = true;
    }
    // Environment override
    stream_output = aila::env::read_flag("AILA_STREAM_OUTPUT", stream_output);

    bool should_quit = false;
    auto registry = build_default_commands(gen_config, stream_output, should_quit, &engine, default_reference_path);

    std::string input;
    while (!should_quit) {
        std::cout << "\nUser: ";
        if (!std::getline(std::cin, input)) break;
        input = trim(input);

        if (input.empty()) continue;

        // Try as command
        if (registry.try_handle(input)) continue;

        std::string model_input = normalize_input_for_model(input);

        std::cout << "\nAila: ";
        auto t_start = std::chrono::high_resolution_clock::now();
        int token_count = 0;
        double ttft_ms = 0.0;

        if (stream_output) {
            engine.generate(model_input, gen_config, [&](const std::string& token_text) {
                if (token_count == 0) {
                    auto t_first = std::chrono::high_resolution_clock::now();
                    ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
                }
                ++token_count;
                std::cout << token_text << std::flush;
            });
            auto t_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            double gen_ms = total_ms - ttft_ms;
            double gen_tok_s = (token_count > 1 && gen_ms > 0)
                ? (static_cast<double>(token_count - 1) / gen_ms * 1000.0) : 0.0;
            if (token_count > 0 && total_ms > 0) {
                std::cout << "\n[TTFT: " << std::fixed << std::setprecision(0)
                          << ttft_ms << "ms, " << token_count << " tokens";
                if (gen_tok_s > 0)
                    std::cout << ", " << std::setprecision(1) << gen_tok_s << " tok/s";
                std::cout << "]";
            }
            std::cout << std::endl;
        } else {
            std::string response = engine.generate(model_input, gen_config, nullptr);
            auto t_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << response << std::endl;
            std::cout << "[" << std::fixed << std::setprecision(0) << total_ms << "ms]" << std::endl;
        }
    }

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
