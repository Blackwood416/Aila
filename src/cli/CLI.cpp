#include "CLI.hpp"
#include "engine/Engine.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

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

std::string normalize_input_for_model(const std::string& text) {
    if (text.empty() || is_valid_utf8(text)) return text;

    UINT cp = GetConsoleCP();
    if (cp == 0) cp = GetACP();
    std::string converted = codepage_to_utf8(text, cp);
    if (is_valid_utf8(converted)) return converted;

    converted = codepage_to_utf8(text, GetACP());
    return converted;
}

void setup_console_utf8(bool interactive_terminal) {
    if (!interactive_terminal) return;
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    std::setlocale(LC_ALL, ".UTF-8");
}
#else
std::string normalize_input_for_model(const std::string& text) {
    return text;
}

void setup_console_utf8(bool /*interactive_terminal*/) {}
#endif

} // namespace

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
  --decode-chunk <N>       Decode chunk size (default: 12)
  --stream-chunk <N>       Stream chunk size (default: 4)
  --rep-penalty <F>        Repetition penalty (default: 1.0, >1.0 to penalize)
  --pres-penalty <F>       Presence penalty (default: 0.0)
  --freq-penalty <F>       Frequency penalty (default: 0.0)
  --bench                  Run benchmark mode
  --bench-pp <N>           Benchmark prompt length (default: 512)
  --bench-tg <N>           Benchmark generation length (default: 128)
  --bench-iters <N>        Benchmark iterations (default: 5)
  --bench-warmup <N>       Benchmark warmup iterations (default: 1)
  --bench-sample           Benchmark decode in sampling mode
  --bench-greedy           Benchmark decode in greedy mode (default)
  --messages-json <path>   Single-shot generation from OpenAI-style messages JSON file ('-' = stdin)
  -h, --help               Show this help message
  -v, --version            Show version

Environment Variables:
  AILA_MODEL_DIR           Default model directory
  AILA_MAX_SEQ_LEN         Default max sequence length
  AILA_STREAM_OUTPUT       Force stream mode (0/1)
  AILA_DECODE_CHUNK_SIZE   Default decode chunk size
  AILA_STREAM_CHUNK_SIZE   Default stream chunk size

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
  /config                  Show current configuration
)" << std::flush;
}

void print_version() {
    std::cout << "Aila v0.1.0" << std::endl;
}

bool parse_cli_args(int argc, char** argv, CLIOptions& opts) {
    // Load defaults from environment
    opts.model_dir = aila::env::read_string("AILA_MODEL_DIR", "");
    opts.max_seq_len = aila::env::read_int("AILA_MAX_SEQ_LEN", 4096);
    opts.decode_chunk_size = aila::env::read_int("AILA_DECODE_CHUNK_SIZE", 12);
    opts.stream_chunk_size = aila::env::read_int("AILA_STREAM_CHUNK_SIZE", 4);

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
            continue;
        }
        if (arg == "--sample") {
            opts.do_sample = true;
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
        if (arg == "--decode-chunk" && i + 1 < argc) {
            opts.decode_chunk_size = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--stream-chunk" && i + 1 < argc) {
            opts.stream_chunk_size = std::atoi(argv[++i]);
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
        if (arg == "--messages-json" && i + 1 < argc) {
            opts.messages_json_path = argv[++i];
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
                                       bool& should_quit, InferenceEngine* engine) {
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
            std::cout << "  KV cache tokens:  " << engine->context_length()
                      << " / " << engine->max_context_length() << std::endl;
            std::cout << "  History turns:    " << engine->history().size() << std::endl;
            std::cout << std::endl;
        }
        return true;
    });

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
        std::cout << "  stream_output:      " << (stream_output ? "true" : "false") << std::endl;
        std::cout << "  rep_penalty:        " << gen_config.repetition_penalty << std::endl;
        std::cout << "  pres_penalty:       " << gen_config.presence_penalty << std::endl;
        std::cout << "  freq_penalty:       " << gen_config.frequency_penalty << std::endl;
        if (engine) {
            std::cout << "  context_tokens:     " << engine->context_length()
                      << " / " << engine->max_context_length() << std::endl;
            std::cout << "  history_turns:      " << engine->history().size() << std::endl;
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

int run_interactive(InferenceEngine& engine, GenerationConfig& gen_config, bool stream_output) {
    bool interactive_terminal = detect_interactive_terminal();
    setup_console_utf8(interactive_terminal);

    // Auto-detect stream mode if not explicitly set
    if (!stream_output && interactive_terminal) {
        stream_output = true;
    }
    // Environment override
    stream_output = aila::env::read_flag("AILA_STREAM_OUTPUT", stream_output);

    bool should_quit = false;
    auto registry = build_default_commands(gen_config, stream_output, should_quit, &engine);

    std::string input;
    while (!should_quit) {
        std::cout << "\nUser: ";
        if (!std::getline(std::cin, input)) break;

        if (input.empty()) continue;

        // Try as command
        if (registry.try_handle(input)) continue;

        std::string model_input = normalize_input_for_model(input);

        std::cout << "\nAila: ";
        if (stream_output) {
            engine.generate(model_input, gen_config, [](const std::string& token_text) {
                std::cout << token_text << std::flush;
            });
            std::cout << std::endl;
        } else {
            std::string response = engine.generate(model_input, gen_config, nullptr);
            std::cout << response << std::endl;
        }
    }

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
