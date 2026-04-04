#include "engine/Engine.hpp"
#include "profile/Device.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <stdio.h>
#include <clocale>
#else
#include <unistd.h>
#endif

namespace {

bool read_env_flag(const char* name, bool default_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        bool enabled = (std::atoi(value) != 0);
        free(value);
        return enabled;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    if (!value) return default_value;
    return (std::atoi(value) != 0);
#endif
}

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

void setup_console_utf8(bool interactive_terminal) {
    (void)interactive_terminal;
}
#endif

} // namespace

int main(int argc, char** argv) {
    bool interactive_terminal = detect_interactive_terminal();
    setup_console_utf8(interactive_terminal);

    // Model directory (default or from command line)
    CheckDevice();
    std::string model_dir = "E:\\RiderProjects\\Aila\\Qwen3-0.6B";
    if (argc > 1) {
        model_dir = argv[1];
    }

    // Initialize engine
    InferenceEngine engine;
    if (!engine.init(model_dir)) {
        std::cerr << "Failed to initialize inference engine" << std::endl;
        return 1;
    }

    // Generation config
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 1024;
    gen_config.temperature = 0.7f;
    gen_config.top_k = 15;
    gen_config.do_sample = false;
    gen_config.decode_chunk_size = 12;
    gen_config.stream_chunk_size = 4;
    bool stream_output = interactive_terminal;
    stream_output = read_env_flag("AILA_STREAM_OUTPUT", stream_output);
#ifdef _WIN32
    char* chunk_env = nullptr;
    size_t chunk_len = 0;
    if (_dupenv_s(&chunk_env, &chunk_len, "AILA_DECODE_CHUNK_SIZE") == 0 && chunk_env) {
        int chunk = std::atoi(chunk_env);
        if (chunk > 0) {
            gen_config.decode_chunk_size = chunk;
        }
        free(chunk_env);
    }

    char* stream_chunk_env = nullptr;
    size_t stream_chunk_len = 0;
    if (_dupenv_s(&stream_chunk_env, &stream_chunk_len, "AILA_STREAM_CHUNK_SIZE") == 0 && stream_chunk_env) {
        int stream_chunk = std::atoi(stream_chunk_env);
        if (stream_chunk > 0) {
            gen_config.stream_chunk_size = stream_chunk;
        }
        free(stream_chunk_env);
    }
#else
    if (const char* chunk_env = std::getenv("AILA_DECODE_CHUNK_SIZE")) {
        int chunk = std::atoi(chunk_env);
        if (chunk > 0) {
            gen_config.decode_chunk_size = chunk;
        }
    }
    if (const char* stream_chunk_env = std::getenv("AILA_STREAM_CHUNK_SIZE")) {
        int stream_chunk = std::atoi(stream_chunk_env);
        if (stream_chunk > 0) {
            gen_config.stream_chunk_size = stream_chunk;
        }
    }
#endif

    // Interactive loop
    std::string input;
    while (true) {
        std::cout << "\nUser: ";
        std::getline(std::cin, input);

        if (input.empty()) continue;
        if (input == "/quit" || input == "/exit") break;

        // Handle special commands
        if (input == "/greedy") {
            gen_config.do_sample = false;
            std::cout << "[Config] Switched to greedy decoding" << std::endl;
            continue;
        }
        if (input == "/sample") {
            gen_config.do_sample = true;
            std::cout << "[Config] Switched to sampling (temp="
                      << gen_config.temperature << ", top_k=" << gen_config.top_k << ")" << std::endl;
            continue;
        }
        if (input == "/stream_on") {
            stream_output = true;
            std::cout << "[Config] Stream output enabled" << std::endl;
            continue;
        }
        if (input == "/stream_off") {
            stream_output = false;
            std::cout << "[Config] Stream output disabled" << std::endl;
            continue;
        }
        if (input.rfind("/decode_chunk ", 0) == 0) {
            int v = 0;
            std::istringstream iss(input.substr(14));
            if ((iss >> v) && v > 0) {
                gen_config.decode_chunk_size = v;
                std::cout << "[Config] decode_chunk_size=" << gen_config.decode_chunk_size << std::endl;
            } else {
                std::cout << "[Config] Usage: /decode_chunk <positive_int>" << std::endl;
            }
            continue;
        }
        if (input.rfind("/stream_chunk ", 0) == 0) {
            int v = 0;
            std::istringstream iss(input.substr(14));
            if ((iss >> v) && v > 0) {
                gen_config.stream_chunk_size = v;
                std::cout << "[Config] stream_chunk_size=" << gen_config.stream_chunk_size << std::endl;
            } else {
                std::cout << "[Config] Usage: /stream_chunk <positive_int>" << std::endl;
            }
            continue;
        }

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
