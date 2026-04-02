#include "engine/Engine.hpp"
#include "profile/Device.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
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
    gen_config.max_new_tokens = 512;
    gen_config.temperature = 0.6f;
    gen_config.top_k = 20;
    gen_config.do_sample = false;
    gen_config.decode_chunk_size = 12;
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
#else
    if (const char* chunk_env = std::getenv("AILA_DECODE_CHUNK_SIZE")) {
        int chunk = std::atoi(chunk_env);
        if (chunk > 0) {
            gen_config.decode_chunk_size = chunk;
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

        std::cout << "\nAila: ";
        std::string response = engine.generate(input, gen_config, nullptr);
        std::cout << response << std::endl;
    }

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
