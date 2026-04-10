#pragma once

#include "engine/Types.hpp"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

// ============================================================
// CLI: Command-line argument parsing & interactive command loop
// ============================================================

class InferenceEngine;

// Parsed CLI options
struct CLIOptions {
    std::string model_dir;         // -m, --model
    int max_seq_len = 4096;        // -s, --max-seq
    int max_new_tokens = 1024;     // --max-tokens
    float temperature = 0.7f;      // -t, --temperature
    int top_k = 15;                // -k, --top-k
    float top_p = 0.95f;           // --top-p
    bool do_sample = true;         // --greedy to disable sampling
    uint64_t sampling_seed = 42;   // --seed
    bool use_fixed_seed = false;   // seed enabled?
    bool stream_output = true;     // --stream / --no-stream (auto-detect)
    int decode_chunk_size = 12;    // --decode-chunk
    int stream_chunk_size = 4;     // --stream-chunk
    bool show_help = false;        // -h, --help
    bool show_version = false;     // -v, --version
    bool explicit_stream = false;  // user explicitly set stream mode

    // Penalty parameters
    float repetition_penalty = 1.0f;  // --rep-penalty
    float presence_penalty   = 0.0f;  // --pres-penalty
    float frequency_penalty  = 0.0f;  // --freq-penalty

    // Benchmark mode
    bool bench_mode = false;       // --bench
    int bench_pp = 512;            // --bench-pp
    int bench_tg = 128;            // --bench-tg
    int bench_iters = 5;           // --bench-iters
    int bench_warmup = 1;          // --bench-warmup
    bool bench_sample = false;     // --bench-sample / --bench-greedy

    // Single-shot messages JSON mode
    std::string messages_json_path; // --messages-json
};

// Parse command-line arguments
// Returns true on success, false on error (error printed to stderr)
bool parse_cli_args(int argc, char** argv, CLIOptions& opts);

// Print help message
void print_help();

// Print version
void print_version();

// ============================================================
// Interactive command registry
// ============================================================

class CommandRegistry {
public:
    using Handler = std::function<bool(const std::string& args)>;

    void register_command(const std::string& name, const std::string& help, Handler handler);

    // Try to handle input as a command.
    // Returns true if it was a command (handled), false if it's a regular message.
    bool try_handle(const std::string& input);

    // Print help for all registered commands
    void print_help() const;

private:
    struct CommandEntry {
        std::string name;
        std::string help;
        Handler handler;
    };
    std::vector<CommandEntry> commands_;
};

// Build the default interactive command set
CommandRegistry build_default_commands(GenerationConfig& gen_config, bool& stream_output,
                                       bool& should_quit, InferenceEngine* engine = nullptr);

// Run the interactive loop
int run_interactive(InferenceEngine& engine, GenerationConfig& gen_config, bool stream_output);
