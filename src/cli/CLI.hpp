#pragma once

#include "engine/Types.hpp"
#include "profile/Profiling.hpp"
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
    std::string lora_dir;          // --lora
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
    int thinking_budget_tokens = -1; // --thinking-budget, -1=off, 0=no-think, >0 cap
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

    // Log level
    aila::LogLevel log_level = aila::LogLevel::Info; // --log-level (or AILA_LOG_LEVEL)

    // Single-shot messages JSON mode
    std::string messages_json_path; // --messages-json
    bool chat_output_json = false;   // --chat-output-json
    bool chat_stream_jsonl = false;  // --chat-stream-jsonl

    // ASR transcription mode
    std::string transcribe_path;    // --transcribe
    std::string forced_language;    // --forced-lang
    std::string system_prompt;      // --asr-system
    float segment_sec = 0.0f;       // --asr-segment
    bool past_text_conditioning = false; // --asr-past / --no-asr-past

    // TTS synthesis mode
    std::string tts_text;           // --synthesize
    std::string tts_output_path;    // --output-wav
    std::string tts_reference_path;  // --ref, --speaker (reference audio path)
    std::string tts_speaker_name;   // --speaker (CustomVoice: "vivian", etc.)
    std::string tts_instruct_text;  // --instruct (voice style description)
    std::string tts_language;       // --language ("chinese", "english", etc.)
    std::string tts_ref_cache_dir;  // --ref-cache-dir
    bool tts_stream = false;         // --stream-tts (output raw PCM to stdout)
    int tts_stream_batch = 4;        // --stream-batch <N> (frames per chunk, default 4)

    // ForceAligner mode
    std::string align_text;          // --align-text
    std::string align_audio;         // --align-audio
    std::string align_language = "Chinese"; // --align-lang

    // Qwen3.5 prefill checkpoint step
    int q35_prefill_step = 64;       // --q35-prefill-step

    // KV Cache quantization (FP8)
    bool kv_quant = false;           // --kv-quant
};

// Parse command-line arguments
// Returns true on success, false on error (error printed to stderr)
bool parse_cli_args(int argc, char** argv, CLIOptions& opts);

// Print help message
void print_help();

// Print version
void print_version();

// String processing utilities
std::string normalize_input_for_model(const std::string& text);
std::string trim(const std::string& str);
std::vector<float> load_or_extract_speaker_embedding(InferenceEngine* engine, const std::string& path);

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
                                       bool& should_quit, InferenceEngine* engine = nullptr,
                                       const std::string& default_reference_path = "");

// Run the interactive loop
int run_interactive(InferenceEngine& engine, GenerationConfig& gen_config, bool stream_output,
                    const std::string& default_reference_path = "");
