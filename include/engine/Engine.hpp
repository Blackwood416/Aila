#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/IModelBackend.hpp"
#include "../src/models/Qwen3DenseBackend.hpp"
#include "../src/models/Qwen3Bnb4Backend.hpp"
#include "../src/models/Qwen35HybridBnb4Backend.hpp"
#include "../src/models/DiffusionGemmaBnb4Backend.hpp"
#include "../src/models/Qwen4ExpBnb4Backend.hpp"
#include "../src/generation/DiffusionGemmaGenerator.hpp"
#include "../src/models/Qwen35HybridTextBackend.hpp"
#include "../src/models/Qwen3ASRBackend.hpp"
#include "../src/models/Qwen3ASRBnb4Backend.hpp"
#include "../src/models/Qwen3TTSBackend.hpp"
#include "../src/models/Qwen3ForceAlignerBackend.hpp"
#include "../src/models/Qwen3ForceAlignerBnb4Backend.hpp"
#include "../src/vision/IVisionEncoder.hpp"
#include "../src/vision/Qwen35VisionEncoder.hpp"
#include "../src/vision/Gemma4VisionEncoder.hpp"
#include "../src/vision/Yolo26Detector.hpp"
#include "../src/audio/Qwen3ASRAudioEncoder.hpp"
#include "../src/audio/AudioPreprocessor.hpp"
#include "../src/audio/SpeakerEncoder.hpp"
#include "../src/audio/MimiEncoder.hpp"
#include "../src/utils/ForceAlignerPostProcess.hpp"
#include "../src/audio/GpuSpeakerEncoder.hpp"
#include "../src/lora/LoraLoader.hpp"
#include "../src/lora/LoraConfig.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/ModelConfig.hpp"
#include "../src/utils/ModelSpec.hpp"
#include "../src/utils/SafeTensors.hpp"
#include "../src/chat/AssistantOutputParser.hpp"
#include "../src/chat/ChatFormatter.hpp"
#include "../src/chat/ChatJson.hpp"
#include "../src/chat/ChatSessionState.hpp"
#include "../src/chat/ChatStreamJson.hpp"
#include "../src/chat/StructuredStreamParser.hpp"
#include "../src/chat/ThinkingBudgetController.hpp"
#include "../src/chat/ToolPolicy.hpp"
#include "../src/profile/Profiling.hpp"
#include "../src/utils/EnvUtils.hpp"
#include "../src/utils/JsonParser.hpp"
#include "Types.hpp"
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include <cctype>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstddef>
#include <utility>

#include "engine/AsrText.hpp"

// ============================================================
// Inference Engine: orchestrates loading, tokenization, inference
// ============================================================
class InferenceEngine {
public:
    using ChatStreamCallback = std::function<bool(const aila::chat::StructuredStreamEvent&)>;

    InferenceEngine() = default;

    // Initialize: load model + tokenizer from model directory
    bool init(const std::string& model_dir, int max_seq_len = 4096,
              const std::string& lora_dir = "");

    // ============================================================
    // Context management
    // ============================================================

    void set_system_prompt(const std::string& prompt) { system_prompt_ = prompt; }
    const std::string& system_prompt() const { return system_prompt_; }

    void reset_context();

    int context_length() const { return static_cast<int>(cached_ids_.size()); }
    int max_context_length() const;
    const ChatHistory& history() const { return history_; }
    size_t active_history_message_count() const;
    size_t active_history_turn_count() const;
    int conversation_context_length() const;

    void align_backend_context_to_sequence(const std::vector<int>& prompt_ids,
                                           const std::vector<int>& generated_ids,
                                           int prompt_len,
                                           const char* log_tag);

    Tensor* forward_prompt_tokens(const std::vector<int>& token_ids,
                                  int start,
                                  int len,
                                  int snapshot_after_len = 0,
                                  bool async_h2d = false);

    std::string decode_with_special_tokens(const std::vector<int>& ids) const;

    void debug_dump_prompt_text(const char* label,
                                const std::string& prompt_text,
                                int prompt_tokens) const;

    aila::chat::ChatRequest make_chat_request_from_legacy(
        const std::vector<Message>& messages,
        const GenerationConfig& config,
        const aila::chat::ChatRequest* source_request = nullptr) const;

    aila::chat::ChatFormatInput make_chat_format_input(const aila::chat::ChatRequest& request) const;

    std::string current_model_family_name() const;

    std::string tool_choice_name(const aila::chat::ChatRequest& request) const;

    std::string tool_policy_name(aila::chat::ToolPolicyMode policy) const;

    std::vector<int> thinking_close_token_ids() const;

    // ============================================================
    // Generate response (multi-turn with incremental prefill)
    //
    // Context tracking: we maintain cached_ids_ which is the EXACT
    // token ID sequence currently stored in the KV cache. This avoids
    // decode-then-re-encode mismatches. New turns are built by
    // appending raw token IDs directly, never by re-encoding text.
    // ============================================================
    std::string generate(const std::string& user_message,
                         const GenerationConfig& gen_config = GenerationConfig(),
                         std::function<void(const std::string&)> token_callback = nullptr);

    std::string generate_messages(const std::vector<Message>& messages,
                                  const GenerationConfig& gen_config = GenerationConfig(),
                                  std::function<void(const std::string&)> token_callback = nullptr,
                                  const aila::chat::ChatRequest* structured_request = nullptr);

    std::string generate_chat_request_text(
        const aila::chat::ChatRequest& request,
        std::function<void(const std::string&)> token_callback = nullptr);

    aila::chat::AssistantChatResult generate_chat_request_result(
        const aila::chat::ChatRequest& request,
        std::function<void(const std::string&)> token_callback = nullptr);

    aila::chat::StructuredStreamEvent make_final_stream_event(
        const aila::chat::AssistantChatResult& result) const;

    int generate_chat_request_stream(
        const aila::chat::ChatRequest& request,
        ChatStreamCallback callback);

    std::string generate_messages_json(const std::string& messages_json,
                                       const GenerationConfig& gen_config = GenerationConfig(),
                                       std::function<void(const std::string&)> token_callback = nullptr);

    std::string generate_chat_json(const std::string& chat_request_json,
                                   const GenerationConfig& gen_config = GenerationConfig(),
                                   std::function<void(const std::string&)> token_callback = nullptr);

    // ============================================================
    // Raw prefill for benchmark (no decode, no history)
    // ============================================================

    static int find_split_point(const float* samples, int n_samples, int target_sample, float search_sec);

    static bool should_insert_boundary_space(int prev_ch, int next_ch);

    // ASR transcription of a single raw segment
    std::string transcribe_segment_raw(const std::vector<float>& seg_buf,
                                       const GenerationConfig& gen_config,
                                       const std::string& forced_language,
                                       const std::string& system_prompt,
                                       const std::string& past_text,
                                       std::string* language_out = nullptr);

    // TTS audio code synthesis from text
    bool synthesize_codes(const std::vector<int>& text_tokens,
                          const std::vector<float>& speaker_embedding,
                          const GenerationConfig& gen_config,
                          std::vector<int32_t>& out_codes,
                          int& out_n_frames);

    // TTS audio samples synthesis from text using Mimi Decoder
    bool synthesize_wav(const std::vector<int>& text_tokens,
                        const std::vector<float>& speaker_embedding,
                        const GenerationConfig& gen_config,
                        std::vector<float>& out_samples);

    // TTS audio samples synthesis directly from raw text using Mimi Decoder
    bool synthesize_text_to_wav(const std::string& text,
                               const std::vector<float>& speaker_embedding,
                               const GenerationConfig& gen_config,
                               std::vector<float>& out_samples);

    // Unified TTS speech synthesis supporting Base (voice cloning), CustomVoice
    // (speaker name), and VoiceDesign (instruct text) models.
    // Exactly one of reference_audio_path / speaker_name / instruct_text should be
    // non-empty to select the mode; all empty = default voice.
    /// Tokenize text for TTS (pure tokens without ChatML tags)
    std::vector<int> tokenize_tts_text(const std::string& text) const;

    bool synthesizeSpeech(const std::string& text,
                          const std::string& reference_audio_path,
                          const std::string& speaker_name,
                          const std::string& instruct_text,
                          const std::string& language,
                          const GenerationConfig& gen_config,
                          std::vector<float>& out_samples,
                          const std::string& reference_text = "",
                          VoiceCloneMode clone_mode = VoiceCloneMode::Auto);

    // Streaming TTS: calls callback from background thread with PCM audio chunks.
    // Returns immediately; callback fires from a worker thread as audio is generated.
    std::thread synthesizeSpeechStream(
        const std::string& text,
        const std::string& reference_audio_path,
        const std::string& speaker_name,
        const std::string& instruct_text,
        const std::string& language,
        const GenerationConfig& gen_config,
        std::function<void(const float*, int)> callback,
        int stream_batch_frames = 6,
        const std::string& reference_text = "",
        VoiceCloneMode clone_mode = VoiceCloneMode::Auto,
        std::shared_ptr<std::atomic<int>> status_out = nullptr,
        std::shared_ptr<std::string> err_msg_out = nullptr
    );

    // TTS audio discrete codes decoding using Mimi Decoder
    bool decode_mimi_vocoder(const std::vector<int32_t>& codes,
                             int n_frames,
                             std::vector<float>& out_samples);

    // Extract speaker embedding from a reference audio file for TTS voice cloning.
    // Uses GPU-accelerated ECAPA-TDNN (SYCL kernels) for the forward pass;
    // mel spectrogram is computed on CPU.
    //
    // Caching: results are cached in memory (keyed by audio file path) and
    // optionally persisted to disk.  Disk cache location:
    //   - If ref_cache_dir_ is set: <cache_dir>/<basename>.ref.bin
    //   - Otherwise: <audio_path>.ref.bin alongside the reference audio
    //
    // The audio file must be mono, and will be automatically resampled to 24kHz if needed.
    // Supported formats: WAV, MP3, FLAC.
    // Returns true on success, false on error (call last_error_message() for details).
    bool extractSpeakerEmbedding(const std::string& audio_path,
                                 std::vector<float>& embedding);

    // Forced alignment: given audio samples + text → word-level timestamps.
    std::vector<aila::AlignedWord> align(
        const std::vector<float>& audio_samples, int sample_rate,
        const std::string& text, const std::string& language);

    // Forced alignment with pre-tokenized word list (bypasses built-in tokenizer).
    std::vector<aila::AlignedWord> align_words(
        const std::vector<float>& audio_samples, int sample_rate,
        const std::vector<std::string>& word_list);

    // ASR transcription from WAV file
    std::string transcribe(const std::string& wav_path,
                           const GenerationConfig& gen_config = GenerationConfig(),
                           std::string* language_out = nullptr,
                           const std::string& forced_language = "",
                           const std::string& system_prompt = "",
                           float segment_sec = 0.0f,
                           bool past_text_conditioning = false,
                           std::function<void(const std::string&)> token_callback = nullptr);

    // Real-time ASR transcribe streaming state machine
    class TranscribeStream {
    public:
        TranscribeStream(
            InferenceEngine* engine,
            const GenerationConfig& gen_config,
            const std::string& forced_language,
            const std::string& system_prompt
        );

        void feed_audio(const float* data, size_t count);

        void process();

        void get_text(std::string& out_stable, std::string& out_partial);

    private:
        InferenceEngine* engine_;
        GenerationConfig gen_config_;
        std::string forced_language_;
        std::string system_prompt_;

        std::vector<float> audio_buffer_;
        size_t stable_samples_offset_;

        std::string stable_text_;
        std::string partial_text_;
        std::string past_text_;
    };

    double benchmark_prefill(const std::vector<int>& token_ids);

    // Raw decode N tokens for benchmark (after prefill)
    double benchmark_decode(int num_tokens, const GenerationConfig* decode_config = nullptr);

    std::vector<Detection> detect_file(
            const std::string& path,
            const DetectionConfig& detection_config = DetectionConfig());

    std::vector<Detection> detect_encoded(
            const uint8_t* data, size_t size,
            const DetectionConfig& detection_config = DetectionConfig());

    std::vector<Detection> detect_pixels(
            const ImageView& image,
            const DetectionConfig& detection_config = DetectionConfig());

    int last_detect_image_width() const;
    int last_detect_image_height() const;
    double last_detect_preprocess_ms() const;
    double last_detect_device_wall_ms() const;
    double last_detect_postprocess_ms() const;
    size_t last_detect_peak_device_bytes() const;

    Qwen3Config& config() { return config_; }
    const Qwen3Config& config() const { return config_; }
    const ModelSpec& model_spec() const { return model_spec_; }
    Tokenizer& tokenizer() { return tokenizer_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    bool vision_enabled() const { return vision_backend_enabled_; }
    void clear_error() {
        last_error_code_ = EngineErrorCode::Ok;
        last_error_message_.clear();
    }

    bool reject_yolo_capability(const char* operation) {
        if (model_spec_.family != ModelFamily::Yolo26) return false;
        set_error(EngineErrorCode::ModelCapability,
                  std::string(operation) + " is not supported by a YOLO26 detection model");
        return true;
    }

    void set_error(EngineErrorCode code, const std::string& message) {
        last_error_code_ = code;
        last_error_message_ = message;
    }

    EngineErrorCode last_error_code() const { return last_error_code_; }
    const std::string& last_error_message() const { return last_error_message_; }
    double last_transcribe_duration_s() const { return last_transcribe_duration_s_; }
    double last_transcribe_latency_ms() const { return last_transcribe_latency_ms_; }
    int last_transcribe_tokens() const { return last_transcribe_tokens_; }
    const std::string& model_dir() const { return model_dir_; }

    // Speaker embedding cache: avoid re-extracting the same reference audio.
    // Set via AILA_REF_CACHE_DIR env var or --ref-cache-dir CLI argument.
    void setRefCacheDir(const std::string& dir) { ref_cache_dir_ = dir; }
    const std::string& refCacheDir() const { return ref_cache_dir_; }

    // Look up speaker token ID from spk_id map (CustomVoice).
    // Returns 0 if name is empty or not found.
    int getSpeakerId(const std::string& name) const;

    // Look up language codec token ID from codec_language_id map.
    // Returns 0 if lang is empty or not found (0 = auto/nothink).
    int getLanguageId(const std::string& lang) const;

    // Clear the in-memory speaker embedding cache.
    void clearRefCache() { ref_cache_.clear(); }

    // Build a cache key that includes the embedding dim so different models
    // (e.g. 0.6B=1024-dim vs 1.7B=2048-dim) don't share incompatible cache files.
    static std::string refCacheKey(const std::string& audio_path, int dim);

    // Build disk cache path from audio path and embedding dimension.
    std::string refDiskCachePath(const std::string& audio_path, int dim) const;

    // Convenience: try common embedding dimensions (1024 for 0.6B, 2048 for 1.7B).
    bool lookupRefCache(const std::string& audio_path,
                            std::vector<float>& embedding);

    // Check if a speaker embedding is cached for a specific embedding dimension.
    // The 'dim_hint' is used to construct the cache key; pass the expected
    // embedding dimension (1024 for 0.6B, 2048 for 1.7B).
    bool lookupRefCache(const std::string& audio_path, int dim_hint,
                            std::vector<float>& embedding);

    // Cache an externally-extracted speaker embedding (in-memory + disk).
    void cacheRefEmbedding(const std::string& audio_path,
                               const std::vector<float>& embedding);

    std::string canonicalAudioHash(const std::string& audio_path) const;

    struct CachedRefCodes {
        TTSReferenceCodes codes;
        uint64_t source_size = 0;
        int64_t source_mtime = 0;
        uint64_t tokenizer_size = 0;
        int64_t tokenizer_mtime = 0;
    };

    bool getFileMeta(const std::string& path, uint64_t& out_size, int64_t& out_mtime) const;

    bool getAudioFileMeta(const std::string& path, uint64_t& out_size, int64_t& out_mtime) const;

    // Build disk cache path for 12Hz reference audio codes
    std::string refCodesDiskCachePath(const std::string& audio_path) const;

    bool lookupRefCodesCache(const std::string& audio_path, TTSReferenceCodes& outCodes);

    void cacheRefCodes(const std::string& audio_path, const TTSReferenceCodes& codes);

private:
    // In-memory speaker embedding cache (keyed by audio file path).
    std::unordered_map<std::string, std::vector<float>> ref_cache_;
    // In-memory reference codes cache (keyed by audio file path).
    std::unordered_map<std::string, CachedRefCodes> ref_codes_cache_;
    // Optional persistent cache directory for speaker embeddings.
    std::string ref_cache_dir_;

    std::string model_dir_;
    std::string lora_dir_;
    std::string system_prompt_ = "You are a helpful assistant.";
    Qwen3Config config_;
    ModelSpec model_spec_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<IModelBackend> backend_;
    std::unique_ptr<aila::vision::Yolo26Detector> detector_;
    std::unique_ptr<aila::vision::IVisionEncoder> vision_encoder_;
    std::unique_ptr<aila::audio::Qwen3ASRAudioEncoder> audio_encoder_;
    std::unique_ptr<aila::audio::MimiEncoder> mimi_encoder_;
    aila::chat::ChatFormatter chat_formatter_;
    Tokenizer tokenizer_;
    bool vision_backend_enabled_ = false;

    // Multi-turn conversation state
    ChatHistory history_;
    std::vector<Message> mm_history_;
    aila::chat::ChatSessionState chat_session_;
    // Exact token IDs currently stored in the KV cache.
    // This is the ground truth for incremental prefill.
    std::vector<int> cached_ids_;

    // Benchmark decode seed (token after prefill argmax)
    int benchmark_seed_token_ = -1;
    bool benchmark_seed_ready_ = false;
    EngineErrorCode last_error_code_ = EngineErrorCode::Ok;
    std::string last_error_message_;
    std::string last_generation_finish_reason_ = "stop";
    std::string last_generation_template_name_;
    bool last_generation_forced_think_close_ = false;
    bool last_generation_think_close_truncated_ = false;
    bool generation_abort_requested_ = false;
    double last_transcribe_duration_s_ = 0.0;
    double last_transcribe_latency_ms_ = 0.0;
    int last_transcribe_tokens_ = 0;
};
