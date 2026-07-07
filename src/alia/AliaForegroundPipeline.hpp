#pragma once

#include "alia_api.h"
#include "AliaTtsTextChunker.hpp"
#include "engine/Types.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aila::alia {

class AliaAsrPipeline;
class AliaTtsPipeline;
class ModelSlot;

bool is_valid_generation_config(const AliaGenConfig& config);
GenerationConfig translate_generation_config(const AliaGenConfig* config);
std::string foreground_system_prompt();

enum class ForegroundDecodeMode {
    None,
    LoadedVlm
};

enum class ForegroundTurnState {
    Idle,
    Running,
    Aborting,
    Aborted,
    Completed,
    Failed
};

struct AliaForegroundMetrics {
    int prompt_tokens = 0;
    int prefilled_prompt_tokens = 0;
    int prompt_suffix_tokens = 0;
    int final_cached_prefix_rejected = 0;
    int generated_tokens = 0;
    long long prompt_build_ms = -1;
    long long prompt_prefill_ms = -1;
    long long first_token_delta_ms = -1;
    long long first_content_delta_ms = -1;
    long long first_tts_enqueue_ms = -1;
    long long tts_first_audio_priority_wait_ms = 0;
    long long first_tts_chunk_wait_ms = -1;
    int first_tts_chunk_wait_tokens = -1;
    int first_tts_chunk_pending_chars_at_first_content = -1;
    int first_tts_chunk_pending_chars_at_enqueue = -1;
    long long decode_ms = -1;
    long long model_ms = -1;
    std::string final_cached_prefix_reject_reason;
    std::string final_prefix_path = "fresh_full";
    std::string first_tts_chunk_reason;
};

class AliaForegroundPipeline {
public:
    AliaForegroundPipeline(ModelSlot* vlm_slot,
                           AliaTtsPipeline* tts_pipeline,
                           AliaAsrPipeline* asr_pipeline);
    ~AliaForegroundPipeline();

    bool start_turn(const AliaGenConfig* config,
                    AliaToolCallCallback tool_cb,
                    AliaAudioCallback audio_cb,
                    void* user_data);
    bool start_speculative_turn(const std::string& stable_text,
                                const std::string& partial_text,
                                const AliaGenConfig* config);
    bool commit_speculative_turn(const std::string& stable_text,
                                 const std::string& partial_text,
                                 const AliaGenConfig* config,
                                 AliaToolCallCallback tool_cb,
                                 AliaAudioCallback audio_cb,
                                 void* user_data);
    AliaErrorCode prefill_asr_text(const std::string& stable_text,
                                   const std::string& partial_text);
    bool warmup_loaded_vlm(std::string* error_message = nullptr);
    void request_abort();
    AliaErrorCode rollback_kv_cache(int rollback_tokens);
    void join();

    ForegroundTurnState state() const;
    bool wait_until_idle_for(std::chrono::milliseconds timeout);
    GenerationConfig last_generation_config() const;
    std::string last_user_text() const;
    std::string last_assistant_text() const;
    std::vector<std::string> last_action_tags() const;
    std::string last_tool_call_json() const;
    std::string last_tool_result_text() const;
    std::string last_tool_resume_prompt_text() const;
    ForegroundDecodeMode last_decode_mode() const;
    int last_prompt_token_count() const;
    int last_generated_token_count() const;
    int last_asr_prefill_token_count() const;
    int last_asr_prefill_reused_token_count() const;
    int last_asr_prefill_suffix_token_count() const;
    int last_asr_prefill_candidate_token_count() const;
    int last_asr_prefill_candidate_suffix_token_count() const;
    std::string last_asr_prefill_skip_reason() const;
    int last_asr_prefill_skipped_small_suffix_count() const;
    long long last_asr_prefill_ms() const;
    long long last_first_content_delta_ms() const;
    long long last_first_tts_enqueue_ms() const;
    AliaForegroundMetrics last_metrics() const;
    bool last_speculative_commit_hit() const;
    std::string last_speculative_commit_reason() const;
    std::string last_error_text() const;
    const std::string& last_error() const { return last_error_; }

private:
    void run_turn(AliaGenConfig config,
                  AliaToolCallCallback tool_cb,
                  AliaAudioCallback audio_cb,
                  void* user_data);
    void run_speculative_turn(std::string stable_text,
                              std::string partial_text,
                              AliaGenConfig config);
    void run_commit_speculative_turn(std::string stable_text,
                                     std::string partial_text,
                                     AliaGenConfig config,
                                     AliaToolCallCallback tool_cb,
                                     AliaAudioCallback audio_cb,
                                     void* user_data);
    bool synthesize_committed_text(const std::string& user_text,
                                   const std::string& spoken_text,
                                   const AliaGenConfig& config,
                                   AliaAudioCallback audio_cb,
                                   void* user_data,
                                   std::chrono::steady_clock::time_point turn_start);
    bool can_generate_with_loaded_vlm() const;
    bool generate_with_loaded_vlm(const std::string& user_text,
                                  const GenerationConfig& config,
                                  std::string& assistant_text,
                                  bool reset_session,
                                  bool record_generation_anchor,
                                  bool use_chat_template,
                                  bool stop_on_tool_call,
                                  const std::vector<int>* prompt_override_ids,
                                  int prefilled_prompt_tokens,
                                  const AliaGenConfig* tts_config,
                                  AliaAudioCallback audio_cb,
                                  void* user_data,
                                  std::chrono::steady_clock::time_point turn_start);
    bool process_tool_calls(const std::string& raw_assistant_text,
                            const std::string& user_text,
                            AliaToolCallCallback tool_cb,
                            void* user_data,
                            std::string& spoken_text);
    bool abort_requested() const;
    bool is_busy_locked() const;

    ModelSlot* vlm_slot_ = nullptr;
    AliaTtsPipeline* tts_pipeline_ = nullptr;
    AliaAsrPipeline* asr_pipeline_ = nullptr;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    ForegroundTurnState state_ = ForegroundTurnState::Idle;
    bool abort_requested_ = false;
    std::string last_error_;
    GenerationConfig last_generation_config_;
    std::string last_user_text_;
    std::string last_assistant_text_;
    std::vector<std::string> last_action_tags_;
    std::string last_tool_call_json_;
    std::string last_tool_result_text_;
    std::string last_tool_resume_prompt_text_;
    int generation_start_context_len_ = -1;
    int last_prompt_token_count_ = 0;
    int last_generated_token_count_ = 0;
    long long last_first_content_delta_ms_ = -1;
    long long last_first_tts_enqueue_ms_ = -1;
    AliaForegroundMetrics metrics_;
    std::vector<int> generation_anchor_prompt_ids_;
    std::vector<int> generation_token_ids_;
    bool speculative_ready_ = false;
    std::string speculative_user_text_;
    std::string speculative_assistant_text_;
    std::vector<std::string> speculative_action_tags_;
    std::vector<int> speculative_prompt_ids_;
    std::vector<int> speculative_generated_token_ids_;
    bool last_speculative_commit_hit_ = false;
    std::string last_speculative_commit_reason_;
    std::vector<int> asr_prefill_prompt_ids_;
    std::string asr_prefill_text_;
    int last_asr_prefill_token_count_ = 0;
    int last_asr_prefill_reused_token_count_ = 0;
    int last_asr_prefill_suffix_token_count_ = 0;
    int last_asr_prefill_candidate_token_count_ = 0;
    int last_asr_prefill_candidate_suffix_token_count_ = 0;
    std::string last_asr_prefill_skip_reason_ = "not evaluated";
    int last_asr_prefill_skipped_small_suffix_count_ = 0;
    long long last_asr_prefill_ms_ = -1;
    ForegroundDecodeMode last_decode_mode_ = ForegroundDecodeMode::None;
};

}  // namespace aila::alia
