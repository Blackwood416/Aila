#pragma once

#include "alia_api.h"
#include "engine/Types.hpp"

#include <chrono>
#include <condition_variable>
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
std::vector<std::string> split_spoken_text_for_tts(const std::string& text);

enum class ForegroundDecodeMode {
    None,
    NoModelFallback,
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
    void request_abort();
    AliaErrorCode rollback_kv_cache(int rollback_tokens);
    void join();

    ForegroundTurnState state() const;
    bool wait_until_idle_for(std::chrono::milliseconds timeout);
    GenerationConfig last_generation_config() const;
    std::string last_user_text() const;
    std::string last_assistant_text() const;
    std::string last_tool_call_json() const;
    std::string last_tool_result_text() const;
    std::string last_tool_resume_prompt_text() const;
    ForegroundDecodeMode last_decode_mode() const;
    const std::string& last_error() const { return last_error_; }

private:
    void run_turn(AliaGenConfig config,
                  AliaToolCallCallback tool_cb,
                  AliaAudioCallback audio_cb,
                  void* user_data);
    bool can_generate_with_loaded_vlm() const;
    bool generate_with_loaded_vlm(const std::string& user_text,
                                  const GenerationConfig& config,
                                  std::string& assistant_text,
                                  bool reset_session,
                                  bool record_generation_anchor,
                                  bool use_chat_template,
                                  bool stop_on_tool_call,
                                  const AliaGenConfig* tts_config,
                                  AliaAudioCallback audio_cb,
                                  void* user_data);
    bool process_tool_calls(const std::string& raw_assistant_text,
                            const std::string& user_text,
                            AliaToolCallCallback tool_cb,
                            void* user_data,
                            std::string& spoken_text);
    void synthesize_spoken_text(const std::string& spoken_text,
                                const AliaGenConfig& config,
                                AliaAudioCallback audio_cb,
                                void* user_data);
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
    std::string last_tool_call_json_;
    std::string last_tool_result_text_;
    std::string last_tool_resume_prompt_text_;
    int generation_start_context_len_ = -1;
    int last_generated_token_count_ = 0;
    ForegroundDecodeMode last_decode_mode_ = ForegroundDecodeMode::None;
};

}  // namespace aila::alia
