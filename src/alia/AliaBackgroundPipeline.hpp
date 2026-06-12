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

class ModelSlot;

std::string background_system_prompt();
std::string build_background_extraction_prompt(const std::string& chat_turn_text);
std::string make_background_fallback_json(const std::string& chat_turn_text);
std::string enforce_background_result_schema(const std::string& raw_result_json,
                                             const std::string& chat_turn_text);

enum class BackgroundDecodeMode {
    None,
    NoModelFallback,
    LoadedVlm
};

enum class BackgroundJobState {
    Idle,
    Running,
    Aborting,
    Completed,
    Failed
};

class AliaBackgroundPipeline {
public:
    explicit AliaBackgroundPipeline(ModelSlot* slot);
    ~AliaBackgroundPipeline();

    void register_callback(AliaBackgroundResultCallback callback);
    bool trigger(std::string chat_turn_text);
    void request_abort();
    void join();

    BackgroundJobState state() const;
    bool wait_until_idle_for(std::chrono::milliseconds timeout);
    BackgroundDecodeMode last_decode_mode() const;
    std::string last_prompt_text() const;
    std::string last_result_json() const;
    int last_schema_retry_count() const;
    bool last_schema_repair_applied() const;
    std::string last_schema_diagnostic() const;
    static std::string json_escape(const std::string& value);
    static bool has_required_schema_keys(const std::string& value);
    const std::string& last_error() const { return last_error_; }

private:
    void run_job(std::string chat_turn_text, AliaBackgroundResultCallback callback);
    bool can_generate_with_loaded_vlm() const;
    bool generate_with_loaded_vlm(const std::string& prompt_text,
                                  std::string& result_json);
    bool is_busy_locked() const;

    ModelSlot* slot_ = nullptr;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    AliaBackgroundResultCallback callback_ = nullptr;
    BackgroundJobState state_ = BackgroundJobState::Idle;
    bool abort_requested_ = false;
    std::string last_error_;
    std::string last_prompt_text_;
    std::string last_result_json_;
    int last_schema_retry_count_ = 0;
    bool last_schema_repair_applied_ = false;
    std::string last_schema_diagnostic_;
    BackgroundDecodeMode last_decode_mode_ = BackgroundDecodeMode::None;
};

}  // namespace aila::alia
