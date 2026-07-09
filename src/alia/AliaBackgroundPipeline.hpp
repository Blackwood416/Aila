#pragma once

#include "BackgroundMemoryExtractor.hpp"
#include "alia_api.h"
#include "engine/Types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aila::alia {

class ModelSlot;

std::string background_system_prompt();
std::string build_background_extraction_prompt(const std::string& chat_turn_text);
std::string build_background_schema_repair_prompt(const std::string& chat_turn_text,
                                                  const std::string& invalid_output);
std::string enforce_background_result_schema(const std::string& raw_result_json,
                                             const std::string& chat_turn_text);
std::string normalize_background_model_json(const std::string& raw_result_json);
std::string cleanup_background_result_json(const std::string& raw_result_json,
                                           const std::string& chat_turn_text);

enum class BackgroundDecodeMode {
    None,
    LoadedVlm,
    NativeCpuQ35
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
    explicit AliaBackgroundPipeline(std::unique_ptr<IBackgroundMemoryExtractor> extractor,
                                    size_t queue_capacity = 8);
    ~AliaBackgroundPipeline();

    void register_callback(AliaBackgroundResultCallback callback, void* user_data = nullptr);
    bool trigger(std::string chat_turn_text);
    void request_abort();
    void join();

    BackgroundJobState state() const;
    bool wait_until_idle_for(std::chrono::milliseconds timeout);
    BackgroundDecodeMode last_decode_mode() const;
    std::string last_prompt_text() const;
    std::string last_result_json() const;
    std::string last_error_text() const;
    int last_schema_retry_count() const;
    bool last_schema_repair_applied() const;
    std::string last_schema_diagnostic() const;
    static std::string json_escape(const std::string& value);
    static bool has_required_schema_keys(const std::string& value);
    const std::string& last_error() const { return last_error_; }

private:
    void run_job(std::string chat_turn_text, AliaBackgroundResultCallback callback, void* user_data);
    void worker_loop(AliaBackgroundResultCallback callback, void* user_data);
    bool is_busy_locked() const;

    std::unique_ptr<IBackgroundMemoryExtractor> extractor_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::deque<std::string> queued_jobs_;
    AliaBackgroundResultCallback callback_ = nullptr;
    void* callback_user_data_ = nullptr;
    BackgroundJobState state_ = BackgroundJobState::Idle;
    std::atomic_bool abort_requested_{false};
    size_t queue_capacity_ = 8;
    std::string last_error_;
    std::string last_prompt_text_;
    std::string last_result_json_;
    int last_schema_retry_count_ = 0;
    bool last_schema_repair_applied_ = false;
    std::string last_schema_diagnostic_;
    BackgroundDecodeMode last_decode_mode_ = BackgroundDecodeMode::None;
};

}  // namespace aila::alia
