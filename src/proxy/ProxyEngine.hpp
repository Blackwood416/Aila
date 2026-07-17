#pragma once

#include "aila_api.h"
#include "runtime/WorkerProcess.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace aila::proxy {

class ProxyEngine {
public:
    ProxyEngine() = default;
    ~ProxyEngine() noexcept;

    ProxyEngine(const ProxyEngine&) = delete;
    ProxyEngine& operator=(const ProxyEngine&) = delete;

    bool init(std::string_view model_directory, int max_seq_len);
    void reset_context();
    int context_length();
    bool generate_text(
        std::string_view method,
        std::string_view input,
        const AilaGenConfig* config,
        std::string& output);
    bool generate_text_v2(
        std::string_view method,
        std::string_view input,
        const AilaGenConfigV2* config,
        std::string& output);
    int generate_stream(
        std::string_view method,
        std::string_view input,
        const AilaGenConfig* config,
        AilaTokenCallback callback,
        void* user_data);
    int generate_stream_v2(
        std::string_view method,
        std::string_view input,
        const AilaGenConfigV2* config,
        AilaChatStreamCallback callback,
        void* user_data);
    bool transcribe(
        std::string_view wav_path,
        const AilaGenConfig* config,
        const char* forced_language,
        const char* system_prompt,
        float segment_sec,
        int past_text_conditioning,
        AilaTokenCallback callback,
        void* user_data,
        std::string& transcript,
        std::string& language);
    bool transcribe_stream_create(
        const AilaGenConfig* config,
        const char* forced_language,
        const char* system_prompt,
        uint64_t& worker_session,
        uint64_t& stream_id);
    int transcribe_stream_feed(
        uint64_t worker_session, uint64_t stream_id, const float* samples, int sample_count);
    int transcribe_stream_get_text(
        uint64_t worker_session, uint64_t stream_id,
        std::string& stable, std::string& partial);
    void transcribe_stream_destroy(uint64_t worker_session, uint64_t stream_id) noexcept;

    int last_error_code() const;
    const char* last_error_message() const;
    void record_invalid_argument(std::string message);
    void record_runtime_error(std::string message);

private:
    ipc::Frame request_locked(std::string method, std::string payload_json);
    ipc::Frame request_locked(
        std::string method,
        std::string payload_json,
        std::vector<std::byte> attachment);
    bool accept_lifecycle_response_locked(
        const ipc::Frame& response,
        std::string_view expected_method);
    bool accept_error_response_locked(
        const ipc::Frame& response,
        std::string_view expected_method,
        std::string_view fallback_message);
    bool accept_stream_error_response_locked(
        const ipc::Frame& response,
        std::string_view expected_method,
        std::string_view fallback_message);
    bool generate_payload_locked(
        std::string_view method,
        std::string payload_json,
        std::string& output);
    int stream_payload_locked(
        std::string_view method,
        std::string payload_json,
        AilaTokenCallback token_callback,
        AilaChatStreamCallback structured_callback,
        void* user_data,
        std::unique_lock<std::mutex>& engine_lock);
    void set_error_locked(int code, std::string message);
    void set_stream_busy_error_locked();
    void clear_error_locked();
    bool asr_stream_is_active_locked(uint64_t worker_session, uint64_t stream_id) const;
    void remove_asr_stream_locked(uint64_t stream_id);
    void destroy_remote_asr_stream_locked(uint64_t stream_id);
    void flush_deferred_asr_destroys_locked();
    void shutdown_locked() noexcept;

    mutable std::mutex mutex_;
    runtime::WorkerProcess worker_;
    uint64_t next_request_id_ = 1;
    uint64_t worker_session_generation_ = 0;
    uint64_t last_remote_asr_id_ = 0;
    std::vector<uint64_t> active_asr_stream_ids_;
    std::vector<uint64_t> deferred_asr_destroy_ids_;
    bool initialized_ = false;
    bool stream_active_ = false;
    int error_code_ = 0;
    std::string error_message_;
};

} // namespace aila::proxy
