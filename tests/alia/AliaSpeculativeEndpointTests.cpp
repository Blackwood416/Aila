#include "alia/AliaSpeculativeEndpoint.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(TestResults& results,
                 bool value,
                 const char* expression,
                 const char* file,
                 int line) {
    if (value) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression << '\n';
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

bool wait_for_state(aila::alia::AliaSpeculativeEndpointController& controller,
                    int expected_state,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (controller.metrics().state == expected_state) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return controller.metrics().state == expected_state;
}

aila::alia::AliaSpeculativeEndpointOperations make_operations(
    std::atomic<int>& reset_calls,
    std::atomic<int>& asr_calls,
    std::atomic<int>& prefill_calls,
    std::atomic<int>& start_calls,
    AliaErrorCode prefill_result = ALIA_OK) {
    aila::alia::AliaSpeculativeEndpointOperations operations;
    operations.reset_asr = [&]() { ++reset_calls; };
    operations.get_asr_text = [&](std::string& stable, std::string& partial) {
        ++asr_calls;
        stable.clear();
        partial = "啊";
    };
    operations.get_candidate_asr_text = operations.get_asr_text;
    operations.prefill_asr_text = [&](const std::string&, const std::string&) {
        ++prefill_calls;
        return prefill_result;
    };
    operations.start_foreground = [&](const std::string& user_text,
                                      const AliaGenConfig*,
                                      AliaToolCallCallback,
                                      AliaAudioCallback,
                                      void*) {
        ++start_calls;
        return user_text == "啊";
    };
    operations.prefill_token_count = []() { return 42; };
    operations.prefill_reused_token_count = []() { return 36; };
    operations.prefill_suffix_token_count = []() { return 6; };
    return operations;
}

aila::alia::AliaSpeculativeEndpointConfig enabled_config() {
    aila::alia::AliaSpeculativeEndpointConfig config;
    config.enabled = true;
    config.candidate_silence_frames = 5;
    config.speech_probability_threshold = 0.5f;
    config.allow_cold_candidate = true;
    return config;
}

void utf8_content_filter_accepts_tiny_chinese(TestResults& results) {
    AILA_EXPECT_TRUE(results, aila::alia::contains_spoken_utf8_content("啊"));
    AILA_EXPECT_TRUE(results, aila::alia::contains_spoken_utf8_content("我"));
    AILA_EXPECT_TRUE(results, !aila::alia::contains_spoken_utf8_content("……。！？"));
    AILA_EXPECT_TRUE(results, !aila::alia::contains_spoken_utf8_content("  ... "));
}

void silence_threshold_starts_one_candidate(TestResults& results) {
    std::atomic<int> reset_calls{0}, asr_calls{0}, prefill_calls{0}, start_calls{0};
    aila::alia::AliaSpeculativeEndpointController controller(
        make_operations(reset_calls, asr_calls, prefill_calls, start_calls),
        enabled_config());
    AILA_EXPECT_TRUE(results, controller.begin() == ALIA_OK);
    for (int i = 0; i < 4; ++i) {
        AILA_EXPECT_TRUE(results, controller.observe_vad(0.1f) == ALIA_OK);
    }
    AILA_EXPECT_TRUE(results, prefill_calls.load() == 0);
    AILA_EXPECT_TRUE(results, controller.observe_vad(0.1f) == ALIA_OK);
    AILA_EXPECT_TRUE(results, wait_for_state(controller, ALIA_SPECULATIVE_ENDPOINT_READY));
    AILA_EXPECT_TRUE(results, prefill_calls.load() == 1);
    AILA_EXPECT_TRUE(results, controller.metrics().trigger_count == 1);
    controller.observe_vad(0.1f);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    AILA_EXPECT_TRUE(results, prefill_calls.load() == 1);
}

void resumed_speech_invalidates_inflight_candidate(TestResults& results) {
    std::atomic<int> reset_calls{0}, asr_calls{0}, prefill_calls{0}, start_calls{0};
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool entered = false;
    bool release = false;
    auto operations = make_operations(reset_calls, asr_calls, prefill_calls, start_calls);
    operations.prefill_asr_text = [&](const std::string&, const std::string&) {
        ++prefill_calls;
        std::unique_lock<std::mutex> lock(gate_mutex);
        entered = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&]() { return release; });
        return ALIA_OK;
    };
    aila::alia::AliaSpeculativeEndpointController controller(
        std::move(operations), enabled_config());
    controller.begin();
    for (int i = 0; i < 5; ++i) {
        controller.observe_vad(0.1f);
    }
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        AILA_EXPECT_TRUE(results, gate_cv.wait_for(
            lock, std::chrono::milliseconds(500), [&]() { return entered; }));
    }
    controller.observe_vad(0.9f);
    for (int i = 0; i < 5; ++i) {
        controller.observe_vad(0.9f);
    }
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release = true;
    }
    gate_cv.notify_all();
    const auto stale_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(500);
    while (controller.metrics().stale_completion_count == 0 &&
           std::chrono::steady_clock::now() < stale_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    AILA_EXPECT_TRUE(results, wait_for_state(controller, ALIA_SPECULATIVE_ENDPOINT_INVALIDATED));
    AILA_EXPECT_TRUE(results, controller.metrics().resume_count == 1);
    AILA_EXPECT_TRUE(results, controller.metrics().stale_completion_count == 1);
}

void commit_waits_for_candidate_and_reconciles_final_prefill(TestResults& results) {
    std::atomic<int> reset_calls{0}, asr_calls{0}, prefill_calls{0}, start_calls{0};
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool entered = false;
    bool release = false;
    auto operations = make_operations(reset_calls, asr_calls, prefill_calls, start_calls);
    operations.prefill_asr_text = [&](const std::string&, const std::string&) {
        const int call = ++prefill_calls;
        if (call == 1) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            entered = true;
            gate_cv.notify_all();
            gate_cv.wait(lock, [&]() { return release; });
        }
        return ALIA_OK;
    };
    aila::alia::AliaSpeculativeEndpointController controller(
        std::move(operations), enabled_config());
    controller.begin();
    for (int i = 0; i < 5; ++i) {
        controller.observe_vad(0.1f);
    }
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        AILA_EXPECT_TRUE(results, gate_cv.wait_for(
            lock, std::chrono::milliseconds(500), [&]() { return entered; }));
    }
    auto commit_future = std::async(std::launch::async, [&]() {
        return controller.commit(nullptr, nullptr, nullptr, nullptr);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    AILA_EXPECT_TRUE(results, start_calls.load() == 0);
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release = true;
    }
    gate_cv.notify_all();
    AILA_EXPECT_TRUE(results, commit_future.get() == ALIA_OK);
    AILA_EXPECT_TRUE(results, asr_calls.load() == 1);
    AILA_EXPECT_TRUE(results, prefill_calls.load() == 2);
    AILA_EXPECT_TRUE(results, start_calls.load() == 1);
    AILA_EXPECT_TRUE(results, controller.metrics().commit_prefill_hit == 1);
    AILA_EXPECT_TRUE(results, controller.metrics().final_asr_reused_candidate == 1);
}

void disabled_endpoint_preserves_fresh_commit_baseline(TestResults& results) {
    std::atomic<int> reset_calls{0}, asr_calls{0}, prefill_calls{0}, start_calls{0};
    auto config = enabled_config();
    config.enabled = false;
    aila::alia::AliaSpeculativeEndpointController controller(
        make_operations(reset_calls, asr_calls, prefill_calls, start_calls), config);
    controller.begin();
    for (int i = 0; i < 15; ++i) {
        controller.observe_vad(0.1f);
    }
    AILA_EXPECT_TRUE(results, controller.commit(nullptr, nullptr, nullptr, nullptr) == ALIA_OK);
    AILA_EXPECT_TRUE(results, prefill_calls.load() == 0);
    AILA_EXPECT_TRUE(results, start_calls.load() == 1);
}

void final_prefill_failure_falls_back_to_foreground(TestResults& results) {
    std::atomic<int> reset_calls{0}, asr_calls{0}, prefill_calls{0}, start_calls{0};
    aila::alia::AliaSpeculativeEndpointController controller(
        make_operations(reset_calls,
                        asr_calls,
                        prefill_calls,
                        start_calls,
                        ALIA_ERR_CONTEXT_OVERFLOW),
        enabled_config());
    controller.begin();
    AILA_EXPECT_TRUE(results, controller.commit(nullptr, nullptr, nullptr, nullptr) == ALIA_OK);
    AILA_EXPECT_TRUE(results, prefill_calls.load() == 1);
    AILA_EXPECT_TRUE(results, start_calls.load() == 1);
    AILA_EXPECT_TRUE(results,
                     controller.metrics().final_prefill_rc == ALIA_ERR_CONTEXT_OVERFLOW);
}

}  // namespace

int main() {
    TestResults results;
    utf8_content_filter_accepts_tiny_chinese(results);
    silence_threshold_starts_one_candidate(results);
    resumed_speech_invalidates_inflight_candidate(results);
    commit_waits_for_candidate_and_reconciles_final_prefill(results);
    disabled_endpoint_preserves_fresh_commit_baseline(results);
    final_prefill_failure_falls_back_to_foreground(results);
    std::cout << "AilaAliaSpeculativeEndpointTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
