#include "alia/AliaBackgroundPipeline.hpp"
#include "alia/BackgroundMemoryExtractor.hpp"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(TestResults& results, bool value, const char* expression,
                 const char* file, int line) {
    if (value) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression << '\n';
}

void expect_eq_string(TestResults& results, const std::string& left,
                      const std::string& right, const char* left_expression,
                      const char* right_expression, const char* file, int line) {
    if (left == right) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << left_expression
              << " == " << right_expression << ", got \"" << left
              << "\" vs \"" << right << "\"\n";
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_STRING(results, left, right) \
    expect_eq_string((results), (left), (right), #left, #right, __FILE__, __LINE__)

struct CallbackCapture {
    int calls = 0;
    std::string json;
};

void capture_callback(const char* json, void* user_data) {
    auto* capture = static_cast<CallbackCapture*>(user_data);
    ++capture->calls;
    capture->json = json ? json : "";
}

class FakeExtractor : public aila::alia::IBackgroundMemoryExtractor {
public:
    aila::alia::BackgroundExtractionResult next_result;
    std::string seen_text;
    std::string backend_name_value = "fake";
    int calls = 0;
    int sleep_ms = 0;
    std::mutex mutex;
    std::condition_variable cv;
    int started_calls = 0;

    bool ready() const override { return ready_; }
    const char* backend_name() const override { return backend_name_value.c_str(); }

    void wait_for_started(int expected) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return started_calls >= expected;
        });
    }

    aila::alia::BackgroundExtractionResult extract(
        const aila::alia::BackgroundExtractionRequest& request,
        const std::atomic_bool& abort_requested) override {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++started_calls;
            cv.notify_all();
        }
        if (sleep_ms > 0) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(sleep_ms);
            while (std::chrono::steady_clock::now() < deadline &&
                   !abort_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        ++calls;
        seen_text = request.chat_turn_text;
        return next_result;
    }

    bool ready_ = true;
};

void fake_extractor_success_updates_result_and_callback(TestResults& results) {
    auto fake = std::make_unique<FakeExtractor>();
    FakeExtractor* raw_fake = fake.get();
    raw_fake->next_result.ok = true;
    raw_fake->next_result.result_json =
        "{\"summary\":\"ok\",\"memory_candidates\":[],\"preferences\":[],\"tasks\":[]}";
    raw_fake->next_result.prompt_text = "prompt";
    raw_fake->next_result.schema_diagnostic = "fake accepted";

    aila::alia::AliaBackgroundPipeline pipeline(std::move(fake), 4);
    CallbackCapture capture;
    pipeline.register_callback(capture_callback, &capture);

    AILA_EXPECT_TRUE(results, pipeline.trigger("User: hi\nAssistant: hello"));
    AILA_EXPECT_TRUE(results, pipeline.wait_until_idle_for(std::chrono::seconds(2)));

    AILA_EXPECT_TRUE(results, raw_fake->calls == 1);
    AILA_EXPECT_EQ_STRING(results, raw_fake->seen_text, "User: hi\nAssistant: hello");
    AILA_EXPECT_TRUE(results, capture.calls == 1);
    AILA_EXPECT_EQ_STRING(results, capture.json, raw_fake->next_result.result_json);
    AILA_EXPECT_EQ_STRING(results, pipeline.last_result_json(), raw_fake->next_result.result_json);
    AILA_EXPECT_EQ_STRING(results, pipeline.last_prompt_text(), "prompt");
    AILA_EXPECT_EQ_STRING(results, pipeline.last_schema_diagnostic(), "fake accepted");
}

void fake_extractor_failure_sets_failed_state(TestResults& results) {
    auto fake = std::make_unique<FakeExtractor>();
    fake->next_result.ok = false;
    fake->next_result.error = "fake failed";

    aila::alia::AliaBackgroundPipeline pipeline(std::move(fake), 4);
    CallbackCapture capture;
    pipeline.register_callback(capture_callback, &capture);

    AILA_EXPECT_TRUE(results, pipeline.trigger("User: hi"));
    AILA_EXPECT_TRUE(results, pipeline.wait_until_idle_for(std::chrono::seconds(2)));
    AILA_EXPECT_TRUE(results, capture.calls == 0);
    AILA_EXPECT_EQ_STRING(results, pipeline.last_error_text(), "fake failed");
    AILA_EXPECT_TRUE(results, pipeline.state() == aila::alia::BackgroundJobState::Failed);
}

void cpu_extractor_failure_keeps_decode_mode_diagnostic(TestResults& results) {
    auto fake = std::make_unique<FakeExtractor>();
    fake->backend_name_value = "NativeCpuQ35";
    fake->next_result.ok = false;
    fake->next_result.error = "native CPU Qwen3.5 background extractor inference is not implemented";

    aila::alia::AliaBackgroundPipeline pipeline(std::move(fake), 4);
    CallbackCapture capture;
    pipeline.register_callback(capture_callback, &capture);

    AILA_EXPECT_TRUE(results, pipeline.trigger("User: hi"));
    AILA_EXPECT_TRUE(results, pipeline.wait_until_idle_for(std::chrono::seconds(2)));
    AILA_EXPECT_TRUE(results, capture.calls == 0);
    AILA_EXPECT_TRUE(results,
                     pipeline.last_decode_mode() ==
                         aila::alia::BackgroundDecodeMode::NativeCpuQ35);
    AILA_EXPECT_EQ_STRING(
        results,
        pipeline.last_error_text(),
        "native CPU Qwen3.5 background extractor inference is not implemented");
}

void bounded_queue_rejects_when_pending_capacity_is_full(TestResults& results) {
    auto fake = std::make_unique<FakeExtractor>();
    FakeExtractor* raw_fake = fake.get();
    raw_fake->sleep_ms = 200;
    raw_fake->next_result.ok = true;
    raw_fake->next_result.result_json =
        "{\"summary\":\"ok\",\"memory_candidates\":[],\"preferences\":[],\"tasks\":[]}";

    aila::alia::AliaBackgroundPipeline pipeline(std::move(fake), 1);
    CallbackCapture capture;
    pipeline.register_callback(capture_callback, &capture);

    AILA_EXPECT_TRUE(results, pipeline.trigger("turn 1"));
    raw_fake->wait_for_started(1);
    AILA_EXPECT_TRUE(results, pipeline.trigger("turn 2"));
    AILA_EXPECT_TRUE(results, !pipeline.trigger("turn 3"));
    AILA_EXPECT_TRUE(results, pipeline.wait_until_idle_for(std::chrono::seconds(3)));
    AILA_EXPECT_TRUE(results, raw_fake->calls == 2);
    AILA_EXPECT_TRUE(results, capture.calls == 2);
}

void abort_clears_pending_jobs(TestResults& results) {
    auto fake = std::make_unique<FakeExtractor>();
    FakeExtractor* raw_fake = fake.get();
    raw_fake->sleep_ms = 300;
    raw_fake->next_result.ok = true;
    raw_fake->next_result.result_json =
        "{\"summary\":\"ok\",\"memory_candidates\":[],\"preferences\":[],\"tasks\":[]}";

    aila::alia::AliaBackgroundPipeline pipeline(std::move(fake), 4);
    CallbackCapture capture;
    pipeline.register_callback(capture_callback, &capture);

    AILA_EXPECT_TRUE(results, pipeline.trigger("turn 1"));
    raw_fake->wait_for_started(1);
    AILA_EXPECT_TRUE(results, pipeline.trigger("turn 2"));
    pipeline.request_abort();
    AILA_EXPECT_TRUE(results, !pipeline.trigger("turn 3"));
    pipeline.join();

    AILA_EXPECT_TRUE(results, raw_fake->calls <= 1);
    AILA_EXPECT_TRUE(results, capture.calls <= 1);
}

}  // namespace

int main() {
    TestResults results;
    fake_extractor_success_updates_result_and_callback(results);
    fake_extractor_failure_sets_failed_state(results);
    cpu_extractor_failure_keeps_decode_mode_diagnostic(results);
    bounded_queue_rejects_when_pending_capacity_is_full(results);
    abort_clears_pending_jobs(results);

    std::cout << "AilaAliaBackgroundPipelineTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
