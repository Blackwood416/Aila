#include "alia/BackgroundExtractorFactory.hpp"
#include "alia/CpuQ35BackgroundExtractor.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

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

void set_env_var(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env_var(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

void background_extractor_kind_parser_defaults_to_gpu(TestResults& results) {
    using aila::alia::BackgroundExtractorKind;
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("") ==
                         BackgroundExtractorKind::GpuLoadedVlm);
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("gpu") ==
                         BackgroundExtractorKind::GpuLoadedVlm);
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("loaded-vlm") ==
                         BackgroundExtractorKind::GpuLoadedVlm);
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("unknown") ==
                         BackgroundExtractorKind::GpuLoadedVlm);
}

void background_extractor_kind_parser_accepts_cpu_aliases(TestResults& results) {
    using aila::alia::BackgroundExtractorKind;
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("cpu") ==
                         BackgroundExtractorKind::NativeCpuQ35);
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("native-cpu-q35") ==
                         BackgroundExtractorKind::NativeCpuQ35);
    AILA_EXPECT_TRUE(results,
                     aila::alia::background_extractor_kind_from_string("Native_CPU_Q35") ==
                         BackgroundExtractorKind::NativeCpuQ35);
}

void background_extractor_env_defaults_to_cpu_and_allows_gpu_override(
    TestResults& results) {
    using aila::alia::BackgroundExtractorKind;
    unset_env_var("AILA_BACKGROUND_EXTRACTOR");
    AILA_EXPECT_TRUE(results,
                     aila::alia::read_background_extractor_kind_from_env() ==
                         BackgroundExtractorKind::NativeCpuQ35);

    set_env_var("AILA_BACKGROUND_EXTRACTOR", "gpu");
    AILA_EXPECT_TRUE(results,
                     aila::alia::read_background_extractor_kind_from_env() ==
                         BackgroundExtractorKind::GpuLoadedVlm);
    unset_env_var("AILA_BACKGROUND_EXTRACTOR");
}

void cpu_background_kind_skips_only_background_gpu_slot(TestResults& results) {
    using aila::alia::BackgroundExtractorKind;
    using aila::alia::ModelRole;

    AILA_EXPECT_TRUE(results,
                     aila::alia::should_load_gpu_model_slot(
                         ModelRole::BackgroundVlm,
                         BackgroundExtractorKind::GpuLoadedVlm));
    AILA_EXPECT_TRUE(results,
                     !aila::alia::should_load_gpu_model_slot(
                         ModelRole::BackgroundVlm,
                         BackgroundExtractorKind::NativeCpuQ35));
    AILA_EXPECT_TRUE(results,
                     aila::alia::should_load_gpu_model_slot(
                         ModelRole::Asr,
                         BackgroundExtractorKind::NativeCpuQ35));
    AILA_EXPECT_TRUE(results,
                     aila::alia::should_load_gpu_model_slot(
                         ModelRole::ForegroundVlm,
                         BackgroundExtractorKind::NativeCpuQ35));
    AILA_EXPECT_TRUE(results,
                     aila::alia::should_load_gpu_model_slot(
                         ModelRole::Tts,
                         BackgroundExtractorKind::NativeCpuQ35));
}

void cpu_extractor_reports_not_loaded_without_model(TestResults& results) {
    aila::alia::CpuQ35BackgroundExtractor extractor("", 2048);
    std::atomic_bool abort_requested{false};
    aila::alia::BackgroundExtractionRequest request;
    request.chat_turn_text = "User: hi\nAssistant: hello";

    const aila::alia::BackgroundExtractionResult result =
        extractor.extract(request, abort_requested);

    AILA_EXPECT_TRUE(results, !extractor.ready());
    AILA_EXPECT_EQ_STRING(results, extractor.backend_name(), "NativeCpuQ35");
    AILA_EXPECT_TRUE(results, !result.ok);
    AILA_EXPECT_EQ_STRING(results,
                          result.error,
                          "native CPU Qwen3.5 background extractor is not loaded");
}

}  // namespace

int main() {
    TestResults results;
    background_extractor_kind_parser_defaults_to_gpu(results);
    background_extractor_kind_parser_accepts_cpu_aliases(results);
    background_extractor_env_defaults_to_cpu_and_allows_gpu_override(results);
    cpu_background_kind_skips_only_background_gpu_slot(results);
    cpu_extractor_reports_not_loaded_without_model(results);

    std::cout << "AilaAliaBackgroundExtractorTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
