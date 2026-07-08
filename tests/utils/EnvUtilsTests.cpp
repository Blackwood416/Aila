#include "utils/EnvUtils.hpp"

#include <cstdlib>
#include <iostream>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(
    TestResults& results,
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

void clear_kv_env() {
    unset_env_var("AILA_KV_QUANT");
    unset_env_var("AILA_ASR_KV_QUANT");
    unset_env_var("AILA_TTS_KV_QUANT");
    unset_env_var("AILA_VLM_KV_QUANT");
    aila::env::g_kv_quant_override = false;
}

void scoped_kv_quant_defaults_to_global_off(TestResults& results) {
    clear_kv_env();

    AILA_EXPECT_TRUE(
        results,
        !aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT"));
}

void scoped_kv_quant_inherits_global_on(TestResults& results) {
    clear_kv_env();
    set_env_var("AILA_KV_QUANT", "1");

    AILA_EXPECT_TRUE(
        results,
        aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT"));
}

void scoped_kv_quant_can_disable_global_on(TestResults& results) {
    clear_kv_env();
    set_env_var("AILA_KV_QUANT", "1");
    set_env_var("AILA_VLM_KV_QUANT", "0");

    AILA_EXPECT_TRUE(
        results,
        !aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT"));
}

void scoped_kv_quant_can_enable_global_off(TestResults& results) {
    clear_kv_env();
    set_env_var("AILA_KV_QUANT", "0");
    set_env_var("AILA_VLM_KV_QUANT", "1");

    AILA_EXPECT_TRUE(
        results,
        aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT"));
}

void scoped_kv_quant_can_disable_cli_global_override(TestResults& results) {
    clear_kv_env();
    aila::env::g_kv_quant_override = true;
    set_env_var("AILA_ASR_KV_QUANT", "0");

    AILA_EXPECT_TRUE(
        results,
        !aila::env::read_scoped_kv_quant("AILA_ASR_KV_QUANT"));
}

}  // namespace

int main() {
    TestResults results;
    scoped_kv_quant_defaults_to_global_off(results);
    scoped_kv_quant_inherits_global_on(results);
    scoped_kv_quant_can_disable_global_on(results);
    scoped_kv_quant_can_enable_global_off(results);
    scoped_kv_quant_can_disable_cli_global_override(results);
    clear_kv_env();

    std::cout << "AilaEnvUtilsTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
