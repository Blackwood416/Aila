#include <iostream>
#include <string>
#include <vector>

namespace aila::alia {
std::string build_asr_language_hint_prefix(std::string language);
std::vector<int> asr_backend_warmup_lengths(int target_prompt_tokens);
}

namespace {

int expect_eq(const std::string& name,
              const std::string& actual,
              const std::string& expected) {
    if (actual == expected) {
        return 0;
    }
    std::cerr << name << ": expected [" << expected << "], got ["
              << actual << "]\n";
    return 1;
}

int expect_eq(const std::string& name,
              const std::vector<int>& actual,
              const std::vector<int>& expected) {
    if (actual == expected) {
        return 0;
    }
    std::cerr << name << ": expected [";
    for (size_t i = 0; i < expected.size(); ++i) {
        if (i != 0) std::cerr << ",";
        std::cerr << expected[i];
    }
    std::cerr << "], got [";
    for (size_t i = 0; i < actual.size(); ++i) {
        if (i != 0) std::cerr << ",";
        std::cerr << actual[i];
    }
    std::cerr << "]\n";
    return 1;
}

}  // namespace

int main() {
    int failures = 0;
    failures += expect_eq(
        "normalizes chinese",
        aila::alia::build_asr_language_hint_prefix(" chinese "),
        "language Chinese");
    failures += expect_eq(
        "keeps explicit english",
        aila::alia::build_asr_language_hint_prefix("English"),
        "language English");
    failures += expect_eq(
        "disables auto",
        aila::alia::build_asr_language_hint_prefix("auto"),
        "");
    failures += expect_eq(
        "disables off",
        aila::alia::build_asr_language_hint_prefix("off"),
        "");
    failures += expect_eq(
        "warms prefix append shapes",
        aila::alia::asr_backend_warmup_lengths(128),
        std::vector<int>({24, 32, 48, 64, 80, 128}));
    failures += expect_eq(
        "deduplicates target shape",
        aila::alia::asr_backend_warmup_lengths(64),
        std::vector<int>({24, 32, 48, 64}));
    failures += expect_eq(
        "keeps small target",
        aila::alia::asr_backend_warmup_lengths(20),
        std::vector<int>({20}));

    if (failures != 0) {
        std::cerr << "AilaAliaAsrLanguageHintTests: failed\n";
        return 1;
    }
    std::cout << "AilaAliaAsrLanguageHintTests: passed\n";
    return 0;
}
