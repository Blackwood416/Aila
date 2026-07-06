#include <iostream>
#include <string>

namespace aila::alia {
std::string build_asr_language_hint_prefix(std::string language);
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

    if (failures != 0) {
        std::cerr << "AilaAliaAsrLanguageHintTests: failed\n";
        return 1;
    }
    std::cout << "AilaAliaAsrLanguageHintTests: passed\n";
    return 0;
}
