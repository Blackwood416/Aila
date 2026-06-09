#include "ChatTest.hpp"

#include <iostream>

void run_assistant_output_parser_tests();
void run_chat_json_tests();
void run_chat_template_engine_tests();
void run_chat_formatter_tests();

int main() {
    run_assistant_output_parser_tests();
    run_chat_json_tests();
    run_chat_template_engine_tests();
    run_chat_formatter_tests();

    const auto& results = aila::chat::test::registry();
    std::cout << "AilaChatTests: " << results.passed << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
