#include "ChatTest.hpp"

#include <iostream>

void run_assistant_output_parser_tests();
void run_chat_json_tests();
void run_chat_template_engine_tests();
void run_chat_formatter_tests();
void run_thinking_budget_controller_tests();
void run_structured_stream_parser_tests();
void run_tool_policy_tests();

int main() {
    run_assistant_output_parser_tests();
    run_chat_json_tests();
    run_chat_template_engine_tests();
    run_chat_formatter_tests();
    run_thinking_budget_controller_tests();
    run_structured_stream_parser_tests();
    run_tool_policy_tests();

    const auto& results = aila::chat::test::registry();
    std::cout << "AilaChatTests: " << results.passed << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
