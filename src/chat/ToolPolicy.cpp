#include "chat/ToolPolicy.hpp"

namespace aila::chat {

ToolPolicyValidation validate_tool_policy(
    const ChatRequest& request,
    const AssistantChatResult& parsed_output) {
    ToolPolicyValidation validation;
    const bool has_tool_calls = !parsed_output.tool_calls.empty();

    switch (request.tool_choice) {
    case ToolChoice::None:
        if (has_tool_calls) {
            validation.warnings.emplace_back("tool call returned despite tool_choice none");
        }
        break;
    case ToolChoice::Required:
        if (!has_tool_calls) {
            validation.warnings.emplace_back("tool_choice required but no tool call was returned");
        }
        break;
    case ToolChoice::Function:
        if (!has_tool_calls) {
            validation.warnings.emplace_back(
                "tool_choice function " +
                request.tool_choice_function_name +
                " was requested but no tool call was returned");
            break;
        }
        for (const auto& tool_call : parsed_output.tool_calls) {
            if (tool_call.function.name != request.tool_choice_function_name) {
                validation.warnings.emplace_back(
                    "tool_choice function expected " +
                    request.tool_choice_function_name +
                    " but tool call returned " +
                    tool_call.function.name);
            }
        }
        break;
    case ToolChoice::Auto:
        break;
    }

    if (request.tool_policy == ToolPolicyMode::Strict && !validation.warnings.empty()) {
        validation.hard_error = true;
    }

    return validation;
}

} // namespace aila::chat
