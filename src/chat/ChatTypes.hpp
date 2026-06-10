#pragma once

#include "engine/Types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace aila::chat {

enum class ChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Unknown
};

inline std::string role_to_string(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    case ChatRole::Unknown:
    default:
        return "unknown";
    }
}

inline ChatRole role_from_string(const std::string& role) {
    if (role == "system") {
        return ChatRole::System;
    }
    if (role == "developer") {
        return ChatRole::Developer;
    }
    if (role == "user") {
        return ChatRole::User;
    }
    if (role == "assistant") {
        return ChatRole::Assistant;
    }
    if (role == "tool") {
        return ChatRole::Tool;
    }
    return ChatRole::Unknown;
}

struct ChatContentPart {
    ContentType type = ContentType::Text;
    std::string text;
    std::string uri;
    std::vector<uint8_t> binary_data;
    std::string media_format;

    static ChatContentPart text_part(const std::string& value) {
        ChatContentPart part;
        part.type = ContentType::Text;
        part.text = value;
        return part;
    }
};

struct ChatFunctionCall {
    std::string name;
    std::string arguments_json;
};

struct ChatToolCall {
    std::string id;
    std::string type = "function";
    ChatFunctionCall function;
};

struct ChatMessage {
    ChatRole role = ChatRole::Unknown;
    std::vector<ChatContentPart> content;
    std::string reasoning_content;
    std::vector<ChatToolCall> tool_calls;
    std::string name;
    std::string tool_call_id;
};

struct ChatTool {
    std::string type = "function";
    std::string name;
    std::string description;
    std::string parameters_json = "{}";
    std::string raw_json;
};

enum class ToolChoice {
    Auto,
    None,
    Required,
    Function
};

enum class ToolPolicyMode {
    Warn,
    Strict
};

struct ChatTemplateOptions {
    std::optional<bool> enable_thinking;
    bool preserve_thinking = true;
    bool auto_disable_thinking_with_tools = false;
    // 0 means unlimited.
    int max_tool_arg_chars = 0;
    // 0 means unlimited.
    int max_tool_response_chars = 0;
    std::string template_override_text;
    std::string template_override_path;
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
    ToolChoice tool_choice = ToolChoice::Auto;
    ToolPolicyMode tool_policy = ToolPolicyMode::Warn;
    std::string tool_choice_function_name;
    ChatTemplateOptions template_options;
    GenerationConfig generation_config;
};

struct ChatRenderResult {
    std::vector<int> input_ids;
    std::vector<int> stable_prefix_ids;
    std::string rendered_text;
    std::string template_name;
};

struct ChatResponseMetadata {
    std::string template_name;
    std::string model_family;
    int reasoning_budget_tokens = -1;
    bool reasoning_budget_forced_close = false;
    bool reasoning_budget_truncated = false;
    std::string tool_policy = "warn";
    std::string tool_choice = "auto";
};

struct AssistantChatResult {
    std::string role = "assistant";
    std::string content;
    std::string reasoning_content;
    std::vector<ChatToolCall> tool_calls;
    std::string raw_text;
    std::string finish_reason = "stop";
    std::vector<std::string> warnings;
    ChatResponseMetadata metadata;
};

inline std::string decode_finish_reason(bool hit_loop_guard, bool hit_length, bool has_tool_calls) {
    if (hit_loop_guard) {
        return "loop_guard";
    }
    if (hit_length) {
        return "length";
    }
    if (has_tool_calls) {
        return "tool_calls";
    }
    return "stop";
}

} // namespace aila::chat
