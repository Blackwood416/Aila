#include "chat/AssistantOutputParser.hpp"

#include "simdjson.h"

#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace aila::chat {
namespace {

std::string trim(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::string escape_json_string(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    for (char c : value) {
        switch (c) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += hex[(static_cast<unsigned char>(c) >> 4) & 0xf];
                escaped += hex[static_cast<unsigned char>(c) & 0xf];
            } else {
                escaped += c;
            }
            break;
        }
    }
    return escaped;
}

bool is_raw_json_value(const std::string& value) {
    const std::string trimmed = trim(value);
    if (trimmed.empty()) {
        return false;
    }

    const char first = trimmed.front();
    const char last = trimmed.back();
    const bool candidate =
        (first == '{' && last == '}') ||
        (first == '[' && last == ']') ||
        (first == '"' && last == '"') ||
        trimmed == "true" ||
        trimmed == "false" ||
        trimmed == "null";
    if (!candidate) {
        return false;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element parsed;
    return parser.parse(trimmed).get(parsed) == simdjson::SUCCESS;
}

std::string json_argument_value(const std::string& value) {
    const std::string trimmed = trim(value);
    if (is_raw_json_value(trimmed)) {
        return trimmed;
    }
    return "\"" + escape_json_string(trimmed) + "\"";
}

std::string build_arguments_json(const std::vector<std::pair<std::string, std::string>>& parameters) {
    std::string json = "{";
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (i != 0) {
            json += ",";
        }
        json += "\"";
        json += escape_json_string(trim(parameters[i].first));
        json += "\":";
        json += json_argument_value(parameters[i].second);
    }
    json += "}";
    return json;
}

std::vector<std::pair<std::string, std::string>> parse_xml_like_attributes(const std::string& value) {
    std::vector<std::pair<std::string, std::string>> attributes;
    size_t pos = 0;
    while (pos < value.size()) {
        while (pos < value.size() && std::isspace(static_cast<unsigned char>(value[pos]))) {
            ++pos;
        }
        if (pos >= value.size()) {
            break;
        }

        const size_t name_begin = pos;
        while (pos < value.size() &&
               (std::isalnum(static_cast<unsigned char>(value[pos])) ||
                value[pos] == '_' ||
                value[pos] == '-' ||
                value[pos] == '.')) {
            ++pos;
        }
        if (pos == name_begin) {
            return {};
        }

        std::string name = value.substr(name_begin, pos - name_begin);
        while (pos < value.size() && std::isspace(static_cast<unsigned char>(value[pos]))) {
            ++pos;
        }
        if (pos >= value.size() || value[pos] != '=') {
            return {};
        }
        ++pos;
        while (pos < value.size() && std::isspace(static_cast<unsigned char>(value[pos]))) {
            ++pos;
        }
        if (pos >= value.size()) {
            return {};
        }

        std::string attr_value;
        const char quote = value[pos];
        if (quote == '"' || quote == '\'') {
            ++pos;
            const size_t value_begin = pos;
            const size_t value_end = value.find(quote, value_begin);
            if (value_end == std::string::npos) {
                return {};
            }
            attr_value = value.substr(value_begin, value_end - value_begin);
            pos = value_end + 1;
        } else {
            const size_t value_begin = pos;
            while (pos < value.size() && !std::isspace(static_cast<unsigned char>(value[pos]))) {
                ++pos;
            }
            attr_value = value.substr(value_begin, pos - value_begin);
        }

        attributes.emplace_back(std::move(name), std::move(attr_value));
    }
    return attributes;
}

std::string extract_thinking(const std::string& raw_text, std::string& reasoning_content) {
    std::string content;
    size_t pos = 0;
    while (pos < raw_text.size()) {
        const size_t begin = raw_text.find("<think>", pos);
        if (begin == std::string::npos) {
            content += raw_text.substr(pos);
            break;
        }

        const size_t inner_begin = begin + std::string("<think>").size();
        const size_t end = raw_text.find("</think>", inner_begin);
        if (end == std::string::npos) {
            content += raw_text.substr(pos);
            break;
        }

        content += raw_text.substr(pos, begin - pos);
        if (!reasoning_content.empty()) {
            reasoning_content += "\n";
        }
        reasoning_content += trim(raw_text.substr(inner_begin, end - inner_begin));
        pos = end + std::string("</think>").size();
    }
    return content;
}

bool find_first_think_close(const std::string& content, size_t& pos, std::string& tag) {
    const std::string close_tags[] = {
        "</think>",
        "</thinking>",
        "</think >",
    };

    pos = std::string::npos;
    tag.clear();
    for (const auto& candidate : close_tags) {
        const size_t found = content.find(candidate);
        if (found != std::string::npos && (pos == std::string::npos || found < pos)) {
            pos = found;
            tag = candidate;
        }
    }
    return pos != std::string::npos;
}

std::string recover_orphan_think_close(std::string content, std::string& reasoning_content) {
    size_t close_pos = std::string::npos;
    std::string close_tag;
    if (!find_first_think_close(content, close_pos, close_tag)) {
        return content;
    }

    const std::string before = trim(content.substr(0, close_pos));
    const std::string after = trim(content.substr(close_pos + close_tag.size()));
    if (after.empty()) {
        return content;
    }
    if (before.empty() || before == after) {
        return after;
    }

    if (!reasoning_content.empty()) {
        reasoning_content += "\n";
    }
    reasoning_content += before;
    return after;
}

bool parse_tool_call_block(const std::string& block, ChatToolCall& tool_call) {
    const size_t function_begin = block.find("<function=");
    if (function_begin == std::string::npos) {
        return false;
    }

    const size_t name_begin = function_begin + std::string("<function=").size();
    const size_t open_end = block.find('>', name_begin);
    if (open_end == std::string::npos) {
        return false;
    }

    const std::string function_name = trim(block.substr(name_begin, open_end - name_begin));
    if (function_name.empty()) {
        return false;
    }

    const size_t function_close = block.find("</function>", open_end + 1);
    if (function_close == std::string::npos) {
        return false;
    }

    const std::string body = block.substr(open_end + 1, function_close - open_end - 1);
    std::vector<std::pair<std::string, std::string>> parameters;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t parameter_begin = body.find("<parameter=", pos);
        if (parameter_begin == std::string::npos) {
            if (!trim(body.substr(pos)).empty()) {
                return false;
            }
            break;
        }

        if (!trim(body.substr(pos, parameter_begin - pos)).empty()) {
            return false;
        }

        const size_t parameter_name_begin = parameter_begin + std::string("<parameter=").size();
        const size_t parameter_open_end = body.find('>', parameter_name_begin);
        if (parameter_open_end == std::string::npos) {
            return false;
        }

        const std::string parameter_name =
            trim(body.substr(parameter_name_begin, parameter_open_end - parameter_name_begin));
        const size_t parameter_close = body.find("</parameter>", parameter_open_end + 1);
        if (parameter_name.empty() || parameter_close == std::string::npos) {
            return false;
        }

        parameters.emplace_back(parameter_name, body.substr(parameter_open_end + 1, parameter_close - parameter_open_end - 1));
        pos = parameter_close + std::string("</parameter>").size();
    }

    tool_call.type = "function";
    tool_call.function.name = function_name;
    tool_call.function.arguments_json = build_arguments_json(parameters);
    return true;
}

bool parse_named_xml_tool_call_block(const std::string& block, ChatToolCall& tool_call) {
    std::string text = trim(block);
    if (text.empty() || text[0] != '<' || text.rfind("</", 0) == 0 || text.rfind("<function=", 0) == 0) {
        return false;
    }

    const size_t open_end = text.find('>');
    if (open_end == std::string::npos) {
        return false;
    }

    std::string header = trim(text.substr(1, open_end - 1));
    bool self_closing = false;
    if (!header.empty() && header.back() == '/') {
        self_closing = true;
        header.pop_back();
        header = trim(header);
    }

    size_t name_end = 0;
    while (name_end < header.size() &&
           (std::isalnum(static_cast<unsigned char>(header[name_end])) ||
            header[name_end] == '_' ||
            header[name_end] == '-' ||
            header[name_end] == '.')) {
        ++name_end;
    }
    if (name_end == 0) {
        return false;
    }

    const std::string function_name = header.substr(0, name_end);
    const std::string attr_text = trim(header.substr(name_end));
    std::vector<std::pair<std::string, std::string>> parameters =
        parse_xml_like_attributes(attr_text);
    if (!attr_text.empty() && parameters.empty()) {
        return false;
    }

    if (!self_closing) {
        const std::string close_tag = "</" + function_name + ">";
        const size_t close_pos = text.find(close_tag, open_end + 1);
        if (close_pos == std::string::npos) {
            return false;
        }
        const std::string body = trim(text.substr(open_end + 1, close_pos - open_end - 1));
        const std::string suffix = trim(text.substr(close_pos + close_tag.size()));
        if (!body.empty() || !suffix.empty()) {
            return false;
        }
    }

    tool_call.type = "function";
    tool_call.function.name = function_name;
    tool_call.function.arguments_json = build_arguments_json(parameters);
    return true;
}

std::string extract_tool_calls(
    const std::string& content,
    std::vector<ChatToolCall>& tool_calls,
    std::vector<std::string>& warnings) {
    std::string remaining;
    size_t pos = 0;
    while (pos < content.size()) {
        const size_t begin = content.find("<tool_call>", pos);
        if (begin == std::string::npos) {
            remaining += content.substr(pos);
            break;
        }

        const size_t inner_begin = begin + std::string("<tool_call>").size();
        const size_t end = content.find("</tool_call>", inner_begin);
        if (end == std::string::npos) {
            remaining += content.substr(pos, begin - pos);
            ChatToolCall tool_call;
            const std::string block = content.substr(inner_begin);
            if (parse_tool_call_block(block, tool_call) ||
                parse_named_xml_tool_call_block(block, tool_call)) {
                tool_call.id = "call_" + std::to_string(tool_calls.size());
                tool_calls.push_back(std::move(tool_call));
                warnings.emplace_back("Recovered tool_call block: missing closing </tool_call>");
            } else {
                remaining += content.substr(begin);
                warnings.emplace_back("Malformed tool_call block: missing closing tag");
            }
            break;
        }

        remaining += content.substr(pos, begin - pos);
        const size_t block_end = end + std::string("</tool_call>").size();
        const std::string block = content.substr(inner_begin, end - inner_begin);

        ChatToolCall tool_call;
        if (parse_tool_call_block(block, tool_call) ||
            parse_named_xml_tool_call_block(block, tool_call)) {
            tool_call.id = "call_" + std::to_string(tool_calls.size());
            tool_calls.push_back(std::move(tool_call));
        } else {
            warnings.emplace_back("Malformed tool_call block: leaving raw text in content");
            remaining += content.substr(begin, block_end - begin);
        }

        pos = block_end;
    }

    return remaining;
}

std::string strip_leading_think_close_artifacts(std::string content) {
    content = trim(content);
    const std::string close_tags[] = {
        "</think>",
        "</thinking>",
        "</think >",
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& tag : close_tags) {
            if (content.rfind(tag, 0) == 0) {
                content.erase(0, tag.size());
                content = trim(content);
                changed = true;
                break;
            }
        }
    }
    return content;
}

} // namespace

AssistantChatResult parse_assistant_output(const std::string& raw_text) {
    AssistantChatResult result;
    result.raw_text = raw_text;

    std::string without_thinking = extract_thinking(raw_text, result.reasoning_content);
    without_thinking = recover_orphan_think_close(std::move(without_thinking), result.reasoning_content);
    result.content = strip_leading_think_close_artifacts(
        extract_tool_calls(without_thinking, result.tool_calls, result.warnings));
    return result;
}

} // namespace aila::chat
