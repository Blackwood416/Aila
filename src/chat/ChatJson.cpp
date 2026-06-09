#include "chat/ChatJson.hpp"

#include "simdjson.h"

#include <sstream>
#include <string_view>
#include <utility>

namespace aila::chat {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string elem_to_json(simdjson::dom::element elem) {
    return simdjson::minify(elem);
}

bool read_string(simdjson::dom::object obj, const char* key, std::string& out) {
    simdjson::dom::element elem;
    std::string_view sv;
    if (obj.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }
    if (elem.get_string().get(sv) != simdjson::SUCCESS) {
        return false;
    }
    out = std::string(sv);
    return true;
}

bool read_bool(simdjson::dom::object obj, const char* key, bool& out) {
    simdjson::dom::element elem;
    if (obj.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }
    if (elem.get_bool().get(out) == simdjson::SUCCESS) {
        return true;
    }
    int64_t i64 = 0;
    if (elem.get_int64().get(i64) == simdjson::SUCCESS) {
        out = i64 != 0;
        return true;
    }
    return false;
}

bool read_int(simdjson::dom::object obj, const char* key, int& out) {
    simdjson::dom::element elem;
    int64_t i64 = 0;
    if (obj.at_key(key).get(elem) != simdjson::SUCCESS ||
        elem.get_int64().get(i64) != simdjson::SUCCESS) {
        return false;
    }
    out = static_cast<int>(i64);
    return true;
}

bool read_float(simdjson::dom::object obj, const char* key, float& out) {
    simdjson::dom::element elem;
    double dbl = 0.0;
    if (obj.at_key(key).get(elem) != simdjson::SUCCESS ||
        elem.get_double().get(dbl) != simdjson::SUCCESS) {
        return false;
    }
    out = static_cast<float>(dbl);
    return true;
}

std::string extension_from_uri(const std::string& uri) {
    const size_t dot = uri.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= uri.size()) {
        return {};
    }
    return uri.substr(dot + 1);
}

bool read_uri_field(simdjson::dom::object obj,
                    const char* key,
                    const char* nested_key,
                    std::string& out) {
    simdjson::dom::element elem;
    if (obj.at_key(key).get(elem) != simdjson::SUCCESS) {
        return false;
    }
    std::string_view sv;
    if (elem.get_string().get(sv) == simdjson::SUCCESS) {
        out = std::string(sv);
        return true;
    }
    simdjson::dom::object nested;
    if (elem.get_object().get(nested) == simdjson::SUCCESS && read_string(nested, nested_key, out)) {
        return true;
    }
    return false;
}

bool parse_content(simdjson::dom::element elem,
                   std::vector<ChatContentPart>& out,
                   std::string* error) {
    if (elem.is_null()) {
        return true;
    }

    std::string_view sv;
    if (elem.get_string().get(sv) == simdjson::SUCCESS) {
        out.push_back(ChatContentPart::text_part(std::string(sv)));
        return true;
    }

    simdjson::dom::array arr;
    if (elem.get_array().get(arr) != simdjson::SUCCESS) {
        set_error(error, "message.content must be string, array, or null");
        return false;
    }

    for (auto part_elem : arr) {
        simdjson::dom::object part_obj;
        if (part_elem.get_object().get(part_obj) != simdjson::SUCCESS) {
            set_error(error, "content part must be object");
            return false;
        }

        std::string type;
        if (!read_string(part_obj, "type", type)) {
            set_error(error, "content part.type missing or not string");
            return false;
        }

        if (type == "text" || type == "input_text") {
            std::string text;
            if (!read_string(part_obj, "text", text)) {
                set_error(error, "text content missing text field");
                return false;
            }
            out.push_back(ChatContentPart::text_part(text));
            continue;
        }

        ChatContentPart part;
        std::string uri;
        if (type == "image" || type == "image_url" || type == "input_image") {
            part.type = ContentType::Image;
            if (!read_uri_field(part_obj, "image_url", "url", uri) &&
                !read_string(part_obj, "image", uri)) {
                set_error(error, "image content missing image/image_url field");
                return false;
            }
        } else if (type == "video" || type == "video_url" || type == "input_video") {
            part.type = ContentType::Video;
            if (!read_uri_field(part_obj, "video_url", "url", uri) &&
                !read_string(part_obj, "video", uri)) {
                set_error(error, "video content missing video/video_url field");
                return false;
            }
        } else if (type == "audio" || type == "audio_url" || type == "input_audio") {
            part.type = ContentType::Audio;
            if (!read_uri_field(part_obj, "audio_url", "url", uri) &&
                !read_string(part_obj, "audio", uri)) {
                simdjson::dom::element input_audio_elem;
                simdjson::dom::object input_audio_obj;
                if (part_obj.at_key("input_audio").get(input_audio_elem) == simdjson::SUCCESS &&
                    input_audio_elem.get_object().get(input_audio_obj) == simdjson::SUCCESS) {
                    read_string(input_audio_obj, "format", part.media_format);
                    read_string(input_audio_obj, "data", uri);
                }
            }
            if (uri.empty()) {
                set_error(error, "audio content missing audio/audio_url/input_audio field");
                return false;
            }
        } else {
            set_error(error, "unsupported content part type in chat parser: " + type);
            return false;
        }

        part.uri = uri;
        if (part.media_format.empty()) {
            part.media_format = extension_from_uri(uri);
        }
        out.push_back(std::move(part));
    }
    return true;
}

bool parse_tool_calls(simdjson::dom::object msg_obj,
                      std::vector<ChatToolCall>& out,
                      std::string* error) {
    simdjson::dom::element calls_elem;
    if (msg_obj.at_key("tool_calls").get(calls_elem) != simdjson::SUCCESS || calls_elem.is_null()) {
        return true;
    }

    simdjson::dom::array calls_arr;
    if (calls_elem.get_array().get(calls_arr) != simdjson::SUCCESS) {
        set_error(error, "assistant.tool_calls must be array");
        return false;
    }

    for (auto call_elem : calls_arr) {
        simdjson::dom::object call_obj;
        if (call_elem.get_object().get(call_obj) != simdjson::SUCCESS) {
            set_error(error, "tool_call must be object");
            return false;
        }

        ChatToolCall call;
        read_string(call_obj, "id", call.id);
        read_string(call_obj, "type", call.type);
        if (call.type.empty()) {
            call.type = "function";
        }

        simdjson::dom::element fn_elem;
        simdjson::dom::object fn_obj;
        if (call_obj.at_key("function").get(fn_elem) != simdjson::SUCCESS ||
            fn_elem.get_object().get(fn_obj) != simdjson::SUCCESS) {
            set_error(error, "tool_call.function missing");
            return false;
        }
        if (!read_string(fn_obj, "name", call.function.name)) {
            set_error(error, "tool_call.function.name missing");
            return false;
        }

        simdjson::dom::element args_elem;
        if (fn_obj.at_key("arguments").get(args_elem) == simdjson::SUCCESS) {
            std::string_view args_sv;
            if (args_elem.get_string().get(args_sv) == simdjson::SUCCESS) {
                call.function.arguments_json = std::string(args_sv);
            } else {
                call.function.arguments_json = elem_to_json(args_elem);
            }
        } else {
            call.function.arguments_json = "{}";
        }
        out.push_back(std::move(call));
    }
    return true;
}

bool parse_message(simdjson::dom::element item, ChatMessage& out, std::string* error) {
    simdjson::dom::object obj;
    if (item.get_object().get(obj) != simdjson::SUCCESS) {
        set_error(error, "message item is not an object");
        return false;
    }

    std::string role;
    if (!read_string(obj, "role", role)) {
        set_error(error, "message.role missing or not string");
        return false;
    }
    out.role = role_from_string(role);
    if (out.role == ChatRole::Unknown) {
        set_error(error, "unsupported message.role: " + role);
        return false;
    }

    read_string(obj, "name", out.name);
    read_string(obj, "tool_call_id", out.tool_call_id);
    read_string(obj, "reasoning_content", out.reasoning_content);

    simdjson::dom::element content_elem;
    if (obj.at_key("content").get(content_elem) == simdjson::SUCCESS) {
        if (!parse_content(content_elem, out.content, error)) {
            return false;
        }
    }

    return parse_tool_calls(obj, out.tool_calls, error);
}

bool parse_messages_array(simdjson::dom::array messages_arr,
                          ChatRequest& out,
                          std::string* error) {
    for (auto item : messages_arr) {
        ChatMessage msg;
        if (!parse_message(item, msg, error)) {
            return false;
        }
        out.messages.push_back(std::move(msg));
    }
    return true;
}

void parse_generation_options(simdjson::dom::object root_obj, ChatRequest& out) {
    bool b = false;
    bool has_do_sample = read_bool(root_obj, "do_sample", b);
    if (has_do_sample) {
        out.generation_config.do_sample = b;
    }

    float f = 0.0f;
    if (read_float(root_obj, "temperature", f)) {
        out.generation_config.temperature = f;
        if (f == 0.0f && !has_do_sample) {
            out.generation_config.do_sample = false;
        }
    }
    if (read_float(root_obj, "top_p", f)) {
        out.generation_config.top_p = f;
    }
    if (read_float(root_obj, "repetition_penalty", f)) {
        out.generation_config.repetition_penalty = f;
    }
    if (read_float(root_obj, "presence_penalty", f)) {
        out.generation_config.presence_penalty = f;
    }
    if (read_float(root_obj, "frequency_penalty", f)) {
        out.generation_config.frequency_penalty = f;
    }

    int i = 0;
    if (read_int(root_obj, "max_tokens", i) || read_int(root_obj, "max_new_tokens", i)) {
        out.generation_config.max_new_tokens = i;
    }
    if (read_int(root_obj, "top_k", i)) {
        out.generation_config.top_k = i;
    }
    if (read_int(root_obj, "decode_chunk_size", i)) {
        out.generation_config.decode_chunk_size = i;
    }
    if (read_int(root_obj, "stream_chunk_size", i)) {
        out.generation_config.stream_chunk_size = i;
    }

    simdjson::dom::element seed_elem;
    int64_t seed = 0;
    if (root_obj.at_key("seed").get(seed_elem) == simdjson::SUCCESS &&
        seed_elem.get_int64().get(seed) == simdjson::SUCCESS) {
        out.generation_config.sampling_seed = static_cast<uint64_t>(seed);
        out.generation_config.use_fixed_seed = true;
    }
}

bool parse_tool_choice(simdjson::dom::object root_obj, ChatRequest& out, std::string* error) {
    simdjson::dom::element elem;
    if (root_obj.at_key("tool_choice").get(elem) != simdjson::SUCCESS || elem.is_null()) {
        return true;
    }

    std::string_view choice_sv;
    if (elem.get_string().get(choice_sv) == simdjson::SUCCESS) {
        const std::string choice(choice_sv);
        if (choice == "none") {
            out.tool_choice = ToolChoice::None;
        } else if (choice == "required") {
            out.tool_choice = ToolChoice::Required;
        } else if (choice == "auto") {
            out.tool_choice = ToolChoice::Auto;
        } else {
            set_error(error, "unsupported tool_choice string: " + choice);
            return false;
        }
        return true;
    }

    simdjson::dom::object choice_obj;
    if (elem.get_object().get(choice_obj) != simdjson::SUCCESS) {
        set_error(error, "tool_choice must be string, object, or null");
        return false;
    }

    std::string type;
    read_string(choice_obj, "type", type);
    if (type != "function") {
        set_error(error, "object tool_choice only supports type=function");
        return false;
    }

    simdjson::dom::element fn_elem;
    simdjson::dom::object fn_obj;
    if (choice_obj.at_key("function").get(fn_elem) != simdjson::SUCCESS ||
        fn_elem.get_object().get(fn_obj) != simdjson::SUCCESS ||
        !read_string(fn_obj, "name", out.tool_choice_function_name)) {
        set_error(error, "tool_choice.function.name missing");
        return false;
    }
    out.tool_choice = ToolChoice::Function;
    return true;
}

void parse_template_kwargs(simdjson::dom::object root_obj, ChatRequest& out) {
    simdjson::dom::element elem;
    simdjson::dom::object kwargs;
    if (root_obj.at_key("chat_template_kwargs").get(elem) != simdjson::SUCCESS ||
        elem.get_object().get(kwargs) != simdjson::SUCCESS) {
        return;
    }

    bool b = false;
    if (read_bool(kwargs, "enable_thinking", b)) {
        out.template_options.enable_thinking = b;
    }
    if (read_bool(kwargs, "preserve_thinking", b)) {
        out.template_options.preserve_thinking = b;
    }
    if (read_bool(kwargs, "auto_disable_thinking_with_tools", b)) {
        out.template_options.auto_disable_thinking_with_tools = b;
    }

    int i = 0;
    if (read_int(kwargs, "max_tool_arg_chars", i)) {
        out.template_options.max_tool_arg_chars = i;
    }
    if (read_int(kwargs, "max_tool_response_chars", i)) {
        out.template_options.max_tool_response_chars = i;
    }

    read_string(kwargs, "chat_template", out.template_options.template_override_text);
    read_string(kwargs, "chat_template_path", out.template_options.template_override_path);
}

bool parse_tools(simdjson::dom::object root_obj, ChatRequest& out, std::string* error) {
    simdjson::dom::element elem;
    if (root_obj.at_key("tools").get(elem) != simdjson::SUCCESS || elem.is_null()) {
        return true;
    }

    simdjson::dom::array tools_arr;
    if (elem.get_array().get(tools_arr) != simdjson::SUCCESS) {
        set_error(error, "tools must be array");
        return false;
    }

    for (auto tool_elem : tools_arr) {
        simdjson::dom::object tool_obj;
        if (tool_elem.get_object().get(tool_obj) != simdjson::SUCCESS) {
            set_error(error, "tool item must be object");
            return false;
        }

        ChatTool tool;
        read_string(tool_obj, "type", tool.type);
        if (tool.type.empty()) {
            tool.type = "function";
        }
        tool.raw_json = elem_to_json(tool_elem);

        simdjson::dom::element fn_elem;
        simdjson::dom::object fn_obj;
        if (tool_obj.at_key("function").get(fn_elem) != simdjson::SUCCESS ||
            fn_elem.get_object().get(fn_obj) != simdjson::SUCCESS) {
            set_error(error, "tool.function missing");
            return false;
        }
        if (!read_string(fn_obj, "name", tool.name)) {
            set_error(error, "tool.function.name missing");
            return false;
        }
        read_string(fn_obj, "description", tool.description);

        simdjson::dom::element params_elem;
        if (fn_obj.at_key("parameters").get(params_elem) == simdjson::SUCCESS) {
            tool.parameters_json = elem_to_json(params_elem);
        }
        out.tools.push_back(std::move(tool));
    }
    return true;
}

} // namespace

std::string json_escape_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                out << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
            } else {
                out << static_cast<char>(c);
            }
            break;
        }
    }
    out << '"';
    return out.str();
}

bool parse_chat_request_json(const std::string& request_json,
                             const GenerationConfig& defaults,
                             ChatRequest& out,
                             std::string* error_message) {
    out = ChatRequest{};
    out.generation_config = defaults;

    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.parse(request_json);

        simdjson::dom::array messages_arr;
        simdjson::dom::object root_obj;
        bool has_root_obj = false;
        if (root.get_array().get(messages_arr) == simdjson::SUCCESS) {
            return parse_messages_array(messages_arr, out, error_message);
        }

        if (root.get_object().get(root_obj) != simdjson::SUCCESS) {
            set_error(error_message, "chat request root must be array or object");
            return false;
        }
        has_root_obj = true;

        simdjson::dom::element messages_elem;
        if (root_obj.at_key("messages").get(messages_elem) != simdjson::SUCCESS ||
            messages_elem.get_array().get(messages_arr) != simdjson::SUCCESS) {
            set_error(error_message, "messages field missing or not array");
            return false;
        }

        if (!parse_messages_array(messages_arr, out, error_message)) {
            return false;
        }

        if (has_root_obj) {
            parse_generation_options(root_obj, out);
            parse_template_kwargs(root_obj, out);
            if (!parse_tool_choice(root_obj, out, error_message)) {
                return false;
            }
            if (!parse_tools(root_obj, out, error_message)) {
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("chat JSON parse failed: ") + e.what());
        return false;
    }
}

std::string assistant_result_to_json(const AssistantChatResult& result) {
    std::ostringstream out;
    out << "{";
    out << "\"role\":" << json_escape_string(result.role);
    out << ",\"content\":" << json_escape_string(result.content);
    out << ",\"reasoning_content\":" << json_escape_string(result.reasoning_content);
    out << ",\"tool_calls\":[";
    for (size_t i = 0; i < result.tool_calls.size(); ++i) {
        const auto& call = result.tool_calls[i];
        if (i > 0) {
            out << ",";
        }
        out << "{";
        out << "\"id\":" << json_escape_string(call.id);
        out << ",\"type\":" << json_escape_string(call.type.empty() ? std::string("function") : call.type);
        out << ",\"function\":{";
        out << "\"name\":" << json_escape_string(call.function.name);
        out << ",\"arguments\":" << json_escape_string(call.function.arguments_json);
        out << "}}";
    }
    out << "]";
    out << ",\"raw_text\":" << json_escape_string(result.raw_text);
    out << ",\"finish_reason\":" << json_escape_string(result.finish_reason);
    out << ",\"warnings\":[";
    for (size_t i = 0; i < result.warnings.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << json_escape_string(result.warnings[i]);
    }
    out << "]}";
    return out.str();
}

} // namespace aila::chat
