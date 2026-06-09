#include "chat/ChatTemplateEngine.hpp"

#include "templates/jinja/lexer.h"
#include "templates/jinja/parser.h"
#include "templates/jinja/runtime.h"
#include "templates/jinja/value.h"
#include "simdjson.h"

namespace aila::chat {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

jinja::value mk_string(const std::string& text) {
    return jinja::mk_val<jinja::value_string>(text);
}

jinja::value mk_input_string(const std::string& text) {
    auto value = jinja::mk_val<jinja::value_string>(text);
    value->mark_input();
    return value;
}

jinja::value element_to_jinja(simdjson::dom::element elem) {
    if (elem.is_null()) {
        return jinja::mk_val<jinja::value_none>();
    }

    bool b = false;
    if (elem.get_bool().get(b) == simdjson::SUCCESS) {
        return jinja::mk_val<jinja::value_bool>(b);
    }

    int64_t i64 = 0;
    if (elem.get_int64().get(i64) == simdjson::SUCCESS) {
        return jinja::mk_val<jinja::value_int>(i64);
    }

    double dbl = 0.0;
    if (elem.get_double().get(dbl) == simdjson::SUCCESS) {
        return jinja::mk_val<jinja::value_float>(dbl);
    }

    std::string_view sv;
    if (elem.get_string().get(sv) == simdjson::SUCCESS) {
        return mk_string(std::string(sv));
    }

    simdjson::dom::array arr;
    if (elem.get_array().get(arr) == simdjson::SUCCESS) {
        auto value_arr = jinja::mk_val<jinja::value_array>();
        for (auto item : arr) {
            value_arr->push_back(element_to_jinja(item));
        }
        return value_arr;
    }

    simdjson::dom::object obj;
    if (elem.get_object().get(obj) == simdjson::SUCCESS) {
        auto value_obj = jinja::mk_val<jinja::value_object>();
        for (auto field : obj) {
            value_obj->insert(std::string(field.key), element_to_jinja(field.value));
        }
        return value_obj;
    }

    return mk_string("");
}

jinja::value json_text_to_jinja_or_string(const std::string& json_text) {
    simdjson::dom::parser parser;
    simdjson::dom::element elem;
    if (parser.parse(json_text).get(elem) == simdjson::SUCCESS) {
        return element_to_jinja(elem);
    }
    return mk_string(json_text);
}

jinja::value mk_content_value(const ChatMessage& msg) {
    if (msg.content.size() == 1 && msg.content[0].type == ContentType::Text) {
        return mk_input_string(msg.content[0].text);
    }

    auto arr = jinja::mk_val<jinja::value_array>();
    for (const auto& part : msg.content) {
        auto obj = jinja::mk_val<jinja::value_object>();
        if (part.type == ContentType::Text) {
            obj->insert("type", mk_string("text"));
            obj->insert("text", mk_input_string(part.text));
        } else if (part.type == ContentType::Image) {
            obj->insert("type", mk_string("image"));
            obj->insert("image", mk_input_string(part.uri));
        } else if (part.type == ContentType::Video) {
            obj->insert("type", mk_string("video"));
            obj->insert("video", mk_input_string(part.uri));
        } else if (part.type == ContentType::Audio) {
            obj->insert("type", mk_string("audio"));
            obj->insert("audio", mk_input_string(part.uri));
        }
        arr->push_back(obj);
    }
    return arr;
}

jinja::value mk_messages(const ChatRequest& request) {
    auto arr = jinja::mk_val<jinja::value_array>();
    for (const auto& msg : request.messages) {
        auto obj = jinja::mk_val<jinja::value_object>();
        obj->insert("role", mk_string(role_to_string(msg.role)));
        obj->insert("content", mk_content_value(msg));

        if (!msg.reasoning_content.empty()) {
            obj->insert("reasoning_content", mk_input_string(msg.reasoning_content));
        }
        if (!msg.name.empty()) {
            obj->insert("name", mk_string(msg.name));
        }
        if (!msg.tool_call_id.empty()) {
            obj->insert("tool_call_id", mk_string(msg.tool_call_id));
        }
        if (!msg.tool_calls.empty()) {
            auto calls = jinja::mk_val<jinja::value_array>();
            for (const auto& call : msg.tool_calls) {
                auto call_obj = jinja::mk_val<jinja::value_object>();
                call_obj->insert("id", mk_string(call.id));
                call_obj->insert("type", mk_string(call.type.empty() ? std::string("function") : call.type));
                auto fn = jinja::mk_val<jinja::value_object>();
                fn->insert("name", mk_string(call.function.name));
                fn->insert("arguments", json_text_to_jinja_or_string(call.function.arguments_json));
                call_obj->insert("function", fn);
                calls->push_back(call_obj);
            }
            obj->insert("tool_calls", calls);
        }
        arr->push_back(obj);
    }
    return arr;
}

jinja::value mk_tools(const ChatRequest& request) {
    auto arr = jinja::mk_val<jinja::value_array>();
    for (const auto& tool : request.tools) {
        auto tool_obj = jinja::mk_val<jinja::value_object>();
        tool_obj->insert("type", mk_string(tool.type.empty() ? std::string("function") : tool.type));

        auto fn = jinja::mk_val<jinja::value_object>();
        fn->insert("name", mk_string(tool.name));
        fn->insert("description", mk_string(tool.description));
        fn->insert("parameters", json_text_to_jinja_or_string(tool.parameters_json));
        tool_obj->insert("function", fn);

        arr->push_back(tool_obj);
    }
    return arr;
}

} // namespace

bool ChatTemplateEngine::render(const ChatTemplateRenderInput& input,
                                ChatTemplateRenderOutput& output,
                                std::string* error_message) const {
    output = ChatTemplateRenderOutput{};
    if (!input.request) {
        set_error(error_message, "ChatTemplateEngine input.request is null");
        return false;
    }
    if (input.template_source.empty()) {
        set_error(error_message, "ChatTemplateEngine template_source is empty");
        return false;
    }

    try {
        jinja::lexer lex;
        jinja::lexer_result lexed = lex.tokenize(input.template_source);
        jinja::program program = jinja::parse_from_tokens(lexed);
        jinja::context ctx(input.template_source);

        ctx.set_val("messages", mk_messages(*input.request));
        ctx.set_val("tools", mk_tools(*input.request));
        ctx.set_val("bos_token", mk_string(input.bos_token));
        ctx.set_val("eos_token", mk_string(input.eos_token));
        ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(input.add_generation_prompt));

        const auto& opts = input.request->template_options;
        ctx.set_val("enable_thinking", jinja::mk_val<jinja::value_bool>(opts.enable_thinking.value_or(true)));
        ctx.set_val("preserve_thinking", jinja::mk_val<jinja::value_bool>(opts.preserve_thinking));
        ctx.set_val("auto_disable_thinking_with_tools",
                    jinja::mk_val<jinja::value_bool>(opts.auto_disable_thinking_with_tools));
        ctx.set_val("max_tool_arg_chars", jinja::mk_val<jinja::value_int>(opts.max_tool_arg_chars));
        ctx.set_val("max_tool_response_chars", jinja::mk_val<jinja::value_int>(opts.max_tool_response_chars));

        jinja::runtime runtime(ctx);
        jinja::value_array results = runtime.execute(program);
        jinja::value_string parts = jinja::runtime::gather_string_parts(results);
        output.text = jinja::render_string_parts(parts);
        return true;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("Jinja render failed: ") + e.what());
        return false;
    }
}

} // namespace aila::chat
