#include "chat/ChatFormatter.hpp"

#include "chat/BuiltinTemplates.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>

namespace aila::chat {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return "";
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string chatml_fallback_template() {
    return "{%- for message in messages -%}\n"
           "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n"
           "{%- endfor -%}\n"
           "{%- if add_generation_prompt -%}\n"
           "{{- '<|im_start|>assistant\\n' -}}\n"
           "{%- endif -%}";
}

} // namespace

bool ChatFormatter::render_text(const ChatFormatInput& input,
                                bool add_generation_prompt,
                                ChatFormatTextResult& output,
                                std::string* error_message) const {
    output = ChatFormatTextResult{};
    if (!input.request) {
        set_error(error_message, "ChatFormatter input.request is null");
        return false;
    }

    ChatRequest request_copy = *input.request;
    if (request_copy.tool_choice == ToolChoice::None) {
        request_copy.tools.clear();
    } else if (request_copy.tool_choice == ToolChoice::Function &&
               !request_copy.tool_choice_function_name.empty()) {
        auto keep = [&](const ChatTool& tool) {
            return tool.name == request_copy.tool_choice_function_name;
        };
        request_copy.tools.erase(
            std::remove_if(request_copy.tools.begin(), request_copy.tools.end(),
                           [&](const ChatTool& tool) { return !keep(tool); }),
            request_copy.tools.end());
    }

    if (!request_copy.template_options.enable_thinking.has_value() &&
        input.family == ModelFamily::Qwen35Hybrid) {
        request_copy.template_options.enable_thinking = !input.is_qwen35_0p8b;
    }

    std::string source;
    std::string name;
    if (!request_copy.template_options.template_override_text.empty()) {
        source = request_copy.template_options.template_override_text;
        name = "request-override";
    } else if (!request_copy.template_options.template_override_path.empty()) {
        source = read_file_text(request_copy.template_options.template_override_path);
        name = request_copy.template_options.template_override_path;
    } else {
        const std::string env_template = aila::env::read_string("AILA_CHAT_TEMPLATE", "");
        const std::string env_path = aila::env::read_string("AILA_CHAT_TEMPLATE_PATH", "");
        if (!env_template.empty()) {
            source = env_template;
            name = "AILA_CHAT_TEMPLATE";
        } else if (!env_path.empty()) {
            source = read_file_text(env_path);
            name = env_path;
        } else if (input.family == ModelFamily::Qwen35Hybrid) {
            source = qwen35_fixed_chat_template();
            name = "qwen35-fixed-v20";
        } else if (!input.tokenizer_chat_template.empty()) {
            source = input.tokenizer_chat_template;
            name = "tokenizer_config";
        } else {
            source = chatml_fallback_template();
            name = "chatml-fallback";
        }
    }

    if (source.empty()) {
        set_error(error_message, "selected chat template is empty");
        return false;
    }

    ChatTemplateRenderInput render_input;
    render_input.template_source = source;
    render_input.request = &request_copy;
    render_input.add_generation_prompt = add_generation_prompt;
    render_input.bos_token = input.bos_token;
    render_input.eos_token = input.eos_token;

    ChatTemplateRenderOutput render_output;
    ChatTemplateEngine engine;
    if (!engine.render(render_input, render_output, error_message)) {
        return false;
    }

    output.text = render_output.text;
    output.template_name = name;
    return true;
}

} // namespace aila::chat
