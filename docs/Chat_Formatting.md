# Chat Formatting

Aila's chat formatting layer converts OpenAI-style chat requests into model
prompt text, then parses raw assistant text back into a structured assistant
result. Tool execution is intentionally outside the inference engine.

## Components

| Component | Responsibility |
|-----------|----------------|
| `src/chat/ChatTypes.hpp` | Canonical chat request, message, content part, tool, and assistant result types. |
| `src/chat/ChatJson.*` | Parse OpenAI-style JSON requests and serialize assistant results. |
| `src/chat/ChatTemplateEngine.*` | Render llama.cpp-style Jinja templates from structured chat values. |
| `src/chat/ChatFormatter.*` | Select the template and render text for Engine prompt construction. |
| `src/chat/AssistantOutputParser.*` | Parse `<think>` and Qwen-style `<tool_call>` output blocks. |
| `src/chat/StructuredStreamParser.*` | Split raw streaming text into reasoning, content, and tool-call deltas. |
| `src/chat/ToolPolicy.*` | Validate parsed tool calls against `tool_choice` and emit warning-only policy diagnostics. |
| `src/chat/BuiltinTemplates.*` | Built-in fixed templates, currently including the fixed Qwen3.5 template. |

## Template Selection

`ChatFormatter` chooses a template in this order:

1. Request `chat_template_kwargs.chat_template`.
2. Request `chat_template_kwargs.chat_template_path`.
3. Environment variable `AILA_CHAT_TEMPLATE`.
4. Environment variable `AILA_CHAT_TEMPLATE_PATH`.
5. Built-in fixed Qwen3.5 template for `ModelFamily::Qwen35Hybrid`.
6. Tokenizer-provided `chat_template`.
7. Minimal ChatML fallback.

For Qwen3.5 Hybrid, `enable_thinking` defaults to `false` for the exact 0.8B
spec and `true` for larger specs unless the request overrides it. The legacy
`/think` and `/no_think` suffix commands are not part of the new structured
chat API.

Set `AILA_DEBUG_CHAT_TEMPLATE=1` to log the selected template source, or
`AILA_DEBUG_PROMPT_TEXT=1` to log both the source and rendered prompt text.

The current structured stream parser emits completed reasoning and tool-call
blocks as deltas once their closing tags arrive. Raw token streaming remains
available for callers that need every token immediately.

## Request JSON

The chat JSON parser accepts either a raw messages array or an object with
`messages`. The object form also supports tools, tool choice, template kwargs,
and generation parameters:

```json
{
  "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Search for cats"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "chat_template_kwargs": {
    "enable_thinking": false,
    "preserve_thinking": true,
    "auto_disable_thinking_with_tools": false,
    "max_tool_arg_chars": 0,
    "max_tool_response_chars": 0
  },
  "temperature": 0.7,
  "max_tokens": 128,
  "reasoning_budget": -1
}
```

`tool_choice` may be `"auto"`, `"none"`, `"required"`, or an OpenAI-style
named function object. Aila treats `tool_choice` as prompt guidance plus
warning-only validation: it still never executes tools, but structured results
warn when the assistant returns a tool call despite `"none"`, fails to return a
tool call for `"required"`, or calls a different function than requested.

Thinking budget aliases `reasoning_budget`, `thinking_budget`, and
`thinking_budget_tokens` are accepted in request JSON. `-1` disables budget
control, `0` requests no-think prompt mode, and positive values cap generated
tokens inside `<think>`.

## Structured Output

Use `InferenceEngine::generate_chat_json` or the C API
`aila_generate_chat_json` / `aila_generate_chat_json_ex` when callers need
parsed reasoning or tool calls.
The legacy `generate_messages_json` and `aila_generate_messages` APIs still
return raw assistant text.

Assistant result JSON contains:

```json
{
  "role": "assistant",
  "content": "visible text",
  "reasoning_content": "text from <think>...</think>",
  "tool_calls": [],
  "raw_text": "raw decoded assistant text",
  "finish_reason": "stop",
  "warnings": []
}
```

Tool calls are parsed from Qwen-style XML blocks:

```text
<tool_call>
<function=search>
<parameter=query>cats</parameter>
</function>
</tool_call>
```

Callers execute returned tool calls externally and append their results as
`tool` messages with `tool_call_id`.

`tests/chat_tool_workflow_smoke.ps1` exercises the caller-executed tool loop:
assistant tool call, external tool result, and final answer. It checks
structure rather than exact wording.

`finish_reason` is `"stop"` when decoding ended on EOS, `"length"` when
`max_new_tokens` was exhausted, `"loop_guard"` when Aila stopped repetitive
decode, and `"tool_calls"` when the assistant result contains parsed tool
calls.

## CLI

`--messages-json <path>` keeps returning raw assistant text by default. Add
`--chat-output-json` to print the structured assistant result JSON:

```bash
Aila.exe --model <model-dir> --messages-json request.json --chat-output-json
```

Use `--chat-stream-jsonl` to print structured stream events as
newline-delimited JSON:

```bash
Aila.exe --model <model-dir> --messages-json request.json --chat-stream-jsonl
```
