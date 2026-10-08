# Qwen3.8-Flash-Next PI coding-agent compatibility (2026-10-07)

HTTP/chat compatibility findings. Long-context and session reuse measurements
are recorded in qwen38-server-agent-long-context-2026-10-07.md.

## What was checked

Real PI (pi-coding-agent 1.0.1, npm i -g @earendil-works/pi-coding-agent) drives
an OpenAI-compatible endpoint through its openai-completions API. A stub
endpoint captured the exact request its client sends for a single read turn:

    path /v1/chat/completions
    body keys: max_completion_tokens, messages, model, store, stream,
               stream_options, tools
    store False, stream True, stream_options {"include_usage": true}
    max_completion_tokens 512
    roles ["developer", "user"]

In this configuration PI sends store:false and a leading developer instruction message. It does
not send reasoning_effort unless the model is declared with a thinking level, it
sends strict only when compat.supportsStrictMode is enabled (off by default),
and it omits prompt_cache_key unless the base URL is api.openai.com or long
cache retention is requested.

The Qwen3.8 checkpoint template (tokenizer_config.json, chat_template.jinja)
accepts only system, user, assistant and tool roles and raises on anything else,
and gates reasoning on the enable_thinking variable. Tool calls use the XML
tool_call/function/parameter form the StructuredStreamParser already emits, and
tool_response for tool results.

## Gaps fixed (src/server/ChatCodec.cpp, ChatCodec.hpp)

- Accept store (PI always sends store:false, ignored); reject store:true as
  unsupported instead of silently dropping it.
- Accept reasoning_effort and map non-off values onto enable_thinking; it never
  sets a token budget. Unknown values are rejected.
- Normalize a leading developer message to system so the checkpoint template
  renders the instruction instead of raising on the role. A leading developer
  now renders byte-identically to a leading system message.
- ChatSettings gained maxBodyBytes (64 KiB floor, 1 MiB default, 4 MiB ceiling)
  and prefillChunkSize. Body, rendered prompt, and per-message strings are
  bounded by maxBodyBytes, so a long agent tool result is no longer capped at
  1 MiB by the old per-string default.
- Capacity checks use wire::absolute_limits() (the published elementwise maximum
  of extended_generation_v1 and extended_context_v1) plus the 4 MiB body
  protocol ceiling.
- ChatOutput::Finish validates the worker summary against the prepared prompt
  and generation capacities instead of the 4096/1024 wire defaults, so a
  long-context completion no longer fails at the final frame.

## Smoke helper (tools/server/pi_agent_smoke.py)

New reusable helper that drives the real PI CLI against a local endpoint with a
fully isolated agent directory (PI_CODING_AGENT_DIR, dummy key, no auth.json),
--no-session, no extensions/skills/templates/context files, --tools read, and
the isolated workspace as its working directory. Without --stub it requires the
agent to read one marker file and answer with its contents. --context and
--max-tokens set the advertised model limits; keep them at or below what the
running server was configured for. --stub serves a canned SSE reply on CPU and
can capture the request PI sends.

## Verification

    ctest --test-dir build-refactor-server -C Release -R AilaServerChatTests --output-on-failure
    python tools/server/pi_agent_smoke.py --stub --capture tmp/pi-capture.json

Against the running GPU server:

    python tools/server/pi_agent_smoke.py --endpoint http://127.0.0.1:8080/v1 --model-id qwen4exp --context 32768 --max-tokens 512 --keep

The helper now saves PI JSON events and verifies one successful read of the
expected marker file followed by a correct final assistant answer. Add
--context-file PATH to attach long background text through PI's @file input.
The isolated working directory and read-only tool selection are not a sandbox.
