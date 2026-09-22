"""Function-tool HTTP/SSE acceptance using the real Worker protocol.

Fake mode uses a separate, explicitly named CPU fixture. GPU mode uses an
unmodified model and must be launched by a machine-guarded Windows Job runner.
Each response, request identity and final cleanup report is retained.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import sys
import traceback

import socket_tests as http


def tool(name):
    return {"type": "function", "function": {"name": name, "description": "Look up the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def comparable(value):
    result = copy.deepcopy(value)
    for call in result.get("tool_calls", []):
        call.pop("id")
    return result


class ToolRunner(http.Runner):
    def payload(self, scenario="json", city="Beijing"):
        prompt = f'Use the weather tool with city exactly "{city}". Return only the tool call.'
        if self.mode == "gpu" and self.args.backend.startswith("qwen35_"):
            # The 0.8B NF4 checkpoint can omit <function=weather> for the shorter
            # prompt. Its raw-token failure is retained in the validation
            # evidence; use an explicit format to test transport/decoding,
            # without treating this fixture as a tool-selection quality eval.
            prompt = (f'Use the weather tool for city exactly "{city}". '
                      'Return one complete XML tool call: <tool_call> followed by <function=weather>, '
                      f'<parameter=city>{city}</parameter>, </function>, and </tool_call>.')
        if self.mode == "fake":
            prompt = f"<|tool_{scenario}|>\n" + prompt
        body = self.chat(prompt, 192)
        body["tools"] = [tool("weather"), tool("calc")]
        body["tool_choice"] = {"type": "function", "function": {"name": "weather"}}
        body["parallel_tool_calls"] = False
        return body

    def json_result(self, label, payload):
        value, timing = self.json_call(label, "/v1/chat/completions", "POST", payload)
        semantic = http.validate_completion(value, payload["max_tokens"], self.model_id,
                                            allow_tools=bool(payload.get("tools")) and payload.get("tool_choice") != "none")
        self.remember_id(value["id"])
        for index, call in enumerate(semantic.get("tool_calls", [])):
            http.check(call["id"] == f"call_{value['id']}_{index}", "Tool ID is not scoped to its response")
        self.idle()
        return semantic, {"id": value["id"], "timing": timing, "semantic": semantic}

    def pair(self, label, payload):
        expected, reference = self.json_result(label + "_json", payload)
        streamed = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        client = self.stream(label + "_sse", streamed)
        try:
            summary = client.complete(expected_generated=expected["usage"]["completion_tokens"])
            self.remember_id(summary["id"])
            actual = client.semantic()
            http.check(comparable(actual) == comparable(expected), "JSON and fragmented HTTP SSE differ")
            for index, call in enumerate(actual.get("tool_calls", [])):
                http.check(call["id"] == f"call_{summary['id']}_{index}", "Stream tool ID differs from its response")
                http.check(call["id"] != expected["tool_calls"][index]["id"], "Tool ID crossed independent responses")
        finally:
            client.close()
        self.idle()
        return expected, {"reference": reference, "stream": summary}

    def calls(self):
        results = []
        for scenario in (("json", "xml") if self.mode == "fake" else ("json",)):
            semantic, record = self.pair("tool_" + scenario, self.payload(scenario))
            calls = semantic.get("tool_calls", [])
            http.check(len(calls) == 1 and calls[0]["function"]["name"] == "weather" and
                       semantic["finish_reason"] == "tool_calls", "Named choice did not return one weather call")
            arguments = http.strict_json(calls[0]["function"]["arguments"])
            http.check(arguments == {"city": "北京" if self.mode == "fake" else "Beijing"},
                       f"Tool argument differs from the requested city: {arguments!r}")
            results.append(record)
        return results

    def history(self):
        body = self.payload()
        generated, call_record = self.json_result("history_call", body)
        calls = generated.get("tool_calls", [])
        http.check(len(calls) == 1, "History fixture needs exactly one completed call")
        body["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        body["messages"].append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "Sunny, 23 C."})
        body["messages"].append({"role": "user", "content": ("<|tool_plain|>\n" if self.mode == "fake" else "") +
                                 "Give a short weather summary."})
        body["tool_choice"] = "none"
        result, record = self.pair("history_result", body)
        http.check(result["content"] and "tool_calls" not in result, "Tool result history did not return text")
        if self.mode == "fake":
            http.check(result["content"] == "Tool result processed.", "Fake history result was altered")
        return {"call": call_record, "followup": record}

    def concurrent(self):
        payloads = [self.payload("json", "Beijing"), self.payload("xml", "Paris")]
        references = [self.json_result(f"parallel_reference_{index}", payload)[0] for index, payload in enumerate(payloads)]
        if self.mode == "gpu":
            for reference, city in zip(references, ("Beijing", "Paris")):
                calls = reference.get("tool_calls", [])
                http.check(len(calls) == 1 and calls[0]["function"]["name"] == "weather" and
                           http.strict_json(calls[0]["function"]["arguments"]) == {"city": city},
                           "Independent tool reference did not preserve its requested city")
        clients = [self.stream(f"parallel_sse_{index}", {**payload, "stream": True,
                   "stream_options": {"include_usage": True}}) for index, payload in enumerate(payloads)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(client.complete, expected_generated=reference["usage"]["completion_tokens"])
                           for client, reference in zip(clients, references)]
                stats = [future.result(timeout=self.read_timeout + 10) for future in futures]
            ids = []
            for client, reference, item in zip(clients, references, stats):
                self.remember_id(item["id"])
                http.check(comparable(client.semantic()) == comparable(reference), "Concurrent tools crossed session output")
                ids.extend(call["id"] for call in client.tool_calls)
            http.check(len(ids) == 2 and len(set(ids)) == 2, "Concurrent calls reused IDs or lost a call")
            http.check(max(item["started_at"] for item in stats) < min(item["done_at"] for item in stats),
                       "Tool SSE requests did not overlap")
        finally:
            for client in clients:
                client.close()
        self.idle()
        return {"references": references, "streams": stats}

    def validation(self):
        mutations = [
            ("strict", lambda b: b["tools"][0]["function"].update(strict=True), "unsupported_parameter"),
            ("unknown_choice", lambda b: b["tool_choice"]["function"].update(name="missing"), "invalid_parameter"),
            ("duplicate", lambda b: b["tools"].append(copy.deepcopy(b["tools"][0])), "invalid_parameter"),
            ("orphan", lambda b: b["messages"].append({"role": "tool", "tool_call_id": "orphan", "content": "value"}), "invalid_parameter"),
            ("parallel_type", lambda b: b.update(parallel_tool_calls=1), "invalid_parameter"),
        ]
        records = []
        for label, mutate, code in mutations:
            body = self.payload(); mutate(body)
            value, timing = self.json_call("invalid_" + label, "/v1/chat/completions", "POST", body, expected=400)
            http.check(value.get("error", {}).get("code") == code, "Invalid tool request received the wrong error")
            records.append({"name": label, "error": value, "timing": timing})
            self.idle()
        return records

    def output_errors(self):
        records = []
        cases = [("malformed", {}, "invalid_tool_call"), ("unknown", {}, "invalid_tool_call"),
                 ("unfinished", {}, "invalid_tool_call"), ("plain", {}, "tool_choice_violation"),
                 ("json", {"tool_choice": {"type": "function", "function": {"name": "calc"}}}, "tool_choice_violation"),
                 ("parallel", {"tool_choice": "required"}, "tool_choice_violation"),
                 ("json", {"max_tokens": 20}, "invalid_tool_call")]
        for index, (scenario, options, code) in enumerate(cases):
            body = {**self.payload(scenario), **options}
            label = f"tool_error_{index}_{scenario}"
            value, timing = self.json_call(label + "_json", "/v1/chat/completions", "POST", body, expected=502)
            http.check(value.get("error", {}).get("code") == code and "choices" not in value, "Output error fabricated a completion")
            self.idle()
            client = self.stream(label + "_sse", {**body, "stream": True, "stream_options": {"include_usage": True}})
            try:
                stats = client.complete(allow_error=True)
                http.check(len(stats["errors"]) == 1 and stats["errors"][0]["code"] == code and
                           client.finish_count == 0 and client.usage_count == 0, "Tool stream error has a success terminal")
            finally:
                client.close()
            self.idle()
            records.append({"scenario": scenario, "json": value, "timing": timing, "stream": stats})
        recovery = self.payload("plain"); recovery["tool_choice"] = "none"
        result, record = self.pair("after_tool_errors", recovery)
        http.check(result["content"] == "Tool result processed.", "Tool failure poisoned subsequent inference")
        return {"errors": records, "recovery": record}

    def multiple_and_long(self):
        body = self.payload("parallel"); body.update(tool_choice="required", parallel_tool_calls=True)
        multiple, multiple_record = self.pair("multiple_calls", body)
        http.check([call["function"]["name"] for call in multiple["tool_calls"]] == ["weather", "calc"],
                   "Multiple tool calls were dropped or reordered")
        large, large_record = self.pair("long_arguments", self.payload("long"))
        http.check(http.strict_json(large["tool_calls"][0]["function"]["arguments"]) == {"city": "北京🙂" * 2400},
                   "Long tool arguments changed across bounded SSE frames")
        http.check(large["usage"]["completion_tokens"] == 26 and large_record["stream"]["content_frames"] > 16,
                   "Long token fragments did not exercise tool frame subdivision")
        return {"multiple": multiple_record, "long": large_record}

    def text_compatibility(self):
        body = self.payload("parallel"); body["tool_choice"] = "none"
        result, record = self.pair("tool_none_text", body)
        http.check(result["content"].startswith("<tool_call>") and "tool_calls" not in result,
                   "Disabled tools changed unsolicited text behavior")
        no_tools = copy.deepcopy(body); no_tools.pop("tools"); no_tools.pop("tool_choice")
        second, second_record = self.pair("without_tools_text", no_tools)
        http.check(second["content"] == result["content"], "No-tools output differs from explicit none")
        return {"none": record, "absent": second_record}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mode", choices=("fake", "gpu"), required=True)
    parser.add_argument("--backend", choices=tuple(http.MODEL_IDS), default="qwen3_nf4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--shutdown-timeout", type=float, default=45)
    args = parser.parse_args()
    if args.mode == "gpu" and args.model is None:
        parser.error("GPU mode requires --model and a machine-guarded Windows Job launcher")
    args.context = 4096 if args.mode == "fake" else 2048
    args.model_id = http.MODEL_IDS[args.backend]
    for name in ("server", "worker", "model", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "started_at": http.utc_now(), "mode": args.mode,
              "backend": args.backend, "context": args.context, "passed": False, "tests": [],
              "scope": "Function tool request/history/output semantics; no schema-constrained decoding or tool execution"}
    process = runner = None
    planned = []
    try:
        if args.mode == "fake":
            model = directory / "fake-http-tools-model"
            result["fixture"] = http.write_fake_tokenizer(model, args.backend, tools=True)
        else:
            model = args.model
        process = http.ServerProcess(args, directory, model)
        runner = ToolRunner(args, process, directory, result)
        result["initial_health"] = runner.wait_ready()
        planned = [("tool_request_validation", runner.validation), ("tool_json_xml_sse", runner.calls),
                   ("tool_result_history", runner.history), ("concurrent_tool_isolation", runner.concurrent)]
        if args.mode == "fake":
            planned.extend([("tool_output_errors_and_recovery", runner.output_errors),
                            ("multiple_tools_and_long_arguments", runner.multiple_and_long),
                            ("tools_none_and_absent_text_compatibility", runner.text_compatibility)])
        result["planned_cases"] = [name for name, _ in planned]
        for name, function in planned:
            runner.case(name, function)
        result["passed"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        if runner:
            runner.close_clients()
        if process:
            try:
                cleanup = result["cleanup"] = process.stop()
                http.check(not cleanup.get("forced_kill") and cleanup.get("exit_code") == 0 and not cleanup.get("framework_errors"),
                           "Tool server did not shut down cleanly")
                runner.validate_cleanup(cleanup)
            except BaseException as error:
                result["passed"] = False
                result["cleanup_error"] = f"{type(error).__name__}: {error}"
        result["not_run"] = [name for name, _ in planned if name not in {case["name"] for case in result["tests"]}]
        result["passed"] = result["passed"] and bool(planned) and not result["not_run"] and all(case["passed"] for case in result["tests"])
        result["finished_at"] = http.utc_now()
        http.write_json(directory / "result.json", result)
    print(json.dumps({"passed": result["passed"], "tests": len(result["tests"]), "output": str(directory)}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
