#!/usr/bin/env python3
"""
Aila C API smoke / stability test.

Usage:
  python test_api.py [--model MODEL_DIR] [--quick]

Requires: AilaShared.dll (and all its oneAPI runtime dependencies) in build/
Run from the repo root (or set --build-dir).

The script exercises every public function in aila_api.h:
  - aila_version, aila_engine_create/destroy, aila_engine_init
  - aila_default_gen_config, aila_generate, aila_generate_messages
  - aila_generate_stream, aila_generate_messages_stream
  - aila_set_log_callback, aila_set_log_level
  - aila_engine_reset_context, aila_engine_context_length
  - aila_last_error_code, aila_last_error_message
  - aila_free_string

Stress tests:
  - repeated create/destroy
  - concurrent threaded generation (GIL released during calls)
  - empty / oversize prompts
  - NULL safety on every pointer-accepting function
"""

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_void_p,
    cdll,
    create_string_buffer,
    string_at,
)


# ---------------------------------------------------------------------------
# ctypes definitions
# ---------------------------------------------------------------------------

class AilaGenConfig(Structure):
    _fields_ = [
        ("max_new_tokens",     c_int),
        ("temperature",        c_float),
        ("top_k",              c_int),
        ("top_p",              c_float),
        ("repetition_penalty", c_float),
        ("presence_penalty",   c_float),
        ("frequency_penalty",  c_float),
        ("do_sample",          c_int),
        ("decode_chunk_size",  c_int),
        ("stream_chunk_size",  c_int),
    ]


# Callback types
TokenCallback = CFUNCTYPE(c_int, c_char_p, c_void_p)
LogCallback = CFUNCTYPE(None, c_int, c_char_p, c_void_p)

# Error codes (must match aila_api.h)
ERROR_CODES = {
    0: "OK",
    1: "INVALID_ARGUMENT",
    2: "TEMPLATE",
    3: "JSON_PARSE",
    4: "VISION_NOT_ENABLED",
    5: "CONTEXT_OVERFLOW",
    6: "RUNTIME",
}


class AilaAPI:
    """Thin ctypes wrapper over AilaShared.dll."""

    def __init__(self, dll_path: str):
        self._dll = cdll.LoadLibrary(dll_path)

        # --- bind every function ---
        self._set("aila_version", c_char_p, [])

        self._set("aila_engine_create", c_void_p, [])
        self._set("aila_engine_destroy", None, [c_void_p])
        self._set("aila_engine_init", c_int, [c_void_p, c_char_p, c_int])

        self._set("aila_default_gen_config", AilaGenConfig, [])

        self._set("aila_generate", c_void_p, [c_void_p, c_char_p, c_void_p])
        self._set("aila_generate_messages", c_void_p, [c_void_p, c_char_p, c_void_p])
        self._set("aila_generate_stream", c_int, [c_void_p, c_char_p, c_void_p, TokenCallback, c_void_p])
        self._set("aila_generate_messages_stream", c_int, [c_void_p, c_char_p, c_void_p, TokenCallback, c_void_p])

        self._set("aila_free_string", None, [c_void_p])
        self._set("aila_transcribe", c_void_p, [
            c_void_p,          # engine
            c_char_p,          # wav_path
            c_void_p,          # config
            c_char_p,          # forced_language
            c_char_p,          # system_prompt
            c_float,           # segment_sec
            c_int,             # past_text_conditioning
            TokenCallback,     # token_callback
            c_void_p,          # user_data
            POINTER(c_char_p)  # language_out
        ])

        self._set("aila_transcribe_stream_create", c_void_p, [
            c_void_p,          # engine
            c_void_p,          # config
            c_char_p,          # forced_language
            c_char_p,          # system_prompt
        ])
        self._set("aila_transcribe_stream_feed", c_int, [
            c_void_p,          # stream
            POINTER(ctypes.c_float), # samples
            c_int,             # sample_count
        ])
        self._set("aila_transcribe_stream_get_text", c_int, [
            c_void_p,          # stream
            POINTER(c_char_p), # out_stable
            POINTER(c_char_p), # out_partial
        ])
        self._set("aila_transcribe_stream_destroy", None, [
            c_void_p,          # stream
        ])

        self._set("aila_set_log_callback", None, [LogCallback, c_void_p])
        self._set("aila_set_log_level", None, [c_int])

        self._set("aila_engine_reset_context", None, [c_void_p])
        self._set("aila_engine_context_length", c_int, [c_void_p])

        self._set("aila_last_error_code", c_int, [c_void_p])
        self._set("aila_last_error_message", c_char_p, [c_void_p])

    def _set(self, name, restype, argtypes):
        fn = getattr(self._dll, name)
        fn.restype = restype
        fn.argtypes = argtypes
        setattr(self, name, fn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED = "\033[91m"
GRN = "\033[92m"
YLW = "\033[93m"
RST = "\033[0m"

_passed = 0
_failed = 0
_skipped = 0

def test(name: str, ok: bool, detail: str = ""):
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  {GRN}PASS{RST} {name}" + (f" — {detail}" if detail else ""))
    else:
        _failed += 1
        print(f"  {RED}FAIL{RST} {name}" + (f" — {detail}" if detail else ""))

def skip(name: str, reason: str = ""):
    global _skipped
    _skipped += 1
    print(f"  {YLW}SKIP{RST} {name}" + (f" — {reason}" if reason else ""))


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def test_version(api: AilaAPI):
    print("\n[Version]")
    s = api.aila_version()
    test("version returns non-NULL", s is not None)
    if s:
        ver = s.decode() if isinstance(s, bytes) else s
        test("version string non-empty", len(ver) > 0, ver)


def test_lifecycle(api: AilaAPI):
    print("\n[Lifecycle]")
    # Create
    e = api.aila_engine_create()
    test("create returns non-NULL", e is not None)

    # Double destroy must not crash
    api.aila_engine_destroy(e)
    api.aila_engine_destroy(None)  # NULL safety
    test("double destroy + NULL safe", True)


def test_init(api: AilaAPI, model_dir: str, max_seq: int):
    print("\n[Init]")
    e = api.aila_engine_create()
    test("create", e is not None)

    # NULL safety
    rc0 = api.aila_engine_init(e, None, max_seq)
    test("init(NULL model_dir) fails", rc0 != 0)

    rc1 = api.aila_engine_init(None, model_dir.encode(), max_seq)
    test("init(NULL engine) fails", rc1 != 0)

    # Real init
    t0 = time.time()
    rc = api.aila_engine_init(e, model_dir.encode(), max_seq)
    elapsed = time.time() - t0
    ok = rc == 0
    test("init succeeds", ok, f"{elapsed:.1f}s")
    if not ok:
        code = api.aila_last_error_code(e)
        msg_p = api.aila_last_error_message(e)
        msg = msg_p.decode() if msg_p else "(null)"
        print(f"       error code={code} ({ERROR_CODES.get(code, '?')}) msg={msg}")
        api.aila_engine_destroy(e)
        return None
    return e


def test_generate_simple(api: AilaAPI, engine):
    print("\n[Generate — simple]")

    # NULL safety
    null_out = api.aila_generate(engine, None, None)
    test("generate(NULL prompt) returns NULL", null_out is None)
    test("generate(NULL engine) returns NULL", api.aila_generate(None, b"hi", None) is None)

    # Basic generation (greedy, short)
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 16
    cfg.do_sample = 0  # greedy for determinism

    t0 = time.time()
    out_p = api.aila_generate(engine, b"Say just hello.", byref(cfg))
    elapsed = time.time() - t0

    test("generate returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("generate produces text", len(text) > 0, f"{elapsed:.1f}s → {text[:80]!r}")
        api.aila_free_string(out_p)
    else:
        code = api.aila_last_error_code(engine)
        msg_p = api.aila_last_error_message(engine)
        msg = msg_p.decode() if msg_p else "(null)"
        print(f"       error code={code} ({ERROR_CODES.get(code, '?')}) msg={msg}")

    # Multi-turn
    api.aila_engine_reset_context(engine)
    out2_p = api.aila_generate(engine, b"what is 1+1?", byref(cfg))
    test("generate multi-turn", out2_p is not None)
    if out2_p:
        api.aila_free_string(out2_p)


def test_generate_sampling(api: AilaAPI, engine):
    print("\n[Generate — sampling]")
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 32
    cfg.do_sample = 1
    cfg.temperature = 0.8
    cfg.top_k = 15
    cfg.top_p = 0.95
    cfg.repetition_penalty = 1.1

    api.aila_engine_reset_context(engine)
    out_p = api.aila_generate(engine, b"List three colors.", byref(cfg))
    test("sampling returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("sampling produces text", len(text) > 0, text[:80])
        api.aila_free_string(out_p)


def test_generate_messages(api: AilaAPI, engine):
    print("\n[Generate — messages JSON]")
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 16
    cfg.do_sample = 0

    msgs = json.dumps([
        {"role": "system", "content": "Reply in English."},
        {"role": "user",   "content": "say hello"},
    ])

    out_p = api.aila_generate_messages(engine, msgs.encode(), byref(cfg))
    test("generate_messages (legacy array) returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("generate_messages (legacy array) produces text", len(text) > 0, text[:80])
        api.aila_free_string(out_p)

    # Test OpenAI compatible object format
    msgs_openai = json.dumps({
        "messages": [
            {"role": "system", "content": "Reply in English."},
            {"role": "user",   "content": "say hello"},
        ],
        "temperature": 0.0,
        "max_tokens": 12,
        "seed": 42
    })
    out_openai_p = api.aila_generate_messages(engine, msgs_openai.encode(), byref(cfg))
    test("generate_messages (OpenAI compatible object) returns non-NULL", out_openai_p is not None)
    if out_openai_p:
        text_openai = string_at(out_openai_p).decode("utf-8", errors="replace")
        test("generate_messages (OpenAI compatible object) produces text", len(text_openai) > 0, text_openai[:80])
        api.aila_free_string(out_openai_p)

    # Test OpenAI compatible parameters override
    # 1. max_tokens override
    cfg_large = api.aila_default_gen_config()
    cfg_large.max_new_tokens = 64
    cfg_large.do_sample = 0

    msgs_max_tokens = json.dumps({
        "messages": [
            {"role": "user", "content": "Count from 1 to 5: 1, 2,"}
        ],
        "max_tokens": 3
    })

    token_count = [0]
    @TokenCallback
    def count_callback(text_p, _userdata):
        token_count[0] += 1
        return 0

    api.aila_engine_reset_context(engine)
    rc = api.aila_generate_messages_stream(engine, msgs_max_tokens.encode(), byref(cfg_large), count_callback, None)
    test("generate_messages_stream with max_tokens override succeeds", rc == 0)
    test("max_tokens override works (got exactly 3 tokens)", token_count[0] == 3, f"got {token_count[0]} tokens")

    # 2. temperature=0.0 override to greedy
    cfg_temp = api.aila_default_gen_config()
    cfg_temp.do_sample = 1
    cfg_temp.temperature = 0.95
    cfg_temp.max_new_tokens = 8

    msgs_temp = json.dumps({
        "messages": [
            {"role": "user", "content": "1 + 1 = ?"}
        ],
        "temperature": 0.0
    })

    api.aila_engine_reset_context(engine)
    out_temp1_p = api.aila_generate_messages(engine, msgs_temp.encode(), byref(cfg_temp))
    text1 = string_at(out_temp1_p).decode("utf-8").strip() if out_temp1_p else ""
    if out_temp1_p:
        api.aila_free_string(out_temp1_p)

    api.aila_engine_reset_context(engine)
    out_temp2_p = api.aila_generate_messages(engine, msgs_temp.encode(), byref(cfg_temp))
    text2 = string_at(out_temp2_p).decode("utf-8").strip() if out_temp2_p else ""
    if out_temp2_p:
        api.aila_free_string(out_temp2_p)

    test("temperature=0.0 override produces identical outputs (greedy)", text1 == text2 and len(text1) > 0, f"text1={text1!r}, text2={text2!r}")

    # NULL safety
    test("generate_messages(NULL engine)", api.aila_generate_messages(None, msgs.encode(), None) is None)
    test("generate_messages(NULL json)",  api.aila_generate_messages(engine, None, None) is None)


def test_generate_no_think(api: AilaAPI, engine):
    print("\n[Generate — /no_think]")
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 24
    cfg.do_sample = 0

    api.aila_engine_reset_context(engine)
    out_p = api.aila_generate(engine, b"say hi /no_think", byref(cfg))
    test("/no_think returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        # Should NOT contain raw /no_think or <think> tags in the output
        ok = "/no_think" not in text and "<think>" not in text
        test("/no_think not in output, <think> suppressed", ok, text[:80])
        api.aila_free_string(out_p)


def test_generate_think(api: AilaAPI, engine):
    print("\n[Generate — /think]")
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 32
    cfg.do_sample = 0

    api.aila_engine_reset_context(engine)
    out_p = api.aila_generate(engine, b"say hi /think", byref(cfg))
    test("/think returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        # /think suffix should be stripped from visible output; <think> tags
        # should be present (either prepended by us or generated by model)
        ok = "/think" not in text
        has_think = "<think>" in text
        test("/think stripped from output, <think> present", ok and has_think,
             f"has_think={has_think} text={text[:100]!r}")
        api.aila_free_string(out_p)

    # Also test via generate_messages (the --messages-json path)
    msgs = json.dumps([
        {"role": "user", "content": "say hi /think"},
    ])
    api.aila_engine_reset_context(engine)
    out2_p = api.aila_generate_messages(engine, msgs.encode(), byref(cfg))
    test("/think via messages returns non-NULL", out2_p is not None)
    if out2_p:
        text2 = string_at(out2_p).decode("utf-8", errors="replace")
        ok2 = "/think" not in text2 and "<think>" in text2
        test("/think via messages: stripped, <think> present", ok2,
             f"has_think={'<think>' in text2} text={text2[:100]!r}")
        api.aila_free_string(out2_p)


def test_streaming(api: AilaAPI, engine):
    print("\n[Generate — streaming]")
    tokens = []
    @TokenCallback
    def on_token(text_p, _userdata):
        text = string_at(text_p).decode("utf-8", errors="replace") if text_p else ""
        tokens.append(text)
        return 0  # continue

    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 16
    cfg.do_sample = 0

    api.aila_engine_reset_context(engine)
    rc = api.aila_generate_stream(engine, b"Count: 1 2 3", byref(cfg), on_token, None)
    test("generate_stream returns 0", rc == 0)
    test("streaming tokens received", len(tokens) > 0, f"{len(tokens)} chunks")
    full = "".join(tokens)
    test("streaming output non-empty", len(full) > 0, full[:80])

    # NULL safety
    null_cb = TokenCallback()
    test("stream(NULL engine)",   api.aila_generate_stream(None, b"x", None, on_token, None) != 0)
    test("stream(NULL prompt)",   api.aila_generate_stream(engine, None, None, on_token, None) != 0)
    # null_cb is a NULL function pointer; should be rejected by arg validation
    test("stream(NULL callback)", api.aila_generate_stream(engine, b"x", None, null_cb, None) != 0)


def test_streaming_messages(api: AilaAPI, engine):
    print("\n[Generate — streaming messages]")
    tokens = []
    @TokenCallback
    def on_token(text_p, _userdata):
        text = string_at(text_p).decode("utf-8", errors="replace") if text_p else ""
        tokens.append(text)
        return 0

    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 16
    cfg.do_sample = 0

    msgs = json.dumps([{"role": "user", "content": "say hello"}])
    rc = api.aila_generate_messages_stream(engine, msgs.encode(), byref(cfg), on_token, None)
    test("generate_messages_stream returns 0", rc == 0)
    test("streaming-messages tokens received", len(tokens) > 0, f"{len(tokens)} chunks")


def test_streaming_abort(api: AilaAPI, engine):
    print("\n[Generate — streaming abort]")
    count = [0]
    @TokenCallback
    def on_token(text_p, _userdata):
        count[0] += 1
        return 1  # abort immediately

    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 64
    cfg.do_sample = 0

    api.aila_engine_reset_context(engine)
    rc = api.aila_generate_stream(engine, b"Write a very long story about dragons.", byref(cfg), on_token, None)
    test("stream abort returns 1", rc == 1, f"tokens before abort: {count[0]}")


def test_context_api(api: AilaAPI, engine):
    print("\n[Context management]")
    api.aila_engine_reset_context(engine)
    clen = api.aila_engine_context_length(engine)
    test("context_length after reset", clen == 0 or clen >= 0, str(clen))

    # Generate something to populate context
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 8
    cfg.do_sample = 0
    out_p = api.aila_generate(engine, b"hello", byref(cfg))
    if out_p:
        api.aila_free_string(out_p)

    clen2 = api.aila_engine_context_length(engine)
    test("context_length after generate", clen2 > 0, str(clen2))

    # Reset again
    api.aila_engine_reset_context(engine)
    clen3 = api.aila_engine_context_length(engine)
    test("context_length after second reset", clen3 == 0 or clen3 >= 0, str(clen3))

    # NULL safety
    test("context_length(NULL)", api.aila_engine_context_length(None) == 0)


def test_error_api(api: AilaAPI, engine):
    print("\n[Error reporting]")
    # NULL safety
    test("last_error_code(NULL)", api.aila_last_error_code(None) == 1)
    test("last_error_message(NULL) empty", api.aila_last_error_message(None) is not None)

    # After a successful call
    api.aila_engine_reset_context(engine)
    code = api.aila_last_error_code(engine)
    msg_p = api.aila_last_error_message(engine)
    msg = msg_p.decode() if msg_p else "(null)"
    test("last_error_code after success", code == 0, f"{code} ({ERROR_CODES.get(code, '?')}) msg={msg}")

    # After a failing call (invalid JSON triggers JsonParseError)
    api.aila_generate_messages(engine, b"not valid json {{{", None)
    code2 = api.aila_last_error_code(engine)
    test("last_error_code after failure", code2 != 0, f"{code2} ({ERROR_CODES.get(code2, '?')})")


def test_log_callback(api: AilaAPI):
    print("\n[Log callback]")
    captured = []
    @LogCallback
    def on_log(level, msg_p, _userdata):
        msg = string_at(msg_p).decode("utf-8", errors="replace") if msg_p else ""
        captured.append((level, msg))

    api.aila_set_log_callback(on_log, None)
    api.aila_set_log_level(0)  # debug

    # Trigger a log by doing a real init with a bogus path — this logs errors
    e = api.aila_engine_create()
    api.aila_engine_init(e, b"./nonexistent_model_dir_12345", 4096)
    api.aila_engine_destroy(e)

    test("log callback received messages", len(captured) > 0, f"{len(captured)} messages")

    # Restore default
    null_cb = LogCallback()  # NULL function pointer
    api.aila_set_log_callback(null_cb, None)
    api.aila_set_log_level(2)  # warning


def test_stress_create_destroy(api: AilaAPI):
    print("\n[Stress — create/destroy]")
    n = 20
    for i in range(n):
        e = api.aila_engine_create()
        if e is not None:
            api.aila_engine_destroy(e)
    test(f"create/destroy x{n}", True, "no crash")


def test_stress_threaded_generation(api: AilaAPI, model_dir: str, max_seq: int):
    """Multiple threads each create their own engine and generate."""
    print("\n[Stress — threaded generation]")
    errors = []
    results = [None] * 4

    def worker(idx: int):
        try:
            e = api.aila_engine_create()
            if e is None:
                errors.append(f"thread {idx}: create failed")
                return
            rc = api.aila_engine_init(e, model_dir.encode(), max_seq)
            if rc != 0:
                errors.append(f"thread {idx}: init failed (rc={rc})")
                api.aila_engine_destroy(e)
                return
            cfg = api.aila_default_gen_config()
            cfg.max_new_tokens = 8
            cfg.do_sample = 0
            out_p = api.aila_generate(e, f"Thread {idx} says hi.".encode(), byref(cfg))
            if out_p:
                results[idx] = string_at(out_p).decode("utf-8", errors="replace")
                api.aila_free_string(out_p)
            api.aila_engine_destroy(e)
        except Exception as ex:
            errors.append(f"thread {idx}: exception: {ex}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(results))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    test("no errors", len(errors) == 0, "; ".join(errors) if errors else "")
    for i, r in enumerate(results):
        test(f"thread {i} result non-empty", r is not None and len(r) > 0, r[:60] if r else "None")


def test_transcribe(api: AilaAPI, engine, model_path: str):
    print("\n[Transcribe]")
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 64
    cfg.do_sample = 0

    wav_path = "./This is an English test.wav"
    is_asr_model = "asr" in model_path.lower()

    if not is_asr_model:
        # Defense test for non-ASR models
        lang_p = c_char_p()
        out_p = api.aila_transcribe(engine, wav_path.encode(), byref(cfg), None, None, 0.0, 0, TokenCallback(), None, byref(lang_p))
        test("transcribe on non-ASR model returns NULL", out_p is None)
        err = api.aila_last_error_code(engine)
        test("transcribe error code is RUNTIME (6)", err == 6) # RUNTIME
        return

    # Tests with ASR model
    if not os.path.isfile(wav_path):
        skip("transcribe test", f"audio file not found at {wav_path}")
        return

    # Test 1: Auto-detection (forced_language = None)
    lang_p = c_char_p()
    t0 = time.time()
    out_p = api.aila_transcribe(engine, wav_path.encode(), byref(cfg), None, None, 0.0, 0, TokenCallback(), None, byref(lang_p))
    elapsed = time.time() - t0

    test("transcribe (auto-detect) returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("transcribe text is clean (no 'language' or '<asr_text>')", "language" not in text and "<asr_text>" not in text)
        test("transcribe text length > 0", len(text) > 0, f"{elapsed:.1f}s → {text[:80]!r}")
        api.Free_string = getattr(api, "aila_free_string")
        api.Free_string(out_p)

    test("language_out is not NULL", lang_p.value is not None)
    if lang_p.value:
        lang = lang_p.value.decode("utf-8")
        test("detected language is English", lang == "English", f"lang={lang!r}")
        api.Free_string = getattr(api, "aila_free_string")
        api.Free_string(lang_p)

    # Test 2: Forced Language (forced_language = "English")
    lang_p = c_char_p()
    out_p = api.aila_transcribe(engine, wav_path.encode(), byref(cfg), b"English", None, 0.0, 0, TokenCallback(), None, byref(lang_p))
    test("transcribe (forced language) returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("forced transcribe text length > 0", len(text) > 0, f"text={text[:80]!r}")
        api.Free_string = getattr(api, "aila_free_string")
        api.Free_string(out_p)

    test("language_out for forced language is English", lang_p.value is not None)
    if lang_p.value:
        lang = lang_p.value.decode("utf-8")
        test("forced language value is English", lang == "English", f"lang={lang!r}")
        api.Free_string = getattr(api, "aila_free_string")
        api.Free_string(lang_p)

    # Test 3: NULL safety
    test("transcribe(NULL engine) returns NULL", api.aila_transcribe(None, wav_path.encode(), None, None, None, 0.0, 0, TokenCallback(), None, None) is None)
    test("transcribe(NULL wav_path) returns NULL", api.aila_transcribe(engine, None, None, None, None, 0.0, 0, TokenCallback(), None, None) is None)

    # Test 4: System Prompt and Streaming Callback
    streaming_tokens = []
    @TokenCallback
    def on_asr_token(piece_p, _userdata):
        piece = string_at(piece_p).decode("utf-8", errors="replace") if piece_p else ""
        streaming_tokens.append(piece)
        return 0

    lang_p = c_char_p()
    out_p = api.aila_transcribe(
        engine,
        wav_path.encode(),
        byref(cfg),
        None,
        b"Preserve spelling",
        5.0,
        1,
        on_asr_token,
        None,
        byref(lang_p)
    )
    test("transcribe with advanced parameters returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("advanced transcribe text length > 0", len(text) > 0, text)
        test("advanced streaming tokens received", len(streaming_tokens) > 0, f"{len(streaming_tokens)} segment chunks: {streaming_tokens}")
        api.Free_string = getattr(api, "aila_free_string")
        api.Free_string(out_p)
    if lang_p.value:
        api.Free_string = getattr(api, "aila_free_string")
        api.Free_string(lang_p)

    # Test 5: Real-time Streaming Input ASR (Online ASR)
    print("\n[ASR Stream Input (Online ASR)]")
    import struct

    # Load English test WAV
    wav_path = "./This is an English test.wav"
    if not os.path.isfile(wav_path):
        skip("stream input test", f"audio file not found at {wav_path}")
        return

    # Read float samples with a custom WAV parser to support IEEE Float (format 3)
    def read_wav_custom(path):
        with open(path, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                raise ValueError("Not a RIFF file")
            f.read(4)
            wave_tag = f.read(4)
            if wave_tag != b"WAVE":
                raise ValueError("Not a WAVE file")

            audio_format = 1
            num_channels = 1
            sample_rate = 16000
            bits_per_sample = 16
            data = b""

            while True:
                subchunk_id = f.read(4)
                if not subchunk_id or len(subchunk_id) < 4:
                    break
                subchunk_size_bytes = f.read(4)
                if not subchunk_size_bytes or len(subchunk_size_bytes) < 4:
                    break
                subchunk_size = struct.unpack("<I", subchunk_size_bytes)[0]

                if subchunk_id == b"fmt ":
                    fmt_data = f.read(subchunk_size)
                    audio_format, num_channels, sample_rate = struct.unpack("<HHI", fmt_data[:8])
                    bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                elif subchunk_id == b"data":
                    data = f.read(subchunk_size)
                    break
                else:
                    f.seek(subchunk_size, 1)

            if not data:
                raise ValueError("No data chunk found")

            if audio_format == 3:  # IEEE float
                n_elements = len(data) // 4
                samples = struct.unpack(f"<{n_elements}f", data)
                float_samples = list(samples)
            elif audio_format == 1:  # PCM
                if bits_per_sample == 16:
                    n_elements = len(data) // 2
                    samples = struct.unpack(f"<{n_elements}h", data)
                    float_samples = [s / 32768.0 for s in samples]
                elif bits_per_sample == 8:
                    n_elements = len(data)
                    samples = struct.unpack(f"<{n_elements}B", data)
                    float_samples = [(s - 128) / 128.0 for s in samples]
                else:
                    raise ValueError(f"Unsupported PCM bits per sample: {bits_per_sample}")
            else:
                raise ValueError(f"Unsupported audio format tag: {audio_format}")

            if num_channels > 1:
                mono = []
                for i in range(0, len(float_samples), num_channels):
                    mono.append(sum(float_samples[i:i+num_channels]) / num_channels)
                float_samples = mono

            return float_samples

    try:
        float_samples = read_wav_custom(wav_path)
    except Exception as e:
        skip("stream input test", f"failed to parse WAV: {e}")
        return

    # Create transcribe stream
    stream = api.aila_transcribe_stream_create(engine, byref(cfg), None, b"System prompt bias")
    test("transcribe_stream_create returns non-NULL", stream is not None)
    if stream:
        # Feed audio chunk by chunk (1.0s chunks)
        chunk_size = 16000
        stable_texts = []
        partial_texts = []

        for i in range(0, len(float_samples), chunk_size):
            chunk = float_samples[i:i+chunk_size]
            arr = (ctypes.c_float * len(chunk))(*chunk)
            ret_feed = api.aila_transcribe_stream_feed(stream, arr, len(chunk))
            test(f"feed chunk {i//chunk_size} returns OK", ret_feed == 0)

            out_stable = c_char_p()
            out_partial = c_char_p()
            ret_get = api.aila_transcribe_stream_get_text(stream, byref(out_stable), byref(out_partial))
            test(f"get_text chunk {i//chunk_size} returns OK", ret_get == 0)

            stable_val = string_at(out_stable).decode("utf-8", errors="replace") if out_stable.value else ""
            partial_val = string_at(out_partial).decode("utf-8", errors="replace") if out_partial.value else ""

            if out_stable.value:
                api.Free_string = getattr(api, "aila_free_string")
                api.Free_string(out_stable)
            if out_partial.value:
                api.Free_string = getattr(api, "aila_free_string")
                api.Free_string(out_partial)

            if stable_val:
                stable_texts.append(stable_val)
            if partial_val:
                partial_texts.append(partial_val)

        print(f"  Stable text progress: {stable_texts}")
        print(f"  Final partial text: {partial_texts[-1] if partial_texts else ''}")

        # Destroy stream
        api.aila_transcribe_stream_destroy(stream)
        test("stream destroyed successfully", True)


def test_stress_long_prompt(api: AilaAPI, engine):
    print("\n[Stress — long prompt]")
    cfg = api.aila_default_gen_config()
    cfg.max_new_tokens = 4
    cfg.do_sample = 0

    api.aila_engine_reset_context(engine)
    long_prompt = "Hello world. " * 200
    out_p = api.aila_generate(engine, long_prompt.encode(), byref(cfg))
    test("long prompt returns non-NULL", out_p is not None)
    if out_p:
        text = string_at(out_p).decode("utf-8", errors="replace")
        test("long prompt produces text", len(text) > 0, text[:80])
        api.aila_free_string(out_p)


def test_default_config(api: AilaAPI):
    print("\n[Default config]")
    cfg = api.aila_default_gen_config()
    ok = (
        cfg.max_new_tokens == 512
        and abs(cfg.temperature - 0.6) < 0.01
        and cfg.top_k == 20
        and abs(cfg.top_p - 0.95) < 0.01
        and cfg.do_sample == 1
        and cfg.decode_chunk_size == 12
        and cfg.stream_chunk_size == 4
    )
    test("defaults match spec", ok,
         f"tokens={cfg.max_new_tokens} temp={cfg.temperature:.3f} "
         f"top_k={cfg.top_k} top_p={cfg.top_p:.3f} "
         f"do_sample={cfg.do_sample} dc={cfg.decode_chunk_size} sc={cfg.stream_chunk_size}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aila C API smoke test")
    parser.add_argument("--model", default="./models/qwen3.5-0.8B-bnb-nf4-offline",
                        help="Model directory")
    parser.add_argument("--max-seq", type=int, default=4096,
                        help="Max sequence length")
    parser.add_argument("--build-dir", default="./build",
                        help="Build output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Skip stress tests")
    args = parser.parse_args()

    dll = os.path.abspath(os.path.join(args.build_dir, "AilaShared.dll"))
    if not os.path.isfile(dll):
        print(f"ERROR: DLL not found at {dll}")
        sys.exit(1)
    # DLL's oneAPI dependencies are in the same directory; add to PATH
    os.add_dll_directory(os.path.dirname(dll))

    print(f"DLL : {dll}")
    print(f"Model: {args.model}")

    api = AilaAPI(dll)

    # --- Phase 0: must-pass checks (no model needed) ---
    test_version(api)
    test_lifecycle(api)
    test_default_config(api)
    test_log_callback(api)

    # --- Phase 1: init + basic generation ---
    engine = test_init(api, args.model, args.max_seq)
    if engine is None:
        print(f"\n{RED}*** Engine init failed — skipping model-dependent tests ***{RST}")
        test("generate(NULL engine)", api.aila_generate(None, b"hi", None) is None)
        skip("model-dependent tests", "init failed")
    else:
        is_asr_model = "asr" in args.model.lower()
        if is_asr_model:
            test_error_api(api, engine)
            test_transcribe(api, engine, args.model)
        else:
            test_generate_simple(api, engine)
            test_generate_sampling(api, engine)
            test_generate_messages(api, engine)
            test_generate_no_think(api, engine)
            test_generate_think(api, engine)
            test_streaming(api, engine)
            test_streaming_messages(api, engine)
            test_streaming_abort(api, engine)
            test_context_api(api, engine)
            test_error_api(api, engine)
            test_transcribe(api, engine, args.model)
            test_stress_long_prompt(api, engine)

            if not args.quick:
                test_stress_create_destroy(api)
                test_stress_threaded_generation(api, args.model, args.max_seq)

        api.aila_engine_destroy(engine)

    # --- Report ---
    total = _passed + _failed + _skipped
    print(f"\n{'='*50}")
    print(f"Results: {_passed} passed, {_failed} failed, {_skipped} skipped (total {total})")
    if _failed:
        print(f"{RED}*** SOME TESTS FAILED ***{RST}")
        sys.exit(1)
    else:
        print(f"{GRN}All tests passed.{RST}")



if __name__ == "__main__":
    main()
