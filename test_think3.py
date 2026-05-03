"""Test /think on 4B model with high token count. Also checks CLI output."""
import ctypes, json, os, sys, subprocess, tempfile
from ctypes import c_int, c_float, c_char_p, c_void_p, Structure, byref, cdll, string_at

os.add_dll_directory(os.path.abspath("./build"))
dll = cdll.LoadLibrary("./build/AilaShared.dll")

class AilaGenConfig(Structure):
    _fields_ = [
        ("max_new_tokens", c_int), ("temperature", c_float), ("top_k", c_int),
        ("top_p", c_float), ("repetition_penalty", c_float),
        ("presence_penalty", c_float), ("frequency_penalty", c_float),
        ("do_sample", c_int), ("decode_chunk_size", c_int), ("stream_chunk_size", c_int),
    ]

def bind(fn, restype, argtypes):
    fn.restype, fn.argtypes = restype, argtypes

bind(dll.aila_engine_create, c_void_p, [])
bind(dll.aila_engine_destroy, None, [c_void_p])
bind(dll.aila_engine_init, c_int, [c_void_p, c_char_p, c_int])
bind(dll.aila_default_gen_config, AilaGenConfig, [])
bind(dll.aila_generate, c_void_p, [c_void_p, c_char_p, c_void_p])
bind(dll.aila_generate_messages, c_void_p, [c_void_p, c_char_p, c_void_p])
bind(dll.aila_free_string, None, [c_void_p])
bind(dll.aila_engine_reset_context, None, [c_void_p])
bind(dll.aila_last_error_code, c_int, [c_void_p])

def make_cfg(max_tokens=1024, do_sample=0):
    cfg = AilaGenConfig()
    dll.aila_default_gen_config(byref(cfg))
    cfg.max_new_tokens = max_tokens
    cfg.do_sample = do_sample
    return cfg

def gen(engine, prompt, max_tok=1024):
    cfg = make_cfg(max_tok, 0)
    out = dll.aila_generate(engine, prompt.encode(), byref(cfg))
    if out:
        return string_at(out).decode("utf-8", errors="replace")
    return f"ERROR: {dll.aila_last_error_code(engine)}"

def gen_msgs(engine, msgs, max_tok=1024):
    cfg = make_cfg(max_tok, 0)
    out = dll.aila_generate_messages(engine, json.dumps(msgs).encode(), byref(cfg))
    if out:
        return string_at(out).decode("utf-8", errors="replace")
    return f"ERROR: {dll.aila_last_error_code(engine)}"

def reset(engine):
    dll.aila_engine_reset_context(engine)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/qwen3.5-4B-bnb-nf4-visiondense"
print(f"Model: {MODEL}")
engine = dll.aila_engine_create()
rc = dll.aila_engine_init(engine, MODEL.encode(), 4096)
if rc != 0:
    print(f"INIT FAILED: {rc}")
    sys.exit(1)

TOKENS = 1024

# === Test 1: /think on 4B, interactive path, 1024 tokens ===
print("\n" + "="*60)
print(f"Test 1: /think on 4B, interactive, {TOKENS} tokens")
reset(engine)
r = gen(engine, "Explain quantum computing briefly. /think", TOKENS)
has_open = "<think>" in r
has_close = "</think>" in r
print(f"  length: {len(r)} chars")
print(f"  <think>: {has_open}, </think>: {has_close}")
# Show start
idx_start = r.find("<think>")
idx_end = r.find("</think>")
if idx_start >= 0:
    print(f"  starts-with-think: {r[:20].replace(chr(10),'\\n')}")
if idx_end >= 0:
    print(f"  </think> at position {idx_end}")
    # Show content around </think>
    after = r[idx_end+8:idx_end+8+100].replace(chr(10),'\\n')
    print(f"  after </think>: {after}")
else:
    print(f"  last 150 chars: ...{r[-150:].replace(chr(10),'\\n')}")

# === Test 2: /no_think on 4B, interactive path ===
print("\n" + "="*60)
print(f"Test 2: /no_think on 4B, interactive, {TOKENS} tokens")
reset(engine)
r = gen(engine, "Explain quantum computing briefly. /no_think", TOKENS)
print(f"  length: {len(r)} chars")
print(f"  <think>: {'<think>' in r}, </think>: {'</think>' in r}")
print(f"  start: {r[:120].replace(chr(10),'\\n')}")

# === Test 3: default (no suffix) on 4B ===
print("\n" + "="*60)
print(f"Test 3: default (no suffix) on 4B, {TOKENS} tokens")
reset(engine)
r = gen(engine, "Explain quantum computing briefly.", TOKENS)
print(f"  length: {len(r)} chars")
print(f"  <think>: {'<think>' in r}, </think>: {'</think>' in r}")
print(f"  start: {r[:120].replace(chr(10),'\\n')}")

# === Test 4: /think + multi-turn on 4B ===
print("\n" + "="*60)
print(f"Test 4: multi-turn on 4B, /think then no suffix")
reset(engine)
r1 = gen(engine, "My name is Alice. /think", 256)
print(f"  T1 len={len(r1)}, <think>={'<think>' in r1}, </think>={'</think>' in r1}")
r2 = gen(engine, "What is my name?", 128)
print(f"  T2: {r2[:120].replace(chr(10),'\\n')}")

# === Test 5: CLI output check ===
print("\n" + "="*60)
print("Test 5: CLI output via subprocess (/think on 4B)")
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
    json.dump([{"role":"user","content":"Explain quantum computing in 2 sentences. /think"}], f)
    tmp = f.name
try:
    result = subprocess.run(
        ["./build/Aila.exe", "-m", MODEL, "--messages-json", tmp,
         "--max-tokens", "1024", "--greedy", "--no-stream"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PATH": os.path.abspath("./build") + ";" + os.environ.get("PATH", "")}
    )
    # Filter engine logs, keep only generated output
    lines = result.stdout.split('\n')
    # The response is after "Engine ready!" and before any error/summary
    in_response = False
    response_lines = []
    for line in lines:
        if "Engine ready!" in line:
            in_response = True
            continue
        if in_response and line.strip() and not line.startswith('[') and not line.startswith('Aila:'):
            response_lines.append(line)
    text = '\n'.join(response_lines)
    print(f"  stderr: {result.stderr[:200] if result.stderr else '(none)'}")
    print(f"  length: {len(text)} chars")
    print(f"  <think>: {'<think>' in text}, </think>: {'</think>' in text}")
    # Show first and last parts
    clean = text.replace('\n', '\\n')
    print(f"  start: {clean[:120]}")
    if '</think>' in text:
        idx = text.index('</think>')
        print(f"  after </think>: {text[idx+8:idx+8+80].replace(chr(10),'\\n')}")
    print(f"  end: ...{clean[-120:]}")
finally:
    os.unlink(tmp)

dll.aila_engine_destroy(engine)
