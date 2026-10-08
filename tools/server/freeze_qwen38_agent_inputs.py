"""Freeze repository text once for repeatable Qwen3.8 HTTP/PI measurements."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ("AGENTS.md", 0, 100),
    ("src/inference/Model.cpp", 130, 330),
    ("src/inference/Session.cpp", 0, 210),
    ("src/server/HttpServer.cpp", 420, 690),
    ("src/models/Qwen4ExpExpertCache.cpp", 2010, 2220),
    ("src/models/Qwen4ExpBnb4Backend.cpp", 1730, 1900),
    ("docs/plans/qwen38-bottleneck-2026-10-07.md", 0, 120),
    ("tmp/q4e-htoff-base-worker.log", 900, 950),
]
TASK = "\nExplain how this inference server handles a long request and reuses state on the next turn. Identify the important validation and ownership constraints using the supplied code."
MARKER = "AILA_FIXED_AGENT_READ_20261007"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def token_sha(ids):
    return sha(b"".join(i.to_bytes(4, "little", signed=False) for i in ids))

def load_fixture(folder, tokenizer):
    folder = Path(folder).resolve(strict=True)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["inputs"]:
        data = (folder / item["file"]).read_bytes()
        ids = tokenizer.encode(data.decode("utf-8")).ids
        if sha(data) != item["sha256"] or token_sha(ids) != item["token_sha256"] or len(ids) != item["tokens"]:
            raise ValueError("Frozen input or tokenizer changed: " + item["file"])
    marker = folder / "pi-work/smoke-marker.txt"
    if marker.read_text(encoding="utf-8") != manifest["marker"] + "\n":
        raise ValueError("Frozen PI marker changed")
    return manifest

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    if args.verify:
        print(json.dumps(load_fixture(args.output, tokenizer), ensure_ascii=False, indent=2))
        return
    if args.output.exists():
        raise ValueError("Choose a new output directory; frozen inputs are never overwritten")
    source_records, sections = [], []
    for name, begin, end in SOURCES:
        data = (ROOT / name).read_bytes()
        excerpt = "\n".join(data.decode("utf-8-sig").splitlines()[begin:end])
        sections.append(f"\n--- {name}, lines {begin+1}-{end} ---\n{excerpt}\n")
        source_records.append(dict(path=name, first_line=begin+1, last_line=end,
                                   file_sha256=sha(data), excerpt_sha256=sha(excerpt.encode())))
    corpus = "Repository source and diagnostic evidence for review.\n" + "".join(sections)
    ids = tokenizer.encode(corpus).ids
    if len(ids) < 14000:
        raise ValueError(f"Not enough distinct repository text: {len(ids)} tokens")
    args.output.mkdir(parents=True)
    manifest = dict(version=1, sources=source_records, corpus_tokens=len(ids), inputs=[], marker=MARKER,
                    tokenizer_sha256=sha((args.model / "tokenizer.json").read_bytes()))
    for target in (6000, 14000):
        content = tokenizer.decode(ids[:target-100], skip_special_tokens=False) + TASK
        data = content.encode("utf-8")
        actual = tokenizer.encode(content).ids
        name = f"context-{target}.txt"
        (args.output / name).write_bytes(data)
        manifest["inputs"].append(dict(target=target, file=name, tokens=len(actual),
            sha256=sha(data), token_sha256=token_sha(actual)))
    (args.output / "pi-work").mkdir()
    (args.output / "pi-work/smoke-marker.txt").write_bytes((MARKER+"\n").encode())
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    load_fixture(args.output, tokenizer)
    print(json.dumps(manifest["inputs"], indent=2))

if __name__ == "__main__":
    main()
