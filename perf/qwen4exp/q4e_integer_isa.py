"""Disassemble captured VTune zebins without Windows MAX_PATH symbol failures.
Only an analysis copy's ELF section names are shortened; executable section
bytes are asserted unchanged. These copies are never executed or reassembled.
"""
import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sections(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64 zebin")
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, strings = struct.unpack_from("<HHH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride)
               for i in range(count)]
    base = headers[strings][4]
    for i, header in enumerate(headers):
        start = base + header[0]
        end = data.index(0, start)
        yield i, data[start:end].decode(), start, end, header[4], header[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol-contains", default="integer_projection")
    parser.add_argument("--ocloc", type=Path, default=Path(
        "C:/Program Files (x86)/Intel/oneAPI/ocloc/2026.1/bin/ocloc.exe"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    inputs = sorted((args.result_dir / "data.0").glob("*.zebin"))
    if not inputs:
        raise RuntimeError("No captured zebins")
    for source in inputs:
        original = source.read_bytes()
        altered = bytearray(original)
        names = {}
        text_sections = []
        for i, name, start, end, offset, size in sections(original):
            if name.startswith(".text."):
                short = f".text.k{i}"
                names[short] = name
                altered[start:end] = short.encode() + bytes(end - start - len(short))
                text_sections.append((short, name, offset, size))
        for _, _, offset, size in text_sections:
            assert original[offset:offset + size] == altered[offset:offset + size]
        copy = args.output / source.name
        copy.write_bytes(altered)
        destination = args.output / source.stem
        destination.mkdir(exist_ok=True)
        (destination / "names.json").write_text(json.dumps(names, indent=2))
        result = subprocess.run([str(args.ocloc), "disasm", "-file", str(copy.resolve()),
                                 "-dump", str(destination.resolve()), "-device", "dg2"],
                                capture_output=True, text=True, check=True)
        (destination / "disasm.log").write_text(result.stdout + result.stderr)
        for short, name, offset, size in text_sections:
            if args.symbol_contains not in name:
                continue
            assembly = destination / (short + ".asm")
            text = assembly.read_text()
            records.append(dict(symbol=name, source=str(source), source_sha256=digest(original),
                                machine_code_sha256=digest(original[offset:offset + size]),
                                machine_code_unchanged=True, assembly=str(assembly),
                                assembly_sha256=digest(assembly.read_bytes()),
                                native_dp4a_static_sites=len(re.findall(r"\bdp4a\s+\(", text)),
                                native_dpas_static_sites=len(re.findall(
                                    r"\bdpas(?:w)?(?:\.[A-Za-z0-9]+)*\s+\(", text))))
    if not records:
        raise RuntimeError("No matching kernels found: " + args.symbol_contains)
    (args.output / "summary.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
