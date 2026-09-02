from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements


def bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.asarray(values, dtype=np.float32)
    raw = f32.view(np.uint32)
    rounded = raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_float(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def write_tensor(
    path: Path, dtype: str, shape: tuple[int, ...], payload: bytes
) -> None:
    codec.write(
        path,
        dtype=dtype,
        shape=shape,
        stride=codec.contiguous_stride(shape),
        storage_offset=0,
        payload=payload,
    )


def boundary(name: str, payload: str, dtype: str, shape: tuple[int, ...]) -> dict:
    return {
        "name": name,
        "payload": payload,
        "dtype": dtype,
        "shape": list(shape),
        "stride": list(codec.contiguous_stride(shape)),
        "storage_offset": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    m, n, k, block = 32, 128, 128, 64
    root = args.output.resolve()
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "reference").mkdir(parents=True, exist_ok=True)

    qmap = (np.arange(16, dtype=np.float32) - 7.5) * np.float32(0.125)
    absmax = np.empty((n, k // block), dtype=np.float32)
    for row in range(n):
        for group in range(k // block):
            absmax[row, group] = 0.75 + 0.05 * ((row + group) % 11)

    packed = np.empty((n, k // 2), dtype=np.uint8)
    for row in range(n):
        for half in range(k // 2):
            hi = (row * 3 + half * 5) & 0x0F
            lo = (row * 7 + half * 2 + 1) & 0x0F
            packed[row, half] = (hi << 4) | lo

    input_f32 = np.empty((m, k), dtype=np.float32)
    for row in range(m):
        for column in range(k):
            input_f32[row, column] = (row + 1) * ((column % 13) - 6) * 0.03125
    input_bits = bf16_bits(input_f32)
    input_bf16 = bf16_float(input_bits)

    weights = np.empty((n, k), dtype=np.float32)
    for row in range(n):
        for column in range(k):
            byte = packed[row, column // 2]
            code = (byte >> 4) if column % 2 == 0 else (byte & 0x0F)
            weights[row, column] = bf16_float(
                bf16_bits(np.asarray(qmap[code] * absmax[row, column // block]))
            )
    output_bits = bf16_bits(input_bf16 @ weights.T)

    write_tensor(
        root / "inputs" / "packed.irtensor",
        "uint8",
        packed.shape,
        packed.tobytes(),
    )
    write_tensor(
        root / "inputs" / "qmap.irtensor",
        "float32",
        qmap.shape,
        qmap.tobytes(),
    )
    write_tensor(
        root / "inputs" / "absmax.irtensor", "float32", absmax.shape, absmax.tobytes()
    )
    write_tensor(
        root / "inputs" / "input.irtensor",
        "bfloat16",
        input_bits.shape,
        input_bits.tobytes(),
    )
    write_tensor(
        root / "reference" / "output.irtensor",
        "bfloat16",
        output_bits.shape,
        output_bits.tobytes(),
    )

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": "diffusion-gemma-nf4-gemm-bk64",
        "origin": {"kind": "synthetic", "generator": Path(__file__).name},
        "reproducible": True,
        "inputs": [
            boundary("packed", "inputs/packed.irtensor", "uint8", packed.shape),
            boundary("qmap", "inputs/qmap.irtensor", "float32", qmap.shape),
            boundary("absmax", "inputs/absmax.irtensor", "float32", absmax.shape),
            boundary("input", "inputs/input.irtensor", "bfloat16", input_bits.shape),
        ],
        "outputs": [
            boundary(
                "output", "reference/output.irtensor", "bfloat16", output_bits.shape
            )
        ],
        "nodes": [],
        "values": [],
    }
    manifest["requirements"] = derive_requirements(manifest)
    (root / "testcase.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
