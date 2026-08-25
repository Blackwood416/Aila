#!/usr/bin/env python3
"""Compare Aila's NF4 reference formula with bitsandbytes dequantize_4bit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState, dequantize_4bit
from safetensors import safe_open

import compare_layer0 as aila_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "weight",
        nargs="?",
        default="model.encoder.language_model.layers.0.self_attn.q_proj.weight",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "xpu", "cuda"),
        help="Trusted bitsandbytes backend (auto prefers XPU/CUDA, then CPU)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.xpu.is_available():
            device = torch.device("xpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )
    shard = args.model_dir / index["weight_map"][args.weight]
    state: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(args.weight)
        for key in handle.keys():
            if key.startswith(args.weight + "."):
                state[key] = handle.get_tensor(key)

    quant_state = QuantState.from_dict(state, device)
    dequantized = dequantize_4bit(packed.to(device), quant_state)
    actual = dequantized.float().cpu().numpy()

    aila_reference.MODEL_DIR = args.model_dir
    expected = aila_reference.load_weights(args.weight)
    delta = np.abs(actual - expected)
    relative = delta / np.maximum(np.abs(actual), 1.0e-6)
    cosine = np.dot(actual.ravel(), expected.ravel()) / (
        np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel())
    )
    print(f"shape={actual.shape}")
    print(
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} "
        f"max_rel={relative.max():.9g} cosine={cosine:.12g}"
    )

    if args.input is not None or args.output is not None:
        if args.input is None or args.output is None:
            raise ValueError("--input and --output must be provided together")
        in_features = actual.shape[-1]
        out_features = actual.shape[-2]
        input_raw = np.fromfile(args.input, dtype=np.uint16).copy()
        output_raw = np.fromfile(args.output, dtype=np.uint16).copy()
        input_tensor = torch.from_numpy(input_raw).view(torch.bfloat16).reshape(
            -1, in_features
        )
        aila_output = torch.from_numpy(output_raw).view(torch.bfloat16).reshape(
            -1, out_features
        )
        bnb_output = torch.nn.functional.linear(
            input_tensor.to(device), dequantized
        ).cpu()
        out_delta = (aila_output.float() - bnb_output.float()).abs()
        out_cosine = torch.nn.functional.cosine_similarity(
            aila_output.float().flatten(), bnb_output.float().flatten(), dim=0
        )
        print(
            f"linear_max_abs={out_delta.max().item():.9g} "
            f"linear_mean_abs={out_delta.mean().item():.9g} "
            f"linear_cosine={out_cosine.item():.12g}"
        )
        fused_output = bnb.matmul_4bit(
            input_tensor.to(device), packed.to(device).T, quant_state=quant_state
        ).cpu()
        fused_delta = (aila_output.float() - fused_output.float()).abs()
        fused_cosine = torch.nn.functional.cosine_similarity(
            aila_output.float().flatten(), fused_output.float().flatten(), dim=0
        )
        print(
            f"bnb_matmul_max_abs={fused_delta.max().item():.9g} "
            f"bnb_matmul_mean_abs={fused_delta.mean().item():.9g} "
            f"bnb_matmul_cosine={fused_cosine.item():.12g}"
        )


if __name__ == "__main__":
    main()
