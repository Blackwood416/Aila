#!/usr/bin/env python3
"""Validate DiffusionGemma checkpoint state needed by Aila's NF4 backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    index_path = args.model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    layers = int(config["text_config"]["num_hidden_layers"])

    missing: list[str] = []
    scalar_values: dict[str, list[float]] = {
        "model.encoder.language_model.layers": [],
        "model.decoder.layers": [],
    }
    for prefix, values in scalar_values.items():
        for layer in range(layers):
            name = f"{prefix}.{layer}.layer_scalar"
            shard_name = weight_map.get(name)
            if shard_name is None:
                missing.append(name)
                continue
            with safe_open(
                str(args.model_dir / shard_name), framework="pt", device="cpu"
            ) as handle:
                scalar = handle.get_tensor(name)
            if scalar.dtype != torch.bfloat16 or scalar.numel() != 1:
                raise ValueError(
                    f"{name}: expected one bf16 value, got {scalar.dtype} "
                    f"{tuple(scalar.shape)}"
                )
            values.append(float(scalar.item()))

    quantized = [
        name
        for name in weight_map
        if name.endswith(".quant_state.bitsandbytes__nf4")
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} encoder layer scalars: {missing[:3]}")
    if not quantized:
        raise ValueError("checkpoint has no bitsandbytes NF4 quant_state tensors")
    encoder_scalars = scalar_values["model.encoder.language_model.layers"]
    decoder_scalars = scalar_values["model.decoder.layers"]
    if encoder_scalars != decoder_scalars:
        raise ValueError("encoder and decoder layer_scalar values differ")
    if all(value == 1.0 for value in encoder_scalars):
        raise ValueError("all layer_scalar values are 1.0; persistent buffers look incomplete")

    print(
        f"checkpoint_ok layers={layers} layer_scalars="
        f"{len(encoder_scalars) + len(decoder_scalars)} "
        f"nf4_quant_states={len(quantized)} scalar_min={min(encoder_scalars):.9g} "
        f"scalar_max={max(encoder_scalars):.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
