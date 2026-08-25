#!/usr/bin/env python3
"""Compare Aila's DiffusionGemma soft embedding with the HF expression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import compare_layer0 as layer_reference


def load_bf16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16).copy()
    return torch.from_numpy(raw).view(torch.bfloat16).reshape(shape)


def metrics(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    delta = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1.0e-6)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    print(
        f"{name}: max_abs={delta.max().item():.7g} "
        f"mean_abs={delta.mean().item():.7g} "
        f"max_rel={(delta / denom).max().item():.7g} "
        f"cos={cosine.item():.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--previous-soft-dump", type=Path)
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text())
    layer_reference.MODEL_DIR = args.model_dir
    text_config = config["text_config"]
    vocab = int(text_config["vocab_size"])
    hidden = int(text_config["hidden_size"])
    eps = float(text_config["rms_norm_eps"])

    logits = torch.from_numpy(
        np.fromfile(args.dump_dir / "soft_logits_row0.bin", dtype=np.float32).copy()
    )
    if logits.numel() != vocab:
        raise ValueError(f"expected {vocab} logits, got {logits.numel()}")
    actual_probs = load_bf16(args.dump_dir / "soft_probs_row0.bin", (vocab,))
    expected_probs = logits.to(torch.bfloat16).softmax(dim=-1, dtype=torch.float32).to(
        torch.bfloat16
    )
    metrics("softmax", actual_probs, expected_probs)

    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )
    key = "model.encoder.language_model.embed_tokens.weight"
    shard = args.model_dir / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(key)
    expected_soft = (expected_probs @ embedding) * (hidden**0.5)
    actual_soft = load_bf16(args.dump_dir / "soft_embed_row0.bin", (hidden,))
    metrics("soft_embedding", actual_soft, expected_soft.to(torch.bfloat16))

    final_normed_path = args.dump_dir / "decoder_final_normed.bin"
    logits_bf16_path = args.dump_dir / "decoder_logits_bf16_row0.bin"
    if final_normed_path.exists() and logits_bf16_path.exists():
        final_normed = load_bf16(final_normed_path, (256, hidden))[0]
        actual_lm_logits = load_bf16(logits_bf16_path, (vocab,))
        expected_lm_logits = final_normed @ embedding.T
        metrics(
            "tied_lm_head",
            actual_lm_logits,
            expected_lm_logits.to(torch.bfloat16),
        )

    canvas = 256
    token_embeds = load_bf16(
        args.dump_dir / "decoder_token_embeds.bin", (canvas, hidden)
    )
    decoder_soft = load_bf16(
        args.dump_dir / "decoder_soft_embeds.bin", (canvas, hidden)
    )
    sc_down = load_bf16(args.dump_dir / "decoder_sc_down.bin", (canvas, hidden))
    hidden_after_sc = load_bf16(
        args.dump_dir / "decoder_hidden_after_sc.bin", (canvas, hidden)
    )
    if args.previous_soft_dump is None:
        print(
            f"first_step_zero_signal: soft_max={decoder_soft.float().abs().max().item():.7g} "
            f"sc_down_max={sc_down.float().abs().max().item():.7g}"
        )
        expected_hidden = token_embeds.float()
        expected_hidden *= torch.rsqrt(
            expected_hidden.square().mean(-1, keepdim=True) + eps
        )
        metrics(
            "first_step_post_sc_norm",
            hidden_after_sc,
            expected_hidden.to(torch.bfloat16),
        )
    else:
        previous_soft = load_bf16(
            args.previous_soft_dump / "soft_embed_row0.bin", (hidden,)
        )
        metrics("second_step_soft_input", decoder_soft[0], previous_soft)

        prefix = "model.decoder.self_conditioning"
        pre_norm_weight = torch.from_numpy(
            layer_reference.load_bf16_weight(f"{prefix}.pre_norm.weight")
        )
        signal = decoder_soft[0].float()
        expected_normed = signal * torch.rsqrt(signal.square().mean() + eps)
        expected_normed *= pre_norm_weight
        actual_normed = load_bf16(
            args.dump_dir / "decoder_sc_normed.bin", (canvas, hidden)
        )[0]
        metrics("second_step_sc_norm", actual_normed, expected_normed.to(torch.bfloat16))

        gate_weight = torch.from_numpy(
            layer_reference.load_weights(f"{prefix}.gate_proj.weight")
        )
        up_weight = torch.from_numpy(
            layer_reference.load_weights(f"{prefix}.up_proj.weight")
        )
        down_weight = torch.from_numpy(
            layer_reference.load_weights(f"{prefix}.down_proj.weight")
        )
        actual_gate = load_bf16(
            args.dump_dir / "decoder_sc_gate.bin", (canvas, gate_weight.shape[0])
        )[0]
        actual_up = load_bf16(
            args.dump_dir / "decoder_sc_up.bin", (canvas, up_weight.shape[0])
        )[0]
        metrics("second_step_sc_gate", actual_gate, expected_normed @ gate_weight.T)
        metrics("second_step_sc_up", actual_up, expected_normed @ up_weight.T)
        expected_proj = torch.nn.functional.gelu(
            actual_gate.float(), approximate="tanh"
        ) * actual_up.float()
        actual_proj = load_bf16(
            args.dump_dir / "decoder_sc_proj.bin", (canvas, gate_weight.shape[0])
        )[0]
        metrics("second_step_sc_proj", actual_proj, expected_proj.to(torch.bfloat16))
        actual_down = load_bf16(
            args.dump_dir / "decoder_sc_down.bin", (canvas, hidden)
        )[0]
        metrics("second_step_sc_down", actual_down, actual_proj.float() @ down_weight.T)
        combined = token_embeds[0].float() + actual_down.float()
        expected_hidden = combined * torch.rsqrt(combined.square().mean() + eps)
        metrics(
            "second_step_post_sc_norm",
            hidden_after_sc[0],
            expected_hidden.to(torch.bfloat16),
        )


if __name__ == "__main__":
    main()
