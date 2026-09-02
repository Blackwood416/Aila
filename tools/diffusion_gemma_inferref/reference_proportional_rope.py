"""InferRef reference for DiffusionGemma/Gemma 4 proportional RoPE.

The input is an Aila raw-bf16 q_norm dump plus the layer dump metadata. The
module deliberately mirrors Transformers' full-head cos/sin construction:
only the leading quarter has non-zero frequencies, while rotate_half pairs it
with the second half of the complete 512-wide head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def bf16_raw_to_torch(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16).reshape(shape)
    bits = torch.from_numpy(raw.astype(np.int32)) << 16
    return bits.view(torch.float32).to(torch.bfloat16)


class ProportionalRope(torch.nn.Module):
    def __init__(self, theta: float = 1_000_000.0, rotary_fraction: float = 0.25):
        super().__init__()
        self.theta = theta
        self.rotary_fraction = rotary_fraction

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        head_dim = x.shape[-1]
        active_pairs = int(head_dim * self.rotary_fraction) // 2
        frequencies = 1.0 / (
            self.theta
            ** (torch.arange(active_pairs, dtype=torch.float32) * 2.0 / head_dim)
        )
        angles = positions.float()[:, None] * frequencies[None, :]
        cos = angles.cos().to(x.dtype)[:, None, :]
        sin = angles.sin().to(x.dtype)[:, None, :]
        half = head_dim // 2
        out = x.clone()
        x0 = x[..., :active_pairs]
        x1 = x[..., half : half + active_pairs]
        out[..., :active_pairs] = x0 * cos - x1 * sin
        out[..., half : half + active_pairs] = x1 * cos + x0 * sin
        return out.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    args = parser.parse_args()
    metadata: dict[str, int] = {}
    for part in (args.dump_dir / "layer0.txt").read_text().split():
        if "=" in part:
            key, value = part.split("=", 1)
            metadata[key] = int(value)
    q = bf16_raw_to_torch(
        args.dump_dir / "q_norm.bin",
        (metadata["seq"], metadata["heads"], metadata["head_dim"]),
    )
    positions = torch.arange(metadata["start"], metadata["start"] + metadata["seq"])
    rope = ProportionalRope()
    output = rope(q, positions)
    print(float(output.abs().max()))


if __name__ == "__main__":
    main()
