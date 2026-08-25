"""Compare Aila's dumped layer-0 intermediates against a NumPy reference.

Usage:
  python tools/diffusion_gemma_inferref/compare_layer0.py <dump_dir>

The dump dir contains layer0.txt plus raw bf16 dumps written by
DiffusionGemmaBnb4Backend (AILA_DG_DUMP_DIR).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


MODEL_DIR = Path(r"E:/RiderProjects/Aila/models/DiffusionGemma-26B-A4B-it-BNB-NF4")
LAYER0 = "model.encoder.language_model.layers.0"


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Interpret uint16 raw bf16 bits as float32."""
    out = np.zeros(raw.shape, dtype=np.float32)
    finite = (raw & 0x7F80) != 0x7F80
    sign = np.where((raw & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((raw >> 7) & 0xFF).astype(np.int32)
    mant = (raw & 0x7F).astype(np.float32)
    value = np.where(exp == 0, mant / 128.0 * (2.0 ** -126.0),
                     (1.0 + mant / 128.0) * (2.0 ** (exp - 127)))
    out = sign * value
    out[~finite] = np.nan
    return out


def load_dump(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return bf16_to_f32(raw.reshape(shape))


def load_weights(name: str) -> np.ndarray:
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][name]
    def to_np(t):
        return t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
    with safe_open(str(MODEL_DIR / shard), framework="pt", device="cpu") as f:
        packed = to_np(f.get_tensor(name)).reshape(-1)
        absmax_codes = to_np(f.get_tensor(name + ".absmax")).astype(np.int64)
        nested = to_np(f.get_tensor(name + ".nested_absmax"))
        nqmap = to_np(f.get_tensor(name + ".nested_quant_map"))
        qmap = to_np(f.get_tensor(name + ".quant_map"))
        meta = json.loads(
            bytes(f.get_tensor(name + ".quant_state.bitsandbytes__nf4").tolist()).decode()
        )
    shape = tuple(meta["shape"])
    blocksize = int(meta["blocksize"])
    offset = float(meta.get("nested_offset", 0.0))
    nested_blocksize = int(meta.get("nested_blocksize", 256))
    absmax = nqmap[absmax_codes] * nested[absmax_codes // nested_blocksize] + offset
    rows = int(np.prod(shape[:-1]))
    in_dim = shape[-1]
    packed_rows = packed.reshape(rows, in_dim // 2)
    hi = (packed_rows >> 4) & 0xF
    lo = packed_rows & 0xF
    inter = np.empty((rows, in_dim), dtype=np.int64)
    inter[:, 0::2] = hi
    inter[:, 1::2] = lo
    blocks_per_row = in_dim // blocksize
    am = absmax.reshape(rows, blocks_per_row).repeat(blocksize, axis=1)
    weight = qmap[inter] * am
    return weight.reshape(shape)


def load_bf16_weight(name: str) -> np.ndarray:
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][name]
    with safe_open(str(MODEL_DIR / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().numpy()


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def head_rms_norm(x: np.ndarray, weight: np.ndarray, heads: int, head_dim: int, eps: float) -> np.ndarray:
    x = x.reshape(x.shape[0], heads, head_dim)
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms * weight).reshape(x.shape[0], -1)


def rope(x: np.ndarray, positions: np.ndarray, theta: float, head_dim: int) -> np.ndarray:
    half = head_dim // 2
    seq = x.shape[0]
    x = x.reshape(seq, -1, head_dim)
    out = x.copy()
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2 / head_dim))
    angles = positions[:, None].astype(np.float32) * freqs[None, :]
    cos = np.cos(angles)
    sin = np.sin(angles)
    q0 = x[..., :half]
    q1 = x[..., half:2 * half]
    out[..., :half] = q0 * cos[:, None, :] - q1 * sin[:, None, :]
    out[..., half:2 * half] = q1 * cos[:, None, :] + q0 * sin[:, None, :]
    return out.reshape(x.shape[0], -1)


def compare(name: str, actual: np.ndarray, expected: np.ndarray, tol: float = 1e-2):
    if actual.shape != expected.shape:
        print(f"{name}: SHAPE MISMATCH {actual.shape} vs {expected.shape}")
        return False
    diff = np.abs(actual - expected)
    rel = diff / (np.abs(expected) + 1e-6)
    print(f"{name}: max_abs={diff.max():.5f} max_rel={rel.max():.5f}")
    return diff.max() < tol


def main() -> int:
    dump = Path(sys.argv[1])
    meta = {}
    for line in (dump / "layer0.txt").read_text().splitlines():
        for part in line.split():
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k] = int(v)
    seq = meta["seq"]
    hidden = meta["hidden"]
    head_dim = meta["head_dim"]
    heads = meta["heads"]
    kv_heads = meta["kv_heads"]
    sliding = meta["sliding"] == 1

    hidden_in = load_dump(dump / "hidden_in.bin", (seq, hidden))
    normed = load_dump(dump / "normed.bin", (seq, hidden))

    in_ln = load_bf16_weight(f"{LAYER0}.input_layernorm.weight")
    normed_ref = rms_norm(hidden_in, in_ln, 1e-6)
    compare("normed", normed, normed_ref)

    q_w = load_weights(f"{LAYER0}.self_attn.q_proj.weight")
    k_w = load_weights(f"{LAYER0}.self_attn.k_proj.weight")
    v_w = load_weights(f"{LAYER0}.self_attn.v_proj.weight")
    q = normed @ q_w.T
    k = normed @ k_w.T
    v = normed @ v_w.T
    q_dump = load_dump(dump / "q_proj.bin", (seq, q.shape[1]))
    k_dump = load_dump(dump / "k_proj.bin", (seq, k.shape[1]))
    compare("q_proj", q_dump, q)
    compare("k_proj", k_dump, k)

    q_n = load_bf16_weight(f"{LAYER0}.self_attn.q_norm.weight")
    k_n = load_bf16_weight(f"{LAYER0}.self_attn.k_norm.weight")
    q = head_rms_norm(q, q_n, heads, head_dim, 1e-6)
    k = head_rms_norm(k, k_n, kv_heads, head_dim, 1e-6)
    v = head_rms_norm(v, np.ones(head_dim, dtype=np.float32), kv_heads, head_dim, 1e-6)

    positions = np.arange(seq, dtype=np.int32)
    theta = 10000.0
    rotary_dim = head_dim
    q = rope(q, positions, theta, head_dim)
    k = rope(k, positions, theta, head_dim)
    compare("q_rope", load_dump(dump / "q_rope.bin", (seq, q.shape[1])), q)
    compare("k_rope", load_dump(dump / "k_rope.bin", (seq, k.shape[1])), k)
    compare("v_norm", load_dump(dump / "v_norm.bin", (seq, v.shape[1])), v)

    # Eager attention, fp32.
    qh = q.reshape(seq, heads, head_dim).transpose(1, 0, 2)
    kh = k.reshape(seq, kv_heads, head_dim).transpose(1, 0, 2)
    vh = v.reshape(seq, kv_heads, head_dim).transpose(1, 0, 2)
    group = heads // kv_heads
    scores = np.zeros((heads, seq, seq), dtype=np.float32)
    for h in range(heads):
        scores[h] = qh[h] @ kh[h // group].T
    mask = np.triu(np.ones((seq, seq), dtype=np.float32) * -1e30, k=1)
    scores = scores + mask
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    attn = np.zeros((heads, seq, head_dim), dtype=np.float32)
    for h in range(heads):
        attn[h] = probs[h] @ vh[h // group]
    attn = attn.transpose(1, 0, 2).reshape(seq, -1)
    compare("attn_out", load_dump(dump / "attn_out.bin", (seq, heads * head_dim)),
            attn)

    o_w = load_weights(f"{LAYER0}.self_attn.o_proj.weight")
    o = attn @ o_w.T
    compare("attn_o", load_dump(dump / "attn_o.bin", (seq, hidden)), o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
