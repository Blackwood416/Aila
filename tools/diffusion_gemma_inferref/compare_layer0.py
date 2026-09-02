"""Compare Aila's dumped layer-0 intermediates against a NumPy reference.

Usage:
  python tools/diffusion_gemma_inferref/compare_layer0.py <dump_dir> <model_dir>

The dump dir contains layer0.txt plus raw bf16 dumps written by
DiffusionGemmaBnb4Backend (AILA_DG_DUMP_DIR).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


MODEL_DIR: Path
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
    nested_blocks = np.arange(absmax_codes.size, dtype=np.int64) // nested_blocksize
    absmax = nqmap[absmax_codes] * nested[nested_blocks] + offset
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


def bf16_round(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


class QuantizedWeight:
    """Checkpoint-backed NF4 weight with cheap per-expert dequantization."""

    def __init__(self, name: str):
        index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
        shard = index["weight_map"][name]
        with safe_open(str(MODEL_DIR / shard), framework="pt", device="cpu") as f:
            self.packed = f.get_tensor(name).numpy().reshape(-1)
            self.absmax_codes = f.get_tensor(name + ".absmax").numpy().astype(np.int64)
            self.nested = f.get_tensor(name + ".nested_absmax").float().numpy()
            self.nqmap = f.get_tensor(name + ".nested_quant_map").float().numpy()
            self.qmap = f.get_tensor(name + ".quant_map").float().numpy()
            meta = json.loads(
                bytes(f.get_tensor(name + ".quant_state.bitsandbytes__nf4").tolist()).decode()
            )
        self.shape = tuple(meta["shape"])
        self.blocksize = int(meta["blocksize"])
        self.offset = float(meta.get("nested_offset", 0.0))
        self.nested_blocksize = int(meta.get("nested_blocksize", 256))

    def expert(self, expert: int) -> np.ndarray:
        if len(self.shape) != 3:
            raise ValueError(f"expected 3D expert weight, got {self.shape}")
        experts, rows, in_dim = self.shape
        if not 0 <= expert < experts:
            raise ValueError(f"expert {expert} outside [0, {experts})")
        packed_per_expert = rows * in_dim // 2
        packed_begin = expert * packed_per_expert
        packed = self.packed[packed_begin:packed_begin + packed_per_expert].reshape(rows, in_dim // 2)
        hi = (packed >> 4) & 0xF
        lo = packed & 0xF
        codes = np.empty((rows, in_dim), dtype=np.uint8)
        codes[:, 0::2] = hi
        codes[:, 1::2] = lo

        blocks_per_expert = rows * in_dim // self.blocksize
        block_begin = expert * blocks_per_expert
        block_indices = np.arange(block_begin, block_begin + blocks_per_expert, dtype=np.int64)
        abs_codes = self.absmax_codes[block_indices]
        absmax = (self.nqmap[abs_codes] *
                  self.nested[block_indices // self.nested_blocksize] + self.offset)
        scales = absmax.reshape(rows, in_dim // self.blocksize).repeat(self.blocksize, axis=1)
        return self.qmap[codes] * scales


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


def proportional_rope(x: np.ndarray, positions: np.ndarray, theta: float,
                      head_dim: int, rotary_fraction: float = 0.25) -> np.ndarray:
    """Gemma 4 proportional RoPE: rotate a prefix across full head halves."""
    seq = x.shape[0]
    x = x.reshape(seq, -1, head_dim)
    out = x.copy()
    active_pairs = int(head_dim * rotary_fraction) // 2
    half = head_dim // 2
    freqs = 1.0 / (theta ** (
        np.arange(active_pairs, dtype=np.float32) * 2 / head_dim
    ))
    angles = positions[:, None].astype(np.float32) * freqs[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    x0 = x[..., :active_pairs]
    x1 = x[..., half:half + active_pairs]
    out[..., :active_pairs] = x0 * cos - x1 * sin
    out[..., half:half + active_pairs] = x1 * cos + x0 * sin
    return out.reshape(seq, -1)


def compare(name: str, actual: np.ndarray, expected: np.ndarray, tol: float = 1e-2):
    if actual.shape != expected.shape:
        print(f"{name}: SHAPE MISMATCH {actual.shape} vs {expected.shape}")
        return False
    diff = np.abs(actual - expected)
    rel = diff / (np.abs(expected) + 1e-6)
    print(f"{name}: max_abs={diff.max():.5f} max_rel={rel.max():.5f}")
    return diff.max() < tol


def main() -> int:
    global LAYER0, MODEL_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    dump = args.dump_dir
    MODEL_DIR = args.model_dir
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
    layer_idx = meta.get("layer", 0)
    start_pos = meta.get("start", 0)
    bidirectional = meta.get("bidirectional", 0) == 1
    LAYER0 = f"model.encoder.language_model.layers.{layer_idx}"

    hidden_in = load_dump(dump / "hidden_in.bin", (seq, hidden))
    normed = load_dump(dump / "normed.bin", (seq, hidden))

    in_ln = load_bf16_weight(f"{LAYER0}.input_layernorm.weight")
    normed_ref = rms_norm(hidden_in, in_ln, 1e-6)
    compare("normed", normed, normed_ref)

    q_w = load_weights(f"{LAYER0}.self_attn.q_proj.weight")
    k_w = load_weights(f"{LAYER0}.self_attn.k_proj.weight")
    q = normed @ q_w.T
    k = normed @ k_w.T
    if sliding:
        v_w = load_weights(f"{LAYER0}.self_attn.v_proj.weight")
        v = normed @ v_w.T
    else:
        v = k.copy()
    q_dump = load_dump(dump / "q_proj.bin", (seq, q.shape[1]))
    k_dump = load_dump(dump / "k_proj.bin", (seq, k.shape[1]))
    compare("q_proj", q_dump, q)
    compare("k_proj", k_dump, k)

    q_n = load_bf16_weight(f"{LAYER0}.self_attn.q_norm.weight")
    k_n = load_bf16_weight(f"{LAYER0}.self_attn.k_norm.weight")
    q = head_rms_norm(q, q_n, heads, head_dim, 1e-6)
    k = head_rms_norm(k, k_n, kv_heads, head_dim, 1e-6)
    v = head_rms_norm(v, np.ones(head_dim, dtype=np.float32), kv_heads, head_dim, 1e-6)

    positions = start_pos + np.arange(seq, dtype=np.int32)
    if sliding:
        q = rope(q, positions, 10000.0, head_dim)
        k = rope(k, positions, 10000.0, head_dim)
    else:
        q = proportional_rope(q, positions, 1_000_000.0, head_dim)
        k = proportional_rope(k, positions, 1_000_000.0, head_dim)
    compare("q_rope", load_dump(dump / "q_rope.bin", (seq, q.shape[1])), q)
    compare("k_rope", load_dump(dump / "k_rope.bin", (seq, k.shape[1])), k)
    compare("v_norm", load_dump(dump / "v_norm.bin", (seq, v.shape[1])), v)

    # Eager attention, fp32.  Decoder dumps include the encoder cache prefix;
    # consume those actual K/V tensors to isolate the attention operator from
    # any earlier cache-writing difference.
    q_for_attn = load_dump(dump / "q_rope.bin", (seq, q.shape[1]))
    kv_len = start_pos + seq
    if (dump / "k_full.bin").exists():
        k_for_attn = load_dump(dump / "k_full.bin", (kv_len, kv_heads * head_dim))
        v_for_attn = load_dump(dump / "v_full.bin", (kv_len, kv_heads * head_dim))
    else:
        kv_len = seq
        k_for_attn = k
        v_for_attn = v
    qh = q_for_attn.reshape(seq, heads, head_dim).transpose(1, 0, 2)
    kh = k_for_attn.reshape(kv_len, kv_heads, head_dim).transpose(1, 0, 2)
    vh = v_for_attn.reshape(kv_len, kv_heads, head_dim).transpose(1, 0, 2)
    group = heads // kv_heads
    scores = np.zeros((heads, seq, kv_len), dtype=np.float32)
    for h in range(heads):
        scores[h] = qh[h] @ kh[h // group].T
    scores = bf16_round(scores)
    if (dump / "attention_mask.bin").exists():
        mask_raw = np.fromfile(dump / "attention_mask.bin", dtype=np.float32)
        if mask_raw.size == seq * kv_len:
            mask = mask_raw.reshape(seq, kv_len)
        elif bidirectional and kv_len <= 1024:
            # Older debug dumps wrote only numel*2 bytes even for fp32 masks.
            # For this short bidirectional decoder case every entry is allowed.
            mask = np.zeros((seq, kv_len), dtype=np.float32)
        else:
            raise ValueError(
                f"attention mask has {mask_raw.size} values; expected {seq * kv_len}"
            )
    else:
        mask = np.triu(np.ones((seq, kv_len), dtype=np.float32) * -1e30, k=1)
    scores = scores + mask
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    probs = bf16_round(probs)
    attn = np.zeros((heads, seq, head_dim), dtype=np.float32)
    for h in range(heads):
        attn[h] = probs[h] @ vh[h // group]
    attn = attn.transpose(1, 0, 2).reshape(seq, -1)
    compare("attn_out", load_dump(dump / "attn_out.bin", (seq, heads * head_dim)),
            attn, tol=0.2)

    o_w = load_weights(f"{LAYER0}.self_attn.o_proj.weight")
    o = attn @ o_w.T
    compare("attn_o", load_dump(dump / "attn_o.bin", (seq, hidden)), o)

    # Continue through the dense MLP and router, always feeding the dumped
    # previous stage so each comparison identifies one operator rather than
    # accumulating bf16 error from the beginning of the layer.
    attn_o = load_dump(dump / "attn_o.bin", (seq, hidden))
    post_attn_w = load_bf16_weight(f"{LAYER0}.post_attention_layernorm.weight")
    post_attn_ref = rms_norm(attn_o, post_attn_w, 1e-6)
    post_attn = load_dump(dump / "post_attn_norm.bin", (seq, hidden))
    compare("post_attn_norm", post_attn, post_attn_ref)

    hidden_after_attn = load_dump(dump / "hidden_after_attn.bin", (seq, hidden))
    compare("hidden_after_attn", hidden_after_attn,
            bf16_round(hidden_in + post_attn), tol=0.5)

    pre_ff_w = load_bf16_weight(f"{LAYER0}.pre_feedforward_layernorm.weight")
    pre_ff_ref = rms_norm(hidden_after_attn, pre_ff_w, 1e-6)
    pre_ff = load_dump(dump / "pre_ff_norm.bin", (seq, hidden))
    compare("pre_ff_norm", pre_ff, pre_ff_ref)

    intermediate = 2112
    gate_w = load_weights(f"{LAYER0}.mlp.gate_proj.weight")
    up_w = load_weights(f"{LAYER0}.mlp.up_proj.weight")
    gate = load_dump(dump / "mlp_gate.bin", (seq, intermediate))
    up = load_dump(dump / "mlp_up.bin", (seq, intermediate))
    compare("mlp_gate", gate, pre_ff @ gate_w.T)
    compare("mlp_up", up, pre_ff @ up_w.T)

    proj = load_dump(dump / "mlp_proj.bin", (seq, intermediate))
    compare("mlp_proj", proj, gelu_tanh(gate) * up)
    down_w = load_weights(f"{LAYER0}.mlp.down_proj.weight")
    mlp_out = load_dump(dump / "mlp_out.bin", (seq, hidden))
    compare("mlp_out", mlp_out, proj @ down_w.T)

    ff1_w = load_bf16_weight(f"{LAYER0}.post_feedforward_layernorm_1.weight")
    ff1 = load_dump(dump / "ff1.bin", (seq, hidden))
    compare("ff1", ff1, rms_norm(mlp_out, ff1_w, 1e-6))

    # Router reference, including top-k renormalization and learned per-expert
    # scaling. IDs must match exactly; weights are fp32 on both sides.
    router_scale = load_bf16_weight(f"{LAYER0}.router.scale")
    router_proj = load_bf16_weight(f"{LAYER0}.router.proj.weight")
    per_expert_scale = load_bf16_weight(f"{LAYER0}.router.per_expert_scale")
    router_x = bf16_round(
        rms_norm(hidden_after_attn, np.ones(hidden, dtype=np.float32), 1e-6)
    )
    router_x = bf16_round(router_x * router_scale)
    router_x = bf16_round(router_x / np.sqrt(float(hidden)))
    scores = bf16_round(router_x @ router_proj.T)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    top_values, top_ids = torch.topk(torch.from_numpy(probs), 8, dim=-1)
    top_ids = top_ids.numpy().astype(np.int32)
    top_values = top_values.numpy()
    top_weights = top_values / top_values.sum(axis=-1, keepdims=True)
    top_weights *= per_expert_scale[top_ids]
    actual_ids = np.fromfile(dump / "router_ids.bin", dtype=np.int32).reshape(seq, 8)
    actual_weights = np.fromfile(dump / "router_weights.bin", dtype=np.float32).reshape(seq, 8)
    print(f"router_ids: mismatches={np.count_nonzero(actual_ids != top_ids)}")
    compare("router_weights", actual_weights, top_weights, tol=1e-4)

    # Also execute the trusted module boundaries with torch itself.  This is
    # useful for catching BLAS accumulation and scalar-casting details that a
    # NumPy reconstruction can accidentally hide.
    torch_hidden = torch.from_numpy(hidden_after_attn).to(torch.bfloat16)
    torch_scale = torch.from_numpy(router_scale).to(torch.bfloat16)
    torch_proj = torch.from_numpy(router_proj).to(torch.bfloat16)
    torch_per_expert = torch.from_numpy(per_expert_scale)
    mean_squared = torch_hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6
    torch_routed = (torch_hidden.float() * torch.pow(mean_squared, -0.5)).to(
        torch.bfloat16
    )
    torch_routed = torch_routed * torch_scale * (hidden ** -0.5)
    torch_scores = torch.nn.functional.linear(torch_routed, torch_proj)
    torch_probs = torch.softmax(torch_scores, dim=-1, dtype=torch.float32)
    torch_values, torch_ids = torch.topk(torch_probs, 8, dim=-1)
    torch_weights = torch_values / torch_values.sum(dim=-1, keepdim=True)
    torch_weights = torch_weights * torch_per_expert[torch_ids]
    print(
        "router_ids_torch: "
        f"mismatches={np.count_nonzero(actual_ids != torch_ids.numpy())}"
    )
    compare(
        "router_weights_torch",
        actual_weights,
        torch_weights.numpy(),
        tol=1e-4,
    )

    pre_ff2_w = load_bf16_weight(f"{LAYER0}.pre_feedforward_layernorm_2.weight")
    pre_ff2 = load_dump(dump / "pre_ff2_norm.bin", (seq, hidden))
    compare("pre_ff2_norm", pre_ff2, rms_norm(hidden_after_attn, pre_ff2_w, 1e-6))

    # One token exercises all eight routed expert gate/up/down projections and
    # the weighted accumulation without dequantizing all 128 experts.
    gate_up_q = QuantizedWeight(f"{LAYER0}.experts.gate_up_proj")
    down_q = QuantizedWeight(f"{LAYER0}.experts.down_proj")
    moe_row0 = np.zeros(hidden, dtype=np.float32)
    routed = sorted(
        ((int(expert), slot) for slot, expert in enumerate(top_ids[0])),
        key=lambda item: item[0],
    )
    for expert, slot in routed:
        gate_up = pre_ff2[0] @ gate_up_q.expert(int(expert)).T
        gate_part, up_part = np.split(gate_up, 2)
        expert_hidden = gelu_tanh(gate_part) * up_part
        expert_out = expert_hidden @ down_q.expert(int(expert)).T
        contribution = bf16_round(expert_out * top_weights[0, slot])
        moe_row0 = bf16_round(moe_row0 + contribution)
    moe_actual = load_dump(dump / "moe_raw.bin", (seq, hidden))
    compare("moe_raw_row0", moe_actual[0], moe_row0, tol=2.0)

    ff2_w = load_bf16_weight(f"{LAYER0}.post_feedforward_layernorm_2.weight")
    ff2 = load_dump(dump / "ff2.bin", (seq, hidden))
    compare("ff2_row0", ff2[0], rms_norm(moe_actual[:1], ff2_w, 1e-6)[0])
    ff_sum = load_dump(dump / "ff_sum.bin", (seq, hidden))
    compare("ff_sum", ff_sum, bf16_round(ff1 + ff2), tol=1.0)
    post_ff_w = load_bf16_weight(f"{LAYER0}.post_feedforward_layernorm.weight")
    post_ff = load_dump(dump / "post_ff_norm.bin", (seq, hidden))
    compare("post_ff_norm", post_ff, rms_norm(ff_sum, post_ff_w, 1e-6))
    hidden_out = load_dump(dump / "hidden_out.bin", (seq, hidden))
    try:
        layer_scalar = float(
            load_bf16_weight(f"{LAYER0}.layer_scalar").reshape(-1)[0]
        )
    except KeyError:
        # Legacy exports omitted persistent buffers and therefore implicitly
        # behaved as the module initializer (1.0).
        layer_scalar = 1.0
    expected_hidden_out = bf16_round(
        bf16_round(hidden_after_attn + post_ff) * layer_scalar
    )
    compare("hidden_out", hidden_out, expected_hidden_out, tol=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
