#!/usr/bin/env python3
"""Replay exact 14K prefill routes with prompt-scoped cold-key promotion.

This is a CPU-only analysis tool. It does not import Aila or touch the GPU.
The exact-route input is a small JSONL schema documented in
`schema/route-trace-v1.schema.json`; the current 14K hotset export is aggregate
and therefore intentionally cannot produce four-block staging numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "q4e-prefill-union-replay-v1"
ROUTE_SCHEMA = 1
DEFAULT_SLOT_BYTES = 2_534_400
DEFAULT_SLOTS = (2792, 2400)
DEFAULT_PROMOTION_SLOTS = 512
DEFAULT_MIN_OCCURRENCES = 2
REFERENCE_SLOTS = 2792


class InputError(ValueError):
    """Input does not satisfy the documented receipt/schema contract."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hotset_profile(path: str | Path) -> dict[str, Any]:
    """Read an AILA_Q4E_HOTSET profile without changing its ranking."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 4:
        raise InputError(f"profile has too few lines: {p}")
    magic = lines[0].split()
    if magic != ["AILA_Q4E_HOTSET", "1"]:
        raise InputError(f"unsupported profile header: {p}")
    identity = lines[1].strip()
    dims = lines[2].split()
    if len(dims) != 3:
        raise InputError(f"invalid profile dimensions: {p}")
    layers, experts, count = map(int, dims)
    if layers <= 0 or experts <= 0 or count != layers * experts:
        raise InputError(f"invalid profile shape: {p}")
    rows: list[dict[str, int]] = []
    for lineno, line in enumerate(lines[3:], start=4):
        parts = line.split()
        if len(parts) != 4:
            raise InputError(f"{p}:{lineno}: expected four integers")
        try:
            key, decode, prefill, unique = map(int, parts)
        except ValueError as exc:
            raise InputError(f"{p}:{lineno}: non-integer field") from exc
        if min(key, decode, prefill, unique) < 0:
            raise InputError(f"{p}:{lineno}: negative field")
        rows.append({"key": key, "decode": decode, "prefill": prefill, "unique": unique})
    if len(rows) != count:
        raise InputError(f"{p}: expected {count} rows, got {len(rows)}")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != count or min(keys) != 0 or max(keys) != count - 1:
        raise InputError(f"{p}: profile keys are not a complete 0..{count-1} permutation")
    return {
        "path": str(p),
        "sha256": sha256(p),
        "identity": identity,
        "layers": layers,
        "experts": experts,
        "count": count,
        "rows": rows,
        "ranking": keys,
        "decode": {row["key"]: row["decode"] for row in rows},
        "by_key": {row["key"]: row for row in rows},
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def read_route_trace(path: str | Path) -> dict[str, Any]:
    """Read the compact route-trace JSONL schema."""
    p = Path(path)
    raw: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{p}:{lineno}: invalid JSON") from exc
        _require(isinstance(row, dict), f"{p}:{lineno}: top-level object required")
        raw.append(row)
    _require(bool(raw), f"{p}: empty trace")
    config = raw[0]
    _require(config.get("type") == "config", f"{p}: first record must be config")
    _require(config.get("schema") == ROUTE_SCHEMA, f"{p}: schema must be {ROUTE_SCHEMA}")
    for field in ("layers", "experts", "top_k", "slot_bytes"):
        _require(isinstance(config.get(field), int) and config[field] > 0,
                 f"{p}: config.{field} must be positive")

    requests: OrderedDict[str, dict[str, Any]] = OrderedDict()
    forwards: dict[str, dict[str, Any]] = {}
    route_seen: set[tuple[str, int]] = set()
    forward_ids: set[str] = set()

    for lineno, row in enumerate(raw[1:], start=2):
        kind = row.get("type")
        if kind == "request_begin":
            request = str(row.get("request", ""))
            _require(bool(request), f"{p}:{lineno}: request id required")
            _require(request not in requests, f"{p}:{lineno}: duplicate request")
            prompt_tokens = row.get("prompt_tokens")
            _require(isinstance(prompt_tokens, int) and prompt_tokens > 0,
                     f"{p}:{lineno}: prompt_tokens must be positive")
            requests[request] = {
                "id": request,
                "prompt_tokens": prompt_tokens,
                "forwards": [],
            }
        elif kind == "forward_begin":
            request = str(row.get("request", ""))
            _require(request in requests, f"{p}:{lineno}: unknown request")
            forward = str(row.get("forward", ""))
            _require(bool(forward) and forward not in forward_ids,
                     f"{p}:{lineno}: duplicate/missing forward id")
            forward_ids.add(forward)
            position = row.get("position")
            tokens = row.get("tokens")
            _require(isinstance(position, int) and position >= 0,
                     f"{p}:{lineno}: invalid position")
            _require(isinstance(tokens, int) and tokens > 0,
                     f"{p}:{lineno}: invalid tokens")
            phase = row.get("phase", "prefill")
            _require(phase in ("prefill", "decode"), f"{p}:{lineno}: invalid phase")
            item = {
                "forward": forward,
                "request": request,
                "position": position,
                "tokens": tokens,
                "phase": phase,
                "routes": {},
            }
            forwards[forward] = item
            requests[request]["forwards"].append(forward)
        elif kind == "route":
            forward = str(row.get("forward", ""))
            _require(forward in forwards, f"{p}:{lineno}: route before forward_begin")
            layer = row.get("layer")
            rows = row.get("rows")
            top_k = row.get("top_k", config["top_k"])
            ids = row.get("ids")
            _require(isinstance(layer, int) and 0 <= layer < config["layers"],
                     f"{p}:{lineno}: invalid layer")
            _require(isinstance(rows, int) and rows > 0,
                     f"{p}:{lineno}: invalid rows")
            _require(rows == forwards[forward]["tokens"],
                     f"{p}:{lineno}: route rows differ from forward tokens")
            _require(isinstance(top_k, int) and top_k > 0,
                     f"{p}:{lineno}: invalid top_k")
            _require(top_k == config["top_k"],
                     f"{p}:{lineno}: route top_k differs from config")
            _require(isinstance(ids, list), f"{p}:{lineno}: ids must be an array")
            _require(len(ids) == rows * top_k,
                     f"{p}:{lineno}: ids length must be rows*top_k")
            if any(not isinstance(v, int) or v < 0 or v >= config["experts"] for v in ids):
                raise InputError(f"{p}:{lineno}: route id outside expert range")
            key = (forward, layer)
            _require(key not in route_seen, f"{p}:{lineno}: duplicate layer route")
            route_seen.add(key)
            forwards[forward]["routes"][layer] = {
                "rows": rows,
                "top_k": top_k,
                "ids": ids,
            }
        elif kind == "request_end":
            request = str(row.get("request", ""))
            _require(request in requests, f"{p}:{lineno}: unknown request end")
            requests[request]["ended"] = True
        elif kind in ("note", "config"):
            continue
        else:
            raise InputError(f"{p}:{lineno}: unknown record type {kind!r}")

    for request, item in requests.items():
        _require(bool(item["forwards"]), f"{p}: request {request} has no forwards")
        seen_positions: set[int] = set()
        cursor = 0
        prefill_forwards = []
        for forward in item["forwards"]:
            f = forwards[forward]
            _require(f["position"] not in seen_positions,
                     f"{p}: duplicate position in request {request}")
            seen_positions.add(f["position"])
            if f["phase"] != "prefill":
                continue
            _require(f["position"] == cursor,
                     f"{p}: prefill positions must start at 0 and be contiguous")
            cursor += f["tokens"]
            missing = set(range(config["layers"])) - set(f["routes"])
            extra = set(f["routes"]) - set(range(config["layers"]))
            _require(not missing and not extra,
                     f"{p}: forward {forward} must contain exactly one route per layer")
            prefill_forwards.append(f)
        item["prefill_forwards"] = prefill_forwards
        if cursor != item["prompt_tokens"]:
            raise InputError(
                f"{p}: request {request} prefill tokens {cursor} != prompt_tokens "
                f"{item['prompt_tokens']}"
            )
    return {"path": str(p), "sha256": sha256(p), "config": config, "requests": requests}


def select_request(trace: dict[str, Any], request_id: str | None) -> dict[str, Any]:
    requests = trace["requests"]
    if request_id is not None:
        _require(request_id in requests, f"request {request_id!r} not found")
        return requests[request_id]
    candidates = [r for r in requests.values() if r["prefill_forwards"]]
    _require(bool(candidates), "trace has no prefill request")
    return max(candidates, key=lambda r: r["prompt_tokens"])


def aggregate_route_profile(profile: dict[str, Any], slot_bytes: int) -> dict[str, Any]:
    layers = profile["layers"]
    experts = profile["experts"]
    rows = profile["rows"]
    by_key = profile["by_key"]
    unions = []
    for layer in range(layers):
        base = layer * experts
        unions.append(sum(1 for e in range(experts) if by_key[base + e]["prefill"] > 0))
    nonzero = [n for n in unions if n > 0]
    return {
        "layers": layers,
        "experts": experts,
        "identity": profile["identity"],
        "prefill_nonzero_keys": sum(unions),
        "prefill_occurrences": sum(row["prefill"] for row in rows),
        "unique_observations": sum(row["unique"] for row in rows),
        "per_layer_union": unions,
        "per_layer_union_min": min(nonzero) if nonzero else 0,
        "per_layer_union_median": statistics.median(nonzero) if nonzero else 0,
        "per_layer_union_max": max(nonzero) if nonzero else 0,
        "union_bytes": sum(unions) * slot_bytes,
        "slot_bytes": slot_bytes,
    }


def _top_keys(profile: dict[str, Any], count: int) -> list[int]:
    return profile["ranking"][: max(0, min(count, len(profile["ranking"])))]


def _mass(decode_profile: dict[str, Any], keys: Iterable[int]) -> int:
    decode = decode_profile["decode"]
    return sum(decode.get(key, 0) for key in keys)


def _pct(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else round(100.0 * numerator / denominator, 6)


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return {k: float(ordered[0]) for k in ("min", "p25", "median", "p75", "max", "mean")}
    q = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "min": round(ordered[0], 6),
        "p25": round(q[0], 6),
        "median": round(statistics.median(ordered), 6),
        "p75": round(q[2], 6),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _decode_summary(
    decode_profile: dict[str, Any],
    budget: int,
    promotion_slots: int,
    reference_slots: int,
    peak_missing_slots: int | None,
    peak_mass: int | None,
) -> dict[str, Any]:
    ranking = decode_profile["ranking"]
    decode = decode_profile["decode"]
    reference_keys = set(_top_keys(decode_profile, reference_slots))
    base_n = max(0, min(budget - promotion_slots, len(ranking)))
    base_keys = set(_top_keys(decode_profile, base_n))
    budget_keys = set(_top_keys(decode_profile, budget))
    post_missing = reference_keys - budget_keys
    reference_mass = sum(decode.get(k, 0) for k in reference_keys)
    post_mass = sum(decode.get(k, 0) for k in post_missing)
    borrowed_keys = (reference_keys & budget_keys) - base_keys
    outside_budget_keys = reference_keys - budget_keys
    worst_keys = reference_keys - base_keys
    borrowed_mass = sum(decode.get(k, 0) for k in borrowed_keys)
    outside_budget_mass = sum(decode.get(k, 0) for k in outside_budget_keys)
    worst_mass = sum(decode.get(k, 0) for k in worst_keys)
    return {
        "reference_slots": reference_slots,
        "budget_slots": budget,
        "base_permanent_slots": base_n,
        "promotion_overlay_slots": promotion_slots,
        "borrowed_decode_slots": len(borrowed_keys),
        "borrowed_decode_mass": borrowed_mass,
        "outside_budget_decode_slots": len(outside_budget_keys),
        "outside_budget_decode_mass": outside_budget_mass,
        "reference_decode_mass": reference_mass,
        "worst_case_missing_slots_vs_reference": len(worst_keys),
        "worst_case_decode_mass_at_risk": worst_mass,
        "worst_case_decode_mass_pct": _pct(worst_mass, reference_mass),
        "post_rollback_missing_slots_vs_reference": len(post_missing),
        "post_rollback_decode_mass": post_mass,
        "post_rollback_decode_mass_pct": _pct(post_mass, reference_mass),
        "peak_missing_slots_vs_reference": peak_missing_slots,
        "peak_decode_mass_at_risk": peak_mass,
        "peak_decode_mass_pct": None if peak_mass is None else _pct(peak_mass, reference_mass),
    }


def simulate_budget(
    trace: dict[str, Any],
    request: dict[str, Any],
    route_profile: dict[str, Any],
    decode_profile: dict[str, Any],
    budget: int,
    promotion_slots: int,
    min_occurrences: int,
    slot_bytes: int,
) -> dict[str, Any]:
    _require(route_profile["identity"] == decode_profile["identity"],
             "route and decode profile identities differ")
    _require(route_profile["layers"] == trace["config"]["layers"] and
             route_profile["experts"] == trace["config"]["experts"],
             "profile dimensions differ from trace")
    _require(0 <= promotion_slots <= budget, "promotion slots must be <= budget")
    base_n = budget - promotion_slots
    _require(base_n >= 0, "base slots must be non-negative")
    ranking = decode_profile["ranking"]
    base_keys = set(_top_keys(decode_profile, base_n))
    baseline_keys = set(_top_keys(decode_profile, budget))
    borrowed = _top_keys(decode_profile, budget)[base_n:]
    overlay_cap = min(promotion_slots, max(0, len(ranking) - base_n))
    overlay: OrderedDict[int, str] = OrderedDict((key, "borrowed") for key in borrowed)
    overlay_score: dict[int, int] = {key: 0 for key in overlay}
    reference_keys = set(_top_keys(decode_profile, REFERENCE_SLOTS))
    decode = decode_profile["decode"]
    reference_mass = sum(decode.get(key, 0) for key in reference_keys)

    blocks: list[dict[str, Any]] = []
    layer_totals: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    peak_missing = 0
    peak_mass = 0

    for block_index, forward in enumerate(request["prefill_forwards"]):
        block_missing = 0
        block_base_hits = 0
        block_overlay_hits = 0
        candidate_counts: dict[int, int] = defaultdict(int)
        first_seen: dict[int, int] = {}
        layer_rows: list[dict[str, Any]] = []

        for layer in range(trace["config"]["layers"]):
            route = forward["routes"][layer]
            ids = route["ids"]
            baseline_staged: set[int] = set()
            promoted_staged: set[int] = set()
            base_hits = 0
            overlay_hits = 0
            for e in ids:
                key = layer * trace["config"]["experts"] + e
                seen_key = first_seen.setdefault(key, len(ids) * layer + len(first_seen))
                if key not in baseline_keys:
                    baseline_staged.add(key)
                if key in base_keys:
                    base_hits += 1
                elif key in overlay:
                    overlay_hits += 1
                    overlay.move_to_end(key)
                else:
                    promoted_staged.add(key)
                    candidate_counts[key] += 1
                    _ = seen_key

            base_bytes = len(baseline_staged) * slot_bytes
            promoted_bytes = len(promoted_staged) * slot_bytes
            row = {
                "layer": layer,
                "rows": route["rows"],
                "top_k": route["top_k"],
                "route_occurrences": len(ids),
                "baseline_staged_slots": len(baseline_staged),
                "promoted_staged_slots": len(promoted_staged),
                "saved_slots": len(baseline_staged) - len(promoted_staged),
                "baseline_bytes": base_bytes,
                "promoted_bytes": promoted_bytes,
                "saved_bytes": base_bytes - promoted_bytes,
                "base_hit_occurrences": base_hits,
                "overlay_hit_occurrences": overlay_hits,
                "staged_pct_of_route": _pct(len(promoted_staged), max(1, len(set(ids)))),
            }
            row["saved_pct"] = _pct(row["saved_slots"], max(1, row["baseline_staged_slots"]))
            layer_rows.append(row)
            block_missing += len(promoted_staged)
            block_base_hits += base_hits
            block_overlay_hits += overlay_hits
            totals = layer_totals[layer]
            for field in ("route_occurrences", "baseline_staged_slots",
                          "promoted_staged_slots", "saved_slots", "baseline_bytes",
                          "promoted_bytes", "saved_bytes", "base_hit_occurrences",
                          "overlay_hit_occurrences"):
                totals[field] += row[field]

        # Promotion is block-local: frequencies are known after the block, so
        # block 0 is the no-promotion control and later blocks can reuse keys.
        candidates = [
            (key, count, first_seen.get(key, 0))
            for key, count in candidate_counts.items()
            if key not in base_keys and key not in overlay and count >= min_occurrences
        ]
        candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
        for key, count, _ in candidates:
            if len(overlay) < overlay_cap:
                overlay[key] = "promoted"
                overlay_score[key] = count
            elif overlay_cap > 0:
                victim, origin = next(iter(overlay.items()))
                victim_score = overlay_score.get(victim, 0)
                if origin == "promoted" and count <= victim_score:
                    continue
                overlay.popitem(last=False)
                overlay[key] = "promoted"
                overlay_score[key] = count

        missing_now = reference_keys - base_keys - set(overlay)
        missing_mass = sum(decode.get(key, 0) for key in missing_now)
        peak_missing = max(peak_missing, len(missing_now))
        peak_mass = max(peak_mass, missing_mass)
        baseline_slots = sum(row["baseline_staged_slots"] for row in layer_rows)
        promoted_slots = sum(row["promoted_staged_slots"] for row in layer_rows)
        blocks.append({
            "block": block_index,
            "forward": forward["forward"],
            "position": forward["position"],
            "tokens": forward["tokens"],
            "baseline_staged_slots": baseline_slots,
            "promoted_staged_slots": promoted_slots,
            "saved_slots": baseline_slots - promoted_slots,
            "baseline_bytes": baseline_slots * slot_bytes,
            "promoted_bytes": promoted_slots * slot_bytes,
            "saved_bytes": (baseline_slots - promoted_slots) * slot_bytes,
            "saved_pct": _pct(baseline_slots - promoted_slots, max(1, baseline_slots)),
            "base_hit_occurrences": block_base_hits,
            "overlay_hit_occurrences": block_overlay_hits,
            "overlay_entries_after_block": len(overlay),
            "promoted_entries_after_block": sum(1 for v in overlay.values() if v == "promoted"),
            "missing_vs_reference_slots": len(missing_now),
            "decode_mass_at_risk": missing_mass,
            "layer_distribution": layer_rows,
        })

    layer_distribution = []
    saved_pcts = []
    for layer in sorted(layer_totals):
        total = layer_totals[layer]
        saved_pct = _pct(total["saved_slots"], max(1, total["baseline_staged_slots"]))
        saved_pcts.append(float(saved_pct or 0))
        layer_distribution.append({
            "layer": layer,
            **dict(total),
            "saved_pct": saved_pct,
        })
    summary = {
        "baseline_staged_slots": sum(b["baseline_staged_slots"] for b in blocks),
        "promoted_staged_slots": sum(b["promoted_staged_slots"] for b in blocks),
        "saved_slots": sum(b["saved_slots"] for b in blocks),
        "baseline_bytes": sum(b["baseline_bytes"] for b in blocks),
        "promoted_bytes": sum(b["promoted_bytes"] for b in blocks),
        "saved_bytes": sum(b["saved_bytes"] for b in blocks),
        "saved_pct": _pct(sum(b["saved_slots"] for b in blocks),
                           max(1, sum(b["baseline_staged_slots"] for b in blocks))),
        "layer_saved_pct_distribution": _distribution(saved_pcts),
        "four_block_exact": len(blocks) == 4,
    }
    return {
        "status": "ok",
        "budget_slots": budget,
        "base_permanent_slots": base_n,
        "promotion_slots": promotion_slots,
        "promotion_min_occurrences": min_occurrences,
        "baseline_definition": "top total slots from decode profile, no overlay",
        "blocks": blocks,
        "layer_distribution": layer_distribution,
        "summary": summary,
        "decode_hotset": _decode_summary(
            decode_profile, budget, promotion_slots, REFERENCE_SLOTS,
            peak_missing, peak_mass,
        ),
        "rollback": {
            "applied": True,
            "post_prefill_restores_top_slots": budget,
            "post_rollback_missing_slots_vs_reference": len(
                reference_keys - set(_top_keys(decode_profile, budget))
            ),
        },
    }


def profile_only_budget(
    route_profile: dict[str, Any],
    decode_profile: dict[str, Any],
    budget: int,
    promotion_slots: int,
    slot_bytes: int,
) -> dict[str, Any]:
    _require(route_profile["identity"] == decode_profile["identity"],
             "route and decode profile identities differ")
    base_n = max(0, budget - promotion_slots)
    base_keys = set(_top_keys(decode_profile, base_n))
    reference_keys = set(_top_keys(decode_profile, REFERENCE_SLOTS))
    decode = decode_profile["decode"]
    reference_mass = sum(decode.get(key, 0) for key in reference_keys)
    missing = reference_keys - base_keys
    missing_mass = sum(decode.get(key, 0) for key in missing)
    summary = _decode_summary(
        decode_profile, budget, promotion_slots, REFERENCE_SLOTS,
        len(missing), missing_mass,
    )
    return {
        "status": "blocked_missing_per_block_route_trace",
        "budget_slots": budget,
        "base_permanent_slots": base_n,
        "promotion_slots": promotion_slots,
        "blocks": [],
        "layer_distribution": [],
        "summary": None,
        "decode_hotset": summary,
        "rollback": {
            "applied": True,
            "post_prefill_restores_top_slots": budget,
            "post_rollback_missing_slots_vs_reference": len(
                reference_keys - set(_top_keys(decode_profile, budget))
            ),
        },
    }


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    route_profile = read_hotset_profile(args.route_profile)
    decode_profile = read_hotset_profile(args.decode_profile)
    slot_bytes = args.slot_bytes
    trace = None
    selected = None
    if args.trace:
        trace = read_route_trace(args.trace)
        if trace["config"]["slot_bytes"] != slot_bytes:
            raise InputError(
                f"trace slot_bytes {trace['config']['slot_bytes']} != {slot_bytes}"
            )
        selected = select_request(trace, args.request)
        expected = args.expect_blocks
        if expected > 0 and len(selected["prefill_forwards"]) != expected:
            raise InputError(
                f"expected {expected} prefill blocks, got "
                f"{len(selected['prefill_forwards'])}"
            )
        if (route_profile["layers"] != trace["config"]["layers"] or
                route_profile["experts"] != trace["config"]["experts"]):
            raise InputError("route profile dimensions do not match trace")
        if (decode_profile["layers"] != trace["config"]["layers"] or
                decode_profile["experts"] != trace["config"]["experts"]):
            raise InputError("decode profile dimensions do not match trace")

    aggregate = aggregate_route_profile(route_profile, slot_bytes)
    budgets: dict[str, Any] = {}
    for budget in args.slots:
        if not 0 < budget < len(decode_profile["ranking"]):
            raise InputError(f"slot budget {budget} outside profile range")
        if budget < args.promotion_slots:
            raise InputError(f"slot budget {budget} is below promotion slots")
        if trace is None:
            budgets[str(budget)] = profile_only_budget(
                route_profile, decode_profile, budget, args.promotion_slots, slot_bytes
            )
        else:
            budgets[str(budget)] = simulate_budget(
                trace, selected, route_profile, decode_profile, budget,
                args.promotion_slots, args.min_occurrences, slot_bytes,
            )
    status = "blocked_missing_per_block_route_trace" if trace is None else "ok"
    result = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "inputs": {
            "route_profile": {
                "path": route_profile["path"], "sha256": route_profile["sha256"],
                "kind": "aggregate_hotset_export",
            },
            "decode_profile": {
                "path": decode_profile["path"], "sha256": decode_profile["sha256"],
                "kind": "decode_hotset_profile",
            },
            "route_trace": None if trace is None else {
                "path": trace["path"], "sha256": trace["sha256"],
                "kind": "exact_route_trace",
            },
        },
        "parameters": {
            "slot_bytes": slot_bytes,
            "slot_budgets": list(args.slots),
            "promotion_slots": args.promotion_slots,
            "min_occurrences": args.min_occurrences,
            "request": args.request,
            "expect_blocks": args.expect_blocks,
            "reference_slots": REFERENCE_SLOTS,
            "promotion_slots_source": "AILA_Q4E_HOTSET_PROMPT_SWAPS=512 in xpb config",
        },
        "aggregate_route_profile": aggregate,
        "budgets": budgets,
        "limitations": [
            "A profile is an aggregate count table; it does not encode block "
            "boundaries, row order, or route ids.",
            "Without an exact route trace, blocks/layer distribution and "
            "promotion savings are null by design, not inferred.",
            "Decode mass is a proxy for potential hotset loss; it is not a "
            "quality or throughput prediction.",
            "The simulator is a cache/byte accounting model only.",
        ],
    }
    if trace is None:
        result["todo"] = [
            "Capture four prefill forward_begin/route records for prompt 13944 "
            "(four 4096-row windows plus the 1656-row tail) in route-trace-v1.",
            "Include config.layers=48, experts=512, top_k=10, slot_bytes=2534400.",
            "Provide one complete ids array per layer per forward.",
            "Rerun this tool with --trace and the same two profiles; do not "
            "compare aggregate counts as block results.",
        ]
    return result


def write_csv(path: str | Path, report: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status", "slot_budget", "block", "position", "tokens", "layer",
        "baseline_staged_slots", "promoted_staged_slots", "saved_slots",
        "baseline_bytes", "promoted_bytes", "saved_bytes", "saved_pct",
        "base_hit_occurrences", "overlay_hit_occurrences",
        "decode_mass_at_risk",
    ]
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for budget, data in report["budgets"].items():
            if not data["blocks"]:
                writer.writerow({
                    "status": data["status"], "slot_budget": budget,
                    "block": "profile_only", "position": "", "tokens": "",
                    "layer": "",
                })
                continue
            for block in data["blocks"]:
                for row in block["layer_distribution"]:
                    writer.writerow({
                        "status": data["status"],
                        "slot_budget": budget,
                        "block": block["block"],
                        "position": block["position"],
                        "tokens": block["tokens"],
                        "layer": row["layer"],
                        "baseline_staged_slots": row["baseline_staged_slots"],
                        "promoted_staged_slots": row["promoted_staged_slots"],
                        "saved_slots": row["saved_slots"],
                        "baseline_bytes": row["baseline_bytes"],
                        "promoted_bytes": row["promoted_bytes"],
                        "saved_bytes": row["saved_bytes"],
                        "saved_pct": row["saved_pct"],
                        "base_hit_occurrences": row["base_hit_occurrences"],
                        "overlay_hit_occurrences": row["overlay_hit_occurrences"],
                        "decode_mass_at_risk": block["decode_mass_at_risk"],
                    })


def parse_slots(value: str) -> tuple[int, ...]:
    slots = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not slots or any(slot <= 0 for slot in slots):
        raise argparse.ArgumentTypeError("slots must be positive comma-separated integers")
    return slots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-profile", required=True,
                        help="real 14K aggregate hotset export")
    parser.add_argument("--decode-profile", required=True,
                        help="decode hotset profile used as the resident baseline")
    parser.add_argument("--trace", help="optional exact route-trace-v1 JSONL")
    parser.add_argument("--request", help="request id in the trace")
    parser.add_argument("--slots", type=parse_slots, default=DEFAULT_SLOTS,
                        help="comma-separated total resident budgets")
    parser.add_argument("--promotion-slots", type=int, default=DEFAULT_PROMOTION_SLOTS)
    parser.add_argument("--min-occurrences", type=int, default=DEFAULT_MIN_OCCURRENCES)
    parser.add_argument("--expect-blocks", type=int, default=4)
    parser.add_argument("--slot-bytes", type=int, default=DEFAULT_SLOT_BYTES)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out")
    args = parser.parse_args(argv)
    if args.promotion_slots < 0 or args.min_occurrences < 1 or args.slot_bytes <= 0:
        parser.error("promotion slots must be >=0, min occurrences >=1, slot bytes >0")
    try:
        report = make_report(args)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(args.csv_out, report)
    print(f"status={report['status']}")
    print(f"route_profile={report['inputs']['route_profile']['path']}")
    print(f"budgets={','.join(report['budgets'])}")
    for budget, data in report["budgets"].items():
        decode = data["decode_hotset"]
        print(
            f"slots={budget} base={data['base_permanent_slots']} "
            f"promotion={data['promotion_slots']} "
            f"blocks={len(data['blocks'])} "
            f"worst_decode_mass_pct={decode['worst_case_decode_mass_pct']}"
        )
    if report["status"] != "ok":
        print("result=blocked_missing_per_block_route_trace (see JSON todo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
