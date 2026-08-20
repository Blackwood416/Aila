#!/usr/bin/env python3
"""Convert an Ultralytics YOLO26 detect checkpoint to Aila safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_TOPOLOGY = [
    "Conv", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2", "Conv",
    "C3k2", "SPPF", "C2PSA", "Upsample", "Concat", "C3k2", "Upsample",
    "Concat", "C3k2", "Conv", "Concat", "C3k2", "Conv", "Concat", "C3k2", "Detect",
]
SCALES = {"n", "s", "m", "l", "x"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_scale(source: Path, yaml: dict) -> str:
    scale = str(yaml.get("scale", ""))
    if scale in SCALES:
        return scale
    match = re.search(r"yolo26([nsmlx])(?:[-_.]|$)", source.name.lower())
    if not match:
        raise ValueError("cannot infer YOLO26 scale from checkpoint or model YAML")
    return match.group(1)


def normalized_name(name: str) -> str | None:
    if not name.startswith("model."):
        return None
    parts = name.split(".")
    if len(parts) < 3 or not parts[1].isdigit():
        return None
    layer = int(parts[1])
    tail = ".".join(parts[2:])
    if layer == 23:
        if tail.startswith("one2one_cv2."):
            tail = "box." + tail[len("one2one_cv2."):]
        elif tail.startswith("one2one_cv3."):
            tail = "cls." + tail[len("one2one_cv3."):]
        elif tail.startswith(("cv2.", "cv3.", "dfl.")):
            return None
    return f"aila.layers.{layer}.{tail}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        import torch
        import ultralytics
        from safetensors.torch import save_file
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install torch, ultralytics, and safetensors to run this converter") from exc

    source = args.source.resolve()
    if not source.is_file():
        parser.error(f"checkpoint not found: {source}")

    wrapper = YOLO(str(source), task="detect")
    if wrapper.task != "detect":
        raise ValueError(f"expected detect checkpoint, got {wrapper.task!r}")
    model = wrapper.model
    topology = [module.__class__.__name__ for module in model.model]
    if topology != EXPECTED_TOPOLOGY:
        raise ValueError(f"unsupported YOLO26 topology: {topology}")

    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not bool(head.end2end):
        raise ValueError("checkpoint does not contain the YOLO26 end-to-end Detect head")
    if int(head.reg_max) != 1:
        raise ValueError(f"YOLO26 reg_max must be 1, got {head.reg_max}")
    strides = [int(value) for value in head.stride.detach().cpu().tolist()]
    if strides != [8, 16, 32]:
        raise ValueError(f"YOLO26 strides must be [8, 16, 32], got {strides}")

    names_obj = wrapper.names
    if isinstance(names_obj, dict):
        expected_ids = list(range(len(names_obj)))
        if sorted(int(key) for key in names_obj) != expected_ids:
            raise ValueError("class IDs must be contiguous from zero")
        names = [str(names_obj[index]) for index in expected_ids]
    else:
        names = [str(value) for value in names_obj]
    if len(names) != int(head.nc):
        raise ValueError("class name count does not match Detect.nc")

    scale = infer_scale(source, model.yaml)
    model.eval()
    model.fuse(verbose=False)
    head = model.model[-1]
    head.fuse()
    model.half()

    tensors = {}
    shape_records = []
    for source_name, tensor in model.state_dict().items():
        target_name = normalized_name(source_name)
        if target_name is None:
            continue
        value = tensor.detach().contiguous().cpu()
        if value.is_floating_point():
            value = value.to(torch.float16)
        tensors[target_name] = value
        shape_records.append((target_name, list(value.shape), str(value.dtype)))
    if not tensors:
        raise RuntimeError("conversion produced no tensors")

    topology_payload = json.dumps(
        {"modules": topology, "tensors": sorted(shape_records)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    topology_sha256 = hashlib.sha256(topology_payload).hexdigest()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"), metadata={
        "format": "aila-yolo26-weights",
        "format_version": "1",
        "topology_sha256": topology_sha256,
    })
    config = {
        "model_type": "yolo26",
        "format_version": 1,
        "task": "detect",
        "scale": scale,
        "input_width": 640,
        "input_height": 640,
        "num_classes": len(names),
        "class_names": names,
        "reg_max": 1,
        "end2end": True,
        "dtype": "float16",
        "strides": strides,
        "topology_sha256": topology_sha256,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "format": "aila-yolo26-conversion",
        "format_version": 1,
        "source": str(source),
        "source_sha256": file_sha256(source),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "scale": scale,
        "num_classes": len(names),
        "dtype": "float16",
        "topology_sha256": topology_sha256,
        "license_notice": "Ultralytics code/models are AGPL-3.0 or commercially licensed; they are not bundled with Aila.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
