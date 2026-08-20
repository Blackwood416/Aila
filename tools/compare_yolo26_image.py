#!/usr/bin/env python3
"""Compare Aila and Ultralytics YOLO26 semantics on one image."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aila", type=Path, default=Path("build/Aila.exe"))
    parser.add_argument("--source-root", type=Path, default=Path("models/yolo26/source"))
    parser.add_argument("--model-root", type=Path, default=Path("models/yolo26/aila"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--scales", default="n,s,m,l,x")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--min-iou", type=float, default=0.99)
    parser.add_argument("--max-score-error", type=float, default=0.02)
    args = parser.parse_args()

    from ultralytics import YOLO

    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    report: dict[str, object] = {}
    for scale in scales:
        command = [
            str(args.aila.resolve()), "--model", str((args.model_root / scale).resolve()),
            "--detect", str(args.image.resolve()), "--conf", str(args.conf),
            "--max-det", str(args.max_det), "--log-level", "error",
        ]
        completed = subprocess.run(command, check=True, text=True, encoding="utf-8",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        aila = json.loads(completed.stdout)
        detections = aila["detections"]
        scores = [entry["confidence"] for entry in detections]
        if scores != sorted(scores, reverse=True):
            raise RuntimeError(f"YOLO26{scale}: Aila results are not sorted")

        result = YOLO(str(args.source_root / f"yolo26{scale}.pt")).predict(
            str(args.image), imgsz=640, conf=args.conf, max_det=args.max_det,
            device=args.device, rect=False, verbose=False)[0]
        reference = [
            {
                "box": box,
                "score": score,
                "class_id": class_id,
            }
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                result.boxes.cls.int().cpu().tolist())
        ]
        available = set(range(len(reference)))
        matches = []
        for detection in detections:
            box = [detection[key] for key in ("x1", "y1", "x2", "y2")]
            candidates = [index for index in available
                          if reference[index]["class_id"] == detection["class_id"]]
            if not candidates:
                raise RuntimeError(f"YOLO26{scale}: unmatched class {detection['class_id']}")
            best = max(candidates, key=lambda index: iou(box, reference[index]["box"]))
            overlap = iou(box, reference[best]["box"])
            score_error = abs(detection["confidence"] - reference[best]["score"])
            if overlap < args.min_iou or score_error > args.max_score_error:
                raise RuntimeError(
                    f"YOLO26{scale}: semantic mismatch IoU={overlap:.6f}, "
                    f"score_error={score_error:.6f}")
            available.remove(best)
            matches.append({"class_id": detection["class_id"], "iou": overlap,
                            "score_error": score_error})
        report[scale] = {
            "aila_count": len(detections), "reference_count": len(reference),
            "minimum_matched_iou": min((entry["iou"] for entry in matches), default=1.0),
            "maximum_score_error": max((entry["score_error"] for entry in matches), default=0.0),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
