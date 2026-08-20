#!/usr/bin/env python3
"""Evaluate Aila YOLO26 and its source checkpoint on COCO val2017.

The two paths use the same fixed 640x640, batch-one inference contract.  The
script keeps one Aila Worker/Proxy instance alive per scale, writes prediction
JSON below the gitignored models/yolo26 tree, and fails if Aila loses more than
the configured absolute box mAP50-95 allowance.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path


class DetectionConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("confidence_threshold", ctypes.c_float),
        ("max_detections", ctypes.c_int),
        ("reserved", ctypes.c_int * 8),
    ]


class Detection(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("x1", ctypes.c_float), ("y1", ctypes.c_float),
        ("x2", ctypes.c_float), ("y2", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("class_id", ctypes.c_int),
        ("class_name", ctypes.c_char_p),
        ("reserved", ctypes.c_int * 4),
    ]


class AilaSession:
    def __init__(self, build: Path, model: Path, confidence: float, max_det: int):
        os.environ["AILA_RUNTIME_DLL_DIR"] = str(build)
        self.dll_directory = os.add_dll_directory(str(build))
        self.api = ctypes.WinDLL(str(build / "AilaShared.dll"))
        self.api.aila_engine_create.restype = ctypes.c_void_p
        self.api.aila_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.api.aila_engine_init.restype = ctypes.c_int
        self.api.aila_engine_destroy.argtypes = [ctypes.c_void_p]
        self.api.aila_default_detection_config.restype = DetectionConfig
        pointer = ctypes.POINTER(Detection)
        self.api.aila_detect_file.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(DetectionConfig),
            ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_int),
        ]
        self.api.aila_detect_file.restype = ctypes.c_int
        self.api.aila_free_detections.argtypes = [pointer, ctypes.c_int]
        self.api.aila_last_error_message.argtypes = [ctypes.c_void_p]
        self.api.aila_last_error_message.restype = ctypes.c_char_p
        self.pointer_type = pointer
        self.engine = self.api.aila_engine_create()
        if not self.engine:
            self.close()
            raise RuntimeError("aila_engine_create failed")
        rc = self.api.aila_engine_init(self.engine, str(model).encode("utf-8"), 4096)
        if rc != 0:
            message = self.api.aila_last_error_message(self.engine).decode("utf-8", "replace")
            self.close()
            raise RuntimeError(f"aila_engine_init failed ({rc}): {message}")
        self.config = self.api.aila_default_detection_config()
        self.config.confidence_threshold = confidence
        self.config.max_detections = max_det

    def detect(self, image: Path) -> list[dict]:
        values = self.pointer_type()
        count = ctypes.c_int()
        rc = self.api.aila_detect_file(
            self.engine, str(image).encode("utf-8"), ctypes.byref(self.config),
            ctypes.byref(values), ctypes.byref(count))
        if rc != 0:
            message = self.api.aila_last_error_message(self.engine).decode("utf-8", "replace")
            raise RuntimeError(f"detect_file failed for {image} ({rc}): {message}")
        try:
            return [
                {
                    "class_id": values[index].class_id,
                    "score": float(values[index].confidence),
                    "box": [float(values[index].x1), float(values[index].y1),
                            float(values[index].x2), float(values[index].y2)],
                }
                for index in range(count.value)
            ]
        finally:
            self.api.aila_free_detections(values, count.value)

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine:
            self.api.aila_engine_destroy(engine)
            self.engine = None
        directory = getattr(self, "dll_directory", None)
        if directory:
            directory.close()
            self.dll_directory = None

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def coco_record(image_id: int, category_id: int, box: list[float], score: float) -> dict:
    x1, y1, x2, y2 = box
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        "score": score,
    }


def evaluate(coco, predictions: list[dict], image_ids: list[int]) -> float:
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return 0.0
    result = coco.loadRes(predictions)
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build-yolo26"))
    parser.add_argument("--model-root", type=Path, default=Path("models/yolo26/aila"))
    parser.add_argument("--source-root", type=Path, default=Path("models/yolo26/source"))
    parser.add_argument("--coco-root", type=Path, default=Path("models/yolo26/coco"))
    parser.add_argument("--output", type=Path, default=Path("models/yolo26/coco-results"))
    parser.add_argument("--scales", default="n,s,m,l,x")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-ap-loss", type=float, default=0.005,
                        help="maximum absolute mAP50-95 loss (0.005 = 0.5 AP)")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N images (smoke only; 0 means all)")
    parser.add_argument("--reuse-aila", action="store_true",
                        help="reuse an existing per-scale Aila prediction JSON")
    args = parser.parse_args()

    from pycocotools.coco import COCO
    from ultralytics import YOLO

    annotation = args.coco_root / "annotations" / "instances_val2017.json"
    image_root = args.coco_root / "val2017"
    if not annotation.is_file() or not image_root.is_dir():
        parser.error("COCO val2017 is missing; run tools/prepare_yolo26.py --with-coco")
    scales = [item.strip() for item in args.scales.split(",") if item.strip()]
    if not scales or any(scale not in ("n", "s", "m", "l", "x") for scale in scales):
        parser.error("--scales must be a comma-separated subset of n,s,m,l,x")
    if not 0.0 <= args.confidence <= 1.0 or not 1 <= args.max_det <= 300:
        parser.error("invalid confidence/max-det")

    coco = COCO(str(annotation))
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[:args.limit]
    image_info = {entry["id"]: entry for entry in coco.loadImgs(image_ids)}
    category_ids = sorted(coco.getCatIds())
    if len(category_ids) != 80:
        raise RuntimeError(f"expected 80 COCO categories, found {len(category_ids)}")
    image_paths = [image_root / image_info[image_id]["file_name"] for image_id in image_ids]
    args.output.mkdir(parents=True, exist_ok=True)

    summary_path = args.output / "summary.json"
    report: dict[str, object] = {}
    if summary_path.is_file():
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    for scale in scales:
        aila_path = args.output / f"yolo26{scale}-aila.json"
        if args.reuse_aila and aila_path.is_file():
            aila_predictions = json.loads(aila_path.read_text(encoding="utf-8"))
            print(f"YOLO26{scale} Aila: reused {len(aila_predictions)} predictions", flush=True)
        else:
            aila_predictions: list[dict] = []
            with AilaSession(args.build.resolve(), (args.model_root / scale).resolve(),
                             args.confidence, args.max_det) as session:
                for offset, (image_id, image_path) in enumerate(zip(image_ids, image_paths), 1):
                    for detection in session.detect(image_path.resolve()):
                        class_id = detection["class_id"]
                        if not 0 <= class_id < len(category_ids):
                            raise RuntimeError(f"YOLO26{scale} returned invalid class {class_id}")
                        aila_predictions.append(coco_record(
                            image_id, category_ids[class_id], detection["box"], detection["score"]))
                    if offset % 250 == 0:
                        print(f"YOLO26{scale} Aila: {offset}/{len(image_ids)}", flush=True)
            aila_path.write_text(json.dumps(aila_predictions), encoding="utf-8")

        reference_predictions: list[dict] = []
        model = YOLO(str(args.source_root / f"yolo26{scale}.pt"))
        for offset, (image_id, image_path) in enumerate(zip(image_ids, image_paths), 1):
            result = model.predict(
                source=str(image_path), stream=False, batch=1,
                imgsz=640, rect=False, conf=args.confidence, max_det=args.max_det,
                device=args.device, verbose=False)[0]
            for box, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.int().cpu().tolist()):
                reference_predictions.append(coco_record(
                    image_id, category_ids[class_id], box, float(score)))
            if offset % 250 == 0:
                print(f"YOLO26{scale} Ultralytics: {offset}/{len(image_ids)}", flush=True)

        reference_path = args.output / f"yolo26{scale}-ultralytics-fp32.json"
        reference_path.write_text(json.dumps(reference_predictions), encoding="utf-8")
        aila_map = evaluate(coco, aila_predictions, image_ids)
        reference_map = evaluate(coco, reference_predictions, image_ids)
        loss = reference_map - aila_map
        report[scale] = {
            "images": len(image_ids), "aila_map50_95": aila_map,
            "ultralytics_fp32_map50_95": reference_map, "absolute_loss": loss,
            "passed": loss <= args.max_ap_loss,
        }
        summary_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        del model
        gc.collect()
        try:
            import torch
            torch.xpu.empty_cache()
        except (ImportError, AttributeError):
            pass

    print(json.dumps(report, indent=2))
    return 0 if all(entry["passed"] for entry in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
