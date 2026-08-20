#!/usr/bin/env python3
"""Smoke-test the Windows AilaShared proxy detection ABI against a real model."""

from __future__ import annotations

import argparse
import ctypes
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build-yolo26"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    build = args.build.resolve()
    os.environ["AILA_RUNTIME_DLL_DIR"] = str(build)
    dll_directory = os.add_dll_directory(str(build))
    try:
        api = ctypes.WinDLL(str(build / "AilaShared.dll"))
        api.aila_engine_create.restype = ctypes.c_void_p
        api.aila_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        api.aila_engine_init.restype = ctypes.c_int
        api.aila_engine_destroy.argtypes = [ctypes.c_void_p]
        api.aila_default_detection_config.restype = DetectionConfig
        detection_pointer = ctypes.POINTER(Detection)
        common = [ctypes.c_void_p]
        api.aila_detect_file.argtypes = common + [ctypes.c_char_p, ctypes.POINTER(DetectionConfig),
                                                   ctypes.POINTER(detection_pointer), ctypes.POINTER(ctypes.c_int)]
        api.aila_detect_encoded.argtypes = common + [ctypes.c_void_p, ctypes.c_size_t,
                                                      ctypes.POINTER(DetectionConfig),
                                                      ctypes.POINTER(detection_pointer), ctypes.POINTER(ctypes.c_int)]
        api.aila_detect_pixels.argtypes = common + [ctypes.c_void_p, ctypes.c_size_t,
                                                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                     ctypes.POINTER(DetectionConfig),
                                                     ctypes.POINTER(detection_pointer), ctypes.POINTER(ctypes.c_int)]
        for function in (api.aila_detect_file, api.aila_detect_encoded, api.aila_detect_pixels):
            function.restype = ctypes.c_int
        api.aila_free_detections.argtypes = [detection_pointer, ctypes.c_int]
        api.aila_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        api.aila_generate.restype = ctypes.c_void_p
        api.aila_last_error_code.argtypes = [ctypes.c_void_p]
        api.aila_last_error_code.restype = ctypes.c_int

        engine = api.aila_engine_create()
        if not engine:
            raise RuntimeError("aila_engine_create failed")
        try:
            rc = api.aila_engine_init(engine, str(args.model.resolve()).encode("utf-8"), 4096)
            if rc != 0:
                raise RuntimeError(f"aila_engine_init failed: {rc}")
            config = api.aila_default_detection_config()
            config.max_detections = 10

            def check(rc: int, values, count: ctypes.c_int, label: str) -> None:
                if rc != 0 or count.value <= 0 or not values:
                    raise RuntimeError(f"{label} failed: rc={rc}, count={count.value}")
                scores = [values[index].confidence for index in range(count.value)]
                if scores != sorted(scores, reverse=True):
                    raise RuntimeError(f"{label} results are not confidence sorted")
                for index in range(count.value):
                    values[index].class_name.decode("utf-8")
                api.aila_free_detections(values, count.value)

            values = detection_pointer()
            count = ctypes.c_int()
            check(api.aila_detect_file(engine, str(args.image.resolve()).encode("utf-8"),
                                       ctypes.byref(config), ctypes.byref(values), ctypes.byref(count)),
                  values, count, "detect_file")

            encoded = args.image.read_bytes()
            encoded_buffer = ctypes.create_string_buffer(encoded)
            values, count = detection_pointer(), ctypes.c_int()
            check(api.aila_detect_encoded(engine, encoded_buffer, len(encoded), ctypes.byref(config),
                                          ctypes.byref(values), ctypes.byref(count)),
                  values, count, "detect_encoded")

            from PIL import Image
            with Image.open(args.image) as source:
                rgb_image = source.convert("RGB")
                rgba_image = source.convert("RGBA")
                width, height = rgba_image.size
                rgb = rgb_image.tobytes()
                rgba = rgba_image.tobytes()
            formats = [
                (0, 3, rgb),
                (1, 3, b"".join(rgb[index:index + 3][::-1] for index in range(0, len(rgb), 3))),
                (2, 4, rgba),
                (3, 4, b"".join(bytes((rgba[index + 2], rgba[index + 1], rgba[index], rgba[index + 3]))
                                  for index in range(0, len(rgba), 4))),
            ]
            for pixel_format, channels, packed in formats:
                packed_stride = width * channels
                stride = packed_stride + 5
                padded = b"".join(
                    packed[row * packed_stride:(row + 1) * packed_stride] + b"\0" * 5
                    for row in range(height - 1)
                ) + packed[(height - 1) * packed_stride:]
                pixel_buffer = ctypes.create_string_buffer(padded)
                values, count = detection_pointer(), ctypes.c_int()
                check(api.aila_detect_pixels(engine, pixel_buffer, len(padded), width, height,
                                             stride, pixel_format, ctypes.byref(config),
                                             ctypes.byref(values), ctypes.byref(count)),
                      values, count, f"detect_pixels(format={pixel_format})")

            generated = api.aila_generate(engine, b"not supported", None)
            if generated or api.aila_last_error_code(engine) != 7:
                raise RuntimeError("YOLO26 generation did not return AILA_ERR_MODEL_CAPABILITY")
            print("YOLO26 proxy C ABI smoke test passed")
        finally:
            api.aila_engine_destroy(engine)
    finally:
        dll_directory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
