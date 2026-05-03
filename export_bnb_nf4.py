#!/usr/bin/env python3
"""
Aila BNB NF4 model export script.

Quantizes a Hugging Face model to bitsandbytes NF4 format and exports the
quantized weights as a safetensors file that Aila can load directly.

Requirements:
    torch, transformers, bitsandbytes (Intel XPU backend)

Usage:
    python export_bnb_nf4.py \\
        --source Qwen/Qwen3.5-0.8B \\
        --output ./models/qwen3.5-0.8B-bnb-nf4-offline

    # For multimodal (vision) models:
    python export_bnb_nf4.py \\
        --source Qwen/Qwen3.5-4B \\
        --output ./models/qwen3.5-4B-bnb-nf4-vision-offline \\
        --vision
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import bitsandbytes.cextension as bnb_cext
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a BNB NF4 quantized model for Aila inference."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source model: Hugging Face repo ID or local directory path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the exported safetensors model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory if it already exists.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Model has vision (multimodal) capability. Keeps the visual encoder dense.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for quantization reproducibility (default: 1234).",
    )
    parser.add_argument(
        "--keep-dense",
        action="append",
        dest="keep_dense_modules",
        default=None,
        help="Module prefix to keep in original (dense) precision. Repeatable.",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_keep_dense_modules(
    config: AutoConfig,
    vision: bool,
    requested_modules: list[str] | None,
) -> list[str]:
    modules = list(requested_modules or [])
    if vision:
        modules.append("model.visual")
    return unique_preserve_order(modules)


def build_quant_config(keep_dense_modules: list[str]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=keep_dense_modules or None,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Export directory already exists: {output_dir}\n"
            "Use --overwrite to replace it."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_processor_assets(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source_dir.glob("*processor_config.json")):
        shutil.copy2(path, output_dir / path.name)
        copied.append(path.name)
    return copied


def print_header(source: str, output: str) -> None:
    print(f"torch={torch.__version__}")
    print(f"bitsandbytes={bnb.__version__}")
    print(f"bnb_backend={getattr(bnb_cext, 'BNB_BACKEND', 'unknown')}")
    print(f"device={torch.xpu.get_device_name(0)}")
    print(f"quantization=nf4, compute_dtype=float16, double_quant=true")
    print(f"source={source}")
    print(f"output={output}")


def main() -> int:
    configure_stdout()
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("ERROR: torch.xpu is not available. Ensure Intel oneAPI environment is active.", file=sys.stderr)
        return 1

    source = args.source
    output = Path(args.output)
    device = "xpu:0"

    torch.manual_seed(args.seed)
    print_header(source, str(output))

    # Resolve source: try local path first, then Hugging Face
    source_path = Path(source)
    if source_path.exists() and source_path.is_dir():
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=False)
    else:
        print(f"Downloading from Hugging Face: {source}")
        config = AutoConfig.from_pretrained(source, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        source_path = Path(source)  # not a local directory; keep for reference

    model_type = getattr(config, "model_type", "unknown")
    multimodal = args.vision or (
        model_type == "qwen3_5" and getattr(config, "vision_config", None) is not None
    )
    keep_dense_modules = resolve_keep_dense_modules(config, multimodal, args.keep_dense_modules)
    quant_config = build_quant_config(keep_dense_modules)
    model_loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM

    print(f"model_type={model_type}")
    print(f"multimodal={multimodal}")
    print(f"loader={model_loader.__name__}")
    print(f"keep_dense_modules={keep_dense_modules}")

    prepare_output_directory(output, args.overwrite)

    # Quantize and load
    print("Loading and quantizing model...")
    load_start = time.perf_counter()
    load_path = str(source_path) if source_path.exists() else source
    model = model_loader.from_pretrained(
        load_path,
        quantization_config=quant_config,
        device_map={"": device},
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    print(f"Quantized in {load_seconds:.1f}s")

    # Save
    print(f"Saving to {output}...")
    save_start = time.perf_counter()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    copied_assets = copy_processor_assets(
        source_path if source_path.exists() else Path("."), output
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.1f}s")

    # Verify
    saved_files = sorted(p.name for p in output.iterdir())
    print(f"files={saved_files}")
    saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    has_quant = "quantization_config" in saved_config
    print(f"quant_config_present={has_quant}")

    if not has_quant:
        print("WARNING: quantization_config missing from saved config.json", file=sys.stderr)

    print(f"Done. Model exported to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
