#!/usr/bin/env python3
"""Deterministic audio and speaker-embedding comparison helpers.

The comparison is intentionally dependency-light: only the Python standard
library and NumPy are required.  WAV parsing is implemented here because the
TTS runner emits IEEE-float WAV files (format 3), which ``wave`` on older
Python versions cannot read.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class AudioError(ValueError):
    """Raised when an input WAV is malformed or contains invalid samples."""


def _finite_array(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size and not np.isfinite(values).all():
        raise AudioError(f"{label} contains non-finite samples")
    return values


def _parse_riff_wave(path: str | os.PathLike[str]) -> tuple[int, int, np.ndarray, dict[str, Any]]:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise AudioError(f"cannot read WAV '{p}': {exc}") from exc
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioError(f"'{p}' is not a RIFF/WAVE file")
    pos = 12
    fmt: bytes | None = None
    data: bytes | None = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        pos += 8
        end = pos + chunk_size
        if end > len(raw):
            raise AudioError(f"truncated WAV chunk in '{p}'")
        chunk = raw[pos:end]
        if chunk_id == b"fmt " and fmt is None:
            fmt = chunk
        elif chunk_id == b"data" and data is None:
            data = chunk
        pos = end + (chunk_size & 1)
    if fmt is None or len(fmt) < 16 or data is None:
        raise AudioError(f"WAV '{p}' lacks a complete fmt/data chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if channels <= 0 or sample_rate <= 0 or block_align <= 0 or bits <= 0:
        raise AudioError(f"invalid WAV format in '{p}'")
    # WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts at offset 24.
    if audio_format == 0xFFFE and len(fmt) >= 40:
        sub_format = struct.unpack_from("<H", fmt, 24)[0]
        audio_format = sub_format
    if audio_format not in (1, 3):
        raise AudioError(f"unsupported WAV format {audio_format} in '{p}'")
    bytes_per_sample = (bits + 7) // 8
    if bytes_per_sample <= 0 or block_align != channels * bytes_per_sample:
        raise AudioError(f"invalid WAV block alignment in '{p}'")
    if byte_rate != sample_rate * block_align:
        raise AudioError(f"invalid WAV byte rate in '{p}'")
    if len(data) % block_align != 0:
        raise AudioError(f"WAV data is not frame-aligned in '{p}'")
    frame_count = len(data) // block_align
    if frame_count <= 0:
        raise AudioError(f"WAV '{p}' contains zero frames")
    if frame_count > 0:
        usable = data[: frame_count * block_align]
        if audio_format == 3:
            if bits not in (32, 64) or block_align != channels * (bits // 8):
                raise AudioError(f"unsupported IEEE-float WAV depth {bits} in '{p}'")
            dtype = np.dtype("<f4" if bits == 32 else "<f8")
            samples = np.frombuffer(usable, dtype=dtype).reshape(frame_count, channels).mean(axis=1)
            if samples.size and np.any(np.abs(samples) > 1.000001):
                raise AudioError(f"IEEE-float WAV '{p}' contains out-of-range samples")
        else:
            if bits == 8:
                vals = np.frombuffer(usable, dtype=np.uint8).astype(np.float64)
                vals = (vals - 128.0) / 128.0
            elif bits == 16:
                vals = np.frombuffer(usable, dtype="<i2").astype(np.float64) / 32768.0
            elif bits == 24:
                b = np.frombuffer(usable, dtype=np.uint8).reshape(-1, 3)
                unsigned = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
                signed = np.where((unsigned & 0x800000) != 0, unsigned - 0x1000000, unsigned)
                vals = signed.astype(np.float64) / 8388608.0
            elif bits == 32:
                vals = np.frombuffer(usable, dtype="<i4").astype(np.float64) / 2147483648.0
            else:
                raise AudioError(f"unsupported PCM depth {bits} in '{p}'")
            samples = vals.reshape(frame_count, channels).mean(axis=1)
    samples = _finite_array(samples, str(p))
    meta = {
        "path": str(p.resolve()),
        "audioFormat": int(audio_format),
        "channels": int(channels),
        "sampleRate": int(sample_rate),
        "byteRate": int(byte_rate),
        "blockAlign": int(block_align),
        "bitsPerSample": int(bits),
        "frameCount": int(frame_count),
        "durationSeconds": float(frame_count / sample_rate),
        "bytes": int(len(raw)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return int(sample_rate), int(channels), samples, meta


def load_pcm16(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Load a WAV as finite mono float samples and return ``(x, sample_rate)``.

    The historical name is retained for callers that used PCM16; PCM8/24/32
    and IEEE float32/64 generated by the TTS runner are accepted as well.
    """
    sample_rate, _channels, samples, _meta = _parse_riff_wave(path)
    return samples, sample_rate


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 1.0
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    x, y = a[:n], b[:n]
    xc, yc = x - x.mean(), y - y.mean()
    nx, ny = float(np.linalg.norm(xc)), float(np.linalg.norm(yc))
    if nx == 0.0 or ny == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    value = float(np.dot(xc, yc) / (nx * ny))
    return float(np.clip(value, -1.0, 1.0)) if math.isfinite(value) else 0.0


def _log_mel(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    n_fft, hop, bands = 1024, 256, 80
    if samples.size == 0:
        return np.zeros((bands, 1), dtype=np.float64)
    count = max(1, int(math.ceil(max(1, samples.size - n_fft) / hop)) + 1)
    total = n_fft + (count - 1) * hop
    padded = np.pad(samples, (0, max(0, total - samples.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:count]
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    hz = np.linspace(0.0, sample_rate / 2.0, power.shape[1])
    lo, hi = 20.0, max(20.0, sample_rate / 2.0)
    mel_lo = 2595.0 * np.log10(1.0 + lo / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + hi / 700.0)
    points = np.linspace(mel_lo, mel_hi, bands + 2)
    freq = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    matrix = np.zeros((bands, power.shape[1]), dtype=np.float64)
    for i in range(bands):
        left, center, right = freq[i : i + 3]
        rising = (hz - left) / max(center - left, np.finfo(np.float64).eps)
        falling = (right - hz) / max(right - center, np.finfo(np.float64).eps)
        matrix[i] = np.maximum(0.0, np.minimum(rising, falling))
    energies = power @ matrix.T
    return np.log(np.maximum(energies, 1e-10)).T


def compare_pcm(reference: Sequence[float] | np.ndarray, candidate: Sequence[float] | np.ndarray, sample_rate: int, candidate_sample_rate: int | None = None) -> dict[str, Any]:
    """Compute deterministic waveform and log-mel metrics."""
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise AudioError("sample_rate must be a positive integer")
    if candidate_sample_rate is not None and int(candidate_sample_rate) != int(sample_rate):
        raise AudioError("sample-rate mismatch")
    a = _finite_array(np.asarray(reference), "reference")
    b = _finite_array(np.asarray(candidate), "candidate")
    if (a.size and np.any(np.abs(a) > 1.000001)) or (b.size and np.any(np.abs(b) > 1.000001)):
        raise AudioError("PCM samples must be normalized to [-1, 1]")
    n = min(a.size, b.size)
    if n:
        rms_ref = float(np.sqrt(np.mean(a[:n] * a[:n])))
        rms_cand = float(np.sqrt(np.mean(b[:n] * b[:n])))
    else:
        rms_ref = rms_cand = 0.0
    denom = max(rms_ref, 1e-12)
    mel_a, mel_b = _log_mel(a, int(sample_rate)), _log_mel(b, int(sample_rate))
    mel_frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_mae = float(np.mean(np.abs(mel_a[:, :mel_frames] - mel_b[:, :mel_frames]))) if mel_frames else 0.0
    clipping_a = float(np.mean(np.abs(a) >= 0.999)) if a.size else 0.0
    clipping_b = float(np.mean(np.abs(b) >= 0.999)) if b.size else 0.0
    silence_a = float(np.mean(np.abs(a) < 1e-4)) if a.size else 1.0
    silence_b = float(np.mean(np.abs(b) < 1e-4)) if b.size else 1.0
    return {
        "sample_rate": int(sample_rate),
        "reference_frames": int(a.size),
        "candidate_frames": int(b.size),
        "duration_delta_frames": int(abs(a.size - b.size)),
        "correlation": _correlation(a, b),
        "relative_rms_delta": float(abs(rms_cand - rms_ref) / denom),
        "reference_rms": rms_ref,
        "candidate_rms": rms_cand,
        "reference_clipping_ratio": clipping_a,
        "candidate_clipping_ratio": clipping_b,
        "clipping_ratio_delta": abs(clipping_a - clipping_b),
        "reference_silence_ratio": silence_a,
        "candidate_silence_ratio": silence_b,
        "silence_ratio_delta": abs(silence_a - silence_b),
        "log_mel_mae": mel_mae,
    }


def cosine_similarity(reference: Sequence[float], candidate: Sequence[float]) -> float:
    a, b = _finite_array(np.asarray(reference), "reference embedding"), _finite_array(np.asarray(candidate), "candidate embedding")
    if a.size != b.size or a.size == 0:
        raise AudioError("embedding dimensions must match and be non-zero")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise AudioError("embedding norm must be non-zero")
    return float(np.dot(a, b) / (na * nb))


def _extract_embedding(build_dir: str, model_dir: str, audio_path: str) -> dict[str, Any]:
    """Extract one speaker embedding in this process only.

    Callers must launch a fresh process for each oneAPI stack.  Keeping this
    routine single-stack avoids loading sycl8.dll and sycl9.dll together.
    """
    build = Path(build_dir).resolve()
    model = Path(model_dir).resolve()
    audio = Path(audio_path).resolve()
    dll_path = build / "AilaShared.dll"
    for label, path, kind in (("build", build, "dir"), ("model", model, "dir"), ("audio", audio, "file"), ("AilaShared.dll", dll_path, "file")):
        ok = path.is_dir() if kind == "dir" else path.is_file()
        if not ok:
            raise AudioError(f"{label} {kind} not found: {path}")
    os.environ["PATH"] = str(build) + os.pathsep + os.environ.get("PATH", "")
    dll_cookie = os.add_dll_directory(str(build)) if hasattr(os, "add_dll_directory") else None
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from test_api import AilaAPI  # existing, authoritative ctypes wrapper

        api = AilaAPI(str(dll_path))
        engine = api.aila_engine_create()
        if not engine:
            raise AudioError("aila_engine_create returned NULL")
        try:
            rc = api.aila_engine_init(engine, os.fsencode(model), 4096)
            if rc != 0:
                msg = api.aila_last_error_message(engine)
                raise AudioError(f"aila_engine_init failed ({rc}): {(msg or b'').decode('utf-8', 'replace')}")
            ptr = ctypes.POINTER(ctypes.c_float)()
            dim = ctypes.c_int()
            rc = api.aila_extract_speaker_embedding(engine, os.fsencode(audio), ctypes.byref(ptr), ctypes.byref(dim))
            if rc != 0 or not ptr or dim.value <= 0:
                msg = api.aila_last_error_message(engine)
                raise AudioError(f"aila_extract_speaker_embedding failed ({rc}, dim={dim.value}): {(msg or b'').decode('utf-8', 'replace')}")
            try:
                values = np.ctypeslib.as_array(ptr, shape=(dim.value,)).astype(np.float64, copy=True)
            finally:
                api.aila_free_samples(ptr)
        finally:
            api.aila_engine_destroy(engine)
    finally:
        if dll_cookie is not None:
            dll_cookie.close()
    values = _finite_array(values, "speaker embedding")
    norm = float(np.linalg.norm(values))
    if values.size == 0 or norm == 0.0:
        raise AudioError("speaker embedding is empty or has zero norm")
    raw = np.asarray(values, dtype="<f8").tobytes()
    return {"dimension": int(values.size), "norm": norm, "sha256": hashlib.sha256(raw).hexdigest(), "values": values.tolist()}


def _self_test() -> None:
    sr = 24000
    t = np.arange(2400) / sr
    sine = np.sin(2 * np.pi * 440 * t)
    exact = compare_pcm(sine, sine.copy(), sr)
    assert exact["correlation"] > 0.999999
    assert exact["relative_rms_delta"] < 1e-9
    assert exact["duration_delta_frames"] == 0
    assert exact["log_mel_mae"] < 1e-9
    amp = compare_pcm(sine, 0.5 * sine, sr)
    assert math.isfinite(amp["correlation"]) and amp["relative_rms_delta"] > 0.4
    short = compare_pcm(sine[:3], sine[:2], sr)
    assert short["duration_delta_frames"] == 1
    silence = compare_pcm(np.zeros(8), np.ones(8), sr)
    assert silence["correlation"] == 0.0 and math.isfinite(silence["log_mel_mae"])
    clipped = compare_pcm(np.ones(8) * 0.5, np.ones(8), sr)
    assert clipped["candidate_clipping_ratio"] == 1.0
    try:
        compare_pcm(sine, sine, sr, sr + 1)
    except AudioError:
        pass
    else:
        raise AssertionError("sample-rate mismatch not rejected")
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-12
    assert all(math.isfinite(float(v)) for v in exact.values() if isinstance(v, (int, float)))
    # Stereo PCM16 loading, generated in a temporary owned directory.
    with tempfile.TemporaryDirectory(prefix="aila-audio-selftest-") as td:
        wav_path = Path(td) / "stereo.wav"
        frames = (np.column_stack((sine[:100], -sine[:100])) * 32767).astype("<i2").tobytes()
        with wave.open(str(wav_path), "wb") as out:
            out.setnchannels(2); out.setsampwidth(2); out.setframerate(sr); out.writeframes(frames)
        loaded, loaded_sr = load_pcm16(wav_path)
        assert loaded_sr == sr and loaded.size == 100 and np.max(np.abs(loaded)) < 1e-12
        empty_path = Path(td) / "empty.wav"
        with wave.open(str(empty_path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(sr)
        try:
            load_pcm16(empty_path)
        except AudioError:
            pass
        else:
            raise AssertionError("zero-frame WAV not rejected")
        huge_path = Path(td) / "huge-float.wav"
        float_data = struct.pack("<ff", 1e30, -1e30)
        fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
        huge_path.write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(float_data)) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(float_data)) + float_data)
        try:
            load_pcm16(huge_path)
        except AudioError:
            pass
        else:
            raise AssertionError("out-of-range IEEE-float WAV not rejected")
    print("compare_audio self-test: PASS")


def _compare_files(reference_path: str, candidate_path: str) -> dict[str, Any]:
    sr_a, _ca, a, meta_a = _parse_riff_wave(reference_path)
    sr_b, _cb, b, meta_b = _parse_riff_wave(candidate_path)
    metrics = compare_pcm(a, b, sr_a, sr_b)
    if not all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float, np.integer, np.floating))):
        raise AudioError("audio metrics contain non-finite values")
    return {"reference": meta_a, "candidate": meta_b, "metrics": metrics}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--output", help="atomically write comparison JSON")
    parser.add_argument("--extract-embedding", action="store_true")
    parser.add_argument("--build-dir")
    parser.add_argument("--model")
    parser.add_argument("--audio")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test(); return 0
    if args.extract_embedding:
        if not args.build_dir or not args.model or not args.audio or not args.output:
            parser.error("--extract-embedding requires --build-dir, --model, --audio, and --output")
        result = _extract_embedding(args.build_dir, args.model, args.audio)
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=str(out.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh: fh.write(payload)
            os.replace(temp, out)
        finally:
            if os.path.exists(temp): os.unlink(temp)
        return 0
    if not args.reference or not args.candidate:
        parser.error("--reference and --candidate are required unless --self-test is used")
    result = _compare_files(args.reference, args.candidate)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=str(out.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh: fh.write(payload)
            os.replace(temp, out)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AudioError, OSError, ValueError) as exc:
        print(f"compare_audio: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
