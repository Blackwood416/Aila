#!/usr/bin/env python3
"""Download YOLO26 checkpoints and optional COCO val2017 assets.

All assets are kept in the repository-local, gitignored models/yolo26 tree.
Ultralytics is an optional development dependency and is never used by Aila at
runtime. Users are responsible for complying with the Ultralytics license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path


SCALES = ("n", "s", "m", "l", "x")
COCO_URLS = {
    "val2017.zip": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip":
        "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def download_checkpoints(root: Path, scales: tuple[str, ...]) -> list[dict]:
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        import ultralytics
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for checkpoint download. Install the optional "
            "conversion dependencies first."
        ) from exc

    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with working_directory(source):
        for scale in scales:
            name = f"yolo26{scale}.pt"
            resolved = Path(attempt_download_asset(name)).resolve()
            destination = (source / name).resolve()
            if resolved != destination:
                shutil.copy2(resolved, destination)
            records.append({
                "name": name,
                "path": str(destination),
                "sha256": sha256(destination),
                "ultralytics_version": ultralytics.__version__,
            })
    return records


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"unsafe path in {archive}: {member.filename}")
        bundle.extractall(destination)


def download_coco(root: Path) -> list[dict]:
    coco = root / "coco"
    coco.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for filename, url in COCO_URLS.items():
        archive = coco / filename
        if not archive.exists():
            print(f"Downloading {url} -> {archive}")
            urllib.request.urlretrieve(url, archive)
        safe_extract(archive, coco)
        records.append({"name": filename, "url": url, "sha256": sha256(archive)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("models/yolo26"))
    parser.add_argument("--scales", default=",".join(SCALES))
    parser.add_argument("--with-coco", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()

    scales = tuple(part.strip() for part in args.scales.split(",") if part.strip())
    if not scales or any(scale not in SCALES for scale in scales):
        parser.error("--scales must be a comma-separated subset of n,s,m,l,x")

    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "aila-yolo26-assets",
        "format_version": 1,
        "checkpoints": download_checkpoints(args.root, scales),
        "coco": download_coco(args.root) if args.with_coco else [],
    }
    manifest_path = args.root / "assets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
