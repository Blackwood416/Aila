# YOLO26 preparation and validation / YOLO26 准备与验证

Aila 0.2.0 supports the official YOLO26 detection topology in n/s/m/l/x scales.
The first release is deliberately strict: batch 1, 640×640, strides 8/16/32,
`reg_max=1`, and the default end-to-end one-to-one NMS-free head. Custom class
counts and UTF-8 class names are supported when the topology is otherwise
unchanged. P2/P6, custom backbones, segmentation, pose, OBB, classification,
YOLOE, training, tracking, video batching, and dynamic input shapes are rejected.

Aila 0.2.0 支持官方 YOLO26 detection 的 n/s/m/l/x scale。首版固定 batch=1、
640×640、stride 8/16/32、`reg_max=1` 和 one-to-one NMS-free head。同拓扑下
可使用自定义类别数与 UTF-8 类名；其他任务或拓扑会被严格拒绝。

## Prepare assets / 准备资产

The scripts place all external data below the ignored `models/yolo26/` directory:

```powershell
python tools/prepare_yolo26.py
# Optional, large download for accuracy validation:
python tools/prepare_yolo26.py --with-coco

python -m venv models/yolo26/.venv
models/yolo26/.venv/Scripts/pip install ultralytics safetensors
foreach ($scale in 'n','s','m','l','x') {
  models/yolo26/.venv/Scripts/python tools/export_yolo26.py `
    "models/yolo26/source/yolo26$scale.pt" "models/yolo26/aila/$scale"
}
```

The converter validates task, scale, exact 24-layer topology, end-to-end mode,
regression width, strides, and contiguous class IDs. It fuses Conv+BN, discards
the one-to-many/training-only state, stores FP16 tensors under Aila-owned stable
names, and writes `model.safetensors`, `config.json`, and `manifest.json`.
Runtime loading revalidates the config and its manifest/topology hash.

Ultralytics is an optional conversion-time dependency and is neither linked into
the runtime nor included in Aila releases. Check the current
[Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
and the checkpoint/dataset terms for your use. Aila releases do not contain
Ultralytics, official weights, COCO, Python, InferRef, or generated traces.

## Run / 运行

```powershell
build/Aila.exe --model models/yolo26/aila/n --detect image.jpg
build/Aila.exe --model models/yolo26/aila/m --detect image.png `
  --conf 0.4 --max-det 100 --save-detect annotated.png
build/Aila.exe --model models/yolo26/aila/s --detect image.jpg --bench
```

Successful stdout is one stable JSON line containing original oriented image
dimensions, scale, class count, and confidence-sorted detections. `--save-detect`
writes an original-resolution PNG. JPEG/PNG EXIF orientation is applied before
letterboxing and annotation. Preprocessing uses the official centered letterbox
rule, bilinear resize, RGB/NCHW normalization, and padding value 114.

The C++ API exposes `detect_file`, `detect_encoded`, and `detect_pixels`; see
[C_API.md](C_API.md) for the ABI-safe equivalents. Videos should call the raw
pixel API once per frame.

## InferRef correctness workflow

Use a fresh output directory for every adapter run. Do not assign a fabricated
contract to YOLO regions; InferRef 0.7.1 has no built-in YOLO26 contract.

```powershell
inferref doctor --device xpu --verify-plugins
inferref agent capabilities --json
inferref trace models/yolo26/trace_yolo26n.py `
  --output models/yolo26/inferref/traces/yolo26n-full --device xpu
inferref agent context models/yolo26/inferref/traces/yolo26n-full
inferref region create models/yolo26/inferref/traces/yolo26n-full `
  --name conv_silu_layer0 --module model.0 --semantic ConvSiLU --engine-op yolo26_conv_silu
inferref testcase extract models/yolo26/inferref/traces/yolo26n-full `
  --region conv_silu_layer0 --output models/yolo26/inferref/cases/conv_silu_layer0
inferref agent run models/yolo26/inferref/cases/conv_silu_layer0 `
  --adapter tests/inferref/yolo26.adapter.json `
  --runs-dir models/yolo26/inferref/runs --rtol 1e-2 --atol 1e-3
```

Create analogous named regions for depthwise Conv, C3k2, SPPF, C2PSA, raw Detect,
decode/top-k, and full forward. Use `rtol=1e-4, atol=1e-6` for FP32 softmax,
`rtol=2e-2, atol=1e-3` for C2PSA, and `rtol=1e-2, atol=1e-3` for other blocks.
Final detections require semantic comparison of class, ordering, score, box IoU,
and pixel error; an elementwise tensor pass alone is insufficient.

For a fast semantic check across all five scales:

```powershell
models/yolo26/.venv/Scripts/python tools/compare_yolo26_image.py `
  --aila build/Aila.exe --image models/yolo26/bus.jpg
```

For COCO val2017, compare each converted scale with the same checkpoint in
Ultralytics FP32. The merge gate is no more than 0.5 absolute AP loss in box
mAP50-95. The validator fixes both engines to square `640x640`, batch-one
preprocessing and keeps one Aila worker alive for the full scale:

```powershell
models/yolo26/.venv/Scripts/pip install pycocotools
models/yolo26/.venv/Scripts/python tools/validate_yolo26_coco.py `
  --build build-yolo26 --scales n,s,m,l,x
```

Use `--limit 100` only as a wiring smoke test; it is not an accuracy gate.
Record A770 device/wall/full-pipeline median, p10/p90, peak device
memory, driver/oneAPI/oneDNN versions, and the PyTorch XPU baseline. Performance
is reported rather than used as a hard merge gate.

## Validation snapshot (2026-08-20)

The implementation was validated on an Intel Arc A770 with oneAPI 2026.1 and
oneDNN 3.11.2. InferRef 0.7.1 reported PASS with Python 3.13.11, NumPy 2.2.6,
and PyTorch 2.11.0+xpu. COCO numbers below use all 5,000 val2017 images,
`conf=0.001`, `max_det=300`, fixed square 640x640 preprocessing, and batch 1.

| Scale | Aila FP16 mAP50-95 | Ultralytics FP32 | Absolute loss | Gate |
|---|---:|---:|---:|---:|
| n | 0.402502 | 0.402649 | 0.000147 | pass |
| s | 0.477226 | 0.477181 | -0.000045 | pass |
| m | 0.524844 | 0.524909 | 0.000066 | pass |
| l | 0.540732 | 0.540862 | 0.000130 | pass |
| x | 0.568524 | 0.568507 | -0.000018 | pass |

The same A770, using five measured iterations after one warmup on `bus.jpg`,
reported the following Aila medians. Full-pipeline time includes WIC decode,
letterbox, device execution, result transfer, and postprocessing.

| Scale | Full pipeline (ms) | Device wall (ms) | Peak device bytes |
|---|---:|---:|---:|
| n | 42.24 | 6.08 | 121,424,168 |
| s | 42.94 | 6.75 | 248,342,808 |
| m | 46.21 | 8.72 | 492,798,328 |
| l | 48.39 | 11.82 | 636,642,168 |
| x | 55.84 | 17.69 | 1,024,017,528 |
