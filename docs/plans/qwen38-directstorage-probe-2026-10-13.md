# DirectStorage 独立实验：NVMe 读 vs 现方案（2026-10-13）

问题：Windows 上能否用 DirectStorage 直接从 NVMe 读数据，比现有方案（mmap 缺页 / 显式多线程
ReadFile）更有性能优势？结论：**机制可行、获取简单，但在 Aila 的实际加载形态（单文件顺序读）
下吞吐与 16 线程 ReadFile 持平（+2~3%），只省 CPU；不值得为此接入**。

## 1. SDK 怎么搞（无需系统组件）

- 本机**没有**系统级 DirectStorage 运行时（全盘无 `dstorage.dll`），Windows SDK（D 盘）也只有
  `d3d12.h`、没有 `dstorage.h`。
- 官方分发渠道 = **NuGet 包** `Microsoft.Direct3D.DirectStorage`（本次用 1.3.0）：
  `https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/1.3.0/...nupkg`
  （nupkg = zip，解压即得）：
  - `native/include/dstorage.h` + `dstorageerr.h`
  - `native/lib/x64/dstorage.lib`
  - `native/bin/x64/dstorage.dll` + `dstoragecore.dll`（`distributable_files.txt` 明确可再分发）
- 无需安装、无需 D3D12 设备即可做 CPU 内存目标的读（`DSTORAGE_QUEUE_DESC.Device = nullptr`）。
  已解压到 `tmp/q4e-directstorage/sdk/`，探针为 `tmp/q4e-directstorage/ds_probe.cpp`。

## 2. 探针与口径

- 两臂读**互不重叠的区域**（DS 读 [base, base+bytes)，ReadFile 读 [base+bytes, base+2*bytes)），
  避免互相暖缓存；两臂都计入 CPU 时间（`GetProcessTimes`）。
- DS：`factory->OpenFile` + `CreateQueue(capacity=depth)` + 每批 `EnqueueRequest/EnqueueStatus/
  Submit`，双缓冲深度 2×depth，1 MiB 请求。
- ReadFile：16 线程、4 MiB 块、每线程独立句柄。

## 3. 结果（C: = LASTING NVMe，模型所在盘）

17GB 测试文件（12 个模型分片拼接），每次 4 GiB 冷区：

| 配置 | DirectStorage | ReadFile ×16 | DS CPU / RF CPU |
|---|---:|---:|---:|
| depth=128, 1 MiB 请求 | **2.80 GB/s**（1.43 s） | 2.73 GB/s（1.47 s） | 0.20 / 0.56 cpu-s/GiB |
| depth=512, 1 MiB 请求 | **2.82 GB/s**（1.42 s） | 2.78 GB/s（1.44 s） | 0.27 / 0.45 cpu-s/GiB |
| 模型分片 0.4-0.75 GiB（1 MiB 请求） | 2.33 GB/s | 1.95-2.03 GB/s | 0.29-0.39 / 0.67-1.41 |

- **单文件顺序读的上限约 2.8-3.2 GB/s**（与裸 `ReadFile` 16 线程一致），DS 只快 2-3%，CPU 省一半。
- 参照：**16 个不同文件并行读可达 8.6-9.8 GB/s**（`nvme_read_bench.py`）；瓶颈是"单文件/单流"，
  不是 API。Aila 的 arena fill 正是"逐层单文件顺序读"，所以吃不到那个 8.6。
- 坑 1：第一次把测试文件建在 **E:（USB 盘，Disk 2）**上，两臂都只有 0.40 GB/s——做 IO 实验先确认卷。
- 坑 2：队列容量 `<128` 非法（64 → `E_DSTORAGE_INVALID_QUEUE_CAPACITY` 0x89240003）；128/512 正常。

## 4. 结论与建议

1. **不接入** DirectStorage 做模型加载：对逐层单文件读，DS ≈ 现方案吞吐，收益只有 CPU（一次性
   加载里无关紧要），却要新增一个可再分发 DLL 依赖。现有 `AILA_Q4E_ARENA_READ_THREADS`（16 线程
   显式读，冷缓存 116.6→63.8 s）已经把 mmap 的 0.66 GB/s 提到 ~2.8 GB/s，路径相同。
2. 值得保留的洞见：**多文件并行**才能把盘吃满（8.6-9.8 GB/s）。若将来把 loader 改成"多分片并行
   预读 + 按层重排"，那才是数量级的启动收益；届时 DS 的深队列与低 CPU 会更有价值（且它天然
   bypass 系统缓存，有助于控制峰值内存）。
3. "DS 直读 VRAM"（GPU 目标）仍需要 D3D12 资源 + 与 L0/USM 的 interop（未验证），且两跳都是
   ~10 GB/s 量级，收益主要在省 RAM——维持既有评估，优先级低。

## 5. 复现

```powershell
# SDK
Invoke-WebRequest https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/1.3.0/microsoft.direct3d.directstorage.1.3.0.nupkg -OutFile dstorage.nupkg.zip
Expand-Archive dstorage.nupkg.zip sdk
# 构建（cl + dstorage.lib；运行时把 bin/x64 的两个 DLL 放到 exe 旁）
cmd /c tmp\q4e-directstorage\build.cmd
# 跑：<file> <gib> <depth> <request_kib> <threads> <base_gib>
tmp\q4e-directstorage\ds_probe.exe C:\QwenModels\_ds_test_16g.bin 4 128 1024 16 0
```

遗留：`C:\QwenModels\_ds_test_16g.bin`（13.0 GB，本次实验的拼接文件）仍在 C:，确认后可删。
