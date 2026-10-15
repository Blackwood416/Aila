# Q4E Nested-Scale 固定 256 特化：本轮工作交接

> 当前接手状态：nested-256 反序复核已保留，随后 H4 初始驻留和 staging 并行拷贝分别提交为 `c5d5805`、`3962519`，正式 Worker/Server 已同步后者并通过验收。最新证据与后续候选见 [专家供给优化交接](qwen38-expert-supply-handoff-2026-10-09.md)。下文为 nested 实验历史。

> 接手更新：Server 构建阻塞已在新 PowerShell 会话解开，A1→B1→B2→A2 生产筛选与反序复核已完成；14K 完整请求两对分别缩短 2.62%/2.18%，中位数缩短 2.40%，四次输出/工作量一致。用户决定保留，源码已提交为 `98cc037`，官方 Worker/Server 双构建已同步该身份；PI/续轮/decode 没有一致收益。最新状态与二进制身份见 [Server 筛选与反序报告](qwen38-nested-div256-screen-2026-10-09.md)。下文保留原交接时的历史状态，§7 的未提交回退方法已经过时。

日期：2026-10-09。承接 `docs/plans/qwen38-optimization-handoff-2026-10-09.md`（基点 `main@88be93c`）。本轮只做了**候选 A**（nested scale 运行时整数除法消除）；Worker 侧已实现并通过位一致微测，**卡在 Server 重建的本机 CMake/Ninja 环境异常上**。工作区当前为**未提交**状态，全部改动可用 `git diff` 查看， revert 方法见 §6。

## 1. 一句话状态

- 专家权重 nested-scale 反量化的 64 位运行时除法已在 MoeExpertOps GPU kernel 中编译期消除（`index / blocksize → index >> 8`，仅当调用方声明本模型固定几何 256 时走特化实例化，其他调用方逐字节不变）。
- 微测：decode 完整专家链墙钟 **-4.6%**，prefill 80 行完整链墙钟 **-4.9%**（kernel -5.2%），generic 与 fixed 输出**位一致**，回归测试通过。
- 预期整轮收益约 1–2%（14K 请求 74.8s → 约 73–74s 量级；这是推算，不是测量）。
- **阻塞点**：`tools/server/build.ps1` 在本机反复生成缺少 `CMakeFiles/rules.ninja` 的 Server 构建树，导致 AilaServer.exe 无法重建；当前 Worker（候选版，build ID `88be93c…`）与现存 Server（基线版，stamp `a83cd80…`）握手不匹配，**不能直接配对跑生产测量**。基线 Server/Worker 已归档（§5）。

## 2. 源码改动（全部未提交）

### 2.1 `src/ops/MoeExpertOps.cpp`
- `load_expert_absmax`（原 ~404）模板参数扩为 `<bool UseNested, bool FixedNested256 = false>`；`if constexpr (FixedNested256)` 时用 `index >> 8`，否则保留原 `index / static_cast<size_t>(nested.blocksize)`。两个 f32 FMA 舍入步骤一字未动。
- 15 个 kernel launcher 模板追加 `bool FixedNested256`（追加在参数表安全位置，`Resident`/`Compact`/`WorkGroup` 既有含义不变）；~41 处 helper 调用改为双参数。
- 15 个公开 entry（`moe_resident_gate_up/down(_q8_64)`、`moe_decode_gate_up_swiglu_grouped`、`moe_decode_down_to_rank_f32`、`moe_decode_down_weighted_accumulate`、`moe_prefill_gate_up_rows/down_rows`、`moe_expert_major_prefill_gate_up/down(_2token/_4token)`）追加尾参 `bool nested_fixed256 = false`（默认值只写在 hpp，cpp 定义不带默认，否则 redefinition 编译错误），nested 分支据此双实例化 dispatch。
- **未改**：`MoeGroupedXmx.cpp`（`GROUPED_XMX_MIN_ROWS=0` 生产不可达）、legacy `Qwen4ExpExpertCache` 主机侧、`Qwen4ExpExpertReader`（host 循环本已按 run 摊销除法）、MTP 与 DiffusionGemma 相关调用（保持默认 `false`，行为逐字节不变）。

### 2.2 `src/ops/MoeExpertOps.hpp`
对应 14 处声明追加 `bool nested_fixed256 = false`。inline 包装重载未加参（默认走 generic）。

### 2.3 `src/models/Qwen4ExpBnb4Backend.cpp`（生产热路径传参）
- `RunResidentDecode(count)` 的 adaptive 块（~1040，生产 decode 路径）：新增 `const bool nested_fixed256 = true`，传入 hot/staging 4 处调用。依据：该路径描述符只出自 `Qwen4ExpAdaptiveCache::GetResidentSlotDev/GetStagingSlotDev → MakeResidentSlot`（`Qwen4ExpAdaptiveCache.cpp:327` 硬编码 256，且 :318 对 `!=256` 直接报错）。
- `RunGroupedExperts`（~1351，adaptive prefill 的 gate/up+down）：`nested_fixed256 = use_adaptive_pipeline_`，传入两处（down 处注意参数序 `nested, np, nq, /*compact_output=*/false, nested_fixed256`）。
- 已核实为**非自适应死路径**、有意保持默认 `false`：1161 起的 lease 版 `RunResidentDecode` 重载、1925–1946 的 `resident_epoch` 块（其创建条件 `!use_adaptive_pipeline_ && resident_decode_mode_==2 && count==1`）、`moe_resident_gate_up/down(_q8)` legacy 站点、MTP draft。本轮 Workflow 的 coverage 代理独立复核了这一点。

### 2.4 微测工具（`perf/qwen4exp/`）
- `q4e_moe_bench.cpp`：新增 `--mode 2`（generic/fixed 每样本交错 A/B + 两路输出缓冲 + D2H 后 `memcmp` 位一致断言；曾把 device USM 直接 memcmp 导致段错误，已修）。`--mode 0` 行为与原版一致。
- `q4e_prefill_supply_bench.cpp`：`run_pacing` 增加 `fixed` 维度与 `--pacingab` 选项；generic/fixed 交错计时；新增 “Fixed-256 specialization changed gate/up|down bits” 位一致门禁；`--pacing` 原行为保持（内部 execute 显式传 fixed=false）。
- 补丁脚本留存可重放：`tmp/q4e-nested-div-20261009/{patch_ops.py,patch_moe_bench.py,patch_supply_bench.py}`。

## 3. 测量证据（本机，驱动 8724，A770，一次一个 GPU 压力目标）

`[MEASURED]` decode 链（48 层 × 10 专家，真实几何，nested=1，20 samples × 3 repeats，交错）：

| 指标 | generic | fixed256 | Δ |
|---|---:|---:|---:|
| wall median | 11.317 ms | 10.799 ms | **-4.6%** |
| gate device sum | 6.963 | 6.618 | -5.0% |
| down device sum | 3.942 | 3.757 | -4.7% |
| bitwise h / out | — | **1 / 1** | 位一致 |

修改前基线（同 exe 旧版）wall=11.400、streaming floor 427.3 GB/s——generic 臂与基线一致，重构本身零回归。记录：`tmp/q4e-nested-div-20261009/bench-base-nested1.log`、`bench-ab-nested1.log`。

`[MEASURED]` prefill 完整链（pacing bench，rows=80，batch=32，256 专家 648.8 MB H2D+kernel）：wall 136.76/136.79/136.78 → **130.31/129.997/130.27 ms（-4.9%）**，kernel 127.7→121.1（-5.2%）；rows=8 kernel -3.6%、wall 约持平（供给/启动受限）。位一致门禁通过。`pacing-base.log`、`pacing-ab.log`。

`[MEASURED]` codegen 独立验证（Workflow `q4e-nested-div-evidence`，产出物在 `tmp/q4e-nested-div-20261009/codegen/`）：部署的 AilaWorker.exe 提取 44 个 SPIR-V 模块，`mod11`（MoeExpertOps TU）内层循环含 **56 处 `UDiv i64`**；`>>8` 对照构建 **55→0**、`SDiv i32` 不变；ocloc GEN ISA（dg2-g10-a0）无硬件 div，i64 除法是软件倒数/乘高修正序列；nested 相对非 nested kernel 指令 **+39.4%**。注意遗留未处理：`byte_offset / (blocksize/2)` 的 32 位运行时除法仍在（36 处，nested/非 nested 都有；`blocksize=64` 是字面实参但 launcher 形参运行时），以及 `if (blocksize==64)` 实分支使 else 臂翻倍 scale 代码——留作后续小候选，不属于本轮最小改动。

`[CORRECTNESS]` `AilaQwen4ExpGpuQ8Tests.exe` 全过（含 nested=0/1 独立 oracle、block32/参数拒绝）；两个 bench 的 CPU 参考误差 0.000。日志 `q8-tests.log`。

`[HYPOTHESIS]` 整轮预期 1–2%：74.8s 的 14K 请求中专家 kernel 供给与 staging/DMA 重叠，-5% kernel 不会全额转成墙钟；未测，不得当成绩引用。

## 4. 当前二进制身份

```text
build-refactor/AilaWorker.exe            421d6a5c…  = 候选（含特化），内嵌 stamp
                                          aila-0.2.0-abi1-88be93cf79cc…（已从 exe 字节验证）
build-refactor-server/AilaServer.exe     020c8a89…  = 基线（stamp a83cd80a…）→ 与候选 Worker 握手不匹配，勿混跑
tmp/q4e-nested-div-20261009/base/AilaWorker.exe 8ce1efd9…（=10-09 交接基线）
tmp/q4e-nested-div-20261009/base/AilaServer.exe 020c8a89…
```
build-refactor 内其余 bench/test exe 均已按新 lib 重链。Server 树 `build-refactor-server` 的 CMake 状态已碎（见 §5），但旧 `AilaServer.exe` 与 objects 仍在。

## 5. 唯一阻塞：Server 构建树生成不出 rules.ninja

现象：`pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies` 的 configure 步骤报成功（exit 0、`Build files have been written`、新目录得到 92 KB 的 `build.ninja`，第 35 行 `include CMakeFiles\rules.ninja`），但 **`CMakeFiles/rules.ninja` 从未被写出** → `cmake --build` 的 ninja 立即失败。时间线事实：本会话第一次运行脚本（16:30）就已是该错误；此后 4 个全新 `-BuildRoot`（server4/6/7 等，含 `--fresh` 与手工全参数各试遍）全部复现。16:41 我对旧 `build-refactor-server` 做的缓存清理发生在首错之后，不是肇因（当时只删了 CMakeCache/CMakeFiles 生成物，未动 exe/tmp 记录）。

已排除/已知：
- cmake：脚本经 VsDevCmd 后 `Get-Command cmake.exe` = `C:/Program Files/Microsoft Visual Studio/18/Community/…/cmake.exe`（4.1.2-msvc8）。**同一个 cmake 在 (a) VS bat 环境里的最小项目（`--fresh`，正常写出 rules.ninja）和 (b) 本仓库 worker 树 `build-refactor`（15:47 “Re-running CMake” 正常写出 93 KB rules.ninja）都工作**，唯独 tools/server 项目在本会话所有尝试中不写。
- 用户提示的 oneAPI 先行初始化（`. perf/PerfCommon.ps1; Set-AilaProcessEnvironment (Get-AilaOneApiStackEnvironment …)` 后再调脚本）也试过，同样缺文件。
- 两树 `CMAKE_MAKE_PROGRAM` 同为 `Python313/Scripts/ninja.exe`（1.13.0.git.kitware.jobserver-pipe-1）——worker 正常 server 异常，该差异不能单独解释，但**这是与会话前 13:40 成功构建之间唯一找到的可疑变量**（最小项目在 VsDevCmd 环境里 cmake 自选的可能是 VS 自带 ninja：`C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe`）。
- 磁盘空间、目录权限、临时文件残留（无 *.tmp、无 AV 重命名迹象）、CMakeLists 中的 showIncludes 探测函数（server6 手工全参数 configure exit 0 同缺）均已查。

建议接手探针（按成本排序，勿改 build.ps1/握手逻辑本身）：
1. 在**全新 pwsh（不经本 Git Bash 的 PATH）**直接跑 `tools/server/build.ps1 -SkipDependencies -BuildRoot <新目录>`；若成功则是本会话 PATH/ninja 污染。
2. 或 `PATH` 前置 VS 的 `Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja`（让脚本 `Get-Command ninja.exe` 命中 VS ninja）后重跑。
3. 或给脚本 configure 传 `-DCMAKE_MAKE_PROGRAM=<VS ninja>` 等价物：脚本自身从 PATH 取，仍可用 PATH 顺序实现。
4. 若仍失败：在最小复现（§资产 `cmn2` 方法：VsDevCmd bat + `--fresh` 最小 CMakeLists）与真实项目间二分（把 tools/server/CMakeLists.txt 逐段裁剪）——但先做 1–3。
5. 绝对禁止：改 `server-build-info.json`/`build_info.json` 伪造握手 ID（原交接 §3.3/§12 已两次教训）。Server 配好后内嵌 stamp 必须等于 Worker 的 `aila-0.2.0-abi1-88be93cf79cc…`（脚本从 `build-refactor/build_info.json` 的 git.fullCommit 推导；该文件已由 tmp/build.ps1 以 88be93c 重写，接手时先核对其 schemaVersion=2 与 fullCommit 完整 40 位）。

## 6. 解锁后的剩余步骤（顺序执行，勿并行 GPU 任务）

1. `pwsh tools/server/build.ps1 -SkipDependencies` 成功 → 校验两 exe 内嵌/记录的 build ID 一致；把候选对归档 `tmp/q4e-nested-div-20261009/cand/`（含 sha256）。
2. 生产筛选（runner：`python tmp/q4e-resident-speed-20261009/run.py <新tag> --agent`；probe 固定读取 `build-refactor/-server` 路径，需**成对换装**）：
   - 先装基线对（`tmp/.../base/`）跑一次 drift/基线 tag；再装候选对跑 cand tag。比对 14K 完整请求、首 PI、续轮、启动。
   - 判据沿用原交接 §11.5：整轮 ≥~5% 才补反序 + 相关功能检查；1–5% 之间方向一致且稳定属"小幅真实收益"，是否保留由用户拍板；低于噪声或无整轮收益 → 按 §7 revert 归档，保留本轮全部测量资产。
   - 注意：本轮改动位一致，输出 hash 应与基线完全相同，验证成本远低于候选 B/C。
3. `[TOOLCHAIN]` 提交/构建身份：若保留，正式提交后按官方双构建流程重编（tmp helper 可用），Worker/Server 身份同步变化。

## 7. 回退方法

```powershell
git restore src/ops/MoeExpertOps.cpp src/ops/MoeExpertOps.hpp src/models/Qwen4ExpBnb4Backend.cpp perf/qwen4exp/q4e_moe_bench.cpp perf/qwen4exp/q4e_prefill_supply_bench.cpp
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor -Jobs 8   # + Server 重编后恢复基线对
```
bench 与补丁脚本副本已留存 tmp，重放即得本轮状态。

## 8. 本轮资产索引（`tmp/q4e-nested-div-20261009/`）

| 路径 | 内容 |
|---|---|
| `base/` | 哈希核验过的基线 Worker+Server 对 |
| `bench-base-nested1.log` / `bench-ab-nested1.log` | decode 链基线与交错 A/B（含 bitwise 1/1） |
| `pacing-base.log` / `pacing-ab.log` | prefill 完整链基线与交错 A/B |
| `q8-tests.log`、`smoke-ab.log` | 回归与冒烟 |
| `codegen/` | Workflow 产物：44 个 SPIR-V 模块提取、mod11 审计 json、ocloc GEN asm、div→shift 对照审计 |
| `patch_*.py`、`cfg*.log`、`server-build*.log`、`cdb*.log` | 补丁脚本与 Server 故障全程日志 |
| Workflow run | `.qoder/sessions/5b450e43-…/workflows/runs/wf_6ed40779-ea0/output.json`（codegen/coverage 双确认；harness/bc_screen 两代理因 schema 未交回结果，其问题已由主会话自查覆盖） |

## 9. 给接手 agent 的边界提醒

- 原交接文档的纪律全部有效：一次一个候选、一次一个 GPU 压力目标、失败归档不堆积开关、正式测量关诊断、启动+首用合并报告。
- 本轮微测数字是**kernel 链级**成绩，禁止当作 Server 整轮或 decode tok/s 引用；当前 Server 整轮基线仍是原交接 §6 的 74.80 s / 4.16 s / ~13 tok/s。
- `nested_fixed256` 默认 `false` 是对其他模型/legacy 测试的兼容保证；任何调用方传 `true` 前必须确认描述符出自已校验 256 几何的缓存。
- 遗留小机会（未验证、非本轮范围）：`byte_offset / (blocksize/2)` 与 launcher `blocksize` 形参的模板常量化；`if (blocksize==64)` else 死臂的实例化拆分；候选 B（H4 profile 未进 `InitializeResidency`，证据已由原交接 §4.4 给出）仍是下一个最大不确定收益点。
