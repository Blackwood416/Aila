# E0 探针：真实 kernel 上的设备侧参数表 ABI（2026-10-16）

状态：**源码完成、已构建、已通过 CPU 冒烟验证**（全程未占用 GPU；GPU 运行由主代理调度）。
上游：`docs/plans/qwen38-graph-experiment-roadmap-2026-10-16.md` §3.3（参数表 ABI 落地清单）与 §4 E0/E3。
生产源码零改动：本探针与运行脚本只落在 `tmp/q4e-structural-20261016/`，外加本文件。

## 1. 目的

用**真实 decode 热路径 kernel**回答 roadmap 的两个门槛问题：

1. 把动态指针/标量从 kernel 参数搬到"固定地址 USM 参数表"后，输出是否仍与生产 kernel **逐位一致**（风险 R2）？
2. 参数表 ABI 的**提交成本**相对直接提交差多少（Variant B 是否值得做）？

## 2. 被测 kernel 与实现

被测对象（生产路径、未修改）：`ops::hc_decode_down_inject`
（`src/ops/HyperConnectionOps.cpp:453` `hc_decode_down_inject_impl<false, 256>`；Qwen4Exp decode
hyper-connection "down + inject" GEMV，每 SG16 一行；生产账本里最大的 decode 块 hc 6.6 ms/token）。

三臂共用同一个 in-order queue（`Context`）与同一批缓冲区：

| 臂 | 内容 | 动态参数来源 |
|---|---|---|
| **A 直接（参照）** | 直接调用 AilaLib 生产入口 `ops::hc_decode_down_inject` | kernel 参数 |
| **B 参数表 + SYCL** | 表 ABI kernel（逐行副本），普通 `q.submit` | 固定地址 128 B Host USM 表 |
| **C 参数表 + Limited Graph** | 录制一次，之后每次重放前改写表内容 | 同上 |

- 表 ABI（`probe_paramtable.cpp:67`，`alignas(64)`、128 B）：scalar 头
  `version/sequence/lowrank/group_dim/groups/rows`（字节 0–24），指针块 `N/w_down/w_inject/act/inj/echo`
  （32–72）。kernel 签名只有 `const ParamTable*`，表字段全部 volatile 读（`probe_paramtable.cpp:89`），
  与 A770-Graph-Experiment 的 Executor ABI v2 同构（该仓库 `phase2/parameters.hpp`）。
- kernel 体是 `hc_decode_down_inject_impl<false,256>` 的逐行副本，**只改"参数从哪里来"**：
FMA 顺序、`reduce_over_group`、`bf16` 舍入点、vec8 对齐/非对齐两条分支都未动。
- 表内 `echo` 指针把本条 kernel 实际读到的 `version/sequence` 回写设备内存，用于**证明重放确实在读
  live 表**（否则 bit-exact 可能只是"表没生效、仍在用旧参数"）。
- 形状（默认 3 个）：
  - 0 `lowrank=320 group_dim=2560 groups=4`（Qwen4Exp-0.8B decode 真实形状，k_dim=10240，21 WGs）
  - 1 同形状但 `w_down` 偏移 1 个 bf16（强制走非 vec8 对齐分支）
  - 2 `lowrank=64 group_dim=1152 groups=4` 且 `w_inject=0`（表内空指针路径，4 WGs）

## 3. bit-exact 判定逻辑在哪里

- 参照值：臂 A 用 slot0/slot1 两套输入各跑一次，D2H 后按字节保存
  （`probe_paramtable.cpp:448`–`456`）；`:457` 的 `ref[0] != ref[1]` 证明两套输入输出确实不同
  （否则"指针交替"测试是空转），该断言计入 failures。
- 臂 B：每轮先改写表（交替 slot 指针 + 递增 sequence/version）再提交；比较
  `payload() != ref[slot]`（`:492`，`act` 全部 lowrank 项 + `inj` 全部 groups 项，`std::memcmp` 语义）；
  echo 校验（`:496`）；输出缓冲尾部 64 槽哨兵校验（`guard_ok`，`:330`）用于捕获越界写。
- 臂 C：同上，但执行是 `q.ext_oneapi_graph(exec)` 重放（`:512` 录制、`:567`/`:571` 校验），默认 1000 次。
- 汇总：`[PT ABI DONE] all_bitexact=… failures=…`（`:504`/`:578` 累加，`:603` 决定退出码），退出码 0（全绿）/1（任一失败）。判定门槛即
  "A==B==C 且 echo/哨兵全通过"，与 E3 的位精确门一致。

## 4. 编译状态

构建（首次即通过，无报错）：

```powershell
pwsh -NoProfile -File tmp/q4e-bottleneck-audit-20261016/build_probe.ps1 `
     -Source tmp/q4e-structural-20261016/probe_paramtable.cpp
```

产物：`E:\RiderProjects\Aila\tmp\q4e-structural-20261016\probe_paramtable.exe`

CPU 冒烟（`ONEAPI_DEVICE_SELECTOR=*:cpu`，不碰 GPU）已跑两遍，全绿：
`smoke-cpu-default.log`、`smoke-cpu-profiling.log` —— 3 形状 × 3 臂，每臂 1000 次重复全部
`mismatches=0 echo_fail=0 guard_fail=0 slots_differ=1`，`[PT ABI DONE] all_bitexact=1`，exit=0。
（CPU 数字只有跑通意义：A/B/C wall p50 918/1004/1011 µs 是 Xeon 上的 kernel 时间。）

`--wait-every` 版本（2026-10-16 22:21 重建）复跑默认路径（`wait_every=1`，行为与重建前一致）与
`wait_every=8` 扫描路径（`smoke-cpu-wait8.log`，iterations=64/replays=32，仍 `all_bitexact=1`）。
CPU 上该开关已显现方向性：臂 A p50 25.6 → 13.2 µs、臂 B 18.7 → 7.2 µs（drain-per-call 的开销）。

## 5. 请主代理运行（GPU）

```powershell
# 默认配置：3 形状 × 3 臂，200 次计时迭代 + 1000 次 bit-exact 重放
pwsh -NoProfile -File tmp/q4e-structural-20261016/run_paramtable_probe.ps1 `
     -Log tmp/q4e-structural-20261016/gpu-run.log

# 想要 device span（臂 B 的 profiling event）时加一个开关（CPU 已验证不破坏 Graph 臂）
pwsh -NoProfile -File tmp/q4e-structural-20261016/run_paramtable_probe.ps1 `
     -ProbeArgs '--profiling' -Log tmp/q4e-structural-20261016/gpu-run-profiling.log

# wait-every 扫描（3.7 µs vs 23.8 µs 归因）：N=1 是默认的 drain-per-call
foreach ($k in 1,2,4,8,16,32) {
  pwsh -NoProfile -File tmp/q4e-structural-20261016/run_paramtable_probe.ps1 `
       -WaitEvery $k -Iterations 200 -Replays 200 `
       -Log "tmp/q4e-structural-20261016/gpu-waitevery-$k.log"
}
# 等价地直接跑 exe（需 oneAPI/Level Zero 环境）：probe_paramtable.exe --iterations 200 --replays 200 --wait-every 1|2|4|8|16|32
```

`--wait-every N`（默认 1）语义与 `tmp/q4e-structural-20261016/probe_submit_shape.cpp` 的 `measure()` 一致：
每提交 N 次 wait 一次（`0` = 只在最后 wait 一次），**wait 不计入被测量的 submit 区间**，只作用于臂 A/B
（臂 C 图重放固定 drain-per-call，其行打印 `wait_every=-1`）。N>1 时参数表在每个 group 起点写一次
（上一 group drain 之后），`wall_us_p50` 变为 group wall 的每提交摊薄值；正确性校验循环始终逐次 drain，
不受影响。

脚本会先检查 `AilaWorker/AilaServer/Aila` 是否占用 GPU，再设置 oneAPI 2026.1 + 生产 L0 环境
（regular command list）。也可直接用既有 runner：
`pwsh -NoProfile -File tmp/q4e-bottleneck-audit-20261016/run_probe.ps1 -Exe tmp/q4e-structural-20261016/probe_paramtable.exe -Log <log>`

## 6. 预期输出格式与判读

```
[PT ABI CONFIG] device=… limited_graph=1 full_graph=0 profiling_queue=0 iterations=200 replays=1000 wait_every=1 burst=64
[PT ABI SHAPE 0] tag=qwen4exp-0.8b lowrank=320 group_dim=2560 groups=4 k_dim=10240 total_rows=324 num_wgs=21 inject=1 w_off_bytes=0
[PT ABI ARM A] shape=0 wait_every=1 host_submit_us_p50=… host_submit_us_p50_nowait=… p90=… table_write_us_p50=… wall_us_p50=… burst_us_per_call=… device_us_p50=…
[PT ABI CHECK] shape=0 arm=B replays=1000 mismatches=0 echo_fail=0 guard_fail=0 slots_differ=1 memcmp=equal
[PT ABI CHECK] shape=0 arm=C replays=1000 mismatches=0 echo_fail=0 guard_fail=0 memcmp=equal
[PT ABI VERDICT] shape=0 bitexact=1 (A==B==C, replays=1000)
[PT ABI DONE] all_bitexact=1 failures=0
```

- 逐位一致门槛：`mismatches=0 echo_fail=0 guard_fail=0` 且 `exit=0`。
- 提交成本对照：`ARM A/B/C` 的 `host_submit_us_p50` 与 `burst_us_per_call`（后者是连续 64 次
  不等待的摊薄值，最接近"纯提交吞吐"）。`table_write_us_p50` 是每轮写 128 B Host 表的 CPU 成本
  （GE 记录 ~0.1 µs）。臂 A 无 `device_us`（生产入口不返回 event）。
- `host_submit_us_p50` 统计**全部** submit 区间（与 Boole 的 `probe_submit_shape.cpp` median 口径一致，
  可直接对比 3.7 µs）；`host_submit_us_p50_nowait` 只统计没跟 wait 的提交（`wait-every=1` 时为 `-1`）。
  行内新增 `wait_every=` 字段便于扫描脚本按列收集。

## 7. 限制与 TODO

1. **native L0 regular command list 臂未实现**：GE 实测它最快（64 层 135.7 µs vs Graph 253.9 µs），
   Aila 侧要复现需 `zeModuleCreate` + `sycl::make_kernel_bundle<L0, executable>` 互操作，
   模板见 `A770-Graph-Experiment/graph_experiment.cpp`（`Runtime`/`Pipeline`）与
   `phase2/parameters.hpp`（`Recorded`）。本次按任务要求先交 B/C 两臂。
2. **几何仍由 host 计算**：`num_wgs` 用的是与表相同的标量，但换形状要重录图；真正的"变几何重放"
   需要 mutable group count（该驱动支持，GE bit1）或 indirect dispatch（GE-P4 显示无稳定收益）。
3. 表放在 **Host USM**（GE 结论里最省的一档）；Shared/Device 表未测（GE-P1 已覆盖，必要时再补）。
4. `--profiling` 下 Graph 重放 event 无 profiling info（CPU 上为 `-1`），GPU 上是否可读由本次运行记录。
5. 本探针只覆盖 1 个 kernel（E0 范围）；子链级（S1 6 提交 / S2 4 提交）见 roadmap E2/E3。
