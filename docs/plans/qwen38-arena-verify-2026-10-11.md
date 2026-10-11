# B4-①：adaptive 多行 CPU verifier 重接（2026-10-11）

基点 `HEAD=5ef4e1a` + 补丁（`tmp/q4e-arena-verify-20261011/candidate.patch`，6 文件 +502/−10）。
env：`AILA_Q4E_ARENA_VERIFY_ROWS`（read_flag，默认 0=关；非生产几何/无 arena 自动回退原路径，默认逐字节不变）。

## 背景（为什么这是 MTP 的硬前置）

MTP（dense ≤2048）需要在一次 2..4 行 verify forward 中由 CPU 计算冷专家；但 `CanExecuteVerifyRows` 一直绑定退役 `Qwen4ExpExpertCache`（`ExExecutor.cpp:1059/:1069`，唯一喂入口是死分支 `cpu_verify_async`），adaptive pipeline 无路可走——这是与"T1 对齐"并列的历史阻断项。

## 实现

- 新增 `CanExecuteArenaVerifyRows(arena, rows, top_k)` / `StartArenaVerifyRows(arena, layer, rows, ids, route_event)` / `JoinArenaVerifyRows(float* rank_output)`：从 **host arena（RAM 补集）** 解析冷专家、一次提交 rows 个 token 的 CPU 批任务（沿用既有 worker/lease 语义），join 后按 rank 写入输出；算术仍是未改动的 `cpu_expert_batch_forward`，仅源槽换成 arena。
- 修一个容量错误：cold rank/id/slot 数组原为 16，多行批可含 40 个任务。
- 接线：adaptive `RunResidentDecode` 的 `decode_verification`（count 2..4）在 env 开时使用新路径（GPU 常驻热集 + 单次多行 CPU 冷批 + 一次 ordered merge）；`PrepareDecodeVerification` 一次性预分配 verifier scratch。

## 证据

- 新 GPU 测试 `AilaQwen4ExpArenaVerifyTests`：**rows=2/3/4 输出与逐行 T1 CPU 参考逐位一致**（含 partial tail / empty tail / full capacity：16/21/30/40 CPU 任务），GPU ranks 不受影响；`arena.Release()` 后 gate 正确失效。重跑复现（`verify-rerun.log`）。
- CPU/ABI：19 项 100% 通过（`ctest-cpu-abi.log`）。

## 未覆盖（明确边界）

- **B4-②**：GPU 多行核的 T1 对齐（GDN 8vs4 累加、RoPE、QKV prepare、dense-QSA/attention）——本轮未动。
- **B4-③**：MTP 开启的 128-token 验收；未做全模型 2 行 forward 冒烟（无内建目标可在不加载 30GB 模型的前提下驱动 `ForwardImpl(count=2, decode_verification=true)`）。

结论：**B4-② 的进入门槛（多行 CPU verifier 可达 + 语义对齐 + 逐位正确）已满足。**

## 证据位置

`tmp/q4e-arena-verify-20261011/{candidate.patch,REPORT.md,arena-verify-test.log,verify-rerun.log,build.log,ctest-cpu-abi.log}`
