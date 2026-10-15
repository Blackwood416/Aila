# INT8 KV 生产链路：使能、QSA prefill 移植与 A/B/B/A 结果（B3，2026-10-12）

基点：`93562fd` + `8101788`（IPC/守卫/预算）+ `c67edc7`（QSA prefill int8）。协议：生产序列
`run.py <tag> --agent`（14K prefill + PI agent + cancel-recover），A/B/B/A 单变量。

## 1. 使能链（两个阻塞依次修掉）

1. `8101788`：capability `qwen4exp_nf4_int8_kv` + `--kv-type int8`/`AILA_SERVER_KV_TYPE` +
   `allowQuantizedKv` 显式放行 + **reservation 按 int8 布局重算**（1B K/V + 两组 per-token F32 scales，
   逐字节对齐 `Session::CacheBytes()`）+ `AILA_Q4E_MAX_GPU_SLOTS`（默认 2800）可抬。
   修前：`Session cache reservation did not match allocation`（每个 session 都被拒）。
2. `c67edc7`：`attention_prefill_qsa_batched` 模板化 + `k_at/v_at` 访问器（bf16 纯 load；int8 = code × per-token
   scale，一次 scale load 供 d0..d3 四条 lane）。修前：`requires bf16 KV cache`（生产 prefill 主路径）。

默认路径零变化：bf16 生产冒烟（14K+PI+cancel-recover）通过；QSA 内核测试 bf16 三臂逐条不变，
新增 int8 臂 max_diff **3.05e-05**（对照的 host 参考跑在反量化值上，检验 scale 应用本身）。

## 2. 生产 A/B/B/A（14K prompt，generate 32，同二进制）

| 指标 | A1 bf16 | B1 int8 | B2 int8 | A2 bf16 | 判读 |
|---|---:|---:|---:|---:|---|
| 14K decode rate (tok/s) | 15.583 | **15.753** | **15.746** | 15.360 | int8 两轮均高于两轮 A → **+1.8%** |
| 14K ttft (s) | 70.636 | 71.374 | 71.326 | 70.574 | int8 一致慢 ~1.1%（QSA prefill int8 读法更贵） |
| 续轮 rate (tok/s) | 16.627 | 16.571 | 16.333 | 16.012 | 噪声内，无方向性 |

结论：**低于 +5% 采纳线 → int8 KV 维持 opt-in**。收益主要来自 attention 读取（`6deaefc`），
prefill 侧因 QSA 内核每元素多一次 scale 乘而略负。

## 3. 容量收益为何未兑现（2026-10-13 更正：已修复，`548b0d8`）

> 更正：本节原结论"hotset profile 只排 2792 个 key"**错误**——profile 实为全量 24,576 key，
> 日志 `ranked_keys=` 打印的是预算截取后的前 N。真正的缺口是 backend 专家缓存 admission 的 `kv_bytes`
> 仍按 bf16 计。已在 `548b0d8` 修复（`SessionMemory` 单源 `qwen4exp_kv_bytes`），实测：
> INT8 臂 `KV=384 MB (element_bytes=1) + scales 6 MB`、**admitted_slots 2792→2948（+156）**；
> 端到端 A/B/B/A：14K ttft −0.8%、续轮 rate +0.8~+1.7%（区间不重叠）→ 结构收益兑现，端到端 ~1%。

诊断 run（`b3-int8-diag`，`AILA_Q4E_MAX_GPU_SLOTS=3200`）日志：
`[Q4E Adaptive Residency] ... ranked_keys=2792 resident_slots=2792`，
`[Q4E ExecutionConfig] ... slots=2792`。

`Qwen4ExpBnb4Backend.cpp:764`：`resident_rank_count = min(num_gpu_slots, policy.ranking.size())`
——**hotset profile 只排了 2792 个 key**，所以 KV 省下的 384 MiB 没有变成常驻槽（仍 2792）。
且诊断显示 hits=244/misses=236、staging=0：冷专家全走 CPU（每 token ~0.65 GB DRAM 读），
与 bf16 完全相同。

后续（若继续这条线）：把 hotset profile 重新生成为 ≥2949 键（或在 profile 键不足时用剩余键续排），
再跑一次 A/B 才能测到"多 157 槽"的实际价值；否则 int8 KV 只值 attention 的 ~2%。

## 4. 证据

- 生产：`tmp/q4e-resident-speed-20261009/{b3-bf16-check,b3-int8-b2,b3-int8-b3,b3-bf16-check2}/`
- 诊断：`tmp/q4e-resident-speed-20261009/b3-int8-diag{,-worker.log}`
- 内核：`AilaQwen4ExpKernelTests.exe --qsa-prefill-only`（bf16 3 臂 + int8 臂全过）
- 失败定位：`tmp/q4e-resident-speed-20261009/b3-int8-b{0,1}/`（reservation / QSA 两个阻塞的原始现场）
