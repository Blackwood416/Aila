# Qwen3.8 候选 2a：swap 预算单变量筛选

日期：2026-10-10。基点 `HEAD=3962519`，正式二进制对（Worker `18cea21c…`、Server `ab04ae7d…`，与 2026-10-09 归档 pair 一致）。
顺序 **A1 → B1 → B2 → A2**，全部 `run.py <tag> --agent`（PI 工具往返 + 14K 新请求 + 缓存续轮 + 取消恢复），同一冻结语料、32K context、4096 chunk、MTP=0、reserve3584、H4 profile、`AILA_Q4E_STAGE_COPY_PARALLEL=1`。
唯一切换项：`AILA_Q4E_MAX_SWAPS` **2（A）对 8（B）**；`AILA_Q4E_ADAPTIVE_SWAP_FREQ=1` 两臂固定。四轮 worker_environment 差异经程序比对仅此一项。

## 结果

| 轮次 | 14K 首字(s) | 14K 总耗时(s) | 14K decode rate(tok/s) | 续轮首字(s) | 续轮总耗时(s) | 续轮 rate(tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| swap-a1 | 69.614 | 71.868 | 13.749 | 1.455 | 3.600 | 14.450 |
| swap-b1 | 69.469 | 71.996 | 12.263 | 1.485 | 3.977 | 12.442 |
| swap-b2 | 69.543 | 72.050 | 12.367 | 1.502 | 3.981 | 12.509 |
| swap-a2 | 69.540 | 71.794 | 13.749 | 1.471 | 3.628 | 14.372 |
| A 中位数 | 69.577 | 71.831 | 13.749 | 1.463 | 3.614 | 14.411 |
| B 中位数 | 69.506 | 72.023 | 12.315 | 1.494 | 3.979 | 12.476 |
| 变化 | -0.10% | +0.27% | **-10.43%** | +2.13% | +10.10% | **-13.43%** |

四轮 `passed=true`、PI 与取消恢复通过、无 timeout、自然关闭；最低可用 RAM 2.76–3.50 GiB。
14K 请求 payload/输入/输出 hash 四轮一致；续轮输出在臂内一致（A=`795ea167…`，B=`3d4f184d…`），跨臂不同属预期——swap 预算改变驻留轨迹，GPU/CPU 浮点次序不同，只能作策略比较（usage、finish_reason、缓存长度跨臂一致）。

## 结论

- 提高同步 swap 预算到 8 在 14K prefill 上无可测影响（-0.10%，符合 prefill 无 swap 的结构），但 decode 速率一致下降 10–13%。
- 边际 6 次 swap/token 的成本 ≈ 10.9 ms/token（≈1.8 ms/swap），且未换来可观察的命中率收益；当前同步 swap 实现（逐 swap D2H+H2D host wait）不具备放大预算的价值。
- 默认维持 `MAX_SWAPS=2`。未测的更低预算（1）或 `FREQ=2/0` 属于新假设；若走 swap 方向，下一步应先证明"减少 swap"有收益或"异步化 swap（候选 2b）"能把 1.8 ms/swap 的暴露成本降下来，否则该方向冻结。

## 原始证据

`tmp/q4e-resident-speed-20261009/swap-{a1,b1,b2,a2}/`（result.json、逐请求 SSE/事件、monitor）与同名 `swap-*-run.log`；分析脚本与输出：`tmp/q4e-resident-speed-20261009/swap_analyze.py`。
