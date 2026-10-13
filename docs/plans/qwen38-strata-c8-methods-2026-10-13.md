# Strata C8：工程方法学包落地（2026-10-13）

范围：扫描文档 C8 的可操作项。目标：让"微基准好看、端到端不动"这类结论在流程上被拦住。

## 1. 已交付

| 交付物 | 作用 |
|---|---|
| `perf/templates/bench-receipt-v1.json` | 收据 schema（Strata `A770_SPEED_SPEC.md:130-138` 的等价物）：workload/environment/artifacts/protocol/arms/pairs_won/verdict/correctness；缺值必须显式 `null`+note |
| `tools/perf/ab_receipt.py` | 从 runner 日志的 JSON 行生成收据：median/p10/p90、逐对 winner、`pairs_won`、median ratio、git 状态；A/A 时即噪声底 |
| `AilaQwen4ExpKernelTests --queue-overlap-probe --json` | 探针的机器可读输出（一行 JSON，可直接进收据） |
| 本文件 | 发射规则与用法 |

示例（A/A 噪声底，两条基线 14K run）：

```
python tools/perf/ab_receipt.py --metric first_content_seconds --label context-14000 \
  --arm baseline=tmp/q4e-strata-c1/c1-a04-run.log \
  --arm baseline2=tmp/q4e-strata-c1/c1-d0-run.log \
  --out tmp/q4e-strata-c4/aa-noise-receipt.json
→ median ratio 1.0008（0.08%），pairs 1:1
```

## 2. 规则（采纳为 Aila 的实验纪律）

1. **3% 发射规则**：一个改动只有在两个被测形状/配置上都 ≥3% 才允许改默认；形状间结论不一致
   或打平 → 保持 opt-in。（与 Aila 既有"端到端 ≥+5% 才默认"一致，对 kernel 级放宽到 3%。）
2. **先证测量正确**：每次 A/B 先跑 A/A（两条同配置臂）得噪声底；本机 14K TTFT 噪声底
   ≈0.1-0.8%（见上例与 roadmap ±0.6%）。
3. **pairs-won 必须报**：只看中位数会掩盖交替顺序的漂移；收据里逐对记录。
4. **反证优先**：任何"kernel 是瓶颈"的假设先用 GPU 占用/事件重叠反证——本机已两次踩中
   （P4 的 `_nolut` 上限、C1 的 host fill 被设备掩盖），新探针必须给出"去掉它墙钟是否变"的对照。
5. **原始 token 收据**：输出会变的候选（tail skip、CPU 分担、量化路径）另存 raw token id 列表
   并附独立校验脚本（`EXCHANGE_ROTATION.md:105-138` 的做法）。
6. **设备跨度与 host wall 并列**：Aila 的 Context 队列未开 profiling；当前只有探针类
   （queue-overlap/DPAS/ESIMD）能报每轮设备时间，runner 类仍是 host wall。开 profiling 需
   opt-in，属后续项（列入本文件而非另开候选）。

## 3. 与本轮候选的关系

- C1/C2 的结论（transfer 与内存型 compute 相加）就是按规则 4 得到的：先给上界（oracle）、
  再做单变量（parallel fill / direct DMA）、最后用"墙钟是否变"判定。
- 后续 C3/C6/C9/C11/C13 全部按 `bench-receipt-v1` 出收据；kernel 级先过 3%，再谈端到端。
