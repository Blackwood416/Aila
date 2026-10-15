# Qwen3.8 H4 初始驻留接入：生产筛选与反序复核

> 后续更新：候选3a（staging 并行 memcpy）已独立测量、反序复核并保留为 `3962519`，见 [staging 报告](qwen38-staging-copy-screen-2026-10-09.md)。本文保留 H4 单变量实验与 `c5d5805` 正式身份；当前交接见 [专家供给优化交接](qwen38-expert-supply-handoff-2026-10-09.md)。

日期：2026-10-09。基点为 nested-256 已保留的 `98cc037`；H4 单独提交为 `c5d5805ef788802929b857a11a62f194dc635daf`。
使用三个 Luna xhigh 子代理分别实现 H4、审计主机供给、审计 swaps/路由；GPU测量和构建由主代理串行协调。本机链路按用户确认的 PCIe 3.0 x16，不采用 PCIe 4.0 的带宽推算。

## 改动与正确性范围

Adaptive 初始化现在读取 `AILA_Q4E_HOTSET_PROFILE`，复用 `Qwen4ExpHotsetPolicy::Load` 的完整维度、identity、key唯一性/范围与尾随数据检查，将排名前实际准入的 2792 个 key 传入 `InitializeResidency`。未设置变量仍沿用原默认顺序。身份 FNV 算法从 legacy cache 提取共享，保留原采样契约；它不是全量密码学模型校验。HostArena source 名称选择与 identity 收集一致（base 或 `.weight`）。

GPU/RAM继续互补：2792个GPU-only专家、21784个原生大页RAM专家，51.46 GiB RAM补集，小型Host staging 77.3 MiB；没有全量RAM/GPU双份常驻。四轮monitor的available RAM最低2.247–2.743 GiB；两版准入预算相同，未发生OOM。slot预算、swap预算/周期、bank、PLE、router与专家数学均未改。

通过的定向检查：`AilaQwen4ExpExpertReadinessTests --hotset-profile-only`（旧身份算法等价、完整排名顺序、错误identity/重复key拒绝）；`AilaQwen4ExpAdaptiveCacheTests`（非平凡排名顺序、互斥owner、payload/CPU-GPU数值来源、consumer fence、连续owner交换、2800唯一key、稳态无新增USM）。一次新增测试曾误用旧固定cold索引，已将排名检查与原数学场景隔离后重编复跑通过。

提交版官方Worker和Server构建exit=0；`ctest --test-dir build-refactor -L 'cpu|abi'`：18通过、1条件skip、0失败；Server CTest：4/4通过。测试结束后才归档retained exe/metadata。

## 生产对照

顺序 A1→B1→B2→A2；全部 `run.py <tag> --agent`，相同冻结自然语料、32输出、32K context、4096 chunk、MTP=0、reserve3584、相同profile环境和socket harness。真实PI read-marker工具往返、HTTP新14K请求、缓存24token续轮、取消恢复与自然关闭均通过。每轮启动时实测exe SHA与pair record核对。

四次请求的完整payload、输出hash、usage、cache长度、finish与errors一致；PI归一化输出、调用和usage也一致。下表为两次A和两次B的中位数。它是有限输入上的placement策略对照，不代表CPU/GPU浮点数学逐位等价或长期decode稳定收益。

| 指标 | 原版 | H4 | 变化 |
|---|---:|---:|---:|
| 14K新请求 首字(s) | 70.5942 | 70.3661 | -0.32% |
| 14K新请求 总耗时(s) | 73.0968 | 72.7108 | -0.53% |
| 14K新请求 首字后短decode(tok/s) | 12.3877 | 13.2279 | +6.78% |
| 24-token续轮 首字(s) | 1.7221 | 1.4663 | -14.85% |
| 24-token续轮 总耗时(s) | 4.1594 | 3.6886 | -11.32% |
| 24-token续轮 首字后短decode(tok/s) | 12.7217 | 13.9588 | +9.72% |

续轮两对总耗时分别降低12.24%/10.40%，因此保留。新14K总耗时仅约0.53%改善，不能宣布prefill大幅加速。PI分别79.726→78.903s、79.879→78.979s（约1.03%/1.13%）；启动131.749→135.485s、139.648→133.852s，方向不一致，不宣称启动优化。

## 匹配诊断：与生产计时分开

诊断采用相同14K冻结输入和8输出，未先运行PI，因而包含冷JIT/缓存成本；不能把其wall数并入上表。日志中的CPU compute/join、router wait、experts host、fill与bank wait有重叠，不能相加。`h2d_mb`实际上按MiB计数，不是物理总线计数器。

| 轮次 | 单行样本 | GPU命中 | CPU任务中位数 | compute(ms) | join(ms) | submitted(ms) |
|---|---:|---:|---:|---:|---:|---:|
| h4-ad1 | 16 | 14.49% | 411 | 23.082 | 17.361 | 75.308 |
| h4-bd1 | 16 | 50.81% | 240 | 14.271 | 9.058 | 65.942 |

候选运行明确打印与profile一致的identity `nf4-original-expert-order-48x512-2560x640-b64-nested1-sample-fnv64-14151072976976870495` 和 `ranked_keys=2792 resident_slots=2792`。初始化key顺序hash、每层slot数与profile SHA见JSON中的profile manifest；运行期swap会继续改变placement。历史profile自身的decode计数覆盖14.42%→65.13%不作为新语料命中率预测。

大块prefill仍触达约19.3k–19.6k个cold unique experts，上传约46GiB/4096块，fill约7.3s。route hits按出现次数统计，而DMA按每层去重专家计；提高加权路由命中率不能同比减少distinct专家上传。当前原版也包含PLE lookahead，旧13s PLE区间不能当成本轮完全可节省成本。

## 其余候选的分配与边界

- 候选3a：仅并行staging memcpy，复用现有受限TBB arena、保留2×16 banks/事件/顺序，准备在 `tmp/q4e-staging-parallel-20261009/`；按真实槽几何、大页source/HostUSM destination做20组交错memcpy+memcmp微测，再单独生产screen。扩大ring是另一个变量。
- 候选4：仅并行PLE token gather，补丁/全量memcmp微测在 `tmp/q4e-ple-parallel-20261009/`，尚未应用；不加PVM、不改ticket/lookahead数学。没有测量支持“每行必然page fault”或“13秒全是SSD”。
- 候选2a/2b：预算策略与异步H2D分开。当前swap有D2H+H2D，计数字节只含H2D；`MAX_SWAPS=0`表示无上限而非禁用。双ticket设计、owner commit/reader/source/释放约束见 `tmp/q4e-h4-residency-20261009/swap-route-audit.md`，尚未改runtime。
- 候选5：shared expert已经在router wait前入队；不能再以移动它声称新增重叠。预测next route须确认下一层真实x依赖，暂缓实施。

CPU job协调线程、PLE worker与staging loop是串行入口，但CPU专家数学内部已有TBB并行。Strata其他GPU/CPU/量化格式数字不转换成本机预期收益。

## 归档与身份

所有原始输入、SSE、PI事件、launch、result、monitor在 `tmp/q4e-resident-speed-20261009/h4-{a1,b1,b2,a2,ad1,bd1}/`及同名日志。原版/候选exe及元数据分别在 `tmp/q4e-h4-residency-20261009/{base,cand}/`，测量版同为98cc037 stamp但SHA不同，禁止仅凭stamp辨认。提交版另归档retained；SHA如下（有值时表示官方双构建完成，不将其混称为已测候选exe）。

```json
{
  "worker": {
    "path": "E:\\RiderProjects\\Aila\\build-refactor\\AilaWorker.exe",
    "sha256": "de7ef9fa774c7c2237851e02b40347e3b60be5ab559021c2bd528fbde8cb3439",
    "build_id": "aila-0.2.0-abi1-c5d5805ef788802929b857a11a62f194dc635daf"
  },
  "server": {
    "path": "E:\\RiderProjects\\Aila\\build-refactor-server\\AilaServer.exe",
    "sha256": "decf26095d306d73ce277c1e290674f309ec2ba3c396ac3e34bcb9f8ad3a3709",
    "build_id": "aila-0.2.0-abi1-c5d5805ef788802929b857a11a62f194dc635daf"
  }
}
```

机器可读证据：[JSON](qwen38-h4-residency-screen-2026-10-09.json)。
