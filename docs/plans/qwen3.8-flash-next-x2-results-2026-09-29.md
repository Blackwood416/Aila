# X2：单次 prefill 与 interior checkpoint 验收

日期：2026-09-29。基线 commit `42b52181643ab3af054f01eb3b1777ba4fb797ed` 加本轮工作树。
本轮止于 X2；X3 grouped MoE、X4 chunked GDN 尚未开始。计算路径保持 GPU。

## 交付与默认策略

- 初始纯文本 `count=30, checkpoint_after=26` 默认执行一次 forward，在第 26 token 捕获前缀；其他形状默认仍分段。
- `AILA_Q4E_INLINE_CHECKPOINT=0` 强制恢复原分段行为；`=1` 允许满足状态条件的通用锚点，不再用 `{26,28}` bitwise 白名单限制 opt-in。
- 状态条件：4–64 token、前后各至少 2 token、至少两个快照槽、容量足够、无视觉 embedding override/MRoPE、处于 dense QSA 区间。
- 资格查询和直接 `Forward` 使用同一判定，且先检查、后申请运行缓冲。修正旧 `<=2052` 边界：默认 ratio=4、budget=2048 时，2052 已有 513 个完整块，进入 sparse；最后合法 dense 长度为 2051。
- XMX 默认开关不变；本轮分别固定 oneDNN / XMX，测量 checkpoint 自身的收益。
- 没有新增生产 scratch、专家槽或快照槽；继续复用已预分配的快照与状态。

## 性能证据

`[MEASURED]` A770，驱动 `32.0.101.8860`，oneAPI `2026.1.0`、oneDNN `3.11.2`。模型 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`，max_seq=512，story 30 token。
两条路径各预热一次，再做三组交错 A/B。主机专家缓存热，GPU 专家缓存仍受容量限制；没有宣称全 GPU cache hit。所有数值是 host wall time。

| 投影路径 | 指标（ms，中位数） | 26+4 | 单次 30 |
|---|---|---:|---:|
| oneDNN | 固定 token prefill | 1676.73 | 1436.02 |
| oneDNN | prefill + 47 次固定 token decode | 4892.87 | 4648.41 |
| XMX | 固定 token prefill | 1566.89 | 1374.65 |
| XMX | prefill + 47 次固定 token decode | 4749.12 | 4552.88 |
| oneDNN | Engine 首个非空文本 callback / TTFT | 1675.14 | 1431.76 |
| oneDNN | Engine 48-token 总耗时 | 4994.08 | 4801.42 |
| XMX | Engine 首个非空文本 callback / TTFT | 1570.24 | 1373.71 |
| XMX | Engine 48-token 总耗时 | 4837.20 | 4647.98 |

XMX 下固定 token prefill 减少 12.3%，固定工作负载总时间减少 4.1%。对应 fused 调用 `292→146`、整矩阵解量化 `96→48`。oneDNN 下整矩阵解量化 `388→194`。

固定 token 轮次不下载 logits、不做质量计算，仍按正常模型重算 routing。三次 XMX 单次 prefill 为 `1373.05/1374.65/1376.48` ms，总时间为 `4552.88/4552.96/4550.67` ms。三样本仅是本规划要求的轻量 canary，不是广泛 shape sweep。

自由生成会分叉。首次单次 prefill 的自由生成总耗时有冷路由/缓存影响（oneDNN `5631.69`、XMX `4920.13` ms），保留在原始样本中，不剔除也不冒充稳定热结果；主要归因使用固定 token 对照。两边实际均输出 48 token。

## 数值与状态

固定同一份 NF4 权重和 48 个外部 continuation token：

| 投影路径 | 平均 KL(ref‖cand) | 平均 NLL 增量 nat/token | 最大 logits NRMSE |
|---|---:|---:|---:|
| oneDNN | 0.00126094 | -0.00564947 | 6.90% |
| XMX | 0.00208547 | +0.00925565 | 7.51% |

全部 logits 有限。XMX NLL 增量低于但接近方案中的 0.01 初始预警线；它不是质量认证阈值。这里的 NRMSE 是模型输出，不与算子层 1% 初始预警线混为一谈。没有用 token 相同或几个关键词作为准入条件。

状态测试包含：

1. 锚点 26/28/16/2：同样的 30-token shape，更换锚点之后的所有 token，GDN、conv、PLE 快照及可见 KV 前缀逐位不变。
2. restore 后重新 capture，快照全部 float payload 逐字节一致；同一快照连续两次 replay 的 logits 逐位一致。
3. restore 与独立分段前缀对照：26/28 无 logits 差异；16/2 分别 NRMSE 2.66%/2.89%，记录为不同计算 shape 的数值差异。
4. 非零起始状态：先执行 4 token，下一次 forward 的相对 anchor=16 必须发布绝对位置 20 的快照；恢复逐位一致。与独立分段 continuation 比较，KL=0.00553003、NLL 增量=-0.0197336、logits NRMSE=9.71%。不据此把任意非零起点改成默认。
5. 256-token canary：`252+4` fallback 在 opt-in 开关两侧 logits 逐位一致。首轮 30.21 秒、次轮 4.94 秒包含首次长形状 JIT/缓存差异，**不是 checkpoint 提速结果**。

## 验证与复现

目标构建：`Aila`、`AilaQwen4ExpToleranceTests`、`AilaQwen4ExpGenerationTests`、`AilaQwen4ExpModuleTests`。

```powershell
ctest --test-dir build-refactor -R '^AilaQwen4Exp(Module|Generation|Tolerance)Tests$' --output-on-failure
pwsh -NoProfile -File perf/qwen4exp/q4e_checkpoint_acceptance.ps1 -Tag x2-final
```

三个定向 CTest 全部通过；真实模型验收退出 0。未重复全量 CTest。

原始数据：`tmp/x2-final.log`、`tmp/x2-final.err.log`、`tmp/x2-final.meta.json`。验收 exe SHA256 为 `B6D9CF036FE9764F789403214D547C5787F4DA5F772F4C2EEDBEBD4C2BB5BD8E`。
持久化中位数、MAD、全部样本、质量指标和身份信息见 [JSON evidence](qwen3.8-flash-next-x2-results-2026-09-29.json)。后续为内存诊断扩展了同一个测试可执行文件，因此当前 exe hash 可能不同，以上证据绑定验收时的 metadata。

## 长文 RAM 跟进

用户在上述 256-token 测试时观察到系统 RAM 占用超过 90%。复现定位到默认 mmap reader：长文激活更多专家时，源权重映射页进入进程 working set，同时已有约 16 GiB pinned host expert pool 保留另一份数据。不是 CPU 异构计算引入的问题。

`[MEASURED]` 用独立进程、max_seq=512、同一份 256-token lighthouse 文本做两次 forward，500 ms 间隔记录内存。保留 8 GiB 系统可用 RAM 的保护，内存不足即终止自有测试进程：

| 路径 | 峰值进程 working set | 峰值 private commit | 最低系统可用 RAM | 完成情况 |
|---|---:|---:|---:|---|
| 显式 mmap | ≥46.80 GiB | 31.37 GiB | 7.71 GiB | 首次 256-token prefill 中保护停止，不是完整峰值 |
| 新 Windows 默认 ReadFile | 21.18 GiB | 31.36 GiB | 28.93 GiB | 两轮完整结束，exit 0 |

两者 device/pinned-host 分配相同。新默认在 256 token 后 device USM 为 14,295,005,206 B（约 13.31 GiB），pinned host 为 17,360,819,204 B（约 16.17 GiB，其中 expert pool 约 16 GiB），第二轮不再增长。这里是 USM 分配账本，不是 DXGI 实际 resident VRAM。

修改仅将 Windows 默认 `AILA_Q4E_EXPERT_READER` 设为 `readfile`，记录实际 reader 名称；非 Windows 仍为 mmap，显式 `=mmap` 及缺失 file-span 的已有 fallback 都保留。未缩小 host/GPU expert cache，也未更改 kernel 或权重格式。ReadFile 仍使用 Windows 文件缓存；不能将进程 working set 降幅等同于整个系统绝对物理内存节省，但本机系统可用内存同样明显增加。

新默认两次 256-token 输出的全词表 logits hash 均为 `1271920800156241593`，与显式 ReadFile 对照一致，mmap fallback=0。诊断 wall time 为 4204.14/3898.10 ms；mmap 被保护停止，因此不报告 reader 提速倍数。首次显式 ReadFile 运行的计量脚本在结束后汇总 OrderedDictionary 时失败，原始内存样本和完整输出保留；已修正脚本，新默认确认运行退出 0。

reader 修改后再次运行 `Module|Generation|Tolerance|ExpertReader|DmaTiming` 五个定向 CTest，5/5 通过（30.25 秒），包含 reader 字节一致性和 DMA 生命周期覆盖。

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_checkpoint_acceptance.ps1 -Mode memory -Tag x2-memory-default
```

持久化证据：[内存 JSON](qwen3.8-flash-next-x2-memory-2026-09-29.json)。原始采样在 `tmp/x2-memory-{mmap,readfile,default}.memory.json`，最终身份和退出状态在 `tmp/x2-memory-default.meta.json`。没有再扩大输入来施压。上面的 X2 性能表是 reader 修改前的隔离结果，不把两项收益相加。
