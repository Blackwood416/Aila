# Qwen3.8-Flash-Next：P2 供给隔离与 P3 固定段 graph 验证

日期：2026-09-29。基线为 `0293b3b` 加前一轮观测工具。A770 16 GB、Windows 驱动 32.0.101.8860、oneAPI 2026.1。用户已明确：**中长输入追求 Prefill 100+ tok/s；短输入检查延迟与回归，不强制 100+。单请求 decode 目标仍为 20 tok/s。** 计算继续全部放在 GPU。

本轮通过实测收束 P2 的 copy/计算干扰问题，并开始 P3 的真实模型固定段 graph 测试。新增隔离 benchmark、512/1024-token canary、graph 状态验证，没有改变生产默认值。

## 1. 中长输入暖 Prefill 已超过 100 tok/s

[MEASURED] lighthouse 文本、max_seq=max(512,tokens)、ReadFile、dense XMX=1、fill_parts=8。每配置一个独立进程，初次输入单列，之后 reset 并完整处理相同输入三次。引擎默认外层 chunk 仍为 512：1024 输入实际执行 `pos=0,count=512` 和 `pos=512,count=512`，不是单个 1024-row kernel，更不是复用已缓存的 prompt token。

| 输入 | 配置 | 三次暖输入 wall（ms） | 中位数 | tok/s |
|---|---|---|---:|---:|
| 512 | 整层 acquire | 5287.77 / 5271.79 / 5273.92 | 5273.92 | 97.08 |
| 512 | B6=32 | 5052.92 / 5020.46 / 5015.66 | 5020.46 | **101.98** |
| 1024 | 整层 acquire | 10065.30 / 10031.50 / 9992.93 | 10031.50 | 102.08 |
| 1024 | B6=32 | 9204.12 / 9233.89 / 9261.57 | 9233.89 | **110.90** |

512/1024 的时延分别缩短约 4.8% / 8.0%。同一长度的基线、B6 及所有重复的全词表最终 logits 哈希一致：512 为 `7370615997528717082`，1024 为 `4163727905343504607`。canary 现在会直接断言重复输入哈希一致，而非仅打印。

上述 wall 包含最后 logits 下载/检查，初始化 warmup 不计入。它是同进程暖模型供给状态下的结果，不是冷启动或任意请求的保证。例如 1024 的首次输入分别为 12193.3 / 16400.9 ms，初次 shape 准备、host pool 和 OS 文件页未单独隔离，不能拿暖样本掩盖这一差异。单进程三次样本也不等于三个独立进程的发布验收。

非 profiler 模型进程峰值 working set 约 21.21 GiB、最低系统可用 RAM 约 27.65 GiB。1024 的 device/pinned-host 分配计数为 14,438,490,134 / 17,368,254,468 B，host pool 仍为 17,178,163,200 B；B6 没有通过扩大池容量获得收益。分配计数不是 DXGI resident VRAM。

## 2. P2：主要是并发 copy 干扰，拆小 batch 自身不是主因

新增 `AilaQwen4ExpPrefillSupplyBench`，调用生产 four-token expert gate/up，H=2560、I=640、256 个专家、每专家 8 行；共 2048 个激活，NF4+nested codes，完整 slot 2,534,400 B。每轮 copy 载荷固定 648,806,400 B，数据为非均匀确定性值。

只改变批量与依赖/传输方式，测试 256/64/32 专家批量。CPU 抽样参考 72 点通过；所有调度方式的完整 GPU 输出位级一致，未使用的 rank/tail 哨兵完整。正式计时每变体 3×20 样本，变体在同一进程轮换，报告 host submit→两队列完成的 wall 以及 SYCL event 时间。不是与另一个数学算子比较，也没有据此选择新计算 kernel。

[MEASURED] 32-expert 批量，每格列三个重复的中位数：

| 方式 | gate/up event 合计（ms） | 含全部 copy 的 wall（ms） |
|---|---|---|
| 全部 resident，无 copy | 5.19 / 5.20 / 5.23 | 5.46 / 5.43 / 5.46 |
| 向无关设备地址并发 copy，每专家一次 | 28.69 / 28.09 / 27.69 | 69.08 / 69.08 / 69.01 |
| copy 本批权重，kernel 直接依赖完成事件 | 24.99 / 25.91 / 25.24 | 69.57 / 69.73 / 69.62 |
| 同样依赖，经独立 compute barrier | 25.38 / 25.28 / 26.45 | 69.69 / 69.75 / 69.72 |
| 每批合成一次约 81 MB copy，直接依赖 | 57.87 / 53.34 / 50.36 | 70.48 / 70.69 / 69.34 |

无 copy 时，256/64/32 三种划分的总 kernel 时间都在约 5 ms。无关 copy 没有数据依赖或写入冲突，仍复现明显增长；独立 barrier 未消除它。因此，上一轮 B6 的 gate/up 变长不能主要归因于小 batch 的计算形状，也不是仅把等待时间记到 kernel 上的问题。**证据支持当前路径存在并发 H2D 对计算的干扰；还不能仅凭这些时间确定硬件仲裁、缓存还是驱动中的具体机制，更不能宣称显存带宽已饱和。**

### 更细 DMA 切片也不准入

进一步把每个专家拆成最多 512 KiB 的五次 copy，同一工作量再做 3×20 配对。直接依赖模式的 kernel 合计降到 12.50 / 12.53 / 12.72 ms，但 wall 升到 93.04 / 92.82 / 92.83 ms；同轮普通单专家 copy 约 69.5 ms。更多提交带来的开销抵消了计算干扰减少量。

**决策：保持单专家 payload 的现有 copy 方式；不加入大块合并或细分 DMA 的生产分发规则。** 这些负方向保留在独立 benchmark 中。B6 的长输入收益有效，但不能按理想完全重叠估算下一轮收益；减少实际 miss 字节仍有价值。

## 3. P3：静态 graph 能正确执行，本固定段候选却更慢

新增可选 `--front-graph-model` 实模型测试。固定段使用真实 layer 0 的 HC → linear_all projection → GDN → output projection → HC → router，与生产 decode 的默认融合路径相同。它使用测试自己的 recurrent/conv state，排除 PLE、expert acquire/MoE、完整模型和采样。

设备报告 `limited_graph=1`、`full_graph=0`。为 conv head 0/1/2 分别录制静态图，避免错误地认为捕获的 host scalar 会自动更新。三个不同输入/初始状态、1/2/7 步执行，对照 hidden4、MLP 输入、router scores、ids/weights、inject 系数以及全部 recurrent/conv state，均位级一致；graph、host 数组和 device buffer 均在队列排空后才释放。

[MEASURED] 每模式每轮 20 样本，每样本连续执行同一 front 36 次，计时在 reset/copies 之后。这 36 次重复**不是 36 个不同层，也不是完整 decode 吞吐**。两次独立进程均复现 graph 变慢；第二次补齐 inject 对照后的时间：

| 重复轮次 | 普通提交 ms/front | graph ms/front |
|---|---:|---:|
| 0 | 约 0.42 | 0.705 |
| 1 | 0.421 | 1.050 |
| 2 | 0.427 | 1.314 |

普通路径稳定，graph 的代价随重复增长。未进一步把原因归结为某个驱动缺陷，也没有将该局部数据外推为整模型延迟。**该固定段 graph 不准入，不把“减少 API 调用”当作 decode 加速证据。** 这不排除其他更小分段或其他栈下的 graph，但后续候选必须重新测量。

## 4. 当前状态与后续门槛

- 本轮中长输入的暖 Prefill 已验证超过 100 tok/s，依赖显式 dense XMX 和 B6=32；没有变更默认开关。广泛形状/不同 prompt 的默认准入仍未完成。
- decode 20 tok/s 仍未实现，本轮不声称 decode 提速。P3 后续应在固定 continuation、控制缓存热度的基线上衡量完整收益；不能直接推广本次被拒绝的 front graph。MTP 的 acceptance、验证和状态恢复成本仍是独立待验证工作。
- 新增可重现隔离工具及扩展 canary；所有有效采集进程退出 0，无 watchdog 停止。最新 `AilaQwen4ExpToleranceTests` CTest 通过（约 4.07 秒）；模型长度、位级调度和 graph 状态检查另在上述可选测试中完成。
- 证据与源文件/二进制/输入身份见 [结果 JSON](qwen3.8-flash-next-p2-supply-p3-front-results-2026-09-29.json)。改动尚未提交。

## 5. 复现

```powershell
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload supply -Tag supply
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload supply -Sliced -Tag supply-sliced
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload memory -MemoryTokens 512 -MemoryRepeats 4 -FillParts 8 -Tag pp512-base
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload memory -MemoryTokens 512 -MemoryRepeats 4 -FillParts 8 -DemandMicrobatch 32 -Tag pp512-mb32
# 1024 使用相同命令，修改 MemoryTokens 与 Tag；引擎仍按默认 512 分块。
pwsh -NoProfile -File perf/qwen4exp/q4e_execution_probe.ps1 -Workload front_graph -Tag front-graph
```
