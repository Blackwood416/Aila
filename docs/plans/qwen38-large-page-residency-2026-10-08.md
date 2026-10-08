# Qwen3.8-Flash-Next：大页驻留与单流长请求

本轮直接实现 Windows `MEM_LARGE_PAGES` 权重驻留，目标是让 CPU 专家和 GPU 上传共用不可分页的 RAM 补集，PLE 继续按需读取。仅针对本机 Qwen3.8-Flash-Next；不新增其他模型的兼容路径。

## 实现与容量

- 启动时启用 `SeLockMemoryPrivilege`，原生大页分配失败直接报错，不回退到可分页内存。`Context` 按实际页对齐容量记账和释放。
- 专家 payload 按完整专家分块分配，每块不超过 256 MiB；本模型每块 105 个专家，页对齐后为 254 MiB。先分配连续的 embedding/元数据，再分配专家补集，没有减少驻留范围。
- 初始 GPU cache 2824 槽，逻辑热集 2808，其中 2312 个专家只在 GPU；可借出的 496 个热专家同时有 RAM 后备，避免 prefill 借槽后回盘。RAM 覆盖 22264 个专家。
- RAM 保存原始 packed NF4 gate/up、down、nested codes，以及独立复制的 nested 元数据和 embedding。CPU 路由使用实时所有权表，GPU 仍通过原有小型 USM staging 上传。
- 每层复制结束立即 trim Worker 的普通工作集，使已消费的 mmap 源页可回收；原生大页不在该工作集中，不受 trim 影响。避免在整套大页之外继续累积几十 GiB 的活跃源页。
- 热专家替换时把被替换的唯一 GPU 副本回写到新入专家原来的 RAM 槽位。完成驻留后关闭 expert file reader；缺失所有者直接报错，不借 mmap 掩盖覆盖缺口。
- 回写复用现有 staging，保留覆盖 H2D 的整批屏障与 GPU reader fence。不新增线程或 USM 缓冲，不改推理 IPC 协议。

| RAM 项目 | 字节 |
|---|---:|
| 专家 payload | 56,425,881,600 |
| nested 元数据 | 29,589,504 |
| embedding | 1,271,398,400 |
| 实际页对齐总分配 | **57,776,537,600（53.81 GiB）** |

普通进程 Working Set 不包括这类大页占用，不能用约 1 GiB 的 Worker 工作集判断驻留失败。相关小型检查用 `QueryWorkingSetEx` 验证了原生 LargePage 位。

## 已定位的原生崩溃

早期真实长请求在热集交接附近终止，HTTP 只看到断开的 Worker pipe。增加原始 IPC 错误和退出码后，确认退出码是 `0xC0000409`，子码为 `FAST_FAIL_FATAL_APP_EXIT`。

使用 `D:/Windows Kits/10/Debuggers/x64/cdb.exe` 和 Release 链接符号抓到调用链：

```text
ucrtbase!abort / terminate
ur_adapter_level_zero -> ur_loader!urEventWait
sycl::event::wait_and_throw
Qwen4ExpExpertCache::ExchangeResidentOwnership + 0x1a0
NormalizeResidentCore -> InstallHotset -> HandoffPromptHotset
```

反汇编把触发点定位到对旧 `incoming.dma_event` 的重复等待。`acquire` 的整批完成屏障已经覆盖这些 H2D；取消冗余的逐槽 DMA 等待，继续保留该整批屏障与 GPU 消费者 fence。修正后，逐次回写版本的 6K/6K/14K/14K 四个真实请求全部完成、Worker 正常退出。

这记录的是 oneAPI 2026.1 本机适配器的实际终止路径，不把它解释为已证实的物理内存不足。调试运行不用于性能对照。

## 验证范围

只运行大页所有权、完整专家编号规模热集交接，以及真实几何 CPU source 检查，没有运行全套模型或并发验收。覆盖：原始源被覆盖后的 CPU/GPU payload、nested 元数据、同批连续替换、借槽/恢复、move/释放、尾部 guard、页对齐记账；完整编号检查另外逐位核对全部 RAM 专家 payload。CPU 检查使用 H2560/I640/E512 和新建 executor，原始源与大页源输出逐位一致。

## 已撤回的尝试与分配修正

尝试过把同一 microbatch 的 D2H 回写合并等待。检查通过，但真实 14K 请求为 103.73 秒，之前逐次回写样本为 103.62 秒；6K 也只相差约 2%。中途还必须修正分配粒度，因此这不是隔离充分的批量回写因果比较。没有取得支持保留复杂度的正面证据，撤回批处理结构，未增加实验开关。原实现和结果归档在 `tmp/q4e-largepages-rejected-batch-cache.*`、`tmp/q4e-largepages-b6/result.json`。

单次约 54 GiB 的原生大页申请曾返回 Win32 1450。改为 1 GiB 分块后多次启动成功，但在一次新的物理内存状态下，仍在约 44 GiB 处失败，当时还有约 13 GiB available。单独的 256 MiB 分块探针随后成功申请全部 54 GiB，约 6.93 秒，全部释放。最终改用不超过 256 MiB 的专家块，连续 embedding 优先申请；后续完整模型加载成功。不能把单次成功解释为 Windows 在任意碎片状态下都保证分配成功。失败仍明确报错，不转用磁盘/可分页后备。

另一次从 mmap 基线重启到大页版本时，分配成功，但加载阶段 available 连续约 80 秒只有 0.17 GiB，并触发 Server 普通 RPC 的 120 秒超时。最终增加逐层工作集回收；Qwen4Exp 模型加载单独使用 10 分钟 deadline，普通控制 RPC 的 120 秒限制不变。预加载时间继续全额记入启动账本，不用加长 timeout 代替速度结果。

修正后的 B8 实际启动 79.17 秒，加载期可用物理内存最低 **4.00 GiB**，没有低于 1 GiB 的采样点；运行中最低约 3.84 GiB，四个请求全部完成，Worker 退出码为 0。

## 测量约束

- A770 16 GiB，驱动 32.0.101.8724；E5-2696 v3，18C/18T，HT 关闭；约 63.91 GiB RAM。
- oneAPI 2026.1.0（20260617）、oneDNN 3.11，Release 计算优化参数未变。
- 单会话、32K 容量、chunk4096、reserve3584、CPU 专家18线程、BF16 KV、MTP 关闭。
- WorkSummary、stage/component profile、cache trace、Worker 日志捕获全部关闭。
- 固定 `tmp/q4e-agent-corpus-v1` 自然代码/说明/日志语料；实际请求 5968 和 13979 prompt tokens，各生成 16 tokens。保留内容、token 与请求 hash。
- HTTP 序列关闭前缀复用，重复项仍然完整 prefill。未清空 OS 文件缓存，首个请求表示 Worker ready 后首次请求，不代表冷启动重启机器。
- 磁盘数据是 `Win32_PerfRawData_PerfDisk_PhysicalDisk` 累计值差，列出模型所在物理盘 `0 C: D:`；包含该盘其他进程和分页文件，不是逐文件归因，更不是 H2D 流量。
- 首字、整请求、启动/预加载分别记录；16-token 短输出不作为稳定 decode 20+ 的证据。

## 结果

完整同输入对照为 A4/B8，使用相同 Worker 和 Server 二进制；四项输入 hash、输出 token 数及输出 hash 均一致。只有一对完整序列，不把接近的差异解释为稳定加速。早期非驻留 6K 首请求曾为 96.80/97.22 秒，本对为 111.99 秒，说明冷页和首次触达存在波动。

| 阶段 | 原路径首字/总耗时 s | 大页首字/总耗时 s | 原路径读盘 GiB | 大页读盘 GiB |
|---|---:|---:|---:|---:|
| 6K 首次 | 109.70 / 111.99 | 73.18 / 74.82 | 46.678 | 6.539 |
| 6K 重复 | 31.93 / 33.24 | 32.63 / 33.93 | 0.158 | 0.003 |
| 14K 首次 | 103.12 / 104.89 | 101.56 / 103.15 | 10.289 | 7.422 |
| 14K 重复 | 73.11 / 74.39 | 73.95 / 75.26 | 2.957 | 0.045 |

启动由 **16.34 秒增加到 79.17 秒**，启动读盘为 10.71 / 64.97 GiB。把启动计入首个 6K 任务，总耗时为 **128.32 / 154.00 秒**；启动加四请求合计为 **340.84 / 366.34 秒**。不能据启动后的首请求较快，宣称冷启动任务整体加速。

**结论：完整驻留与显著减少运行中读盘已实现；暖态速度基本持平，计入预加载后尚未端到端胜出。保持大页功能入口，但暂不修改本机默认启动配置。** `AILA_Q4E_LARGE_PAGE_RESIDENT=1` 可显式启用，旧 `HOT_RAM_MB=0` 指关闭扩展 Host pool，并不表示这个大页补集为零。

```powershell
$env:AILA_Q4E_LARGE_PAGE_RESIDENT = '1'
pwsh -NoProfile -File tools/server/start-qwen38.ps1
```

## 真实 Agent、接续与取消

首轮 PI 实际成功返回标记，但多读了已附带的背景文件，不满足严格的一次 read 检查；快速工具续轮还撞到一次 429。保留这个失败记录，没有放宽检查来宣称通过。随后只澄清任务说明中背景已完整提供，继续要求恰好一次 marker read。由于提示和输出长度不同，不比较前后 PI 耗时来声称固定工作量加速。

429 的原因是已结束推理的请求仍因 TCP 排空占用推理准入槽。修正为：在发送正常结束帧之前释放推理句柄、完成前缀状态交接；排空中的连接仍受原有连接数、缓冲区和 deadline 限制，但不再占用推理准入。继续使用一个推理槽和原有单线程任务池，没有新增并行推理路径。

这项 HTTP 交接修正位于上述成对测速之后，Worker 二进制没有变化。Server 另用单槽六轮连续请求、保留先前响应 socket 的 CPU fake 检查，以及以下真实 PI/请求序列验证；不将前一 Server 二进制 hash 冒充最终 Server hash。

- 真实 PI：启动后的第一个任务，固定背景 + 标记文件，**恰好一次 read**，答案为 `AILA_FIXED_AGENT_READ_20261007`，整轮 **90.27 秒**。Worker 启动另计 **77.05 秒**，合计 **167.32 秒**。
- PI 首轮输入 6931、输出 26 tokens；工具返回续轮复用 6957、新增输入 32、输出 14 tokens。没有 429 或推理错误。
- 取消一次流式请求后，新请求正确返回 `READY`，耗时 1.51 秒。
- 14K 长请求：13979 输入、64 输出，总耗时 **106.73 秒**。
- 续轮：14067 输入，其中复用 **14043**、新增 **24**，输出 64；首字 **1.70 秒**，总耗时 **6.41 秒**，首字后速率 **13.39 tok/s**。
- 最终 Server 统计：完成 5、取消 1、失败 0、拒绝 0；正常退出。

**decode 20+ 尚未达到。** 本轮停止追加优化变体；数据已经把运行期专家读盘从主要疑点中分离出来，后续计算/传输优化须以完整任务时间为准。

## 原始证据

- `tmp/q4e-largepages-cdb3/cdb-output.log`、`tmp/q4e-largepages-exchange-disassembly.log`：终止调用栈与具体等待位置。
- `tmp/q4e-largepages-event-fix-ownership.log`、`tmp/q4e-largepages-event-fix-handoff.log`：事件修正后的检查。
- `tmp/q4e-largepages-b4/result.json`：逐次回写版本，完整四请求及退出码。
- `tmp/q4e-largepages-batch-ownership.log`、`tmp/q4e-largepages-batch-handoff.log`：批量回写、连续替换与全 RAM payload 检查。
- `tmp/q4e-largepages-async-test.log`：真实几何 CPU 源一致性检查。
- `tmp/q4e-largepages-final-ownership.log`、`tmp/q4e-largepages-final-handoff.log`：撤回批处理后的最终所有权与完整 payload 检查。
- `tmp/q4e-large-page-capacity-256.log`：256 MiB 分块容量探针。
- `tmp/q4e-largepages-b7/result.json`：加载阶段源页压力与 120 秒超时的失败样本。
- `tmp/q4e-largepages-load-trim-ownership.log`：逐层回收后的大页内容/所有权检查。
- `tmp/q4e-largepages-a4/result.json`、`tmp/q4e-largepages-b8/result.json`：同二进制正式对照。
- `tmp/q4e-agent-handoff-fake/result.json`：单槽、六轮、保留旧 socket 的交接检查。
- `tmp/q4e-largepages-agent-final/result.json`：保留的严格 PI 检查失败及 429 证据。
- `tmp/q4e-largepages-agent-handoff/result.json`：最终 PI、取消恢复、14K 续轮验证。
- 汇总数据：`docs/plans/qwen38-large-page-residency-2026-10-08.json`。
