# Qwen4Exp Server：实验性 MTP T2 接入

2026-10-07，MTP 已从探针迁入 shared-model/Session/Request/Worker 执行路径，HTTP 仍使用原来的逐 token IPC/SSE。默认关闭；这是实验性 T2，不能称为严格无损或已达 Server 20+。

## 实现

- Qwen4ExpMtpDraft：由 DeepSeek V4.1 Flash 子代理提取原计算。模型持有 draft 权重、固定 64-slot 专家缓存和 scratch；session 独占 draft K/V、indexer 和最后一个 target hidden。删除了 teacher 生成、全专家扫描/预热和 dump 流程。
- Qwen4ExpMtpRuntime：共享预分配 checkpoint、GDN 输入、PLE prefix 和 greedy 缓冲，复用 backend 持有的 attention bundle。每次窗口不分配设备内存。
- DecodeWindowResult 显式报告 consumedInputs、cacheLength、firstTokenCached 与未缓存 bonus。Request 在提交前决定第一个输出是否可接受；拒绝或遇到 stop 时只提交原 pending 输入，并恢复/重放 GDN 与 PLE 的单输入前缀。
- 接受的 draft 已缓存，绝不重复 forward；bonus 下一次 Step 才发布并成为 pending。队列不足两个空位、输出短尾、采样/penalty、thinking budget、显式 host/legacy 模式走 T1。取消使用过 MTP 的请求会 reset session，避免复用超前状态。
- Reset 同步重置 draft 可见长度；Truncate 到非零旧 checkpoint 后该 session 保持 T1，直到 Reset/replay。长 context、非默认 GDN chunk 或 Q8 head 的模型配置也保持 T1。
- draft session 设备 payload 计入 admission/CacheBytes；C=2048 时额外 4870144 bytes，总计 181104640 bytes。IPC 的 S=0 host reservation 仍为 117669888 bytes。模型共享 scratch 与 draft 权重不乘 session 数。

## 验证与尚未解决的问题

RequestState/Scheduler CPU 测试通过，覆盖接受/拒绝、首个和 bonus stop、1..5 token 短尾、窗口期间/之后取消、队列不足两个位置以及既有 scheduler 行为。真实 shared-session 检查通过两 session 交错、追加、Reset/Truncate、内存限额及先释放 model/runtime 后继续使用；18 个窗口接受 6 次，销毁后显式 USM=0。

真实 HTTP health、24-token JSON/SSE 一致性、8-token 无 usage stream 和优雅回收通过。随后原 1833-token prompt 生成 128 tokens 也通过，128 个内容帧、无流错误、正常退出。

**T1 等价性没有通过。** 首次交错 MTP/T1 对照的 8 个输出中，第 6 个不同（7/8 相同）；在后续不同 prompt/hotset 历史下重复 T1 对照，8 个输出有 2 个不同。相同 MTP 模式的交错及追加对照通过，但这不能证明与 T1 严格等价，也未定位本轮分歧的首个算子。历史探针已有 B1/B2 数值漂移记录，当前仍需区分算术路径、hotset 历史和状态影响。保留初始失败日志，不将实验性结果描述为无损。

## 服务侧时延

| 原 prompt / 128 outputs | 首字 | 请求总时延 | 首字后输出速度 |
|---|---:|---:|---:|
| 初次迁移配置，遗漏 hidden reuse | 82.242 s | 92.464 s | 12.425 tok/s |
| 补回已验证的 hidden reuse 开关 | 50.511 s | 58.987 s | 14.984 tok/s |

速率定义为 127 / (SSE done − first content)，包含 IPC 和最后的 pending token forward，不是纯 kernel decode。两次均为新服务进程的首个长请求；驱动/文件缓存状态未严格控制，不能把全部差异归因于 hidden reuse。两次内容 SHA256 相同：87c0b8d46205410a75bfc6827f39f36f78d35d27e0ddbf80efd8a187d78875f1。

因此尚不能把探针的 20.5929 tok/s 搬作 Server 成绩。服务目前使用 512-token prefill chunk；后续先调查 T1 分歧和服务侧差距，避免重新开启大范围 kernel 搜索。

## 运行

仓库根目录加载本轮修正后的配置，再启动现有 Server：

~~~powershell
$config = Get-Content docs/plans/qwen38-server-mtp-2026-10-07.json -Raw | ConvertFrom-Json
foreach ($entry in $config.worker_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($entry.Name, [string]$entry.Value, 'Process')
}
./build-refactor-server/AilaServer.exe --worker ./build-refactor/AilaWorker.exe --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 --backend qwen4exp_nf4 --context 2048 --sessions 2 --decode-batch-size 1
~~~

AILA_Q4E_SERVER_MTP 是启动时读一次的开关，不接受请求期间修改。设为 0 并重启 Worker 恢复普通 T1。环境中的 profile 路径为本机路径。

构建记录：tmp/q4e-server-mtp-final-build.log，包含 Aila、AilaShared、AilaInference、Worker 和 fake Worker。CPU：tmp/q4e-server-mtp-cpu-final.log。真实 session：tmp/q4e-server-mtp-session-final.log。HTTP：tmp/q4e-server-mtp-http/result.json。长请求：tmp/q4e-server-mtp-long-reuse/result.json。完整环境和测量保存在同名 JSON。
