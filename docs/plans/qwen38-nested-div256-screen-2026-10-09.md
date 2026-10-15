# Q4E nested scale 固定 256：Server 筛选与反序复核

2026-10-09 接续 [原交接](qwen38-nested-div256-handoff-2026-10-09.md)。实测使用 `88be93cf79cc95e4efec1ed95ba0853000d23d48` 上的未提交候选构建；五个候选源码文件已正式提交为 **`98cc037f865a4f9cc3ff5d1f88a764f73a2fdf12`**。本次没有增加新的运行时优化；测量结束后仅在头文件补充固定 256 参数的调用前置条件注释。

## 最终反序复核与保留状态

用户在首对筛选后明确选择保留候选并执行一次反序复核。实际顺序 **A1 → B1 → B2 → A2**，基线及候选各两次使用各自完全相同的归档二进制。Luna xhigh 协助核验调用覆盖、比较器和最终证据，并提供四次聚合工具。

`[MEASURED]` 14K 完整请求两对分别缩短 **2.62% / 2.18%**，中位数 **75.295 → 73.486 秒（-2.40%）**；首字中位数 **72.770 → 70.875 秒（-2.60%）**。收益方向在反序中重现，因此按用户决定保留该小幅长 prefill 优化。它不是所有提示、完整 32K 输入或 Server decode 的性能保证。

| 运行顺序 | 启动 s | 首个 PI s | 启动 + PI s | 14K 首字 s | 14K 完整 s | 续轮完整 s |
|---|---:|---:|---:|---:|---:|---:|
| A1 基线 | 126.582 | 79.099 | 205.681 | 72.752 | 75.286 | 4.315 |
| B1 候选 | 128.111 | 79.636 | 207.747 | 70.761 | 73.315 | 4.185 |
| B2 候选 | 123.876 | 79.172 | 203.049 | 70.990 | 73.658 | 4.238 |
| A2 基线 | 128.179 | 80.444 | 208.623 | 72.787 | 75.303 | 4.138 |

续轮在首对缩短 3.01%、反序慢 2.42%；中位数 **4.226 → 4.211 秒（-0.36%）**，没有一致收益。续轮首字后 tok/s 中位数 **12.419 → 12.487（+0.54%）**，仍约 12–13 tok/s。14K 新请求的短 decode 窗口 tok/s 中位数 **12.277 → 11.880（-3.24%）**，两对分别 -0.78% / -5.68%；不能宣称 decode 提速。以上每个窗口仅输出 32 tokens，没有为 decode 波动继续追加测量。

PI 首对慢 0.68%、反序快 1.58%，没有一致 PI 收益。启动加 PI 首对慢 1.00%、反序快 2.67%，描述性中位数 **207.152 → 205.398 秒（-0.85%）**；启动单项分别慢 1.21%、快 3.36%，同样不归因于特化提速。runner 全序列 A1/B1/B2/A2 为 294.071/294.076/290.070/298.078 秒，其两侧中位数约 296.074/292.073 秒；这含启动、PI、取消、请求和退出开销，波动受启动/PI影响，不用它宣称整序列稳定收益。

两对全部门禁及跨四次 fixture、环境、socket harness、二进制身份检查通过；四次 14K、续轮和 READY 恢复请求的完整 payload/输出/usage 一致，PI 规范化输出及用量也一致。四次均通过取消恢复，runner exit=0/timeout=false、服务 exit=0/forced_kill=false。没有额外长 prefill 首字前取消检查或全 CTest 重跑。原有定向 GPU 微测门禁与本次 4/4 Server CPU 检查作为功能证据。

最低 available RAM 的两秒采样为 A1=1.269、B1=2.436、B2=2.149、A2=2.063 GiB。系统内存状态不同，只有两对筛选，不声称已经建立普遍稳定收益或内存收益。

**当前状态：候选保留并已提交；官方 Worker/Server 双构建与身份同步完成，当前部署路径使用 98cc037 正式配对，服务进程已退出。** 测量用基线及候选归档不覆盖，没有开始候选 B/C。源码已跟踪文件干净，交接及结果文档仍为未跟踪文件。下文保留首对筛选与构建排障明细。

提交后执行 `build.ps1 -BuildDir build-refactor -Jobs 8` 与 `tools/server/build.ps1 -SkipDependencies -Parallelism 4`，两者成功；正式 Server 再次核验 32 个 object 的有效 header dependencies。CPU/ABI 筛选套件 **18 项通过、1 项 GPU 测试按配置跳过、0 失败**；Server **4/4 通过**。日志为 `tmp/q4e-nested-div-20261009/{build-final-worker.log,build-final-server.log,cpu-abi-final.log,cpu-server-final.log}`。没有在提交后重复 GPU 性能实验；本报告收益仍引用上述测量配对，正式重编只同步身份，运行行为不变。

## 结果与当前决定边界

`[MEASURED]` 一对生产筛选 A1 → B1 中，14K 暖态完整请求 **75.286 → 73.315 秒（-2.62%）**，首字 **72.752 → 70.761 秒（-2.74%）**。续轮 **4.315 → 4.185 秒（-3.01%）**。启动、首个 PI 及启动加 PI 没有改善。两个 runner 全序列墙钟均约 294.07 秒，未显示整体序列净收益；该墙钟还包含取消、客户端和退出开销，不能替代分项请求指标。

本节数据是**首对筛选**的历史明细。2.62% 落在原交接 §6 的 1–5% 小幅候选范围。用户随后明确选择“保留候选，做一次反序复核”；复核已完成，最终结果及当前状态见上节。

| 指标 | 基线 A1 | 候选 B1 | 候选变化 |
|---|---:|---:|---:|
| 启动至就绪 s | 126.582 | 128.111 | +1.21% |
| 首个 PI 工具任务 s | 79.099 | 79.636 | +0.68% |
| 启动 + PI s | 205.681 | 207.747 | +1.00% |
| PI 后 14K 新请求首字 s | 72.752 | 70.761 | -2.74% |
| PI 后 14K 新请求完整 s | 75.286 | 73.315 | -2.62% |
| 14K 新请求首字后 tok/s | 12.235 | 12.139 | -0.78% |
| 14K 前缀续轮首字 s | 1.739 | 1.729 | -0.57% |
| 14K 前缀续轮完整 s | 4.315 | 4.185 | -3.01% |
| 续轮首字后 tok/s | 12.035 | 12.624 | +4.89% |

启动与短 decode 的差异不作因果归因；Server decode 20+ 仍未达到。不能将已有微测的约 5% 内核链收益当成服务收益。

## Server 构建阻塞处理

`[TOOLCHAIN]` 当前 PowerShell 中直接运行新 pwsh 的官方脚本，在全新构建目录成功生成 `CMakeFiles/rules.ninja`、构建全部五个 Server 目标，并核验 32 个 C++ object 的有效非空 header dependencies：

```powershell
pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies -BuildRoot tmp/q4e-nested-div-20261009/server-resume-clean -Parallelism 4
```

随后原 `build-refactor-server` 也用官方脚本成功重建。实际仍使用 VS CMake 4.1.2 和 `Python313/Scripts/ninja.exe`，没有切换 Ninja、修改 build.ps1、握手逻辑或伪造 metadata。上一会话异常在当前会话未复现；现有证据不能确认根因是 Ninja 版本或旧会话 PATH。

首次 Server CPU 检查 3/4 通过；Pipe 测试失败原因是 Fake Worker 仍内嵌 a83cd80 身份，而新 Server 要求 88be93c。用现有定向 build helper 重建 `AilaFakeInferenceWorker`，再用官方 Server 脚本同步构建记录。之后 **4/4 CPU 测试通过**，含真实 named-pipe Fake Worker hello/隔离检查。成功日志归档于 `tmp/q4e-nested-div-20261009/cpu-server-resume.log`。真实 Worker 的候选哈希未因 Fake Worker 重编而改变。

## 固定工作量与功能门禁

`[CORRECTNESS]` Luna xhigh 子代理完成两轮只读审查：生产 true 参数只用于 adaptive cache 已校验的 256 描述符；decode hot/staging 与 prefill gate/up/down 覆盖完整；generic 默认值、非 nested 分支与两个 FMA 舍入步骤保留。已有两种 bench 的完整缓冲区位一致门禁有效，本次未重复 GPU 微测。

本机 Intel Arc A770 16 GiB，Windows 驱动 32.0.101.8724，oneAPI 2026.1.0、oneDNN 3.11.2；模型为 `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`。基线、候选使用同一冻结语料、相同 worker 环境、32768 capacity/chunk4096/reserve3584/CPU18/BF16 KV/MTP0/single-session 配置。首个业务请求均为真实 PI，随后取消恢复、14K 摘要和前缀续轮。正式 probe 清除了相同的 `AILA_PROFILE_Q4E_DECODE` 与 `AILA_Q4E_PROFILE_COMPONENTS`，没有诊断日志捕获。构建与 GPU 生产测量没有重叠。

比较器检查实际 launch 记录的 Worker/Server SHA256 与逐次配对记录、完整 request payload hash、messages hash、content/reasoning/tool-call hash、完整 semantic hash、usage、cached tokens、finish reason、错误列表、fixture 和 socket harness hash。PI 比较移除随机工具调用 ID 后的助手事件内容、函数名/参数、prompt hash、最终 marker 和完整 token 用量。**所有门禁为 true**。

- PI：恰好一次 `read(smoke-marker.txt)`，答案均为 `AILA_FIXED_AGENT_READ_20261007`；首轮 input=6931/output=26，第二轮新增 input=32/output=14/cacheRead=6957。
- 14K：prompt=13979/output=32/cached=0；续轮 prompt=14035/output=32/cached=14011，新增 24 tokens。两轮完整 payload 与输出一致。
- 断开后均正确返回 READY；清理 exit=0、forced_kill=false，runner exit=0、timeout=false。没有额外测试首字前长 prefill 取消。
- 每两秒采样的最低 available RAM 为 A1=1.269 GiB、B1=2.436 GiB。系统内存状态并非完全相同；这些采样不是精确内存峰值，也不作为候选改善内存占用的证据。

低层 C++ API 的 `nested_fixed256` 是调用方承诺，dispatch 本身不校验 descriptor blocksize。当前生产描述符构造有校验，头文件现已注明前置条件，未来其他调用方不能在非 256 几何下传 true。Luna 还指出 decode 微测固定 generic 在前/fixed 在后，故原微测保留为方向证据，不新增稳定性宣称。

## 二进制与复现资产

实测候选对已归档于 `tmp/q4e-nested-div-20261009/cand/`，提交后正式配对在 `retained/`，二者包括 Worker、Server、Fake Worker、两份官方 build metadata 和 `pair.json`。基线对仍在同目录的 `base/`，未改变。

| 配对 | Worker SHA256 | Server SHA256 | 内嵌身份 |
|---|---|---|---|
| 基线 | `8ce1efd990cb64332ebde8421989160e18a86810e916d88eed005c530d2a653d` | `020c8a89cef7146999d1bb5de2e680f84b8a74bf4f891bf3f30b4835e48662e9` | `aila-0.2.0-abi1-a83cd80a421fbe3d3dc789bcd6fe911445a9b9d5` |
| 实测候选 | `421d6a5cb3cc698406d56738b305d85fb40e3c1773dcd431980c3e8bbc988063` | `d09e1f197833a77b47db5f1cba10864ee58ab5b1ca7ca4f6dd16a428ed5ae7e2` | `aila-0.2.0-abi1-88be93cf79cc95e4efec1ed95ba0853000d23d48` |
| 正式配对 / 当前部署路径 | `06e37b91d644325d7ddefa192644af9918ecc42b352ca85961fa8c55db31597c` | `ef19c505a19833af7936203fcc5c84509a5be7b6ddeb3327962c01f8b656ad68` | `aila-0.2.0-abi1-98cc037f865a4f9cc3ff5d1f88a764f73a2fdf12` |

原始运行在 `tmp/q4e-resident-speed-20261009/{nested-a1,nested-b1,nested-b2,nested-a2}/`，对应 `*-monitor.json`、`*-run.log` 保留不覆盖。测量源码补丁在 `tmp/q4e-nested-div-20261009/candidate-source-resume.patch`，补充 API 注释后的补丁在 `candidate-source-final.patch`。

两对比较、聚合结果及全部哈希在 [同名 JSON](qwen38-nested-div256-screen-2026-10-09.json)。本地复现的 pair helper 会拒绝在服务进程运行时换装，并核对归档哈希及匹配身份；临时基线换装期间没有伪造任何 build metadata。

```powershell
python tmp/q4e-nested-div-20261009/pair_resume.py install base
python tmp/q4e-nested-div-20261009/pair_resume.py record <新的基线tag>
python tmp/q4e-resident-speed-20261009/run.py <新的基线tag> --agent
# 等待退出后，成对安装 cand，使用另一新 tag 执行相同序列。
python tmp/q4e-nested-div-20261009/compare_resume.py <基线tag> <候选tag>
# 所有测量进程退出后，恢复正式配对。
python tmp/q4e-nested-div-20261009/pair_resume.py install retained
```

正式提交与双构建同步已经完成。如以后明确决定撤回，应针对 `98cc037` 做 revert 并按官方 Worker/Server 双构建流程配齐新身份；原交接 §7 的未提交文件恢复方法是历史操作，不能再用 `git restore` 当前 HEAD 来移除此提交。全部微测、codegen、补丁与生产筛选资产保留。
