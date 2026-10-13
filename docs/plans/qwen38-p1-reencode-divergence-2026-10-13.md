# P1 prefix-reuse 分叉定位：服务端 BPE 跳过 `#` 开头的 merges（2026-10-13）

## 结论（TL;DR）

- **客户端无责**：`--generate 128` 首轮的 SSE 流文本与续轮请求里的 assistant 消息
  逐字节相同（649/649），pi 客户端没有 trim/normalize/丢特殊 token。
- **分叉点是服务端 tokenizer 缺陷**：`src/utils/Tokenizer.cpp:449` 在加载 `merges.txt` 时
  `line[0] == '#'` 把 **83 条以 `#` 开头的合法合并规则**当注释跳过（本模型 247587 → 247504 条），
  于是 `##`、`###`、`####`、`# include`、`# define`、`# if` 等码文体合并全部失效。
- **本次现场的直接因果**：模型在第 69 个生成 token 选了 `'###'`(id 13962)，服务端重编码把它拆成
  `'#'(2) '#'(2) '#'(2)` → `prompt[14047]=2 ≠ cached[14047]=13962` → `TakeSession` 判定非严格前缀
  → `cached_prompt_tokens=0` → 续轮整段重跑 14K prefill（≈70 s）。
- **最小修复**（一行）：只跳过 `#version` 头行，不再跳过所有 `#` 开头行。
  tmp 补丁副本离线验证：`##`→550、`###`→13962、`####`→794；assistant 文本重编码 130→**128** token，
  第 68 位恢复为 **13962**；prompt1/prompt2 从 13979/14133 收敛到 13956/14108；
  加载合并数 247504 → **247587**（= tokenizer.json 全量，rank 逐行对齐）。
- **残余缺口（独立缺陷，需后续处理）**：C++ `pretokenize` 的“多空白 + 标点/单词”边界仍与
  tokenizer.json 参考不一致。参考 regex 用 `\s+(?!\S)` 回溯，把**空白串的最后一个空格**留给后面的
  `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` / ` ?[^\s\p{L}\p{M}\p{N}]+` 分支（如 `'  -'` → `[220,471]`、
  `'   **'` → `[256,2972]`、缩进 `'      |'` → `[…'ĠĠĠĠĠ','Ġ|']`）；C++ 的空白分支把整段空白一次吞掉
  （`'  -'` → `[256,12]`、`'   **'` → `[262,332]`）。两者 token 数相同、id 不同。
  本次 fixture 上：prompt1 911 处、生成区 1 处（content 120-121）、prompt2 共 912 处；
  **生成区（128 token）只命中索引 120-121 这一处**。因此“只修 `#` 后本现场能否完全恢复复用”
  取决于模型在该处的选择，**离线不可判定，需修复后 live 复跑**（修复本身会改变 prompt1 分词，
  需要新 run 的缓存流；模型在该处的输出也不受离线分析约束）。

## 证据链

1. **线上分叉现场**（`tmp/q4e-agent-plan-audit-20261012/p1-g128/stderr.log`）：

   ```
   [PfxDiag] take slot=0 cached_len=14107 prompt_len=14133 diverge_at=14047 qualifying=0 cached_tok=13962 prompt_tok=2
   ```

   缓存流 = 首轮 prompt1（13979 token）+ 128 个生成 token = 14107；续轮 prompt2 = 14133。
   `diverge_at=14047` ⇒ 生成区索引 0..67（前 68 个 token）全部匹配；索引 68（第 69 个生成 token）
   处不一致：cached=`'###'`(13962) vs prompt=`'#'`(2)。

2. **对照 `--generate 32`**（`tmp/q4e-resident-speed-20261009/p1-g32/stderr.log`，同 fixture 单变量）：

   ```
   [PfxDiag] take slot=0 cached_len=14011 prompt_len=14035 diverge_at=14011 qualifying=1
   ```

   32 token 的生成段内没有 `#` 合并依赖 → 修复前后都应命中。

3. **客户端链路**：`p1_analyze.py textdiff` 对 g128/g32 均输出 `IDENTICAL`
   （SSE 拼接文本 == 续轮 assistant 内容，长度 649/171 相同），排除客户端裁剪/规范化。

4. **服务端渲染复现**：按 checkpoint `chat_template.jinja` 语义重建
   （user/assistant `<think>` 包裹分支 + `enable_thinking=false` 的
   `<think>\n\n</think>\n\n` 生成提示，`p1_render.py`），用 **原样编译的**
   `src/utils/Tokenizer.cpp`（`p1_tokenizer_probe.exe`）编码得到
   **13979 / 14133**，与 PfxDiag 完全一致；`ids[14047] == 2`。

5. **参考实现**：HF `tokenizers`（tokenizer.json）对同一 assistant 文本重编码为 128 token，
   `[68] == 13962`（`'###'`），与 PfxDiag 的 `cached_tok` 相同；即模型在该处输出的 id 与参考分词一致，
   是服务端重编码偏离了它。

6. **根因行**：`src/utils/Tokenizer.cpp:449`

   ```cpp
   if (line.empty() || line[0] == '#') continue;   // 误把合法 '#' 合并行当注释
   ```

   `merges.txt` 中 83 行以 `#` 开头（`# #`、`## ##`、`# include`、`#### ####`、`# define`、
   `# if`、`# endif` …），服务端日志 “Loaded 247504 merge rules” 与 247587 − 83 完全吻合；
   `merges.txt` 本身没有 `#version` 头行（首行即 `Ġ Ġ`）。该行由提交 `a43db06`
   (feat: enable quantized Qwen3.5 vision path) 引入，全仓库仅此一处。

7. **tmp 补丁副本验证**（`make_fixed_copy.py` + `p1_tokenizer_probe_fixed.exe`）：

   | 输入 | 修复前 | 修复后 | tokenizer.json 参考 |
   |---|---|---|---|
   | `##` | `2,2` | `550` | `550` |
   | `###` | `2,2,2` | `13962` | `13962` |
   | `####` | `2,2,2,2` | `794` | `794` |
   | assistant 文本（649 字符） | 130 token，`[68]=2` | **128 token，`[68]=13962`** | 128 token，`[68]=13962` |
   | prompt1 / prompt2 | 13979 / 14133 | 13956 / 14108 | 13944 / 14096 |
   | 加载合并规则 | 247504 | 247587 | 247587 |

   全部由 `p1_verify.py` 断言，日志 `p1-g128/verify-offline.log`（ALL CHECKS PASSED）。

## 修复建议

**最小补丁（唯一必要改动）**：`src/utils/Tokenizer.cpp:449`

```cpp
// before
if (line.empty() || line[0] == '#') continue;
// after: merges.txt 的头行只有 "#version"，其余 '#' 开头行都是合法合并
if (line.empty() || (line[0] == '#' && line.rfind("#version", 0) == 0)) continue;
```

- rank 语义不变：`rank` 只用于相对顺序比较（`bpe_encode` 取最小 rank，平局取最左），
  修复后 rank 恰好等于文件行号，与 tokenizer.json 的 merges 顺序一致。
- 现有 CPU 测试的 fixture `merges.txt` 只写 `#version: 0.2`，修复后仍会被正确跳过，不破坏测试。
- 采纳纪律：重建 AilaServer（Tokenizer.cpp 编入 `AilaServerChat`；worker 侧同样源码入编，
  按既有 pair 纪律两侧同建），先跑 CPU/ABI 测试，再按同一 p1-g128 fixture A/B/B/A：
  比对 prompt token 数、`[PfxDiag] qualifying/cached_prompt_tokens`、续轮首字与 tok/s。

**后续（不阻塞本次修复）**：

1. `pretokenize` 与 tokenizer.json 的预切分尚未完全一致（见 TL;DR 残余缺口清单）。
   在修复 `#` 后，若 live 复跑仍在生成区 miss，第一嫌疑就是它；建议下一轮把
   `\s+` 与“标点/单词前导空格”的边界对齐参考 regex，或引入 conformance 语料逐一比对。
2. 建议加 round-trip 回归用例（真实 checkpoint tokenizer）：
   `encode(decode(model_tokens)) == model_tokens`，样本覆盖 `#`/markdown/缩进代码/多空格标点，
   把这类“等长换 id”问题在 CPU 侧拦住。
3. 不为 P1 引入 prefix trie / 多分支快照池；不对 DeltaNet/PLE 做任意 LCP 截断（边界不变）。

## 离线复现命令（全程零 GPU）

```powershell
$d = 'tmp/q4e-agent-plan-audit-20261012'
# 1. 客户端链路（SSE vs 续轮 assistant 文本）
python $d/p1_analyze.py textdiff `
  --first-sse $d/p1-g128/responses/context-14000.sse `
  --continue-request $d/p1-g128/context-14000-continue-request.json
# 2. 按 checkpoint 模板重建 prompt1/prompt2
python $d/p1_render.py --request $d/p1-g128/context-14000-request.json `
  --out $d/p1-g128/render-prompt1.txt
python $d/p1_render.py --request $d/p1-g128/context-14000-continue-request.json `
  --out $d/p1-g128/render-prompt2.txt
# 3. 编译并运行服务端 tokenizer 探针（原文）与 tmp 修复副本
#    见本目录 obj/、obj-fixed/ 的 cl 命令行；入口 p1_tokenizer_probe.cpp
python $d/p1_verify.py          # 全断言，输出 verify-offline.log
python $d/p1_spans.py --text $d/p1-g128/render-prompt2.txt `
  --cpp-ids $d/p1-g128/fixed-prompt2-ids.txt    # 残余差异清单
```

## 产物索引

- 现场：`tmp/q4e-agent-plan-audit-20261012/p1-g128/{stderr.log,stdout.log,launch.json,responses/}`
- 对照：`tmp/q4e-resident-speed-20261009/p1-g32/{stderr.log,responses/}`
- 脚本：`p1_analyze.py`、`p1_render.py`、`p1_token_probe.py`、`p1_compare.py`、`p1_spans.py`、
  `p1_verify.py`、`make_fixed_copy.py`、`p1_tokenizer_probe.cpp`（均在 `tmp/q4e-agent-plan-audit-20261012/`）
- 探针与日志：`p1_tokenizer_probe.exe`、`p1_tokenizer_probe_fixed.exe`、`Tokenizer_fixed.cpp`、
  `p1-g128/{cpp-prompt1-ids.txt,cpp-prompt2-ids.txt,fixed-*-ids.txt,verify-offline.log}`
- 运行身份：`p1-g128/launch.json`（server sha256 `75f92518…`、worker sha256 `880a9158…`）
