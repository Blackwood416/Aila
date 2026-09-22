# 请求级 thinking budget

`RequestState`、IPC v2 与 C++ HTTP/SSE 服务共用请求级预算。旧 `ThinkingBudgetController` 及旧 C ABI 的生成路径保持原有接口。

HTTP 接受 `max_thinking_tokens`、`thinking_budget`、`thinking_budget_tokens` 三个互斥别名。`-1` 禁用预算，`0` 使用 no-think 模板；正整数上限为服务的生成额度上限。正预算且没有显式 `enable_thinking` 时开启 thinking，显式 false 仍保持 false。预算为 0 时覆盖显式 true。非法类型、重复别名和低于 -1 的值返回 400。结束标记来自服务加载的可信 tokenizer，不能由 HTTP 客户端指定。

```json
{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": "计算 7 乘以 9，并简短解释。"}],
  "max_completion_tokens": 64,
  "max_thinking_tokens": 8,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

预算只统计 thinking 内的内容 token，`<think>`、`</think>` 标记不计。自然关闭后若模型再次开启 thinking，继续使用本请求累计的额度。达到上限后，状态机以普通输出事件注入 tokenizer 编码的 `</think>\n\n`，随后继续生成答案。强制 token 计入 completion usage、penalty history 与 KV，强制输出对应的 forward 不采样、不消耗 RNG。每个 Request 独立持有预算、阶段和 RNG。

总生成额度仍由 `max_tokens` / `max_completion_tokens` 限定，包含 reasoning、结束序列与答案。如果剩余额度不足以容纳完整结束序列，消费最后一个已发布 token 后以 `length` 完成，生成数可以小于请求上限；不会发布半段强制序列，也不额外写入未报告的 token。预算为 0 且 prompt 已打开 thinking 的低层请求，若结束序列完全放不下，则完整消费 prompt 后以 `length` / 0 输出结束。取消仍可在步骤边界中断，包括强制结束序列期间。

IPC 的可选 `thinking_budget_v1` 能力不改变 protocol/ABI 版本和默认必需能力。未请求预算时 submit 的 wire 形状不变；旧 Worker 缺少能力时 Client 在发送前局部拒绝，其他请求仍可使用连接。可选的 submit 字段如下：

```json
"thinkingBudget": {
  "maxTokens": 8,
  "openTokenId": 151667,
  "closeTokenId": 151668,
  "promptOpensThink": true,
  "closeTokenIds": [151668, 271]
}
```

以上 ID 仅是格式示例，必须由相应模型 tokenizer 获取。两个标记必须不同、在实际词表内且不是 stop token；关闭列表长度为 1–16，首项是关闭标记，后续不能包含打开标记、重复关闭标记或 stop token。wire 校验 int32 范围，Client 与 Request 再按真实词表验证。`GenerationConfig::thinking_budget_tokens >= 0` 的直接调用必须同时提供 `RequestOptions::thinkingBudget`，两处显式预算冲突会拒绝。

验收分为 CPU 状态与协议测试、真实 IPC 的四预算并发/续写、HTTP JSON/SSE 的语义与 usage 对比，以及独立旧产物的默认路径回归。具体运行结果记录在 [重构验证](refactor-validation.md)。
