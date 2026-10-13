# Strata C5 / C10 / C12 / C13 判定（前置条件阻断类，2026-10-13）

四项的共同点：机制本身清楚，但在 Aila 当前形态/已知证据下不可兑现或前置未满足。
本文件按扫描文档的口径给结论与依据，不做重复实验（每项注明"重启条件"）。

---

## C5 自适应异步/重叠 + 交换旋转 — NO-GO（当前形态）

1. **重叠**：C1/C2 已证明本机（DG2/Windows/UR）上 H2D 与内存型 compute 在 memory path 上
   **相加**（`pingpong_mem 6.70 ms ≈ 串行和 7.16`、`max 3.73`）；swap 的传输走同一路径，
   把拷贝"藏进窗口"同样藏不住。Aila 自己的 swap 实测是 **decode −8.03%**（13.732→14.835，
   `d7b0649`，因此默认关闭），混合精度/放置改动还会破坏逐位等价（需另立 opt-in 门）。
2. **旋转**：Strata 明说旋转的前置是"每层专家块等大 + 常驻补集与交换缓冲 fully pinned +
   GPU-mapped"；Aila 已验证 **51 GiB 全量 pinned 首请求 `UR_OUT_OF_RESOURCES`**（roadmap:98-101），
   Strata 自己的 WDDM 文档也记录 Windows 专家 arena 的 pin 上限为 8 GiB（扫描文档 §3.11）。
   在 64 GB 补集上做旋转不可行；只 pin 小子集的收益（省一次 host memcpy）远小于其复杂度。
3. **重启条件**：出现"swap 打开后 decode 不低于关闭"的新证据（例如更大 VRAM 的卡、
   或新驱动下传输不再与 compute 相加）时，先重跑 C1 的 `--queue-overlap-probe` 再谈。

---

## C10 batch-MTP 取舍规则 + batch exactness 配方 — NO-GO（前置未满足）

1. **前置**：`--batch-mtp` 类收益要求 MTP 本身可用；Aila 的 MTP 在**等价门**未过
   （首分歧=被拒 verify 窗口 row0，pos=1492，`40d6bf6`），且旧口径下吞吐 15.0-15.3 vs
   T1 15.6 已净负（扫描文档 §1-C10 与 `qwen38-mtp-align-2026-10-11.md`）。Strata 自己也有
   "专家全驻时 batch-MTP −21.9%"的反例（2×R9700），说明该机制只在 miss 重时为正。
2. **可搬的只有规则**：Aila 已具备"串行窗口 vs 批量窗口"的开关族（MTP 相关 env 与
   `AILA_Q4E_WINDOW_ACCOUNT_LOG`），扫描文档要求的是"两条默认互斥并打印原因"的工程决定——
   由于 MTP 默认关，这条现在无实际冲突可解。
3. **重启条件**：B4-3 的等价门修复（或接受"verify 窗口退化为逐行"）后，按扫描文档的
   exactness 配方（固定 seed、禁 swap、禁 borrow、固定 PCIe 取向）先证"批量行 == 单行逐 token"。

---

## C12 小 chunk 的 CPU 分担 — NO-GO（与 Aila 最优 chunk 冲突）

1. Strata 的收益区间是 **≤1K token 的块**（512/1000 token −19~−35%；2K/4K/16K 不变，
   `docs/DETAILS.md:1564-1607`）。Aila 的 chunk 扫描是 **4096 最优**（2048/4096/8192 =
   83.89/70.57-70.64/89.22 s，roadmap:104-110），且 CPU 专家核已近 DRAM 墙。
2. Aila 唯一直接受益面是 14K prompt 的**末块（~1.7K）**；按 Strata 规则外推上界 ≈ ttft −0~3%，
   且输出会因 CPU/GPU rounding 不同而变（需 opt-in 门），不满足 C8 的 3% 发射规则。
3. **重启条件**：chunk 策略重估（例如高并发/多会话改变 chunk 选择）时一并评估。

---

## C13 KV 常驻/流式与 KV 预取 — NO-GO（≤32K 区间）

1. Aila 现状：KV 在 VRAM，int8 KV 已 opt-in（结构性 +156 槽、端到端 ~1%，`qwen38-int8kv-prod`），
   decode 的 staging 已归零、冷专家走 CPU（在 9.6 GB/s 链路上是正确的）。
2. 把 KV 放宿主会反向增加每层/每轮 PCIe，且与 C1/C2 的"传输与流式 compute 相加"叠加；
   Strata 自己的 KV 预取收益出现在 **32K/64K 长上下文**（−5.3%/−4.1%，2×RTX3080、PCIe 3.0、
   INT8 KV），8K 无收益、decode 无收益。
3. **重启条件**：Aila 支持并实际使用 ≥64K 上下文时，按扫描文档做"KV 宿主份额探针"，
   预取走现有 staging 池（复用 release/ready event 契约，遵守取消 drain 纪律）。
