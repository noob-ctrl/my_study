# verl 训练内部机制学习笔记

> 本文档沉淀 verl 训练流程中的一系列核心机制原理（remove padding、TiledMLP、HybridEngine
> 架构、多种 log_prob 与重要性采样、rank/device_mesh 分工、权重 resharding 等），
> 用于日后理解与验证。内容基于对 verl 源码的逐行阅读，关键处均给出**文件:行号**方便复查。
>
> 主线从 `use_remove_padding` 展开，逐步延伸到与之相关的显存优化与训练架构。
>
> 相关文档：
> - Ray 相关：[`ray_notes.md`](./ray_notes.md)
> - verl 架构：[`verl_architecture_notes.md`](./verl_architecture_notes.md)

---

## 目录

- [0. 一句话结论](#0-一句话结论)
- [1. 两种 padding：别混为一谈](#1-两种-padding别混为一谈)
- [2. 数据处理阶段的 padding（对齐用）](#2-数据处理阶段的-padding对齐用)
- [3. `use_remove_padding` 到底做了什么](#3-use_remove_padding-到底做了什么)
- [4. 边界问题：cu_seqlens 是核心](#4-边界问题cu_seqlens-是核心)
- [5. cu_seqlens 从哪来：position_ids 反推](#5-cu_seqlens-从哪来position_ids-反推)
- [6. 为什么只需适配 attention 层](#6-为什么只需适配-attention-层)
- [7. HF 原生 unpad vs verl 全程pack](#7-hf-原生-unpad-vs-verl-全程-pack)
- [8. monkey patch 机制（FSDP 侧）](#8-monkey-patch-机制fsdp-侧)
- [9. Megatron 侧：thd 格式与 PackedSeqParams](#9-megatron-侧thd-格式与-packedseqparams)
- [10. 完整数据流图](#10-完整数据流图)
- [11. 常见误解与陷阱](#11-常见误解与陷阱)
- [12. 验证清单（怎么自己确认）](#12-验证清单怎么自己确认)
- [13. 附：TiledMLP —— MLP 层的省显存优化](#13-附tiledmlp--mlp-层的省显存优化)
- [14. HybridEngine：actor/rollout/ref 如何共存同一批 GPU](#14-hybridengineactorrolloutref-如何共存同一批-gpu)
- [15. 每张卡如何知道自己该干哪份活（rank + device_mesh）](#15-每张卡如何知道自己该干哪份活rank--device_mesh)
- [16. 三种 log_prob 与重要性采样](#16-三种-log_prob-与重要性采样)
- [17. vLLM rollout 深入：多实例部署与请求收发（承接 §14）](#17-vllm-rollout-深入多实例部署与请求收发承接-14)

---

## 0. 一句话结论

> verl 的 `use_remove_padding` = **全程把一个 micro-batch 的多条序列拼成一根无 padding 的长串
> `(1, total_nnz)` 送进模型计算**，从而端到端省显存/省算力；
> 由于只有 **attention** 需要区分序列边界，verl 只需适配 attention 层（让它从 `position_ids`
> 推出 `cu_seqlens`，把注意力限制在每条序列内部），其余模块（MLP/LN/logits）天然兼容长串。

---

## 1. 两种 padding：别混为一谈

初学最容易把两个"padding"当成矛盾，其实它们在**不同阶段、管不同的事**，是配合关系：

| 参数 | 位置 | 阶段 | 作用 |
|---|---|---|---|
| `max_prompt_length` + 左 padding | 数据处理（Dataset） | 进模型**之前** | 把不等长 prompt **对齐**成规整 batch 张量 |
| `use_remove_padding` | 模型前向（forward） | 进模型**之后** | 计算时把padding **去掉**，避免浪费算力/显存 |

- 前者负责"对齐"（不得不加的填充）。
- 后者负责"在计算时把对齐的代价消掉"。
- **一个在进模型前，一个在进模型后，不冲突。**

配置位置：
- FSDP：`actor_rollout_ref.model.use_remove_padding`
- Megatron：`actor_rollout_ref.actor.megatron.use_remove_padding`

---

## 2. 数据处理阶段的 padding（对齐用）

深度学习一个 batch 必须是规整矩形张量 `(batch, seqlen)`，但真实序列不等长，所以要补齐。
verl 用**左 padding**（`P` 为 padding token）：

```
seq A: [P P a1 a2]      真实 2 token
seq B: [b1 b2 b3 b4]    真实 4 token
seq C: [P P P c1]       真实 1 token
```

> 为什么左 padding？生成（rollout）要在序列**右侧**续写，左padding 让真实内容贴右对齐，方便续写。

这一步产生了大量无意义的 `P`，正是下一步 `use_remove_padding` 要消除的浪费。

---

## 3. `use_remove_padding` 到底做了什么

以 FSDP 的 `DpActor._forward_micro_batch` 为例（`verl/workers/actor/dp_actor.py`）。

### 3.1 摘 padding、拼长串

```python
# verl/workers/actor/dp_actor.py:164-168
if self.use_remove_padding:
    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
        input_ids.unsqueeze(-1), attention_mask
    )  # input_ids_rmpad (total_nnz, ...)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
```

`unpad_input`（来自 `flash_attn.bert_padding`）根据 `attention_mask` 返回三件套：

- **`input_ids_rmpad`**：所有真实 token 拼成的长串，`(bsz, seqlen)` → `(1, total_nnz)`
- **`indices`**：每个真实 token 在原 padding 矩阵里的位置（**还原时用**）
- **`cu_seqlens`**：每条序列的边界（**切边界时用**）

沿用上面的例子（`total_nnz = 2+4+1 = 7`）：

```
input_ids_rmpad = [a1 a2 b1 b2 b3 b4 c1]   形状 (1, 7)
```

### 3.2 送进模型时不再传 attention_mask

```python
# verl/workers/actor/dp_actor.py:244-251
output = self.actor_module(
    input_ids=input_ids_rmpad,
    attention_mask=None,          # ← 注意：不再传 mask，边界靠 position_ids
    position_ids=position_ids_rmpad,
    ...
)
```

### 3.3 结果也是长串，最后 pad 回去

```python
# logits 全程是长串形状，这是省显存最直观的证据：
# verl/workers/actor/dp_actor.py:258
logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)← 不是 (bsz, seqlen, vocab)!
```

```python
# 算完用 indices 还原回 (bsz, seqlen)：
# verl/workers/actor/dp_actor.py:331-336
full_log_probs = pad_input(
    hidden_states=log_probs.unsqueeze(-1),
    indices=indices,        # ← 用当初记录的 indices 还原
    batch=batch_size,
    seqlen=seqlen,
)
```

> 注意算 log_prob 前会把 label 左移一位（预测下一个 token）：
> `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`（`dp_actor.py:210`）。
>拼接串边界处会有一个 token 错位，但 loss 时用 `response_mask` 屏蔽，不影响正确性。

---

## 4. 边界问题：cu_seqlens 是核心

拼成长串后，怎么保证 `b1` 不去attend `a1 a2`（跨序列串味）？靠 **`cu_seqlens`（累积序列长度）**。

对例子：

```
各 seq 真实长度:  [2, 4, 1]
cu_seqlens:      [0, 2, 6, 7]← 前缀和
```

含义：
- 第 1 条 = 长串 `[0:2]` = `a1 a2`
- 第 2 条 = 长串 `[2:6]` = `b1 b2 b3 b4`
- 第 3 条 = 长串 `[6:7]` = `c1`

**FlashAttention 的 varlen 接口（`flash_attn_varlen_func`）拿到 `cu_seqlens` 后，把注意力严格限制在每条序列内部**，`b1` 绝不会看到 `a1 a2` 或 `c1`。这就是"拼长串不串味"的原因。

---

## 5. cu_seqlens 从哪来：position_ids 反推

代码里没有显式把 `cu_seqlens` 传给模型（只传了 `position_ids`）。因为它是**模型内部从 `position_ids` 现场推导**的。

```python
# verl/models/transformers/qwen2_vl.py:164-179
def prepare_fa2_from_position_ids(query, key, value, position_ids):
    ...
    position_ids = position_ids.view(-1)
    cu_seqlens = torch.cat(
        (
            (position_ids == 0).nonzero().view(-1).to(torch.int32),   # ★ 关键
            torch.tensor(position_ids.size(), ...),
        )
    )
    max_length = cu_seqlens.diff().max()
    return (query, key, value, (cu_seqlens, cu_seqlens), (max_length, max_length))
```

关键点：unpad 后每条序列的 `position_ids` **各自从 0 开始编号**，拼接后长这样：

```
position_ids_rmpad = [0 1 | 0 1 2 3 | 0]
                      ↑     ↑         ↑   每个"0"= 一条新序列的起点
```

所以 `(position_ids == 0)` 的下标 `[0, 2, 6]` + 总长 `7` → `cu_seqlens = [0, 2, 6, 7]`。
**边界信息被"编码"进了 position_ids，模型内部反推即可。** 这也是为什么 `position_ids` 必须同步 unpad
（它既管 RoPE，又兼职承载边界）。

### 模型怎么判断"这是不是一根拼接长串"？

verl patch 后的 attention 通过判断 **position_ids 是否单调不减**：

```python
# verl/models/transformers/qwen2_vl.py:228 附近
if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
    # 有下降（交界处 1→0）→ 判定为 packed 长串 → 走 flash_attn_varlen_func
    ...
else:
    # 普通路径
    attn_output = _flash_attention_forward(...)
```

- 普通序列 `[0,1,2,3,4]`：全程递增 → 普通路径。
- 拼接长串 `[0,1,0,1,2,3,0]`：交界处 `1→0` 出现下降 → packed 路径。

---

## 6. 为什么只需适配 attention 层

**核心洞察：一根长串里，只有 attention 需要知道"边界在哪"，其他模块都不需要。**

| 模块 | 是否跨 token 交互 | 是否需要边界 | 长串下要适配 |
|---|:---:|:---:|:---:|
| Embedding | 否 | 否 | 不用 |
| **Attention** | **是** | **是** | **需要（verl patch）** |
| MLP / FFN | 否 | 否 | 不用 |
| LayerNorm | 否 | 否 | 不用 |
| logits 输出层 | 否 | 否 | 不用 |

- **MLP/LN/Embedding/logits 都是"逐 token 独立计算"**：第 `i` 个 token 的输出只由它自己决定，
  后面跟的是 `b1` 还是别的对结果毫无影响 → 对拼接完全无感 → 不需要边界。
- **Attention 是唯一"跨 token 交互"的模块**（`q_i` 和所有 `k_j` 做点积）→ 必须知道边界。

所以 verl 策略很精准：**全程长串省显存，但只在唯一在乎边界的 attention 处适配，其余模块原封不动。**

---

## 7. HF 原生 unpad vs verl 全程 pack

一个关键澄清：**HF 原生 `_flash_attention_forward` 其实也会unpad**（内部调 `unpad_input` +
`flash_attn_varlen`），不是靠大mask 硬算。所以在 **attention 那一步**，HF 和 verl 省算力效果接近。

真正差异在于**unpad 的"范围"**：

| | HF 原生（方式 A） | verl（方式 B） |
|---|---|---|
| unpad 发生在 | **每层 attention 内部**（进attn 前 unpad，算完立刻 pad 回） | **全程**（进模型第一步就 unpad，直到最后才 pad 回） |
| attention 之外（MLP/LN/logits/层间激活） | 仍是 `(bsz, seqlen)`，**带 padding，占显存** | `(1, total_nnz)`，**无padding** |
| 能否叠加 Ulysses SP | 不能 | 能（长串才能按 token 切多卡） |
| 能否配合 dynamic batching（按 token 数组batch） | 不方便 | 能 |

**结论**：HF 只做到"attention 局部临时 unpad"，达不到"全程 pack 端到端省显存"，也无法叠加 SP。
verl 要的这种全程 pack 能力，HF 原生前向不具备 —— 这才是 verl 出手 patch 的真正原因。

> 省显存的**根本增益在 attention 之外**（MLP/logits/激活全程无 padding，尤其 logits
> `(total_nnz, vocab)` 而非 `(bsz, seqlen, vocab)`，省得最多）。
> attention patch 只是"让全程长串能正确运行"的**必要配套** + **叠加 SP** 的新能力，
> 它本身不是省显存的主因。

---

## 8. monkey patch 机制（FSDP 侧）

###8.1 做法：运行时替换类的方法

monkey patch = 运行时用自己的函数覆盖掉库里已有的方法。verl 的核心操作就是**赋值替换**：

```python
# verl/models/transformers/monkey_patch.py:394-399（VLM 例子）
if use_remove_padding or ulysses_sp_size > 1:
    from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward
    Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward
    Qwen2VLAttention.forward = qwen2_vl_attn_forward
```

```python
# verl/models/transformers/monkey_patch.py:483-491（纯文本模型的通用兜底分支）
if use_remove_padding or ulysses_sp_size > 1:
    if hasattr(module, "_flash_attention_forward"):
        module._flash_attention_forward = _ulysses_flash_attention_forward
    else:
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
```

### 8.2 几个关键事实

1. **触发条件是 `use_remove_padding or ulysses_sp_size > 1`**：开了 remove padding 或 SP 才 patch。
2. **逐模型手工适配**：VLM（`qwen2_vl`/`qwen2_5_vl`/`qwen3_vl`/`glm4v`/`kimi_vl`…）走各自的
   `if/elif` 分支（patch 更重，连 `Model.forward` 都换）；纯文本模型走最后的通用分支。
   **文本模型也适配**，不是不适配，只是更轻。
3. **patch 加的能力**：① 从 `position_ids` 推 `cu_seqlens`（适配无 mask 长串）；
   ② Ulysses SP 的 all-to-all（HF 完全没有）；③ 版本兼容。**不是"新增 unpad 能力"**——unpad 本身
   patch 前后都调 `flash_attn_varlen`。
4. **调用时机**：FSDP worker 构建模型后调用 `apply_monkey_patch(model,ulysses_sp_size, use_remove_padding, ...)`。
5. **不开 SP 时**：patch 仍会挂上，但 `_ulysses_flash_attention_forward` 内部
   `if ulysses_sp_size > 1` 不成立 → 退化为透传给 HF 原生实现。

---

## 9. Megatron 侧：thd 格式与 PackedSeqParams

### 9.1 用 data_format 开关

```python
# verl/workers/actor/megatron_actor.py:683
data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
```

- **`bshd`** = `[batch, seq, head, dim]`：常规带 padding 的 4D 张量。
- **`thd`** = `[total_tokens, head, dim]`：去 padding、拼成长串的 packed 格式（`t` = total tokens）。

即 **`thd` 就是 Megatron 版的 remove padding**。

### 9.2 关键认知：框架提供"消费 thd 的能力"，verl 负责"生产 thd"

Megatron-Core / TransformerEngine 的 attention **能计算 thd**，但需要 verl 把数据整理好、
连同 `PackedSeqParams` 一起喂进去。它不会自己去猜 padding 在哪、每条序列多长。

```python
# verl/models/mcore/util.py:43-130preprocess_packed_seqs（节选）
seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)     # 真实长度只有上层知道
tp_size = mpu.get_tensor_model_parallel_world_size()
cp_size = mpu.get_context_parallel_world_size()
align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size        # 要对齐并行策略
...
cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0) # verl 亲手构造 cu_seqlens
...
packed_seq_params = PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=cu_seqlens_padded,
    max_seqlen_q=max_seqlen_in_batch,
    cu_seqlens_kv=cu_seqlens_padded,
    ...
)
```

**为什么必须 verl 手动 pack、Megatron 不能自动转？**

1. **"哪里是padding"只有上层知道**：attention 算子拿到的是纯张量，不知道 `attention_mask`。
2. **packing 要和 TP/CP/FP8 对齐**：长度要凑 `tp_size*cp_size*2` 的倍数；CP 还要把每条序列切成
   `2*cp` 块按特定顺序分卡（因果掩码负载均衡）；FP8 还要凑 128 的倍数。这些**强耦合分布式配置**，
   只有掌握全局并行配置的 verl 层能做。
3. **算完还要 unpack**：`postprocess_packed_seqs`（`mcore/util.py:133`）把 thd 结果还原成
   `(batch, seq, ...)`。

### 9.3 FSDP vs Megatron 对照

| 维度 | FSDP 侧 | Megatron 侧 |
|---|---|---|
| attention 计算能力 | verl monkey patch 注入（HF + flash-attn） | Megatron-Core/TE **原生**（thd） |
| 谁做 unpad/pack | verl（`unpad_input`/`pad_input`） | verl（`preprocess_packed_seqs`/`postprocess_packed_seqs`） |
| cu_seqlens 来源 | 模型内部从 `position_ids` 反推 | verl 显式构造，塞进 `PackedSeqParams` |
| 额外并行处理 | Ulysses SP 的 all-to-all | TP/CP/FP8 对齐 + CP 分块 |

> 哲学差异：**FSDP 侧是"改造别人的模型（patch HF）来支持 unpad"；
> Megatron 侧是"用框架自带的 unpad 能力（thd），verl 只负责把数据喂对"。**
> 两边都需要 verl 准备数据，区别只是 attention 计算本身谁来实现。

---

## 10. 完整数据流图

```
输入（规整矩阵，含 padding）
   input_ids: (bsz, seqlen)  +  attention_mask
        │
        ▼  unpad_input()                         ←摘 padding、拼长串
   input_ids_rmpad: (1, total_nnz)   ┐
   indices（记录原位置，用于还原）    ├─三件套
   cu_seqlens（记录每个seq 边界）     ┘
        │
        ▼  position_ids 同步 unpad               ← 位置编码对齐，且承载边界（每条从 0 起）
        │
        ▼  model(input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad)
             ├─ Embedding / MLP / LN：逐 token 独立，长串透明，无需边界
             └─ Attention：从 position_ids 推 cu_seqlens，flash_attn_varlen 严格隔离每条 seq
        │
        ▼  logits_rmpad: (total_nnz, vocab)← 全程无 padding（省显存关键）
        │
        ▼  算 log_prob / entropy（都在长串上）
        │
        ▼  pad_input(indices, ...)               ← 用 indices 还原
   full_log_probs: (bsz, seqlen)
```

三个"配套数据"：**`indices` 负责还原、`cu_seqlens` 负责切边界、unpad 后的 `position_ids` 负责位置对齐 + 承载边界**。

---

## 11. 常见误解与陷阱

### 11.1 "`max_prompt_length` 的 padding 和 `use_remove_padding` 冲突"
❌ 不冲突。一个在进模型前（对齐），一个在进模型后（去填充），是配合关系。

### 11.2 "HF transformers 完全不支持 unpad"
❌ 错。HF `_flash_attention_forward` 内部也会 unpad + 调 `flash_attn_varlen`。
verl patch 的目的是"全程 pack + 从 position_ids 推边界 + 叠加 SP"，不是"教HF 学会 unpad"。

### 11.3 "省显存主要发生在 attention 层"
❌ 错。attention 那步 HF 也省了。verl 多省的是 **attention 之外**（MLP/LN/logits/层间激活），
靠"全程长串"实现，尤其 logits `(total_nnz, vocab)` 省得最多。

### 11.4 "verl 只适配 VLM，文本模型不适配"
❌ 错。文本模型走 `monkey_patch.py` 最后的通用兜底分支（替换 `_flash_attention_forward`），
同样被 patch，只是比 VLM 轻。

### 11.5 "代码里没传 cu_seqlens，所以没做边界隔离"
❌ 错。`cu_seqlens` 由模型内部从 `position_ids`（值为 0 的位置）现场推导，不需要显式传。

### 11.6 "拼成长串会导致序列之间信息串味"
❌ 错。attention 靠 `cu_seqlens` + varlen 严格隔离，`position_ids` 各序列从 0 起，RoPE 也正确。

### 11.7 "Megatron 支持 remove padding，所以能自动把数据转成 thd"
❌ 错。Megatron 提供"消费 thd 的能力"，但"生产 thd +构造 cu_seqlens"依赖 attention_mask 语义
和 TP/CP/FP8 全局配置，必须由 verl 手动做（`preprocess_packed_seqs`）。

### 11.8 "`use_remove_padding=False` 就一定更稳"
⚠️ 关掉是为了规避某些模型/自定义结构与 varlen/unpad 的兼容性问题。若开True 能跑通且结果正常，
通常建议开 True（纯收益的加速，无精度损失）。

---

## 12. 验证清单（怎么自己确认）

想亲自验证上面的结论，可以查这些点：

1. **确认全程长串（省显存证据）**：
   `verl/workers/actor/dp_actor.py:258` → `logits_rmpad` 是 `(total_nnz, vocab_size)`，
   不是 `(bsz, seqlen, vocab_size)`。

2. **确认不传 attention_mask**：
   `verl/workers/actor/dp_actor.py:244-251` → `attention_mask=None`。

3. **确认 cu_seqlens 从 position_ids 推**：
   `verl/models/transformers/qwen2_vl.py:164-179` → `(position_ids == 0).nonzero()`。

4. **确认 packed 判定条件**：
   `verl/models/transformers/qwen2_vl.py:228` 附近 → `torch.diff(position_ids) >= 0` 的判断。

5. **确认 patch 触发条件与逐模型分支**：
   `verl/models/transformers/monkey_patch.py:286`（`apply_monkey_patch` 定义）、
   `:352 / :406 / :440 / :467`（VLM 分支）、`:483-491`（文本模型通用分支）。

6. **确认 Megatron thd 开关与数据准备**：
   `verl/workers/actor/megatron_actor.py:683`（`data_format`）、
   `verl/models/mcore/util.py:43`（`preprocess_packed_seqs`）、
   `:118`（`PackedSeqParams` 构造）、`:133`（`postprocess_packed_seqs`）。

7. **运行期打印验证**：patch 生效时日志会打印
   `Monkey patch ... attention layer` / `Monkey patch _flash_attention_forward in ...`，
   以及 `Actor use_remove_padding=True`（`dp_actor.py:67`）。

---

## 13. 附：TiledMLP —— MLP 层的省显存优化

> 配置位置：`verl/trainer/config/model/hf_model.yaml` 的 `tiled_mlp` 段。
> 实现：`verl/models/transformers/tiled_mlp.py`。
> 与 remove padding 同属"省显存"家族，且都用 monkey patch 生效，故一并沉淀于此。

### 13.1 配置项

```yaml
# verl/trainer/config/model/hf_model.yaml:69-77
tiled_mlp:
  enabled: False       # 是否开启，默认关
  num_shards: 4        # 切成几块，越多越省显存但略慢
```

一句话：**TiledMLP 是"用时间换显存"的优化——把 MLP 的前向/反向按序列维切成`num_shards` 小块
依次计算，从而降低 MLP 那一步的峰值显存。**

### 13.2 为什么 MLP 是显存大头

Transformer 的 MLP（SwiGLU 风格）：

```python
# verl/models/transformers/tiled_mlp.py:161
return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))
```

`gate_proj(x)` / `up_proj(x)` 的输出维度是 **intermediate_size（通常是 hidden 的 3~4 倍）**。
输入 `(total_tokens, hidden)` 时，这两个中间激活 `(total_tokens, intermediate)` 非常大，
且反向传播要保存它们算梯度 —— 开了 remove padding 后 `total_tokens=total_nnz` 可能很大，MLP 中间激活容易成为单层显存瓶颈。

### 13.3 原理：分块 + 重算

**前向：切块 → 逐块算（不存激活）→ 拼接**

```python
# verl/models/transformers/tiled_mlp.py:101-106
x_shards = list(torch.chunk(x, chunks=shards, dim=-2))   # 沿序列维切shards 块
with torch.no_grad():                                     # ★ 前向不保存计算图/激活
    output_shards = [fn(module, x_shard) for x_shard in x_shards]
output_unsharded = torch.cat(output_shards, dim=-2)
```

每块只有 `total_tokens/num_shards` 个 token，中间激活也只有 `1/num_shards` 大。

**反向：逐块重算 + 累积梯度**

```python
# verl/models/transformers/tiled_mlp.py:134-151
for i, x_shard in enumerate(x_shards):
    ...
    with torch.enable_grad():
        output = fn(module, x_shard)          # 重新前向这一块（重计算）
    torch.autograd.backward(output, incoming_grad_shard)
grad_accumulator.cleanup()
```

反向时逐块重新前向、算梯度，再把各块对权重的梯度**累加**（`GradientAccumulator`）。
任意时刻显存里只有"一块"的中间激活。**本质 = activation checkpointing（重计算）+ 分块**，
用"多算几遍前向"换"峰值显存只需一块"。

### 13.4 怎么生效：monkey patch MLP 类

```python
# verl/models/transformers/tiled_mlp.py:229-236
def _patch_mlp_class(mlp_class, forward_fn, num_shards):
    def tiled_forward(self, x):
        compute_params = [p for p in self.parameters() if p.requires_grad]
        return TiledMLP.apply(forward_fn, self, x, num_shards, compute_params)
    mlp_class.forward = tiled_forward
```

入口在 `monkey_patch.py`，只有 `enabled=True` 才调用：

```python
# verl/models/transformers/monkey_patch.py:312-317
if use_tiled_mlp:
    from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch
    model_type = getattr(model.config, "model_type", None)
    apply_tiled_mlp_monkey_patch(num_shards=tiled_mlp_shards, model_type=model_type)
```

### 13.5 支持的模型（重要限制）

只支持枚举出来的 SwiGLU风格 MLP，其他类型会**直接报错**：

```python
# verl/models/transformers/tiled_mlp.py:169-174
_MODEL_TYPE_TO_MLP_CLASS = {
    "llama": (..., "LlamaMLP"),
    "qwen2": (..., "Qwen2MLP"),
    "qwen2_5": (..., "Qwen2MLP"),
    "qwen3": (..., "Qwen3MLP"),
}
```

即 **Llama / Qwen2 / Qwen2.5 / Qwen3**。其他模型可自行扩展这个映射表。

### 13.6 与其他省显存手段的关系

| 优化项 | 作用对象 | 机制 |
|--------|---------|------|
| `enable_gradient_checkpointing` | 整层 | 反向时重算整层激活 |
| `enable_activation_offload` | 激活 | 把激活挪到 CPU |
| `use_remove_padding` | 全程 | 去掉 padding token，减少计算量 |
| **`tiled_mlp`** | **仅 MLP 层** | **沿序列维分块 + 重算，降 MLP 峰值显存** |

它与 `use_remove_padding` **互补**：后者减少 token 总量，前者在给定 token 量下再削 MLP 峰值。

### 13.7 使用建议

1. 默认关闭，只在**显存吃紧、尤其长序列**时开。
2. 从 `num_shards: 4` 起，不够再往上加（8/16），代价是速度略降。
3. 仅 Llama/Qwen 系列有效。
4. 已开 gradient checkpointing 仍 OOM 时，可叠加 TiledMLP 进一步压MLP 峰值。

> 小结：**tiled_mlp = 沿序列维把 MLP 切成 `num_shards` 块、前向不存激活、反向逐块重算并累积梯度，
> 用少量额外计算时间换 MLP 层峰值显存的显著下降；默认关闭，仅支持 Llama/Qwen2/2.5/3。**

---

## 14. HybridEngine：actor/rollout/ref 如何共存同一批 GPU

> 相关代码：`verl/workers/fsdp_workers.py`（`ActorRolloutRefWorker`）、
> `verl/workers/rollout/vllm_rollout/`。

### 14.1 不是全局单例，而是"每卡一个实例"

`ActorRolloutRefWorker` **不是全局一个对象**。在 single_controller 下，一个 WorkerGroup 有
`world_size` 个 worker（每张 GPU 一个 Ray actor 进程）：

- 16 张卡 → **16 个 `ActorRolloutRefWorker` 实例**（16 个进程，每卡一个）。
- "actor/rollout/ref 共存"的准确含义：**每个 worker 进程内部同时持有 actor 模型 + rollout 引擎
  + ref 模型三个组件**（同进程、同 GPU）。

### 14.2 核心：同一批 GPU，两套不同的 device_mesh

actor 和 rollout 的并行配置可以不同（副本数不同），靠的是**在同一批物理卡上建立两套不同的逻辑视图
（device_mesh）**：

**actor 的 mesh（训练用）**
```python
# verl/workers/fsdp_workers.py:177-183
self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
dp = world_size // self.ulysses_sequence_parallel_size
if self.ulysses_sequence_parallel_size > 1:
    self.ulysses_device_mesh = init_device_mesh(
        device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
    )
```

**rollout 的 mesh（推理用）**
```python
# verl/workers/fsdp_workers.py:694-704
infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
infer_pp = self.config.rollout.pipeline_model_parallel_size
infer_world_size = infer_tp * infer_pp
dp = self.world_size // infer_world_size
assert self.world_size % infer_world_size == 0   # ← 约束：推理并行度必须整除总卡数
rollout_device_mesh = init_device_mesh(
    device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
)
```

关键点：
- **同一批 GPU（`world_size` 张）被复用**，两套 mesh 只是"切法不同"。
- **副本数（dp）自然不同**：actor 的 dp = `world_size / sp`；rollout 的 dp = `world_size / (tp*pp)`。
- **约束**：`world_size % infer_world_size == 0`。

### 14.3 例子：16 卡，rollout TP=2，actor 纯 FSDP

| | 并行配置 | 在 16 卡上的划分 | 副本数 |
|---|---|---|---|
| actor（训练） | FSDP dp=16 | 16 卡 1 个 FSDP 组 | dp=16 |
| rollout（推理） | TP=2 | 每 2 卡 1 个 vLLM 实例 | 8 个副本 |

同一批 16 卡：训练时看成 16 路 FSDP 组，生成时看成 8 个 TP=2 的 vLLM 引擎。
**物理卡不变，逻辑视图在 train/generate 间切换 —— 这就是 HybridEngine 的精髓。**

### 14.4 两套配置如何协调

1. **dispatch/collect 按角色分别注册**：
```python
# verl/workers/fsdp_workers.py:711-718
is_collect = (rollout_device_mesh["infer_tp"].get_local_rank() == 0
              and rollout_device_mesh["infer_pp"].get_local_rank() == 0)
self._register_dispatch_collect_info("rollout",
    dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect)
```
   actor 和 rollout 各注册一套切分规则，controller 下发数据时按调用的角色选对应规则。

2. **权重 resharding**：train→generate 切换时，把 FSDP 分片的权重重组后灌给 vLLM（见 §14.2 两种
   切法不同，必须转换）。这是 verl 特有的胶水逻辑，FSDP/vLLM 无法自动完成。

---

## 15. 每张卡如何知道自己该干哪份活（rank + device_mesh）

核心思想（SPMD）：**16 个进程跑完全相同的代码，唯一差异是各自的 `rank`（工号）；
每个进程拿 rank 去查 device_mesh（分工表），自己算出该干哪份活，无需中央调度员逐个通知。**

### 15.1 第一步：rank 从哪来

Ray 拉起进程时注入 `RANK/WORLD_SIZE` 等环境变量，进程启动后读取：

```python
# verl/workers/fsdp_workers.py:155-162
if not torch.distributed.is_initialized():
    rank = int(os.environ.get("RANK", 0))            # "我是几号"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.distributed.init_process_group(backend=..., rank=rank, world_size=world_size, ...)
```

16 个进程代码相同，但 `os.environ["RANK"]` 各不相同（0~15）—— 这是"分化"的起点。

### 15.2 第二步：actor 侧靠 FSDP 按 rank 自动分片

```python
# verl/workers/fsdp_workers.py:172-174
world_size = torch.distributed.get_world_size()
self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=...)
```

FSDP 就像"自动分工的组长"：16 个人调 `FSDP(同一个模型)`，它根据每人rank 自动发给每人 1/16 参数。
你只管调 `FSDP(model)`，"我该拿哪份"由它按 rank 算好。

### 15.3 第三步：rollout 侧靠 rank 查表算角色

```python
# verl/workers/fsdp_workers.py:711-718（同§14.4）
# get_local_rank() = "我在这个维度的小组里排第几"
```

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py:84-85
if replica_rank == -1:
    self.replica_rank = rank // rollout_world_size   # 纯算术：算出"我属于第几个 vLLM 副本"
```

例（16 卡 TP=2）：rank 0/1 → 实例 0（rank 0 是主卡 `is_collect=True`，rank 1 是从卡）；
rank 2/3 → 实例 1……每个进程自己拿 rank 一算就知道归属，无需通知。

### 15.4 为什么 actor/rollout 不同分工却不乱

同一个 rank 在两张表里查出的位置可以不同：

| rank | actor 表（FSDP dp=16） | rollout 表（TP=2, 8 实例） |
|---|---|---|
| 0 | actor 参数第 0 片 | vLLM 实例 0 主卡 |
| 1 | actor 参数第 1 片 | vLLM 实例 0 从卡 |
| 2 | actor 参数第 2 片 | vLLM 实例 1 主卡 |

同一张卡训练时按 actor 表办事、生成时按 rollout 表办事；切换时做权重 resharding 对齐。

### 15.5 分层总结

| 层级 | 谁负责 | 干什么 |
|------|--------|--------|
| 资源/进程层 | **Ray**（verl 调用） | 拉起进程，注入 `RANK/WORLD_SIZE/MASTER_ADDR...` |
| 进程组层 | **verl** | 读 rank，`init_process_group`，建 device_mesh |
| 分片/并行层 | **FSDP / vLLM** | 拿 rank + mesh，**自动**完成分片/TP 划分 |
| 桥接层 | **verl** | train↔generate 的**权重 resharding**（FSDP/vLLM 无法自动） |

> 一句话：**Ray 发工号，verl 定分工 + 架权重同步的桥，FSDP/vLLM 照分工表各自自动干活。**

---

## 16. 三种 log_prob 与重要性采样

### 16.1 重要性采样（Importance Sampling）

用从分布 `q` 采样的样本估计分布 `p` 下的期望，通过权重 `p/q` 修正偏差：

\[
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
\]

PPO 里：想优化当前策略 `π_θ`，但数据是旧策略 `π_old` 采的→ 用比值修正 off-policy 偏差：

\[
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}, \quad
L^{CLIP} = \mathbb{E}\left[\min\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
\]

### 16.2 三种 log_prob 对照

| log_prob | 符号 | 何时算 | 哪个引擎 | 更新中是否变 | 用途 |
|---|---|---|---|:---:|---|
| `rollout_log_probs` | π_rollout | rollout 生成时顺带产出 | **推理引擎**(vLLM/SGLang) | — | mismatch校正参考 |
| **`old_log_probs`** | **π_old** | 更新**开始前算一次** | **actor(训练)模型**，纯前向 | **固定** | PPO 比值**分母** |
| `ref_log_prob` | π_ref | old 之后算一次 | **ref 模型**，纯前向 | 固定 | 算 KL 惩罚 |
| **`log_probs`(new)** | **π_θ** | **每个mini-batch** 都算 | **actor(训练)模型**，前向+反向 | **每步变** | PPO 比值**分子** |

### 16.3 时序

```
① rollout 生成 → rollout_log_probs (π_rollout, 推理引擎)
②更新前算一次 → old_log_probs (π_old, 训练引擎) ← 固定分母
   （用 log_prob_micro_batch_size_per_gpu 控制 batch，纯前向可设大）
③ ref 模型算一次 → ref_log_prob (π_ref)
④ mini-batch 更新循环：每步 actor 算 log_probs (π_θ, 分子)
   → r_t = exp(log_probs - old_log_probs) → clip loss → 反向更新
```

对应代码：
```python
# verl/trainer/ppo/ray_trainer.py:1402-1404（old，更新前算一次）
else:  # Recompute old_log_probs
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
```

### 16.4 为什么 old 和 new 都用 actor 算、不用 rollout

必须让分子(new)和分母(old)**来自同一引擎**（都用 actor 训练引擎），这样更新第一步
（`π_θ` 未动时）`r_t` 严格等于 1，PPO clip 语义才正确。
若分母用 `rollout_log_probs`（推理引擎算的），会因**推理/训练引擎数值mismatch** 导致 `r_t ≠ 1`。

### 16.5 关于 `log_prob_micro_batch_size_per_gpu` 的归属

- 控制的是"**用 actor 模型重算 old_log_probs**"那一步的每卡 micro-batch（纯前向、无反向，可设大）。
- 它定义在 `rollout.yaml` 里（`verl/workers/config/rollout.py:184-187`），但**逻辑上属于 actor**：
```python
# verl/workers/config/rollout.py:183
# note that the logprob computation should belong to the actor  ← 源码注释自认
```
- 原因：HybridEngine下 actor/rollout 同worker，重算紧接rollout 之后；历史沿革。
  新 worker 实现（`engine_workers.py:521-523`）已在把它迁回 actor 侧。
- 实际执行：读出后塞进 `meta_info["micro_batch_size"]`，交给 `dp_actor.compute_log_prob`
  （`dp_actor.py:451`）执行。

---

## 17. vLLM rollout 深入：多实例部署与请求收发（承接 §14）

> 本章深入 HybridEngine 的 rollout 侧：8 卡 TP=N 如何部署多个 vLLM 实例、三个 rollout 相关对象
> 的区别、谁去发请求、返回 None 如何被 collect 处理。
> 相关代码：`verl/workers/rollout/replica.py`、`verl/workers/rollout/vllm_rollout/`、
> `verl/experimental/agent_loop/agent_loop.py`、`verl/workers/rollout/base.py`。

### 17.1 纠正常见误解：一批卡会起多个 vLLM 实例，不会浪费

"8 卡 TP=4 只用 4 张、剩 4 张空闲"是**错的**。verl 会起 **`8/4 = 2` 个vLLM 实例**填满全部卡：

```
8 卡 TP=4：
  ┌──实例0(replica 0) ──┐   ┌── 实例1(replica 1) ──┐
  │  GPU 0 1 2 3         │   │  GPU 4 5 6 7         │
  │  TP=4，1 个完整模型   │   │  TP=4，另一份副本    │
  └─────────────────────┘   └─────────────────────┘
```

两个实例是**数据并行**关系（各持完整模型，并行处理不同 prompt 提高吞吐）。

计算实例数的代码：
```python
# verl/experimental/agent_loop/agent_loop.py:952-973
rollout_world_size = tp * dp * pp            # 一个实例需要几张卡
world_size = worker_group.world_size          # 总卡数
num_replicas = world_size // rollout_world_size   # 8 // 4 = 2
self.rollout_replicas = [
    self.rollout_replica_class(replica_rank=replica_rank, ...)
    for replica_rank in range(num_replicas)   # 循环创建 N 个 replica
]
```

每个 replica 按 `replica_rank` 切走属于自己的一批 worker：
```python
# verl/workers/rollout/replica.py:134-136
self.workers = worker_group.workers[
    self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
]
```

### 17.2 三个 rollout 相关对象（最容易混淆，重点）

verl 里有**三个都跟 "rollout"沾边**的对象，务必分清（用"点外卖"比喻）：

| 对象 | 真实类名 | 比喻 | 数量(8卡TP=2) | 职责 | 在哪 |
|------|---------|------|:---:|------|------|
| `ActorRolloutRefWorker.rollout` | **`ServerAdapter`** | 外卖 App（客户端） | 8（每卡一个） | 向 vLLM server 发请求，**自己不推理** | worker 进程内 |
| `RolloutReplica` | `vLLMReplica` | 区域经理（开店） | 4（每实例一个） | 把多卡 worker 组织成 1 个 vLLM server | trainer/driver 侧 |
| vLLM server | vLLMHttpServer + EngineCore | 餐厅（做菜） | 4 | 真正在 GPU 上跑推理 | 独立 actor/子进程 |

**关键澄清**：
- `ActorRolloutRefWorker.rollout` **不是** `RolloutReplica`，而是 `ServerAdapter`
  （`verl/workers/rollout/vllm_rollout/vllm_rollout.py:61`）。**没有 `vLLMRollout` 这个类**。
- `ServerAdapter` 的定位（源码 docstring）：**"serve as a client to request vLLM server"** —— 是**客户端**。
- 注册表：`("vllm","async") -> "...vllm_rollout.ServerAdapter"`（`verl/workers/rollout/base.py:83-87`）。

### 17.3 两条"初始化路径"（解答"到底谁初始化 rollout"）

容易混淆的根源：存在两条独立路径，都叫初始化 rollout：

**路径 A（worker 内，每卡一个客户端）**
```python
# verl/workers/fsdp_workers.py:720-724
self.rollout = get_rollout_class(...)(config=..., model_config=..., device_mesh=...)
# → self.rollout 是 ServerAdapter（客户端/遥控器）
```

**路径 B（trainer 侧，开真正的 server）**
```python
# verl/trainer/ppo/ray_trainer.py:838-843
self.async_rollout_manager = AgentLoopManager.create(
    config=self.config, worker_group=self.actor_rollout_wg, ...)
# → 内部创建 RolloutReplica，切 workers，拉起 vLLM server
```

- 路径 A 装的是"外卖 App"（客户端），每卡一个。
- 路径 B 开的是"餐厅"（vLLM server），几张卡合开一家。
- 两件不同的事，都叫 rollout 才混淆。

### 17.4 客户端如何找到 server：约定名字 + 懒加载

`ServerAdapter` 初始化时 `server_handle = None`（还没有 server）。首次使用时，用**约定的名字**
自己去Ray 里查出 server 句柄（懒加载），**不是被 RolloutReplica 推送地址**：

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py:129-137
if self.rollout_rank != 0:
    return None                                   # ★ 非 0 号卡不发请求
# Lazy init：http server在 hybrid engine 之后才启动，故用名字现查
if self.server_handle is None:
    self.server_handle = ray.get_actor(f"vllm_server_{self.replica_rank}_{self.node_rank}")
future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
```

对接靠"店名"约定：RolloutReplica 开店时给 server 起名 `vllm_server_{replica_rank}_{node_rank}`
（`vllm_async_server.py:839-840`），ServerAdapter 用同名 `ray.get_actor` 查。

### 17.5 谁去发请求：只有 TP 组 0 号卡（is_collect 的一体两面）

一个 TP=2 实例有 2 张卡、2 个 ServerAdapter，但**只有组内 0 号卡真正发请求**：

```python
# vllm_rollout.py:129-130
if self.rollout_rank != 0:
    return None      # 非 0 号卡直接返回 None，不发请求
```

这与"谁收结果"用的是**同一个 rank 判断**：
```python
# verl/workers/fsdp_workers.py:712-717
is_collect = (rollout_device_mesh["infer_tp"].get_local_rank() == 0
              and rollout_device_mesh["infer_pp"].get_local_rank() == 0)
```

| 卡 | 发请求 | 返回值 | is_collect | 结果被收 |
|---|:---:|:---:|:---:|:---:|
| TP 组 0 号卡 | ✅ |真实结果 | True | ✅ |
| TP 组非 0 号卡 | ❌ | `None` | False | ❌ |

**对外一个入口（0 号卡下单/报结果），对内 2卡 TP 协作推理** —— "下单一个人，做菜俩厨师"。

### 17.6 返回的 None 如何被 collect 处理：被 collect_mask 过滤

非 0 号卡返回的 `None` 在收集阶段被**直接过滤**，不会进入最终结果：

```python
# verl/single_controller/base/decorator.py:235-246
def collect_nd_compute(collect_mask, worker_group, output):
    assert len(output) == worker_group.world_size    # 每卡都返回一个（含 None）
    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        if collect_mask[global_rank]:                # ★ 只采纳 is_collect=True 的卡
            output_in_dp.append(output[global_rank])
    return output_in_dp                # None 的卡被跳过
```

**天然对齐**：返回 None 的卡 == `is_collect=False` 的卡（都是 "TP 组非 0 号卡"），用的是同一套
rank 判断，所以 None 永远不会混进结果。

数量验证（8 卡 TP=2）：`output` 长度 8（4 真实 + 4 个 None）→ `collect_mask =
[T,F,T,F,T,F,T,F]` → 过滤后 4 份真实结果 → concat 成完整 batch。正好 4 个数据并行实例各一份。

### 17.7 verl 在多实例部署中做了哪些工作

| # | 工作 | 代码 |
|---|------|------|
| 1 | 算实例数 | `num_replicas = world_size // rollout_world_size`（agent_loop.py:963） |
| 2 | 循环建 N 个 RolloutReplica | agent_loop.py:965-973 |
| 3 | 按 replica_rank 切 worker 分配资源 | replica.py:134-136 |
| 4 | 反查每个 worker 的物理 GPU | `worker.__ray_call__`（vllm_async_server.py:817-829） |
| 5 |拼 CUDA_VISIBLE_DEVICES 交给 vLLM | vllm_async_server.py:833-838, 109 |
| 6 | 并发拉起所有 server | `asyncio.gather(*[server.init_hybrid(...)...])`（agent_loop.py:976） |
| 7 | 管理多实例地址/分发/生命周期 | AgentLoopManager |

而"在给定卡上建 TP 通信组、切权重、跑推理"是**vLLM 自己**做的
（`distributed_executor_backend="mp"`，vllm_async_server.py:306）。

### 17.8 关于通信组：一批卡上有两套

- **训练通信组**：verl 建 8 个 worker 时 `init_process_group` 建的 **8 卡大组**（FSDP/Megatron 用）。
- **推理通信组**：vLLM 自己建的**若干 2 卡小组**（TP=2 → 4 个独立小组），与训练大组无关、互不复用。
- 两者并行拓扑不同（训练 all-reduce 全体 /推理组内 TP），故无法共用；用不同进程、不同 master
  port、不同 NCCL 域隔离（verl 还设 `NCCL_CUMEM_ENABLE=0` 防冲突）。

> **小结**：一批 GPU 起多个 vLLM 实例（`总卡数 /每实例卡数`）；三个对象各司其职
> —— **RolloutReplica 开店、vLLM server 做菜、ServerAdapter 点菜**；客户端靠"约定店名 +
> 懒加载"找server；**只有 TP 组 0 号卡发请求**，非 0 号卡返回 None 且在 collect 时被
> `collect_mask` 过滤；训练与推理**两套通信组并存、隔离**。

---

>一句话总结：**remove padding = 把多条序列拼成无padding 长串全程计算；
> MLP/LN/logits 逐 token 独立、对长串透明；唯一需要边界的 attention 靠 cu_seqlens 隔离；
> FSDP 侧靠 monkey patch HF + position_ids 反推边界，Megatron 侧靠原生 thd + verl 构造 PackedSeqParams。**
