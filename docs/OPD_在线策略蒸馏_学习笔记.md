# OPD（On-Policy Distillation，在线策略蒸馏）学习笔记

> 本文是对 verl 中 OPD 相关内容的通俗总结与答疑沉淀，重点覆盖学习过程中容易困惑的地方，方便日后复习。
>
> 关联源码 / 文档：
> - `docs/algo/opd.md`（官方算法文档）
> - `verl/workers/config/distillation.py`（配置 dataclass）
> - `verl/trainer/distillation/losses.py`（loss 主逻辑、注册表）
> - `verl/trainer/distillation/fsdp/losses.py` / `megatron/losses.py`（top-k forward KL 后端实现）
> - `verl/workers/engine/{fsdp,megatron}/transformer_impl.py`（前向 + logits processor）

---

## 目录

1. [OPD 是什么（一句话 + 直觉）](#一opd-是什么)
2. [KL 散度：公式与 forward / reverse 的区别](#二kl-散度公式与-forward--reverse-的区别)
3. [GKD OPD（监督式蒸馏）](#三gkd-opd监督式蒸馏)
4. [PG OPD（策略梯度蒸馏）](#四pg-opd策略梯度蒸馏)
5. [Stop-Gradient（停止梯度）为什么必须加](#五stop-gradient停止梯度为什么必须加)
6. [为什么 reverse KL 能单样本估计，forward KL 不能](#六为什么-reverse-kl-能单样本估计forward-kl-不能)
7. [GKD OPD 和 SFT / 传统 KD 的异同](#七gkd-opd-和-sft--传统-kd-的异同)
8. [实现细节：student 的 log_prob 从哪来](#八实现细节student-的-log_prob-从哪来)
9. [配置与关键约束速查](#九配置与关键约束速查)
10. [架构两段式速览](#十架构两段式速览)
11. [博客《On-Policy Distillation》补充视角](#十一博客on-policy-distillation补充视角)

---

## 一、OPD 是什么

**一句话**：让 student 从**自己**的策略采样 rollout，在这些"student 自己走出的状态"上，用 teacher 给出的下一 token 分布 / log-prob 作为监督信号进行蒸馏。

**三点动机**（对比其他范式）：

1. 蒸馏：把 teacher 的知识迁移到 student。
2. 相比 **SFT / 传统 KD**：减少 **exposure bias**（暴露偏差）——让训练时的状态分布 = 推理时的状态分布。
3. 相比 **RLVR**：提供 dense 的 **token 级**监督，而不是稀疏的、序列级的 0/1 结果奖励。

**关键词 On-Policy 的含义**：监督信号（teacher 分布）是"外来的（off）"，但**状态采样是 on-policy 的**（跟着 student 当前策略走）。

verl 里实现了两种变体：

- **GKD OPD** —— 监督式，直接反向传播 forward KL（`use_policy_gradient=false`, `loss_mode=forward_kl_topk`）
- **PG OPD** —— 把 reverse KL 的负估计当 reward，走 PPO 策略梯度（`use_policy_gradient=true`, `loss_mode=k1`）

---

## 二、KL 散度：公式与 forward / reverse 的区别

### 2.1 通用公式

对任意两个分布 P、Q（定义在同一取值集合，例如整个词表 V 上）：

```
KL(P || Q) = Σ_v  P(v) · log( P(v) / Q(v) )
      = E_{v~P} [ log( P(v) / Q(v) ) ]
```

记忆要点：

- **竖线 `||` 左边的分布 P，就是求和 / 求期望的"加权分布"（采样分布）**。
- KL **不对称**：`KL(P||Q) ≠ KL(Q||P)`，方向不同行为差很多。

### 2.2 forward / reverse 相对于谁

约定以 **teacher 分布 ν（目标分布）** 为基准，student 为 π_θ：

**Forward KL** —— teacher 在左（以 teacher 为加权分布）：

```
KL(ν || π_θ) = Σ_v  ν(v) · [ log ν(v) − log π_θ(v) ]
           = E_{v~ν} [ ... ]
```

- 加权分布 = teacher。
- 行为：**mass-covering / 覆盖式**——逼 student 覆盖 teacher 所有"可能的词"，宁可广撒网。
- 对应 verl 的 **GKD OPD**。

**Reverse KL** —— student 在左（以 student 为加权分布）：

```
KL(π_θ || ν) = Σ_v  π_θ(v) · [ log π_θ(v) − log ν(v) ]
             = E_{v~π_θ} [ log π_θ(v) − log ν(v) ]
```

- 加权分布 = student。
- 行为：**mode-seeking / 找峰式**——student 集中到 teacher 的某个主峰，追求"说得准"而非"说得全"。
- 对应 verl 的 **PG OPD**。

> **重要澄清**：forward KL **也能**写成期望形式（任何 KL 都能）。真正的区别不是"能不能写成期望"，而是 **"这个期望对谁采样"**：
> - forward KL：`E_{v~ν}`（对 **teacher** 采样）
> - reverse KL：`E_{v~π_θ}`（对 **student** 采样）
>
> 这决定了能否用"student 采样出来的那个 token"做单样本估计（见第六节）。

### 2.3 对比表

| | Forward KL `KL(ν‖π_θ)` | Reverse KL `KL(π_θ‖ν)` |
|---|---|---|
| 加权 / 采样分布 | teacher ν | student π_θ |
| 行为 | 覆盖 teacher 所有可能（广） | 集中到 teacher 主峰（准） |
| 需要的信息 | teacher **完整分布**（top-k 近似） | 仅采样 token 上两者的 logprob |
| verl 变体 | GKD OPD | PG OPD |

---

## 三、GKD OPD（监督式蒸馏）

**GKD = Generalized Knowledge Distillation**。配置：`use_policy_gradient=false`, `loss_mode=forward_kl_topk`。

**做法**：在 student 走出的每个状态 s_t 上，取 teacher 的 **top-k** 候选词及其概率，直接最小化 forward KL：

```
L_GKD^(k)(s_t) = Σ_{v ∈ TopK(ν)}  ν(v) · [ log ν(v) − log π_θ(v) ]
```

这个 loss **直接对 student 概率反向传播**（普通监督式梯度下降，无 RL 机制）。

**为什么只用 top-k**：推理引擎（vLLM/SGLang）只能返回"采样 token + top-k token"的 logprob，无法在任意指定 token 上取值。所以只能做 teacher-top-k 的 forward KL。

**特点**：用了 teacher 完整（top-k）分布，信号密集、稳定；适合追求稳定、充分利用 teacher 分布的场景。

---

## 四、PG OPD（策略梯度蒸馏）

**PG = Policy Gradient**（Thinking Machines 的 On-Policy Distillation 做法）。配置：`use_policy_gradient=true`, 推荐 `loss_mode=k1`。

**三步做法**：

1. reverse KL 的单样本（k1）估计，对采样 token y_t：
   ```
   D ≈ log π_θ(y_t | s_t) − log ν(y_t | s_t)
   ```
2. 取负值当 **reward**（散度越小奖励越高）：
   ```
   r_t = log ν(y_t | s_t) − log π_θ(y_t | s_t)
   ```
3. 用这个 reward 走 **PPO 式 clipped 策略梯度**更新（代码里把 `-distillation_losses` 当 `advantages` 喂给 `ppo_loss`）。

**直觉**：不再逐 token 对齐 teacher 整个分布，而是——"你（student）说的这个词，老师也认可（log ν 高）就给正奖励，鼓励多说；老师不认可就给负奖励"。然后用 RL 提升好行为概率。

**特点**：只需采样 token 上两个 logprob，计算 / 通信轻量；天然 on-policy；但只对采样 token 施加梯度，用不上 top-k 分布信号——**不要**和 `forward_kl_topk` 一起用（代码会警告）。

---

## 五、Stop-Gradient（停止梯度）为什么必须加

**stop-gradient**（记作 `sg(·)`，PyTorch 里就是 `.detach()`）：某个量在**前向数值计算**里照常参与，但**反向传播时梯度不往它里面回传**，即把它当成"冻结常数"。

### 5.1 核心原因：奖励在策略梯度里必须是常数

策略梯度优化"最大化期望奖励"`J(θ) = E_{y_t~π_θ}[ r_t ]`，其标准梯度公式（REINFORCE / policy-gradient theorem）：

```
∇_θ J(θ) = E_{y_t~π_θ} [ r_t · ∇_θ log π_θ(y_t | s_t) ]
```

- 梯度**只从 `∇_θ log π_θ(y_t)` 这一项产生**；
- `r_t` 只是**标量系数**（给动作打分：好不好、好多少），**本身不参与求导**；
- 这套推导成立的**前提就是 r_t 与 θ 无关（是常数）**。

所以必须用 `.detach()` 把 reward 冻结成常数，才能让自动微分算出的梯度与上式一致。

verl 实现：先 `.detach()` 再当 advantage：

```python
# verl/trainer/distillation/losses.py
advantages = -distillation_losses.detach()
```

### 5.2 不加会怎样（重点难点，拆开推导）

reward 组成：`r_t = log ν(y_t) − log π_θ(y_t)`

- `log ν(y_t)`：teacher **冻结**，对 θ 求导 = 0，本来就带不出梯度。
- `log π_θ(y_t)`：是 θ 的函数，可求导。

PG OPD 的 loss（简化 vanilla，不含 clip）：`L = − r_t · log π_θ(y_t)`

**正确（有 sg）**：r_t 是常数，只对后面的 log π_θ 求导：
```
∇_θ L = − r_t · ∇_θ log π_θ(y_t)      ← 干净的策略梯度
```
梯度的正负 / 大小由 r_t（老师的评价）调制：老师认可(r_t>0)推高概率，不认可(r_t<0)压低。**老师信号通过 r_t 影响梯度。**

**错误（无 sg）**：reward 里的 log π_θ 也参与求导。代入：
```
L_wrong = − ( log ν − log π_θ ) · log π_θ
```
对 θ 求导（log ν 为常数，用乘积法则）：
```
∇_θ L_wrong = − [  (∇_θ(−log π_θ)) · log π_θ        ← 多出来的伪梯度项（只含 log π_θ）
    + ( log ν − log π_θ ) · ∇_θ log π_θ  ← 正确的那一项
    ]
```

**关键观察**：多出来的第一项 `(∇_θ(−log π_θ)) · log π_θ` **只含 log π_θ，完全不含 log ν**——它是 student "自己和自己"作用出来的梯度，与老师无关。

**极端例子直观理解**：假设 student 已和 teacher 完全一致，`log π_θ(y_t) = log ν(y_t)`，则 `r_t = 0`，正确梯度应趋于 0（已学好，不该再动）。但错误做法里即使 r_t=0，那个伪梯度项**仍不为 0**，还在瞎推 student 概率，方向只由 student 自己的 log π_θ 决定，跟"老师觉得对不对"毫无关系。

**结论**：不加 stop-gradient，梯度会混入一项"只由 student 自身 logprob 决定、与 teacher 无关"的伪梯度，把老师的监督信号淹没 / 架空，蒸馏退化成 student 自娱自乐（无脑放大 / 缩小自己已有概率）。加上 sg 就切断这条错误求导路径，使梯度严格等于"老师评价 × 策略梯度"。

---

## 六、为什么 reverse KL 能单样本估计，forward KL 不能

### 6.1 蒙特卡洛单样本估计的前提

估计任意期望 `E_{v~p}[f(v)]`：从 p 采样若干样本对 f 取平均；只采一个样本就是"单样本估计"：

```
E_{v~p}[f(v)] ≈ f(v¹),   v¹ ~ p
```

**无偏的硬性前提：样本必须来自 p 这个分布本身。**

### 6.2 套到 reverse KL（能用）

```
KL(π_θ || ν) = E_{v~π_θ} [ log π_θ(v) − log ν(v) ]
```

OPD 里 student rollout 时每个 token y_t **本来就是从 π_θ(·|s_t) 采样出来的**——恰好满足"样本来自 π_θ"。于是直接拿现成 y_t：

```
KL(π_θ || ν) ≈ log π_θ(y_t) − log ν(y_t),   y_t ~ π_θ
```

只需采样 token 上 student、teacher 各一个 logprob 相减 = **k1 估计器**。这就是 PG OPD 计算 / 通信轻量的根本原因（老师只需返回采样 token 的单个 logprob）。

### 6.3 套到 forward KL（不能白嫖）

```
KL(ν || π_θ) = E_{v~ν} [ log ν(v) − log π_θ(v) ]
```

无偏单样本估计要求样本来自 **teacher ν**，但现成的 y_t 来自 **student π_θ**——**分布对不上**。硬要做只有两条路，都不划算：

1. **重要性采样纠偏**：乘权重 `ν(y_t)/π_θ(y_t)`，方差可能极大，训练不稳。
2. **枚举分布**（GKD 的选择）：不做采样估计，直接拿 teacher **top-k 完整分布**显式求和 `Σ_{v∈TopK} ν(v)[...]`。

所以 GKD OPD 需要老师返回 top-k 概率清单，而非一个采样 token 的 logprob。

### 6.4 「词表维度」vs「状态轨迹维度」（易混淆点）

学习中最容易混淆的一点：

- **状态轨迹维度**：teacher 若自己生成序列，会走上和 student 不同的句子路径（exposure bias 的来源）。—— OPD 已经用"**统一采用 student 的 rollout 轨迹**"把状态 s_t 统一了：teacher 和 student **共享同一个状态 s_t**。
- **词表候选维度**（forward/reverse 真正的差异所在）：站在**同一个固定状态 s_t** 上，问"下一个词填什么"，forward KL 在词表上以 ν 加权 / 采样，reverse KL 以 π_θ 加权 / 采样。

> **"样本不是从 teacher 采的"的准确含义**：在同一个共享状态 s_t 上，现成的候选词 y_t 来自 π_θ 而非 ν——是**词表候选层面**的差异，**不是**"整条序列由谁生成"层面的差异。序列由谁生成的问题已被"统一用 student 轨迹"解决。

**teacher 如何拿到该状态的分布**：student 生成完整 rollout（prompt+response）后，把整条序列喂给 teacher 做**一次前向**（`max_tokens=1` + `prompt_logprobs`），teacher 即可返回序列每个位置上它自己的下一 token 分布（top-k）。于是在同一 s_t 上同时拥有 `ν(·|s_t)`（top-k）和 `π_θ(·|s_t)`，可枚举 top-k 求和。

### 6.5 汇总表

| | Forward KL（GKD） | Reverse KL（PG） |
|---|---|---|
| 期望对谁采样 | v ~ ν（teacher） | v ~ π_θ（student） |
| OPD 现成 token y_t 来自 | π_θ（**对不上**） | π_θ（**正好对上**） |
| 能否单样本无偏估计 | 不能（要么重要性采样、要么枚举） | 能，直接用 y_t |
| 实际做法 | 枚举 teacher top-k 分布显式求和 | 单样本 `log π_θ(y_t) − log ν(y_t)` |
| 老师返回什么 | top-k 概率清单 | 采样 token 上一个 logprob |

---

## 七、GKD OPD 和 SFT / 传统 KD 的异同

### 7.1 相似（loss 形式层面）

| | 目标信号 | loss 形式 |
|---|---|---|
| SFT | one-hot 硬标签（真实下一个词） | 交叉熵 `−log π_θ(y_t^真)` |
| 传统 KD | teacher 软分布 ν(·\|s_t) | forward KL / 交叉熵对齐 teacher |
| GKD OPD | teacher 软分布 ν(·\|s_t)（top-k） | forward KL 对齐 teacher |

从 loss 看，**GKD OPD ≈ 传统 KD**；SFT 是"目标分布为 one-hot"的特例。三者都是**直接反向传播的监督式 loss**，无 RL 机制。

### 7.2 根本区别（数据 / 状态从哪来：on-policy vs off-policy）

- **SFT / 传统 KD（off-policy）**：前缀 y_<t 来自**固定数据集 / teacher 生成的答案**；student 在"别人走出的状态"上学习；训练态 ≠ 推理态。
- **GKD OPD（on-policy）**：前缀 y_<t 是 **student 用当前策略 π_θ 实时 rollout** 出来的；teacher 只在这些状态上打分；训练态 = 推理态。

### 7.3 为什么重要：exposure bias

例子：student 偏好**代数证明**，teacher 偏好**几何证明**。

- **传统 KD**：只教了"几何证明状态下 teacher 怎么走"；推理时 student 走上代数证明路，从没被教过，一旦走偏无人纠正 → exposure bias。
- **GKD OPD**：student 自己先走代数证明轨迹，teacher 在**这条 student 实际走出的代数状态**上指导 → 训练态=推理态，无 exposure bias。

> 类比：传统 KD =「老师示范自己的解法，学生背下来」；GKD OPD =「学生自己做题，老师在学生做的每一步旁边实时纠正」。

### 7.4 汇总表

| 维度 | SFT | 传统 KD | GKD OPD |
|---|---|---|---|
| 目标信号 | one-hot 硬标签 | teacher 软分布 | teacher 软分布（top-k） |
| loss 形式 | 交叉熵（监督） | forward KL（监督） | forward KL（监督）—**与传统 KD 基本一致** |
| 状态 s_t 来自 | 数据集 | 数据集 / teacher 生成 | **student 自己 rollout** |
| on / off-policy | off | off | **on-policy** |
| exposure bias | 有 | 有 | **无（训练态=推理态）** |
| 需要 RL 机制 | 否 | 否 | 否（GKD 监督式；PG OPD 才用 RL） |

---

## 八、实现细节：student 的 log_prob 从哪来

**结论：和 RL（PPO/GRPO）一样，student 的 log_prob 是 actor 在训练时对 (prompt+response) 序列重新跑一遍 forward 得到的，不是取自 rollout（vLLM/SGLang）引擎给的那个值。**

### 8.1 为什么必须重新 forward —— 梯度

蒸馏 loss 要对 student 参数 θ **反向传播**，`log π_θ(y_t)` 必须**挂在计算图上、带梯度**。而 rollout 引擎产出的 logprob 是 `no_grad`、脱离训练计算图的纯数值，**不能反传**。所以必须让 actor 重新 forward。

- GKD 甚至需要**完整 logits** 来算 top-k 分布对齐，rollout 引擎更不可能提供。
- PG 虽只用采样 token 的 logprob，但也必须是**带梯度**的那份。

### 8.2 代码证据

GKD（top-k，作为 logits processor 在前向中直接算，`student_logits` 即 actor 前向输出的 `logits_rmpad`）：

```python
# verl/trainer/distillation/fsdp/losses.py
student_log_probs = F.log_softmax(student_logits, dim=-1)
student_topk_ids  = torch.topk(student_log_probs, k=..., dim=-1).indices
student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

# verl/workers/engine/fsdp/transformer_impl.py
outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
```

PG（k1 等）用当次前向算出的 `model_output["log_probs"]`：

```python
# verl/trainer/distillation/losses.py
student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)

# verl/workers/engine/megatron/transformer_impl.py
ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label)
```

rollout 时期的旧概率并非没用，而是作为 PG OPD 里 PPO 的 `old_log_prob`（重要性采样分母 / clip）：

```python
# verl/trainer/distillation/losses.py
log_prob     = no_padding_2_padding(model_output["log_probs"], data)  # 新：actor 重新 forward
old_log_prob = data["old_log_probs"]   # 旧：rollout 时期
```

### 8.3 「哪个 logprob 来自哪」速查

| 量 | 来源 | 是否带梯度 | 用途 |
|---|---|---|---|
| student `log_probs` / `student_logits` | **actor 训练时重新 forward** | 是（计算图上） | 算蒸馏 loss、算梯度 |
| `old_log_probs` | rollout 引擎（推理时） | 否 | PG OPD 的 PPO importance ratio / clip |
| `teacher_logprobs` / `teacher_ids` | teacher 推理引擎（独立资源池） | 否 | 蒸馏的监督目标 |

---

## 九、配置与关键约束速查

配置分三个命名空间（`verl/workers/config/distillation.py`，默认值见 `verl/trainer/config/distillation/distillation.yaml`）：

- `distillation.*` — 总开关 + teacher 资源池
- `distillation.teacher_models.<name>.*` — 每个 teacher 一项
- `distillation.distillation_loss.*` — loss 模式与聚合

**关键开关**：

```yaml
distillation:
  enabled: true
  n_gpus_per_node: 2
  nnodes: 1
  teacher_key: data_source   # 多教师路由字段
  teacher_models:
    teacher_model:        # 单教师约定名
      model_path: Qwen/Qwen3-32B
   inference:
        name: vllm
        gpu_memory_utilization: 0.8
  distillation_loss:
    loss_mode: k1     # 或 forward_kl_topk / k3 ...
    topk: 128       # top-k 模式用
    use_policy_gradient: true       # true=PG OPD, false=GKD OPD
    policy_loss_mode: vanilla    # 目前仅支持 vanilla
    use_task_rewards: true          # 是否叠加 PPO/GRPO 任务奖励
    distillation_loss_coef: 1.0
```

**关键约束（`DistillationLossConfig.__post_init__` 等）**：

- `use_policy_gradient=false` + `loss_mode=k1` → **报错**（k1 对 teacher logprob 无梯度，直接反传无意义）。
- `use_policy_gradient=true` + `loss_mode=forward_kl_topk` → **警告**（PG 只动采样 token，top-k 分布信号浪费）。
- `policy_loss_mode` 目前**只支持 `vanilla`**，其他 → `NotImplementedError`。
- 大多数 OPD run 应关掉 PPO/GRPO 的参考策略 KL，避免同时被 ref 正则 + teacher 蒸馏：
  ```yaml
  actor_rollout_ref.actor.use_kl_loss: false
  algorithm.use_kl_in_reward: false
  ```
- teacher 须与 student **共享 tokenizer / 词表**（通常同族模型，如 Qwen3-32B 教 Qwen3-8B）。

**多教师（MOPD）约束**：

- 教师资源池 GPU 数须严格等于 `Σ(num_replicas × per_replica_world_size)`，否则 `DistillationConfig.__post_init__` 报错。
- `per_replica_world_size = tp × dp × pp`；每个 replica 的 GPU bundle 不能跨节点（否则 `_validate_replica_node_alignment` 报错）。
- 路由：`teacher_key`（默认 `data_source`）须命名 `non_tensor_batch` 上的字段；`sample[teacher_key]` 的值须匹配某 teacher 的 `key`。
- **坑**：单教师约定名 `teacher_model` 在添加其他命名教师时会被**静默 pop 掉**。要么只用它，要么把它改名（如 `teacher_model1`）再加别的。
- 按 data_source 路由时**务必开启 data shuffle**，否则拼接数据集会长时间只激活一个 teacher。

**loss_mode 两大家族**：

- **top-k**：`forward_kl_topk`（teacher top-k 的 forward KL）。
- **单样本 KL 估计器**：`kl / k1 / abs / mse / k2 / low_var_kl / k3`（由 student `log_probs` 和 teacher 单个 `log_prob` 算 reverse KL 的逐 token MC 估计）。

**主要 metrics**（`actor/distillation/*`）：`loss`、`abs_loss`、`loss_min/max`；top-k 模式另有 `student_mass`、`teacher_mass`、`overlap_ratio`、`overlap_token_advantage`。

**调试技巧**：把 student 设成和 teacher 同一个模型，loss 应≈0（非精确 0，因训练/推理引擎差异）。

---

## 十、架构两段式速览

OPD 分两部分：

### 10.1 Teacher logprob 计算（在 Agent Loop 中，随 rollout 交错）

- 跑在**独立的 teacher 资源池**上，vLLM/SGLang 推理。
- 每个样本 rollout 一完成就触发 teacher 调用（无 batch 级 barrier，与其他样本 rollout 重叠）。
- 采样参数：`max_tokens=1` + `prompt_logprobs=topk`（或 0），即对 (prompt+response) 做**一次前向打分**，不生成新 token。
- **温度强制为 1.0**（`prompt_logprobs` 是对已有 token 的前向，温度无影响；配置里若非 1.0 会被忽略并告警）。
- 返回打包成 `teacher_ids`、`teacher_logprobs`，形状 `(S, 1 或 K)`，塞进 rollout 输出，随后拼进 per-batch `DataProto`。

### 10.2 Student 训练（在 actor train workers 上，同 PPO/GRPO）

- `distillation_ppo_loss` 在 distillation 开启时绑定为 actor 的 `loss_fn`。
- **top-k 模式**：前向时作为 **logits processor** 在完整 logits 还在显存时先算逐 token 蒸馏损失（`student_logits is not None` 分支 → `compute_forward_kl_topk`，FSDP/Megatron 各一份实现），把 `distillation_losses / student_mass / teacher_mass / overlap_*` 写回 `model_output`，随后释放完整 logits。
- **最终 loss**（`student_logits is None` 分支）：
  1. `distillation_loss(...)` 按 `loss_mode` 分派算逐 token 蒸馏损失。
  2. 可选 clamp（`loss_max_clamp`，k1 可能为负）。
  3. 聚合方式看 `use_policy_gradient`：
     - `false`（GKD）：对 `distillation_losses` 直接反传。
     - `true`（PG）：把 `-distillation_losses` 当 advantages，走 PPO 式 clipped importance sampling。
  4. 与任务奖励结合：`use_task_rewards=false` 则 PPO 项清零；否则 `policy_loss + distillation_loss_coef × distill_loss`。

### 10.3 另一套独立实现（了解即可）

`recipe/gkd/`（Async On-Policy Distillation）：基于 **Megatron + vLLM + 独立 ZMQ teacher server**，支持 `one_step_off` / `two_step_off` 异步调度，用重叠 generation / teacher 查询 / 更新来提吞吐。与本文主线（`verl/trainer/distillation` + Agent Loop）是两套不同实现。

---

## 十一、博客《On-Policy Distillation》补充视角

> 来源：Thinking Machines Lab《On-Policy Distillation》（Kevin Lu et al., 2025.10）。
> 说明：博客正文讲的就是 **reverse KL + advantage 形式** 的 OPD，对应 verl 的 **PG OPD**（`use_policy_gradient=true`, `loss_mode=k1`）；博客里**没有** "GKD OPD" 一词（GKD 是 verl 额外吸收的监督式变体）。本节补充前面讨论中未覆盖、但博客重点讲到的点。

### 11.1 核心公式对齐（博客 ↔ verl）

博客伪代码：

```python
reverse_kl = sampled_logprobs − teacher_logprobs      # 逐 token，sampled_logprobs 来自 student
trajectories["advantages"] = − reverse_kl
training_client.forward_backward(trajectories, loss_fn="importance_sampling")
```

即 `advantage = −reverse KL`，再用 RL 的 importance sampling（PPO）更新——与第八节 `advantages = -distillation_losses.detach()` 完全一致。

### 11.2 discount factor = 0（设计选择）

博客明确：**每个时刻只优化当前 token 的下一 token 分布，不考虑未来 token**（discount factor = 0）。他们试过 >0 的 discount，实践中没有提升，所以选 0。

理解：per-token reverse KL 是"逐位置独立对齐当前分布"，不是"序列级优化"，所以不需要对未来 token 打折扣。这从侧面印证了 OPD 是 **dense 的 token 级监督**，而非稀疏的序列级 reward。

### 11.3 reverse KL "不可攻击"（为什么不需要 reward model、也不怕 reward hacking）

博客视角：**RL 通常优化的是"由 reward model 诱导的序列级 reverse KL"**（reward model 近似一个目标分布）；而 OPD 的 reverse KL 是**直接、不可攻击的**——KL 越低，在 teacher 视角下"理想行为"的概率就越高，没有可被钻空子的 reward hacking 空间。

这解释了：OPD **不需要 reward model / 标注模型**，且天然免疫 reward hacking（监督信号就是"与 teacher 的 KL"，单调、无歧义）。

### 11.4 SimpleBench 分叉 token 例子（per-token reverse KL 到底惩罚什么）

博客展示了一个真实错误轨迹（物理题被学生错当纯数学题），per-token reverse KL 的惩罚分布是：

- **惩罚最重的是"分叉 token"**——把推理引向错误方向的关键词；
- **最终的错误答案本身反而没被严重惩罚**——因为给定错误前文，那个错答案是"高度可预测"的，学生/教师 logprob 差异小。

直觉价值：**OPD 惩罚的是"走错路的岔路口"，而不是"最终的错答案"**。它把"在哪一步开始错"这个信息用 token 级 KL 自然编码出来，无需人工标注错误位置。这比稀疏的序列级 reward 信息量大得多。

### 11.5 数据效率：单个 prompt 反复用也能恢复

博客实验：**只用一个随机 prompt，连续 20 步、每步 256 个 rollout（共 5120 条教师评分序列），就能大致恢复教师在该任务上的表现**。

原因：OPD 学的是**教师的完整分布（一个函数）**，而不是记忆"特定输入 → 特定输出"配对。这和我们第七节强调的"on-policy 状态对齐"互补：

- 第七节：状态从哪来（on-policy）→ 解决 **exposure bias**；
- 本节：学的是分布而非样本 → 解决 **数据效率**。

### 11.6 持续学习 / 防遗忘：旧版自己当教师

博客第二大实验揭示一个重要应用：**OPD 的"教师"不一定是更大更强的模型，也可以是"旧版的自己"**。

现象：**中训练（mid-training）会破坏后训练（post-training）行为**——

- 直接训内部文档，IF-eval 从 85% 掉到 45%；
- 混 30% chat 数据也恢复不了；
- **LoRA 也无法避免遗忘**（学得更少但照样遗忘）。

解法：以**旧版 Qwen3-8B 为教师**，学生在 Tulu3 prompts 上采样，教师打分恢复指令遵循能力。结果 Internal QA 41% + IF-eval 83%——几乎不损失新知识地恢复旧行为。

### 11.7 成本量化与两个洞见

**量化数据**（学生 Qwen3-8B-Base，教师 Qwen3-32B，AIME'24）：

- 性能：OPD 70% vs RL 67.6%，但成本低得多；
- FLOPs 约为 SFT 的 1/9，GPU hours 约 1/18，计入教师生成离线数据约 1/30；
- 相比 RL，达到相同性能的 gradient steps 少 7–10 倍，compute efficiency 提升 50–100 倍。

**洞见 1：RL 搜索的是语义策略，蒸馏是学习最终策略的捷径。**
RL 的大部分计算花在"搜索策略"而非梯度更新；一旦找到好的最终策略，OPD 可直接学习它，跳过中间的探索/课程过程。类比：科研探索很贵，但一旦发现，教给别人就便宜得多（运动技能则相反，很难语言传授）。

**洞见 2：on-policy SFT 会自我退化，OPD 不会。**
直接对学生自己采样做 SFT，即使 KL 期望为 0，有限 batch 的波动也会产生非零梯度，长序列上错误累积、逐渐偏离原策略。而 OPD 始终保持在学生当前分布上训练、教师固定，能收敛到教师行为，不会自我退化。

---

## 附：一句话记忆卡

- **OPD**：student 自己采样、teacher 在 student 走出的状态上给密集监督（on-policy 采样 + 蒸馏监督）。
- **forward vs reverse**：看期望对谁采样——forward 对 teacher（覆盖式，需完整分布），reverse 对 student（找峰式，可单样本估计）。
- **GKD** = forward_kl_topk，监督式直接反传，像"升级版 KD"；**PG** = k1，把负 reverse-KL 当 reward 走 PPO。
- **stop-gradient**：策略梯度里 reward 必须是常数，否则混入"只由 student 自身 logprob 决定、与 teacher 无关"的伪梯度，蒸馏失效。
- **单样本估计只对 reverse KL 成立**：因为现成 token 来自 π_θ，正好匹配 `E_{v~π_θ}`；forward KL 的 `E_{v~ν}` 对不上，只能枚举 top-k。
- **student logprob 来自 actor 重新 forward**（要梯度），rollout 引擎那份只当 PG 的 `old_log_probs`。
