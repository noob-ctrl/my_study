# Context Parallel（上下文并行，CP）详解

> 本文基于 Megatron-LM 的 CP 实现，系统梳理 CP 的动机、序列切分、Ring K/V 传递、
> online softmax 合并，以及与 TP 的组合方式。适合作为长上下文训练并行原理的学习笔记。

---

## 目录

1. [CP 要解决什么问题](#1-cp-要解决什么问题)
2. [一个通俗例子：多人合读长篇小说](#2-一个通俗例子多人合读长篇小说)
3. [序列切分：Zigzag 负载均衡](#3-序列切分zigzag-负载均衡)
4. [Ring K/V 传递机制](#4-ring-kv-传递机制)
5. [cp_size=2 与 cp_size=4 举例](#5-cp_size2-与-cp_size4-举例)
6. [Online Softmax：如何合并各步结果](#6-online-softmax如何合并各步结果)
7. [为什么必须"边传边算"（关键设计动机）](#7-为什么必须边传边算关键设计动机)
8. [TP + CP 组合切分（带具体形状）](#8-tp--cp-组合切分带具体形状)
9. [四种 CP 通信类型](#9-四种-cp-通信类型)
10. [核心代码位置](#10-核心代码位置)
11. [总结](#11-总结)

---

## 1. CP 要解决什么问题

流水线并行（PP）解决"模型太大装不下"，把**模型的层**切开放到不同 GPU。
但还有另一个痛点：**序列（上下文）太长**。

- 注意力机制的显存和计算量，随序列长度呈平方级 `O(N²)` 增长。
- 训练支持 128K token 超长上下文的模型时，单卡根本存不下这么多 token 的中间激活。

**CP 的思路**：既然模型层能切，序列本身也能切——把一条长序列按 token 切成几段，
分给不同 GPU，每张GPU 只负责其中一段 token 的计算。

**CP 与其他并行的核心区别：**

| 并行方式 | 切什么 | 通俗比喻 |
|---|---|---|
| DP（数据并行） | 数据批次，每卡一份完整模型 | 几个人各读不同的书 |
| PP（流水线并行） | 模型的层（深度） | 一本书的装订流水线 |
| TP（张量并行） | 单层内部的权重矩阵 | 一道复杂计算，几人分工算 |
| SP（序列并行） | 仅 LayerNorm/Dropout 的序列激活 | TP 的小补丁 |
| **CP（上下文并行）** | **整条序列的所有 token 和激活** | **几个人合读同一本超长的书** |

CP 可看作 SP 的"完全体"：SP 只切了 LayerNorm/Dropout 那一小块，CP 把**整个网络的
输入和所有激活的序列维度全都切了**。CP 与 TP/PP/DP 正交，可组合，
总 GPU 数 = `TP × CP × PP × DP`。

---

## 2. 一个通俗例子：多人合读长篇小说

一本 1200 页的小说，4 个人（4 张 GPU）一起读并做笔记（训练）：

```
GPU0: 第 1~300 页    GPU1: 第 301~600 页
GPU2: 第 601~900 页  GPU3: 第 901~1200 页
```

- **大部分工作各干各的**：给每页做摘要（对应 Linear、LayerNorm 等**逐 token、不跨
  token** 的操作），无需交流。
- **唯独"注意力"需要互相交流**：读到第 1000 页的人，要理解这一页，必须知道前面
  1~999 页讲了什么。可前面的内容在别人手里—— 这就是 CP 里GPU 之间的**通信**。

模型里注意力的规则：每个 token 的 **Q（查询）** 要和**所有 token的 K、V（键、值）**
做计算。每张 GPU 只有自己那段的 K/V，所以需要通过通信把其他 GPU 的 K/V 拿过来。

**为什么 attention 需要通信，而其它操作不需要：**

| 操作 | 是否跨 token | 需要 CP 通信 |
|---|---|---|
| LayerNorm | 否，每 token 自己归一化 | 不需要 |
| MLP (fc1/fc2) | 否，每 token 独立 | 不需要 |
| QKV 投影 | 否，每 token 独立投影 | 不需要 |
| **注意力打分** | **是！每 token 要看所有 token** | **需要！** |

---

## 3. 序列切分：Zigzag 负载均衡

### 坑：因果注意力"越靠后越累"

因果注意力里，越靠后的 token 要看的前文越多，计算量越大：
- 第 1 个 token：只看自己，算 1 次。
- 第 1000 个 token：要看前面 999 个，算 999 次。

若简单**顺序切**（GPU0 拿前段、GPU3 拿后段），后段 GPU 会累死，负载不均。

### 解法：Zigzag（之字形）切分

把序列切成 `2 × cp_size` 块，每张 GPU 拿"一前一后"两块：

```
cp_size=2，序列切成 4 块：
  顺序切（不均衡）:Zigzag 切（均衡）:
  GPU0: 块0, 块1             GPU0: 块0 + 块3   ← 一头一尾
  GPU1: 块2, 块3             GPU1: 块1 + 块2   ← 一头一尾
        ↑ GPU1 累死                ↑ 每人都有轻的和重的
```

**规律**：rank i 拿第 `i` 块 和第 `2*cp_size-1-i` 块（对称配对），每张 GPU 的总
计算量都差不多，实现负载均衡。

对应源码 `megatron/core/utils.py`：

```python
index[0].fill_(cp_rank)                    # 拿前面的块
index[1].fill_(2 * cp_size - cp_rank - 1)  # 拿对称的后面的块
val = val.index_select(seq_dim, index)
```

RoPE 位置编码也做同样重排（`rope_utils.py` 的 `get_pos_emb_on_this_cp_rank`）。

---

## 4. Ring K/V 传递机制

### 为什么传 K/V 而不传 Q

因为可用 MQA/GQA 减少 K/V 的头数（K/V 头数往往远少于 Q 头数），**传 K/V 通信量更小**。
Q 留在本地不动，每收到一批别人的 K/V 就和本地 Q 算一次。

### Ring（环形击鼓传花）

所有 CP GPU 围成一个圈：`GPU0 → GPU1 → GPU2 → GPU3 → GPU0`

**每一步（step），每张 GPU 同时做三件事：**
1. 用**当前手里的 K/V** + 本地的 Q，算一部分注意力。
2. 把这块 K/V **发给下一个邻居**。
3. 从**上一个邻居**收一块新的 K/V。

转满 `cp_size` 步（一整圈），每张 GPU 的 Q 就和**所有** GPU 的 K/V 都算过了。

**接力本质**：每一步传出去的 K/V，可能是上一步刚从前一个邻居收来的。K/V 像接力棒
一样一棒接一棒沿环流动。以 cp_size=4 为例，跟踪"每张 GPU 手里持有的 K/V 是谁的"：

| Step | GPU0 持有 | GPU1 持有 | GPU2 持有 | GPU3 持有 |
|:---:|:---:|:---:|:---:|:---:|
| 0 | GPU0 | GPU1 | GPU2 | GPU3 |
| 1 | GPU3 | GPU0 | GPU1 | GPU2 |
| 2 | GPU2 | GPU3 | GPU0 | GPU1 |
| 3 | GPU1 | GPU2 | GPU3 | GPU0 |

- 竖看每列：每张 GPU 依次见过所有 GPU 的 K/V（转一圈）。
- 横看每行：同一时刻 4 块 K/V 在 4 张 GPU 手里，互不重复（4 个接力棒同时跑）。

**收发是同时的（全双工）**：每张 GPU 每步"收到新 K/V → 用本地 Q 算它 → 同时把它
转发给下游"。

**关键优化**：收发通信和计算**同时进行**（专用 CUDA stream `cp_stream`），
一边算这块、一边后台传下一块，通信时间被"藏"在计算里。这是 CP 比朴素 Ring
Attention 快的原因。

---

## 5. cp_size=2 与 cp_size=4 举例

### 例子一：cp_size = 2

序列 1024 token，Zigzag 切成 `2×2=4` 块，每块 256 token：

```
GPU0: 块0(0~255)   + 块3(768~1023)
GPU1: 块1(256~511) + 块2(512~767)
```

Ring 传递（仅 2 步，环形 GPU0 ↔ GPU1）：

```
Step 0（用本地自己的 K/V）:
  GPU0: 用[块0+块3的Q] 算 [块0+块3的K/V]   (本地)
  GPU1: 用[块1+块2的Q] 算 [块1+块2的K/V]   (本地)
  同时后台互相交换 K/V

Step 1（用收到的邻居 K/V）:
  GPU0: 用[块0+块3的Q] 算 [块1+块2的K/V](从GPU1收到)
  GPU1: 用[块1+块2的Q] 算 [块0+块3的K/V](从GPU0收到)
```

因果掩码优化：Step 1 里 GPU0 的块0（靠前）看块1/块2（靠后）→ 不允许，**直接跳过**；
块3（靠后）看块1/块2（靠前）→ 允许，正常计算。Zigzag 让各GPU 跳过量均衡。

### 例子二：cp_size = 4

序列 1024 token，Zigzag 切成 `2×4=8` 块，每块 128 token：

```
GPU0(rank0): 块0 + 块7   ← 最前 + 最后
GPU1(rank1): 块1 + 块6
GPU2(rank2): 块2 + 块5
GPU3(rank3): 块3 + 块4   ← 中间两块
```

（0配7、1配6、2配5、3配4，对称配对 → 负载均衡）

Ring 传递（4 步，环形 GPU0→GPU1→GPU2→GPU3→GPU0），跟踪 GPU0：

| Step | GPU0 手里的 K/V 来自 | 动作 |
|:---:|---|---|
| 0 | 自己（块0+块7） | 算本地；发给GPU1，从 GPU3 收 |
| 1 | GPU3（块3+块4） | 算这批；继续传递 |
| 2 | GPU2（块2+块5） | 算这批；继续传递 |
| 3 | GPU1（块1+块6） | 算这批；一圈结束 |

### 两例对比

| | cp_size=2 | cp_size=4 |
|---|---|---|
| GPU 数 | 2 | 4 |
| Zigzag 切块数 | 4 | 8 |
| 每 GPU 拿几块 | 2 | 2 |
| Ring 步数 | 2 | 4 |
| 每步通信量 | 1 块 K/V | 1 块 K/V |
| 每 GPU 显存 | 全序列 1/2 | 全序列 1/4 |

**规律**：cp_size 越大，每卡分到的 token 越少（越省显存），但 Ring 步数越多。

---

## 6. Online Softmax：如何合并各步结果

### 问题：不能简单相加

注意力 `softmax(QKᵀ/√d)V` 的 softmax 分母 `Σ e^{x_j}` 需要**全序列所有 token 的
指数和**。Ring 传递中每步只看到一块 K/V，算不出正确的全局分母，因此**不能**每步各
算一个完整 softmax 再相加（每步分母都只是局部和，是错的）。

### 解法：Online Softmax（FlashAttention 技术）

每张 GPU 为自己的 Q 维护 3 个累加器：

| 累加器 | 含义 |
|---|---|
| `m` | 目前见过的最大分数（数值稳定，防 `e^x` 溢出） |
| `ℓ` | 目前的 softmax 分母（running 指数和） |
| `O` | 目前的加权输出（running output） |

每来一块新 K/V，算出分数 `S_new = Q·K_newᵀ/√d`，增量更新：

```
① m_new = max(m_old, max(S_new))
② correction = e^{m_old - m_new}          # 修正因子
③ ℓ_new = correction · ℓ_old + Σ e^{S_new - m_new}
④ O_new = correction · O_old + e^{S_new - m_new} · V_new
```

转完整圈后归一化：`O_final = O / ℓ`

**为什么对**：修正因子 `e^{m_old-m_new}` 把旧结果重新缩放到新的最大值基准上。
数学上可证明：无论 K/V 分几块、按什么顺序到达，增量累加结果与"一次性看全部 K/V
算 softmax"**完全相等**。

### 与因果掩码配合

不该看的（未来的）分数置`-∞`（`e^{-∞}=0`），对 `ℓ`、`O` 无贡献= 自动跳过，
省掉下三角掩码的算力。

---

## 7. 为什么必须"边传边算"（关键设计动机）

一个常见疑问：**为什么不等转完一圈拿到全部 K/V，再统一算softmax？**

### 方案对比

| 方案 | 存什么 | 显存量级 | 结果 |
|---|---|---|---|
| 攒齐 K/V 再算 | 全部 K/V `[N,d]` | `O(N·d)` | 白省显存 |
| 攒齐分数矩阵 S 再算 | 分数矩阵 `[n,N]` | `O(n·N)` | 白省，甚至更糟 |
| **Online softmax（边传边算）** | m,ℓ,O 累加器 | `O(n·d)` 常数级 | ✅ 真正省显存 |

- **攒齐 K/V**：转完一圈= 攒齐整个序列 K/V，显存退回没用 CP 的水平，CP 白做。
- **攒齐分数矩阵 S**（设 GPU 有 n 个 Q token，全序列 N token）：分数矩阵形状是
  `[n, N]`，随序列长度膨胀，**通常比 K/V 还大**——这正是标准注意力 `N²` 显存爆炸的
  元凶，等于把它又存回来了，一样白省甚至更糟。
- **Online softmax**：收到一块 → 立刻消化进 m/ℓ/O → 连这块 K/V 和分数都扔掉。
  累加器 `m`、`ℓ` 是标量级，`O` 与一块输出等大，**常数级、不随序列增长**。

### 附带好处：计算与通信重叠

"攒齐再算"必须先纯通信、再纯计算，两段无法重叠。Online 方案每收到一块就立刻算，
算这块时后台传下一块，通信被藏进计算，更快。

### 类比

算 1000 个数的平均值：
- 攒齐再算 = 把 1000 个数全记在本子上再算（大显存）。
- Online = 心里只记"累计和"和"个数"，每报一个数就更新，报完直接得平均
  （小显存，结果一样）。

既然增量算就能得到与攒齐完全相等的结果，就没有理由去存全部数据。

---

## 8. TP + CP 组合切分（带具体形状）

### 配置

模型单层：`hidden=512, heads=8, head_dim=64, ffn=2048, seq=1024, batch=1`
并行：**TP=2, CP=2，共 4 张 GPU**

```
        TP=0     TP=1
CP=0    GPU0     GPU1
CP=1    GPU2     GPU3
```

- **TP 切权重**（横向），**CP 切输入 token**（纵向），两者正交。
- 同一 CP 组内、不同 TP rank 的 GPU 处理**相同的 token**（TP 切权重不切数据）。

### 输入切分（CP 负责）

输入 `[s,b,h]=[1024,1,512]`，Zigzag 切成 4 块（每块 256）：

```
CP rank 0 (GPU0/GPU1): 块0 + 块3  → [512,1,512]
CP rank 1 (GPU2/GPU3): 块1 + 块2  → [512,1,512]
```

### 权重切分（TP 负责，CP 不切权重）

| 权重 | 并行类型 | 切法 | 完整形状 | 每 TP rank |
|---|---|---|---|---|
| QKV 投影 `linear_qkv` | ColumnParallel | 切输出维（head 均分） | `[512,1536]` | `[512,768]`（4 head） |
| 输出投影 `linear_proj` | RowParallel | 切输入维 | `[512,512]` | `[512,256]` |
| MLP `linear_fc1`（升维） | ColumnParallel | 切输出维 | `[512,2048]` | `[512,1024]` |
| MLP `linear_fc2`（降维） | RowParallel | 切输入维 | `[2048,512]` | `[1024,512]` |

**Column→Row 经典搭配**：先列并行各算各的，再行并行合并，一个 block 前向只需 1 次
all-reduce（在 Row 之后）。

### 每张 GPU 的完整分工

| GPU | CP | TP | 处理token | QKV 权重 | MLP fc1 权重 |
|---|:--:|:--:|---|---|---|
| GPU0 | 0 | 0 | 块0+块3 | head 0-3 `[512,768]` | `[512,1024]` |
| GPU1 | 0 | 1 | 块0+块3 | head 4-7 `[512,768]` | `[512,1024]` |
| GPU2 | 1 | 0 | 块1+块2 | head 0-3 `[512,768]` | `[512,1024]` |
| GPU3 | 1 | 1 | 块1+块2 | head 4-7 `[512,768]` | `[512,1024]` |

- 横看（同 CP 不同 TP）：token 相同、权重不同 → TP 分工算同一批 token。
- 竖看（同 TP 不同 CP）：权重相同、token 不同 → CP 分工算不同 token。

### 两种通信的方向

```
          TP通信(横向, 合并head, all-reduce)
        GPU0 ←──────────────→ GPU1
          ↑                    ↑
   CP通信 │                    │ CP通信
 (纵向,   │                    │ (Ring换K/V)
  换K/V)↓                    ↓
        GPU2 ←──────────────→ GPU3
          TP通信(横向, 合并head, all-reduce)
```

- **TP 通信**：在每个 block 内部（Row 层之后 all-reduce），发生在 TP 组内
  （GPU0↔GPU1、GPU2↔GPU3）。
- **CP 通信**：只在注意力核心（Ring 交换 K/V），发生在 CP 组内
  （GPU0↔GPU2、GPU1↔GPU3）。

**口诀**：TP 横向合并 head 结果，CP 纵向交换 K/V，两方向互不干扰。

### GPU0 完整前向流程

```
1. LayerNorm                                          [无通信]
2. QKV投影(列并行)算出块0+块3的Q/K/V(head0-3)         [无通信]
3. 注意力: Ring和GPU2交换K/V, 算head0-3注意力    [★CP通信, GPU0↔GPU2]
4. 输出投影(行并行): all-reduce和GPU1合并head    [★TP通信, GPU0↔GPU1]
5. LayerNorm                                          [无通信]
6. MLP fc1(列并行) 升维各算一半                        [无通信]
7. MLP fc2(行并行): all-reduce和GPU1合并          [★TP通信, GPU0↔GPU1]
```

---

## 9. 四种 CP 通信类型

配置项 `cp_comm_type`（`transformer_config.py`）：

| 类型 | 说明 |
|---|---|
| `p2p`（默认） | ring 拓扑用 P2P 交换 KV 块，异步、可与 attention 计算 overlap（Ring Attention） |
| `all_gather` | attention 前 all-gather 完整 KV 序列；非异步、不可 overlap |
| `a2a` | 类DeepSpeed Ulysses，用 all-to-all 把 attention heads 分散到 CP group，再 gather 得完整 QKV |
| `a2a+p2p` | 分层：低层 CP group用 A2A（如 NVLink），高层用 P2P（如 IB） |

实际的 ring/all-gather/a2a 通信原语在 **Transformer Engine** 内部实现，Megatron 负责
透传 `cp_group`、`cp_global_ranks`、`cp_stream`、`cp_comm_type`。
CP 注意力**只有 `TEDotProductAttention` 支持**，原生实现直接 assert 拒绝。

---

## 10. 核心代码位置

| 文件 | 职责 |
|---|---|
| `megatron/core/parallel_state.py` | CP 进程组创建与管理（`get_context_parallel_group/rank/world_size`） |
| `megatron/core/utils.py` | 序列切分/负载均衡（`get_batch_on_this_cp_rank`及 zigzag 分区函数） |
| `megatron/core/extensions/transformer_engine.py` | `TEDotProductAttention`：透传 CP group/通信类型给 TE 注意力 kernel |
| `megatron/core/transformer/transformer_config.py` | 配置项 `context_parallel_size`、`cp_comm_type` |
| `megatron/core/transformer/attention.py` | 注意力模块透传 `cp_comm_type`、`pg_collection.cp`，RoPE 的 CP 适配 |
| `megatron/core/transformer/dot_product_attention.py` | 原生实现，明确不支持 CP |
| `megatron/core/models/common/embeddings/rope_utils.py` | RoPE 位置编码的 CP 切分 |
| `docs/user-guide/features/context_parallel.md` | CP 官方总览文档 |
| `docs/user-guide/parallelism-guide.md` | 并行策略总指南（DP/TP/PP/CP/EP 对比） |

**启用方式**：命令行设 `context_parallel_size=<CP_SIZE>`（默认 1 = 关闭）。
需要 Megatron Core ≥ 0.5.0 和 Transformer Engine ≥ 1.1。

---

## 11. 总结

CP（上下文并行）把一条超长序列按 token 切成几段，分给多张 GPU 各算一段，突破单卡
显存/算力对序列长度的限制。设计逻辑完整闭环：

1. **为什么要传 K/V**：注意力里每个 token 的 Q 要和全序列的 K/V 计算，而每卡只有
   自己那段 K/V。除注意力外的操作（LayerNorm/MLP/投影）均逐 token 独立，无需通信。
2. **怎么切（Zigzag）**：切成 `2*cp_size` 块，每卡拿"一前一后"两块，均衡因果注意力
   "越靠后越累"的计算量，并支持因果掩码跳过优化。
3. **怎么传（Ring）**：GPU 围成环，Q 固定不动，K/V 像击鼓传花每步传一块，转
   `cp_size` 步后每个 Q 都和全序列 K/V 算过；通信与计算重叠。
4. **怎么合并（Online Softmax）**：维护 m/ℓ/O 三个常数级累加器，每收到一块就带修正
   因子增量累加、随即丢弃，结果与一次性算softmax 完全相等，显存 `O(n·d)` 与序列
   长度无关。
5. **与 TP 组合**：TP 切权重（横向、block 内 all-reduce），CP 切 token（纵向、
   attention 内 Ring 换 K/V），两者正交，通信方向互不干扰。

**一句话**：CP 用 "Zigzag 均衡切分 + Ring 传 K/V + Online Softmax 增量合并"，在几乎不
增加时间开销的前提下，把注意力的显存和计算按 cp_size 均摊到多卡，从而高效训练超长
上下文模型。
