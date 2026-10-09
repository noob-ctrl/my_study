# FlashAttention V1 学习笔记

> 论文：《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（Tri Dao 等，2022）
>
> 本文档在通用介绍之外，**重点解释几个容易困惑的细节**：
> - IO 复杂度推导里那个神秘的常数 `4` 是怎么来的；
> - softmax 到底有没有计算量，为什么说它的算术强度（AI）接近 1；
> - 如何判断一个算子是 memory-bound 还是 math-bound。

---

## 目录

1. [FlashAttention V1 概览](#一flashattention-v1-概览)
2. [核心结论：降的是访存，不是计算](#二核心结论降的是访存不是计算)
3. [HBM 访问复杂度推导](#三hbm-访问复杂度推导)
4. [重点答疑一：公式里的常数 4 是怎么来的](#四重点答疑一公式里的常数-4-是怎么来的)
5. [重点答疑二：softmax 有没有计算量，AI 为什么接近 1](#五重点答疑二softmax-有没有计算量ai-为什么接近-1)
6. [如何判断 memory-bound 还是 math-bound](#六如何判断-memory-bound-还是-math-bound)
7. [逻辑链总结](#七逻辑链总结)

---

## 一、FlashAttention V1 概览

FlashAttention V1 是一种**精确（exact，非近似）** 的注意力计算算法，目标是解决标准 Attention 在长序列上**显存占用大、速度慢**的问题。

标准 Attention 的计算流程：

$$
S = QK^\top,\quad P = \text{softmax}(S),\quad O = PV
$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度，$d$ 是 head 维度。

**问题所在**：中间矩阵 $S$ 和 $P$ 的大小是 $N \times N$：

- **显存开销 $O(N^2)$**：需要把整个 $N\times N$ 矩阵读写到 GPU 的 HBM（高带宽显存）。
- **速度瓶颈**：Attention 实际上是 **memory-bound（访存受限）**，而非 compute-bound。大量时间花在 HBM 读写上，而不是浮点运算上。

**核心思想：IO-Awareness（IO 感知）**

GPU 存储分层：

| 存储 | 容量 | 带宽 |
|---|---|---|
| HBM（高带宽显存） | 几十 GB | 相对慢（约 1.5~2 TB/s） |
| SRAM（片上共享内存） | 每个 SM 约 100+ KB | 极快（约快一个数量级） |

FlashAttention 的洞察：**减少 HBM 的读写次数**才是关键，而不是减少计算量。通过两个关键技术实现：

- **Tiling（分块）**：把 $Q, K, V$ 切分成小块，逐块加载到 SRAM 中计算，避免生成完整的 $N\times N$ 矩阵。整个 Attention 在一个融合 kernel 内完成（kernel fusion），中间结果不写回 HBM。
- **Online Softmax（在线 Softmax）**：增量式地维护并更新每块的最大值 $m$ 和指数和 $\ell$，使得无需一次性看到整行也能得到与标准 softmax **完全一致**的结果。

  $$
  m_{new} = \max(m_{old}, \tilde{m}),\quad
  \ell_{new} = e^{m_{old}-m_{new}}\ell_{old} + e^{\tilde{m}-m_{new}}\tilde{\ell}
  $$

**反向传播**：不保存 $N\times N$ 的 $P$ 矩阵，只保存 softmax 统计量（$m$ 和 $\ell$），反向时在 SRAM 中**重新计算**（recomputation）$S$ 和 $P$。重计算带来的额外 FLOPs 远小于省下的 HBM 访问，整体反而更快。

---

## 二、核心结论：降的是访存，不是计算

> **问题**：FlashAttention V1 是不是没有降低计算复杂度，而是降低了访存复杂度，从而提升了整体速度？

**完全正确。**

### 计算复杂度：没降低，反而略增

- 标准 Attention 的 FLOPs 主要是两个矩阵乘 $QK^\top$（$O(N^2 d)$）和 $PV$（$O(N^2 d)$），总计 $O(N^2 d)$。
- FlashAttention V1 **前向 FLOPs 完全一样**，一个乘法都没省（因为是 exact，不做任何近似）。
- **反向传播 FLOPs 反而增加**——因为要重计算 $S=QK^\top$ 和 $P=\text{softmax}(S)$，多做了一遍部分前向计算。

### 那为什么还更快？

关键：**Attention 是 memory-bound 的**。GPU 算力单元大部分时间在"等数据"，而不是在"算"。FlashAttention 用"多算一点（recomputation）换少访存一大截"，省下的 HBM 访问时间远大于多算的时间。

> **降低的是访存复杂度（$O(N^2) \to O(N^2 d^2/M)$），计算复杂度不变（反向略增），但整体墙钟时间（wall-clock）大幅下降。**

这就是 IO-Awareness 的精髓：优化目标从 "FLOPs" 换成了 "HBM 访问量"。

### 复杂度对比

| | 标准 Attention | FlashAttention V1 |
|---|---|---|
| HBM 访问 | $O(N^2)$ | $O(N^2 d^2 / M)$（M 为 SRAM 大小） |
| 显存占用 | $O(N^2)$ | $O(N)$（线性） |
| 计算精度 | 精确 | 精确（无近似） |
| 前向 FLOPs | $O(N^2 d)$ | $O(N^2 d)$（相同） |
| 速度 | 慢（访存瓶颈） | 快 2~4 倍 |

---

## 三、HBM 访问复杂度推导

符号约定：

- $N$：序列长度
- $d$：head 维度
- $M$：SRAM 大小（能放下的元素个数）

### 标准 Attention：$O(Nd + N^2)$

标准实现要把 $N\times N$ 的中间矩阵 $S, P$ 写回 HBM 再读出来，主导项是 $O(N^2)$。

### FlashAttention 的分块推导

FlashAttention 把 $K, V$ 按行分块，每块设计成**刚好能放进 SRAM**。

**块大小选取**（为让块装进 SRAM）：

$$
B_c = O\!\left(\frac{M}{d}\right),\qquad B_r = O\!\left(\frac{M}{d}\right)
$$

**外层循环遍历 K/V 的块数：**

$$
T_c = \frac{N}{B_c} = O\!\left(\frac{N d}{M}\right)
$$

**关键点**：每处理一个 K/V 块时，都要把整个 $Q$（以及输出 $O$、统计量）从 HBM 完整过一遍，即每轮访问 $O(Nd)$。

**总访问量：**

$$
\text{HBM 访问} = T_c \times O(Nd) = O\!\left(\frac{Nd}{M}\right)\times O(Nd) = O\!\left(\frac{N^2 d^2}{M}\right)
$$

### 为什么这是巨大改进

对比标准的 $O(N^2)$，比值为：

$$
\frac{N^2 d^2 / M}{N^2} = \frac{d^2}{M}
$$

实际中 SRAM 大小 $M$（约 100KB 量级，$\sim 10^4$–$10^5$ 个元素）通常**远大于** $d^2$（$d$ 一般 64/128，$d^2$ 约 $4000$–$16000$）。所以 $d^2/M < 1$，即 FlashAttention 的 HBM 访问**小于**标准实现。

> 直观理解：SRAM 越大（$M$ 越大），每次能塞进去的块越大，来回搬运 $Q$ 的次数（$T_c$）就越少，总访存就越少。这就是为什么复杂度里 $M$ 在分母。

---

## 四、【重点答疑一】公式里的常数 4 是怎么来的

有些文章给出的推导如下：

$$
O(T_c N d) = O\!\left(\frac{N}{B_c} N d\right) = O\!\left(\frac{4Nd}{M} N d\right) = O\!\left(\frac{N^2 d^2}{M}\right)
$$

这里的 `4` 让人困惑，其来源如下。

### 这个 4 来自块大小 $B_c$ 的取值

上式隐含了：

$$
T_c = \frac{N}{B_c} = \frac{4Nd}{M} \quad\Longleftrightarrow\quad B_c = \frac{M}{4d}
$$

而 $B_c = \dfrac{M}{4d}$ 正是 FlashAttention 论文里**人为设定的块大小策略**。原文定义：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad B_r = \min\left(\left\lceil \frac{M}{4d}\right\rceil,\ d\right)
$$

### 这个 4 的物理含义：SRAM 要同时放下 4 组块

在内层循环中，SRAM（大小 $M$）需要**同时容纳**以下数据：

| 变量 | 尺寸 | 说明 |
|---|---|---|
| $Q_i$ 块 | $B_r \times d$ | Query 块 |
| $K_j$ 块 | $B_c \times d$ | Key 块 |
| $V_j$ 块 | $B_c \times d$ | Value 块 |
| $O_i$ 块 | $B_r \times d$ | 输出块 |

即 SRAM 要装下 **Q、K、V、O 大约 4 组** $B \times d$ 的数据（外加 $m, \ell$ 等统计量，是 $O(B)$ 的小量，可忽略）。为保证不溢出：

$$
4 \cdot B \cdot d \le M \quad\Longrightarrow\quad B \le \frac{M}{4d}
$$

所以取 $B_c = \dfrac{M}{4d}$，那个 `4` 就是 **"SRAM 里要同时放 Q/K/V/O 四块数据" 的因子**。

### 关键提醒：4 不影响渐进复杂度

在**大 O 复杂度**意义下，`4` 是个常数，会被吸收掉——所以最终结果 $O\!\left(\dfrac{N^2 d^2}{M}\right)$ 里 `4` 消失了。它只在**中间的精确推导**里出现，用来说明块大小的具体设定。

> 一句话：`4` 不改变渐进复杂度，只是论文为了让 Q/K/V/O 四块都塞进 SRAM 而设的具体常数因子。

---

## 五、【重点答疑二】softmax 有没有计算量，AI 为什么接近 1

> **困惑**：softmax 不是计算量挺大的吗？为什么说它"几乎没有计算量"，而且算术强度（AI）接近 1？

### 澄清：softmax **不是没有计算量**

之前"没有计算量"的说法不严谨。准确说法是：**softmax 的计算量相对它的访存量太小了**。

对一个 $N\times N$ 矩阵做逐行 softmax，实际做了：

- 对每个元素做 $\exp$：$N^2$ 次
- 求每行的和：约 $N^2$ 次加法
- 每个元素除以和：$N^2$ 次除法
- 数值稳定还要减去 max：再 $N^2$ 量级

所以 softmax 的总计算量是 **$O(N^2)$**——从绝对值看确实不小（尤其 $\exp$ 还相对昂贵）。

**但"计算量大不大"不能看绝对值，要看它和访存量的比值，即算术强度 AI。**

### 算 softmax 的算术强度

$$
\text{算术强度 (AI)} = \frac{\text{计算量 (FLOPs)}}{\text{访存量 (Bytes)}}
$$

对 softmax：

- **计算量**：$O(N^2)$ FLOPs（每个元素做常数次运算）
- **访存量**：把整个 $N\times N$ 矩阵从 HBM 读进来（$N^2$ 个元素），算完写回去（$N^2$ 个元素），共 $O(N^2)$ 个元素的读写

$$
\text{AI} = \frac{c \cdot N^2}{N^2 \cdot \text{sizeof(元素)}} = \frac{c}{\text{sizeof}} = O(1)
$$

关键在于：**每个元素只被做了常数次（比如 5~10 次）浮点运算，就要被读一次、写一次。** 计算量和访存量都是 $N^2$ 量级，比值把 $N^2$ 约掉，剩下一个**常数**，所以 AI ≈ 常数（量级 1~10），远低于硬件平衡点（A100 约 160）。

### 对比矩阵乘：为什么 matmul 的 AI 高

对 $QK^\top$（两个 $N\times d$ 矩阵相乘）：

- **计算量**：$O(N^2 d)$ FLOPs（每个输出元素要做 $d$ 次乘加）
- **访存量**：读入 $O(Nd)$，写出 $O(N^2)$

$$
\text{AI}_{matmul} \sim \frac{N^2 d}{N^2 + Nd} \approx O(d)
$$

$d$ 是 64/128，所以 matmul 的 AI 是**几十到 100+**，比 softmax 高一两个数量级。**每读一个元素能复用它做 $d$ 次运算**，这就是矩阵乘"计算密集"的原因。

### 两者对比表

| 操作 | 计算量 | 访存量 | 每元素计算次数 | AI | 瓶颈 |
|---|---|---|---|---|---|
| **matmul** ($QK^\top$) | $O(N^2d)$ | $O(N^2)$ | $d$ 次 | $O(d)$，约几十~100+ | 偏 compute |
| **softmax / dropout / mask** | $O(N^2)$ | $O(N^2)$ | 常数次 | $O(1)$，约 1~10 | 重度 memory-bound |

### 结论

> softmax **不是没有计算量**，它的绝对计算量是 $O(N^2)$ 并不小；但它的**算术强度极低（≈ 常数）**——因为每个数据元素只被复用了常数次运算，搬进搬出的开销主导一切。这就是它 memory-bound 的根本原因。

这正是 FlashAttention 的动机：既然 softmax/dropout/mask 这些逐元素操作是访存瓶颈，就把它们和前后的 matmul **融合进一个 kernel**，让中间的 $N\times N$ 矩阵**根本不落 HBM**，从而把这些低 AI 操作的访存开销彻底省掉。

---

## 六、如何判断 memory-bound 还是 math-bound

靠 **Roofline 模型** 和 **算术强度（Arithmetic Intensity）** 分析。

### 核心工具：算术强度

$$
\text{算术强度 (AI)} = \frac{\text{计算量 (FLOPs)}}{\text{访存量 (Bytes)}} \quad (\text{单位：FLOPs/Byte})
$$

表示"每从内存搬 1 个字节，能做多少次浮点运算"。

### 判断标准：和硬件的 ridge point（平衡点）比

$$
\text{硬件平衡点} = \frac{\text{峰值算力 (FLOP/s)}}{\text{内存带宽 (Byte/s)}}
$$

以 A100 为例：

- 峰值算力 ≈ 312 TFLOP/s（FP16 Tensor Core）
- HBM 带宽 ≈ 1.5~2.0 TB/s
- 平衡点 ≈ $312 / 1.9 \approx 160$ FLOPs/Byte

**判断规则：**

- 若 **算法 AI < 硬件平衡点** → 搬数据比算得慢 → **memory-bound（访存受限）**
- 若 **算法 AI > 硬件平衡点** → 算得比搬数据慢 → **math/compute-bound（计算受限）**

### 用它分析 Attention

整体 Attention 简化估算：

- 计算量：$O(N^2 d)$ FLOPs
- 访存量（标准实现要读写 $N\times N$ 矩阵）：$O(N^2)$ Bytes 量级

$$
\text{AI} \sim \frac{N^2 d}{N^2} = O(d)
$$

$d$ 通常 64/128，AI 大约在几十到 100 出头，**低于** A100 平衡点 $\sim 160$。尤其像 **softmax、dropout、masking 这些逐元素操作**，AI 极低（接近 1），是重度 memory-bound。

**结论：Attention（尤其 softmax/mask/dropout 环节）的算术强度低于硬件平衡点，所以是 memory-bound。**

### 工程上如何实际观察

1. **Profiling**：用 Nsight Compute / nvprof 看 kernel，若观察到计算单元利用率低、而内存带宽利用率接近饱和 → memory-bound 的典型特征。
2. **时间分解**：把 Attention 拆成 matmul、softmax、dropout、mask 等步骤计时，会发现两个 matmul 虽占绝大部分 FLOPs，但**非矩阵乘的逐元素操作却占了大量时间**（因为它们全是访存），直接说明瓶颈在访存而非计算。

---

## 七、逻辑链总结

把前面几个问题串起来的完整逻辑链：

1. **Attention 是 memory-bound**（第六节用算术强度证明：softmax/mask/dropout 的 AI ≈ 1，低于硬件平衡点）。
2. 所以优化目标应该是**减少访存**而非减少计算（第二节）。
3. FlashAttention 通过 **tiling + online softmax + recomputation**，把 HBM 访问从 $O(N^2)$ 降到 $O(N^2 d^2/M)$（第三节），计算量不变（反向略增）。
4. 推导中出现的常数 `4` 来自 "SRAM 要同时放下 Q/K/V/O 四块"，仅影响精确推导、不影响渐进复杂度（第四节）。
5. 最终墙钟时间大幅下降 → **提速 2~4 倍**。

### 一句话记忆

> FlashAttention V1 = 不改计算量、不做近似，只靠"分块 + 在线 softmax + 反向重计算"把 $N\times N$ 中间矩阵挡在 HBM 之外，从而把 memory-bound 的 Attention 大幅提速。

---

## 配套笔记

- 本篇《FlashAttention V1 学习笔记》：算法原理、访存复杂度推导、softmax 算术强度等。
- 《FlashAttention V2 学习笔记》：V2 三大改进、并行维度、warp 划分、autograd 手写反向等。
- 《GPU 执行与存储模型学习笔记》：SM/warp 调度、存储层级、数据搬运、分块≠融合——为本篇的"HBM 访问""memory-bound"等概念提供物理基础。
