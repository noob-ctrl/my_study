# FlashAttention V2 学习笔记（含 GPU 执行模型答疑）

> 论文：《FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning》（Tri Dao，2023）
>
> 本文档承接《FlashAttention V1 学习笔记》，重点讲清楚：
> - V2 相比 V1 到底改了什么，为什么能再快约 2 倍；
> - 一个贯穿全文的重要分层认知：**算法层（原理） vs 实现层（工程优化）**；
> - 几个容易困惑的 GPU 执行模型问题（并行维度、warp/thread block、SIMT、两级切 Q、为什么要手写反向）。

---

## 目录

1. [最重要的认知：算法层 vs 实现层](#一最重要的认知算法层-vs-实现层)
2. [V2 的动机：V1 还不够快在哪](#二v2-的动机v1-还不够快在哪)
3. [V2 的三大改进](#三v2-的三大改进)
4. [V1 vs V2 对比总表](#四v1-vs-v2-对比总表)
5. [答疑一：怎么理解并行维度，为什么 V2 多了序列长度维度](#五答疑一怎么理解并行维度为什么-v2-多了序列长度维度)
6. [答疑二：如何理解 warp 和 thread block](#六答疑二如何理解-warp-和-thread-block)
7. [答疑三：为什么 warp 是 32 个线程，FlashAttention 里怎么划分数据](#七答疑三为什么-warp-是-32-个线程flashattention-里怎么划分数据)
8. [答疑四：两级切 Q——block 层 vs warp 层，谁切的，为什么切](#八答疑四两级切-qblock-层-vs-warp-层谁切的为什么切)
9. [答疑五：PyTorch 能自动求导，为什么还要手写反向](#九答疑五pytorch-能自动求导为什么还要手写反向)
10. [演进脉络与一句话记忆](#十演进脉络与一句话记忆)

---

## 一、最重要的认知：算法层 vs 实现层

学习 FlashAttention 时最关键的一点：**把内容分成两个层次**，它们回答完全不同的问题。

| | **算法层（原理）** | **实现层（工程优化）** |
|---|---|---|
| 回答的问题 | 怎么算出**正确**结果，同时**不产生 $N\times N$ 大矩阵** | 怎么在 GPU 上把这个算法跑得**尽可能快** |
| 核心内容 | tiling 分块、**online softmax**、反向重计算 | 并行维度、**warp 划分（split-Q/split-K）**、Tensor Core、隐藏延迟 |
| 决定的是 | **算得对不对、省不省显存** | **算得快不快、GPU 吃不吃得饱** |
| 换个硬件还成立吗 | ✅ 成立（数学层面，与硬件无关） | ❌ 不一定（强依赖 GPU 架构） |

### 结论：理解 FlashAttention 原理，只需掌握左列

想理解 FlashAttention 的**核心思想**，掌握这三点即可：

1. **Tiling（分块）**：把 Q/K/V 切块，逐块加载进 SRAM 计算，让中间的 $N\times N$ 矩阵**永不落 HBM**。
2. **Online Softmax（在线 softmax）**：**真正的算法难点和精髓**——在只看到部分 K/V 时，通过动态维护、更新 $m$（最大值）和 $\ell$（指数和）并 rescale，得到与标准 softmax **完全一致**的结果。
3. **Recomputation（反向重计算）**：反向传播不存 $N\times N$ 矩阵，只存少量统计量，用时重新算——用少量计算换显存。

> 其中 **online softmax 最值得吃透**：分块的天然障碍是 softmax 要全局归一化，online softmax 正是破解它的钥匙。

### 有力佐证：V1 和 V2 的算法本质完全一样

- 数学结果相同（都是精确 attention）；
- IO 复杂度相同（都是 $O(N^2 d^2 / M)$）；
- V2 相比 V1 改的**几乎全是实现层**：交换循环顺序、增加序列维并行、split-K → split-Q、减少非 matmul 运算。

> **V1 → V2 基本没动算法原理，只动工程实现，就快了 2 倍。** 这恰好证明 warp 划分等属于"性能调优"，不是"算法内核"。

### 什么时候才需要深入实现层（右列）

- 要自己写/魔改 CUDA kernel（算子优化、适配新硬件、写变体 attention）；
- 要分析性能瓶颈、看 profiling、解释"为什么 V2 比 V1 快"；
- 面试被追问 GPU 底层实现。

否则，理解外层切分（tiling + online softmax + 重计算）就够了，warp 切分可以先放一放。

**下面第二~四节讲 V2 改了什么（实现层为主），第五节起是 GPU 执行模型的答疑，供想深入的时候看。**

---

## 二、V2 的动机：V1 还不够快在哪

V1 已把 Attention 从 memory-bound 解救出来，但实测 GPU 利用率只有理论峰值的 **25~40%**，而优化良好的 GEMM 能到 **80~90%**。差距来自三点：

1. **非矩阵乘（non-matmul）运算占比过高**。Tensor Core 算 matmul 极快，但 softmax 的 rescale、$\exp$ 走普通 CUDA Core，吞吐低很多（A100 上 matmul 算力约是非 matmul 的 16 倍）。V1 每处理一个块都要对已累积输出做 rescale，非 matmul 运算太多。
2. **并行度不足**。V1 只在 **batch 和 head** 两个维度并行。当 batch × head 少（长序列、小 batch）时，占不满 GPU 所有 SM。
3. **warp 之间划分不合理**。V1 在 block 内把 K/V 切给不同 warp、Q 共享，导致 warp 间要频繁通过 shared memory 通信和同步。

---

## 三、V2 的三大改进

### 改进 1：减少非矩阵乘运算（算法层微调）

V1 的 online softmax 每来一个新块都要用新的全局最大值 rescale 之前累积的输出 $O$，并维护 $\ell$。

**V2 做法：不再每步都对输出除以 $\ell$，而是把归一化推迟到最后。**

- 循环中间只累积**未归一化**的输出 $\tilde{O}$ 和 $\ell$；
- 块之间的 rescale 尽量简化；
- **整个循环结束后，才用最终 $\ell$ 对 $\tilde{O}$ 做一次除法归一化**。

这样把大量"逐块除法/rescale"（非 matmul 运算）从 $O(T_c)$ 次降到**最后 1 次**。数学结果依然精确，只是把归一化时机往后挪（等价变形）。

此外 V2 还优化了反向的部分非 matmul 运算，并减少统计量存储（反向只需存 logsumexp 一个量，而非分别存 $m$ 和 $\ell$）。

### 改进 2：增加序列长度维度的并行

- **V1**：外层循环遍历 K/V 块，内层遍历 Q 块。并行维度 = batch × head。
- **V2**：**交换循环顺序**——外层遍历 **Q 块**，内层遍历 K/V 块，并**把不同 Q 块分配到不同 thread block 上并行**。

并行维度变成 **batch × head × (序列长度 / Q块大小)**，多了**序列长度维度**。即使 batch × head 很小，只要序列够长也能占满所有 SM。

### 改进 3：更好的 warp 划分（split-K → split-Q）

- **V1（split-K）**：block 内把 **K/V 切给不同 warp**，Q 共享。每个 warp 只得到部分结果，需写 shared memory、同步、再累加 → warp 间通信开销大。
- **V2（split-Q）**：把 **Q 切给不同 warp**，K/V 共享。每个 warp 独立负责一部分 Q 行的完整计算，**warp 间几乎不需通信和同步**，算完直接写回。

> 减少了 shared memory 读写和 `__syncthreads()` 同步，进一步提速。

---

## 四、V1 vs V2 对比总表

| 维度 | FlashAttention V1 | FlashAttention V2 |
|---|---|---|
| **本质思想** | tiling + online softmax + 重计算 | 同 V1，算法本质不变 |
| **计算精度** | 精确（exact） | 精确（exact） |
| **循环顺序** | 外层 K/V，内层 Q | **外层 Q，内层 K/V**（交换） |
| **并行维度** | batch × head | batch × head × **序列长度** |
| **warp 划分** | split-K（切 K/V，Q 共享） | **split-Q**（切 Q，K/V 共享） |
| **softmax 归一化** | 每块都 rescale/除 $\ell$ | **推迟到最后一次性归一化** |
| **非 matmul 运算** | 较多 | **大幅减少** |
| **反向存储** | 存 $m$ 和 $\ell$ | 只存 logsumexp（合并统计量） |
| **GPU 利用率** | 25~40% | **~70%** |
| **相对速度** | 基准 | 比 V1 快 **~2 倍** |
| **HBM 访问复杂度** | $O(N^2 d^2/M)$ | 同量级（IO 复杂度没变） |

> **关键理解**：V1 优化"访存复杂度"（算法层的 IO-Awareness）；V2 优化"GPU 执行效率"（并行度、warp 调度、减少低效运算）。V2 的 IO 复杂度和 V1 一样，提速来自把 GPU 用得更满。

---

## 五、答疑一：怎么理解并行维度，为什么 V2 多了序列长度维度

### 什么是"并行维度"

GPU 有很多 SM（A100 有 108 个）。要喂满 GPU，就得把任务**切成很多互相独立的块**分派到不同 SM 同时算。**"并行维度"就是指沿着哪些维度切分任务，从而产生这些独立块。** 独立块越多，越能占满 SM。

### V1：并行维度 = batch × head

```
for each (batch b, head h) in parallel:      # 并行的部分
    for j in K/V blocks:                      # 外层循环（串行）
        for i in Q blocks:                    # 内层循环（串行）
            计算这一小块的 attention
```

一个 thread block 负责**一个完整 (batch, head) 的整条注意力**，内部所有 Q/K/V 块循环在同一 block 里串行完成。所以独立任务数 = **batch × head**。

**痛点**：若 `batch=1, head=16`，只有 16 个任务，而 A100 有 108 个 SM → 90 多个 SM 闲置。

### V2：多出序列长度维度

```
for each (batch b, head h) in parallel:              # 原有并行维度
    for each Q block i in parallel:                   # V2 新增的并行维度！
        for j in K/V blocks:                          # 内层循环（串行）
            计算这一小块的 attention
```

不同 Q 块被分到不同 thread block，它们**互相独立**（不同 query token 的输出本就互不依赖）。独立任务数 = **batch × head × (序列长度 / Q块大小)**。

例：`batch=1, head=16`、序列 8192、Q 块 128 → Q 块数 64 → 独立任务 `1 × 16 × 64 = 1024` 个，远超 108 个 SM，能喂满 GPU。

### 为什么 V1 不能这么切？—— online softmax 的依赖方向

- **V1（外层 K/V）**：每个 Q 块的最终结果依赖所有 K/V 块循环走完，更新分散在外层每一轮。若把 Q 块拆到不同 block，就得共享/同步跨块累积状态 → 很麻烦。
- **V2（外层 Q）**：固定一个 Q 块，在它自己的 block 内把所有 K/V 走完，累积量完全**本地、自包含**，算完直接输出，不依赖别的 Q 块 → 天然可并行。

> **一句话**：V2 交换循环顺序，使"每个 Q 块的完整计算封装在一个独立 thread block 内"，从而让序列长度维度可被安全并行化。

### 补充：并行是"同时"吗？

不一定是物理同时，而是"逻辑独立、可被并行调度"：

- SM 数量 ≥ 任务数 → 真正同时执行；
- 任务数 > SM 数量 → 分批调度（wave），执行完的 SM 再领取剩下的。

心智模型：**GPU 把任务切成一堆独立 thread block 扔进池子，SM 谁空了就领一个算。** `batch=2, head=8` 的 16 个 block 在 A100 上确实会同时算（16 < 108），但这正是 V1 没吃饱的表现——V2 靠切 Q 把 block 数拉到上千才真正占满。

---

## 六、答疑二：如何理解 warp 和 thread block

### CUDA 执行层级

| 层级 | 是什么 | 数量级 | 对应硬件 |
|---|---|---|---|
| **Grid（网格）** | 一次 kernel 启动的所有线程 | 整个任务 | 整个 GPU |
| **Thread Block（线程块）** | 一组线程，共享 shared memory、能同步 | 每块几百~1024 线程 | 调度到**一个 SM** 上 |
| **Warp（线程束）** | 32 个线程捆一起，同步执行同一条指令（SIMT） | 固定 32 | SM 内调度单位 |
| **Thread（线程）** | 最小执行单位 | 1 | 一个 CUDA Core |

### 工厂比喻

- **SM** = 车间（A100 有 108 个）
- **Thread Block** = 一个工作小组，分到某个车间，组内共用一块白板（shared memory），可互相喊话（`__syncthreads()` 同步）
- **Warp** = 小组里 32 人组成的"班"，必须齐步走——同一时刻做同一动作，只是处理不同数据（SIMT）
- **Thread** = 单个工人

### 关键规则

1. **一个 thread block 只在一个 SM 上运行**（不跨 SM）。所以 block 数量决定能用多少 SM —— 呼应答疑一：V2 产生更多 block 才能占满更多 SM。
2. **一个 block 由多个 warp 组成**（256 线程 = 8 warp）。
3. **同 block 内 warp 之间**可通过 shared memory 通信，但需同步（`__syncthreads()`），有开销。
4. **不同 block 之间**基本无法直接通信。

### 回到 V1 split-K vs V2 split-Q（block 内 warp 分工）

**V1（split-K）**：K/V 切给不同 warp，Q 共享。每个 warp 只算"部分和"，需写 shared memory → 同步 → 累加，通信开销大。

**V2（split-Q）**：Q 切给不同 warp，K/V 共享。每个 warp 独占若干 Q 行、能看到全部 K/V，自己算完完整 softmax，直接写回，几乎不用 warp 间通信/同步。

---

## 七、答疑三：为什么 warp 是 32 个线程，FlashAttention 里怎么划分数据

### 为什么是 32 个线程捆一起，而不是 1 个

**根本原因：用最少的硬件控制成本，换最大的计算吞吐（SIMT 架构）。**

- **CPU**：每个核心有完整控制单元（取指、译码、分支预测），灵活但占面积 → 核心少。
- **GPU**：目标是堆尽可能多的计算单元（ALU）。若每线程都配独立控制单元，面积全被控制逻辑吃掉。
- **GPU 取舍**：让 **32 个线程共享一套控制单元**（取指、译码只做一次），它们被迫执行**同一条指令**，只是处理不同数据。这就是 **SIMT（Single Instruction, Multiple Threads）**。

> 比喻：CPU 是 32 个各自带脑子的工人各干各的；GPU 一个 warp 是一个工头喊口令、32 人齐步走——省掉 31 套"脑子"，把面积全用来堆"手"（ALU）。

附带好处：32 线程执行同一条 load、访问连续地址时，硬件能**合并访存（coalesce）** 成一次大事务，效率极高。

代价：**分支发散（warp divergence）**。遇到 `if/else`，warp 会先执行满足条件的线程（其余等待），再执行另一批，两分支串行走完 → 损失性能。所以要尽量避免 warp 内分支发散。

### FlashAttention 里"不同数据"怎么划分

那句关键的话：**32 个线程执行同一条指令，但各有自己的寄存器、指向不同数据地址**。本质是同一段代码，各线程代入不同 `threadIdx`，访问不同内存位置。

**通用例子（向量加法）**：
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;  // 每线程算出不同 i
C[i] = A[i] + B[i];                              // 同一指令，不同数据
```

**FlashAttention 里**：一个 warp 算一小块输出 tile，按**输出矩阵的元素位置**划分。以算 $S = Q_{tile}K_{tile}^\top$ 的一小块（8×4=32 个输出）为例：

```
          key0   key1   key2   key3
query0   [T0]   [T1]   [T2]   [T3]
query1   [T4]   [T5]   [T6]   [T7]
...
query7   [T28]  [T29]  [T30]  [T31]

每个线程 Tk 负责一个 S[i][j] = 点积(Q第i行, K第j行)
```

32 线程执行同一段点积代码，但各自按 `threadIdx` 算出负责的 `(i,j)`，读 Q/K 不同行、写 S 不同位置。

**真实实现补充**：现代 FlashAttention 的矩阵乘不是让每线程傻算点积，而是调 **Tensor Core**（`mma` 指令）。32 线程**协同**喂给 Tensor Core 一小块矩阵乘（如 16×16×16），每线程持有输入/输出的特定元素（映射由硬件 mma 规格固定，称 fragment layout）。softmax 的行内求和/求 max 则用 **warp shuffle**（`__shfl_xor_sync`）在寄存器间直接归约，不走 shared memory。

> **一句话**：FlashAttention 的"不同数据"就是**输出 tile 的不同元素位置**——32 线程按各自 `threadIdx` 映射到矩阵不同 (行,列)，执行同一段代码却读写不同位置；矩阵乘部分由 32 线程协同喂 Tensor Core。

---

## 八、答疑四：两级切 Q——block 层 vs warp 层，谁切的，为什么切

一个容易混淆的关键点：**Q 在两个层级上被切了两次，是嵌套关系，不矛盾。**

### 两级切分

```
整条序列的 Q
   │
   ├─ 第一层：切给不同 thread block（不同 SM）  ← 解决"并行维度/占满 SM"（答疑一）
   │        thread block 0   thread block 1  ...  thread block 63
   │              │
   │              └─ 第二层：block 内切给不同 warp  ← 解决"warp 划分"（split-Q）
   │                    warp0(32行Q)  warp1(32行Q)  warp2  warp3
   │                    （K/V 由 block 内所有 warp 共享）
```

- **第一层（block 间，粗粒度）**：为产生足够多独立任务**占满所有 SM**。这就是"序列长度并行维度"。
- **第二层（block 内 warp 间，细粒度）**：这才是 V2 "split-Q vs split-K" 讨论的层级——block 内 warp 之间怎么分工。

### 谁切的：FlashAttention 切的，不是 GPU 自动切

| 谁负责 | 做什么 |
|---|---|
| **GPU 硬件自动做** | 1. 把 block 分派到哪个 SM；2. 把 block 内线程每 32 个自动打包成 warp（线程 0~31 = warp0…）；3. 调度 warp 执行 |
| **程序员（FlashAttention）写代码决定** | **每个 warp/线程具体算哪部分数据**（即 split-Q 分工方案） |

GPU 只负责"把 32 线程物理打包成 warp"，但**不知道**每个 warp 该算哪块 Q。分工是 kernel 代码里用 `warpId`/`threadIdx` 显式写的：

```cuda
int warp_id = threadIdx.x / 32;   // 程序员算出我是第几个 warp
int lane_id = threadIdx.x % 32;
int q_row_start = warp_id * ROWS_PER_WARP;  // 分工逻辑：这个 warp 负责哪几行 Q
```

> split-Q vs split-K 是**算法/实现层的设计选择**，V1 选 split-K，V2 改 split-Q，改的就是这段分工代码。GPU 对两种都照单全收，只是 split-Q 更快。

### 为什么要切给多个 warp

**是的，为了并行，但更准确说是"填满 SM 计算资源 + 隐藏延迟"。** 一个 block 分到一个 SM，但一个 SM 内有多个 warp 调度器和大量计算单元，光靠一个 warp 喂不饱。

**原因 1：并行占满 SM 计算单元。** 一个 SM 能同时驻留几十个 warp、有 4 个调度器。只有 1 个 warp 则大部分算力闲置。

**原因 2：隐藏内存延迟（latency hiding，GPU 高吞吐核心机制）。** 某 warp 发起访存要等几十上百周期，这时 SM 立刻切去执行别的 warp：

```
warp0 发起访存 → 等待（卡住）
   ↓ SM 立刻切换
warp1 计算 → 也发起访存 → 等待
   ↓ 切换
warp2 计算 → ...
   ↓（warp0 数据回来）切回 warp0 继续
```

用"别的 warp 的计算"填补"这个 warp 等数据"的空档。warp 越多越能藏延迟。这也是 GPU "超额订阅"的原因——故意让 warp 远多于计算单元。

### 为什么切 Q 而不切 K/V

- **split-Q**：每 warp 独占若干 Q 行、看到全部 K/V → 自己算完完整 softmax → warp 间不用通信/同步。
- **split-K**：每 warp 只有部分 K/V → 只得部分结果 → 必须写 shared memory、同步、合并 → 开销大。

> **一句话**：block 内切 warp 是 FlashAttention kernel 显式写的（GPU 只物理打包 warp，不管分工）；切多 warp 是为了**并行占满 SM + 用 warp 切换隐藏访存延迟**；选按 Q 切（split-Q）是为了让每 warp **独立算完、避免 warp 间同步**。

---

## 九、答疑五：PyTorch 能自动求导，为什么还要手写反向

### 结论

> PyTorch autograd 只对"它能看见的、由 PyTorch 算子拼起来的计算图"生效。FlashAttention 是一个手写 CUDA 的融合自定义算子（custom kernel），autograd 看不见它内部，**所以必须手动实现反向**。

### autograd 为什么对普通代码有效

每调用一个 PyTorch 算子（`torch.matmul`、`torch.softmax`），它会：执行前向 → 后台记录进计算图并保存反向所需中间结果 → `loss.backward()` 时沿图反向、用预定义求导规则自动算梯度。

用标准 PyTorch 写 attention **不用手写反向**：
```python
S = Q @ K.transpose(-1, -2)      # 记入计算图
P = torch.softmax(S, dim=-1)      # 记入计算图
O = P @ V                         # 记入计算图
# backward() 自动求导
```
但它会把 $S$、$P$ 这些 $N\times N$ 中间矩阵**存在 HBM 里**以备反向——显存又回到 $O(N^2)$，这正是 FlashAttention 想避免的。

### 为什么 FlashAttention 必须手写反向

FlashAttention 前向是**一个手写 CUDA kernel**，把 matmul/softmax/dropout 全融合，中间 $N\times N$ 矩阵从不写回 HBM。对 PyTorch 而言它只是**一个黑盒算子**：

```
输入 Q,K,V → [FlashAttention CUDA kernel 黑盒] → 输出 O
```

autograd 看不到黑盒内部（没经过 PyTorch 算子记录），不知道怎么求导。所以必须：
1. **手动推导**反向数学公式（从 $dO$ 算 $dQ, dK, dV$）；
2. **手写反向 CUDA kernel**；
3. 通过 `torch.autograd.Function` 把前向/反向**注册**给 PyTorch。

```python
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, logsumexp = flash_attn_forward_cuda(Q, K, V)   # 手写前向 kernel
        ctx.save_for_backward(Q, K, V, O, logsumexp)       # 只存少量东西
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, logsumexp = ctx.saved_tensors
        # 手写反向 kernel，内部重计算 S、P（recomputation）
        dQ, dK, dV = flash_attn_backward_cuda(dO, Q, K, V, O, logsumexp)
        return dQ, dK, dV
```

注册后，上层调用它时 `loss.backward()` 就能正常工作——autograd 遇到这个 `Function` 会自动调用你写的 `backward`。

### 关键：反向里的"重计算"（呼应 V1 设计）

`save_for_backward` **只存 $Q,K,V,O$ 和 logsumexp，没存 $N\times N$ 的 $S$、$P$**（存了又占 $O(N^2)$）。反向时在 kernel 内用 $Q,K$ 和 logsumexp **重新算 $S$、$P$**（recomputation）——牺牲少量计算换取不存大矩阵，保持 $O(N)$ 显存。

| | 普通 PyTorch attention | FlashAttention |
|---|---|---|
| 前向 | 一串 PyTorch 算子 | 一个手写 CUDA 黑盒 kernel |
| autograd 能否自动求导 | 能（图对它透明） | 不能（黑盒不透明） |
| 反向 | autograd 自动生成 | **必须手写反向 kernel + 注册 Function** |
| 中间矩阵存储 | 存 $N\times N$（$O(N^2)$ 显存） | 不存，反向时重计算（$O(N)$ 显存） |

> **一句话**：autograd 只对"PyTorch 认识的算子"自动求导；FlashAttention 是手写融合黑盒，PyTorch 不认识其内部，所以必须手动实现并注册反向。而手动实现顺便用"重计算"避免了存储大矩阵，正是它省显存的关键。

---

## 十、演进脉络与一句话记忆

| 版本 | 核心优化点 | 提速来源 |
|---|---|---|
| **V1** (2022) | tiling + online softmax + recomputation | 减少 HBM 访问，摆脱 memory-bound |
| **V2** (2023) | 减少非 matmul 运算 + 序列维并行 + split-Q warp 划分 | 提高 GPU 利用率（25~40% → ~70%） |
| **V3** (2024) | 针对 Hopper：异步（warp specialization、TMA）、FP8、乘法与 softmax 重叠 | 利用新硬件特性，H100 进一步提速 |

### 一句话记忆

> - **V1**：用分块把 $N\times N$ 矩阵挡在 HBM 外，解决**访存瓶颈**（算法层）。
> - **V2**：算法本质不变，通过**减少非矩阵乘运算 + 增加序列维并行 + 优化 warp 划分**，把闲着的 GPU **喂饱**，再快 2 倍（实现层）。

### 学习建议（最重要的一条）

> 理解 FlashAttention **原理**只需掌握**算法层**：tiling + online softmax + 重计算（尤其 online softmax）。warp 划分、Tensor Core、隐藏延迟等属于**实现层**的性能工程，与算法正确性、显存复杂度无关，需要做算子优化/性能分析时再深入即可。

---

## 配套笔记

- 《FlashAttention V1 学习笔记》：算法原理、访存复杂度推导、softmax 算术强度等。
- 本篇《FlashAttention V2 学习笔记》：V2 三大改进、并行维度、warp 划分、autograd 手写反向等。
- 《GPU 执行与存储模型学习笔记》：SM/warp 调度、存储层级、数据搬运、分块≠融合——为本篇的 warp 划分、并行维度、隐藏延迟等 GPU 概念提供更完整的物理基础和矩阵乘示例。
