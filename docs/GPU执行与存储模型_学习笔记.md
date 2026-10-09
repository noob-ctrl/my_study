# GPU 执行与存储模型学习笔记（用矩阵乘理解 SM / warp / 存储层级）

> 本文档配套《FlashAttention V1/V2 学习笔记》，把 GPU 底层执行模型系统整理成一篇，回答：
> - GPU 怎么调度 SM 和 warp？（用一个矩阵乘例子从头走一遍）
> - warp 的运算逻辑由谁执行、结果存哪、数据从哪取？
> - GPU 的存储层级（寄存器 / Shared Memory / HBM）是怎么回事？
> - kernel 内部是怎么从 HBM 逐级加载数据的？
> - 一个关键澄清：**"分块（tiling)" ≠ "融合（fusion)"**，标准 Attention 慢在哪。
>
> 这块内容是 FlashAttention 一切优化的物理基础，理解它就理解了 FlashAttention 为什么快。

---

## 目录

1. [GPU 执行层级总览](#一gpu-执行层级总览)
2. [用矩阵乘走一遍 SM 与 warp 的调度](#二用矩阵乘走一遍-sm-与-warp-的调度)
3. [warp 的执行逻辑：谁执行、SIMT 怎么齐步走](#三warp-的执行逻辑谁执行simt-怎么齐步走)
4. [GPU 存储层级：寄存器 / Shared Memory / HBM](#四gpu-存储层级寄存器--shared-memory--hbm)
5. [kernel 内部如何从 HBM 逐级加载数据](#五kernel-内部如何从-hbm-逐级加载数据)
6. [关键澄清：分块（tiling） ≠ 融合（fusion）](#六关键澄清分块tiling--融合fusion)
7. [一切串起来：这些如何解释 FlashAttention 为什么快](#七一切串起来这些如何解释-flashattention-为什么快)

---

## 一、GPU 执行层级总览

| 层级 | 是什么 | 数量级 | 对应硬件 | 谁决定 |
|---|---|---|---|---|
| **Grid（网格）** | 一次 kernel 启动的所有线程 | 整个任务 | 整个 GPU | kernel 启动时 |
| **Thread Block（线程块）** | 一组线程，共享 Shared Memory、能同步 | 每块几百~1024 线程 | 调度到**一个 SM** 上 | **程序员**切分 |
| **Warp（线程束）** | 32 个线程捆一起，同步执行同一条指令（SIMT） | 固定 32 | SM 内调度单位 | 硬件打包 + 程序员分工数据 |
| **Thread（线程）** | 最小执行单位 | 1 | 一个 CUDA Core | 程序员写逻辑 |

**核心分工记忆：**

- **切成多少 block、每个 warp/线程算哪块数据** → **程序员**在 kernel 里写（软件）；
- **block 派到哪个 SM、线程怎么打包成 warp、warp 怎么调度切换** → **GPU 硬件**自动做。

---

## 二、用矩阵乘走一遍 SM 与 warp 的调度

### 例子设定

计算 $C = A \times B$：$A, B, C$ 均为 $512 \times 512$，$C$ 共 $262144$ 个输出元素。

$$
C[i][j] = \sum_{k=0}^{511} A[i][k] \times B[k][j]
$$

即 $C$ 每个元素 = A 第 $i$ 行与 B 第 $j$ 列的点积（512 次乘加）。硬件设为 A100：**108 个 SM**，每个 SM 有 4 个 warp 调度器。

### 第一步：把 C 切成 tile，分给 thread block（程序员做）

把 C 切成 $16\times16$ 的 tile，每个 tile 交给一个 thread block：

```
C (512×512) 切成网格，每个小方格是一个 16×16 tile → 一个 thread block

  ┌────┬────┬────┬─ ... ─┐
  │ B0 │ B1 │ B2 │       │   每行 512/16 = 32 个 tile
  ├────┼────┼────┼─ ... ─┤
  │ B32│ B33│    │       │   共 32 行
  └────┴────┴────┴─ ... ─┘

thread block 总数 = 32 × 32 = 1024 个
```

这 **1024 个 block 就是"独立任务"**（互不依赖），对应**并行维度**（沿 C 的行、列切分）。

### 第二步：GPU 把 1024 个 block 分派到 108 个 SM（硬件做）

- 规则：一个 block 只在一个 SM 上运行；一个 SM 可同时驻留多个 block。
- 1024 > 108，所以**分批（wave）**调度：

```
第 1 波：108 个 SM 各领 block → 同时计算
   ↓ 算完的 SM 释放资源
第 2 波：SM 从剩下的 block 里再领取
   ↓ ... 直到 1024 个 block 全部算完
```

> "GPU 把任务扔进池子，SM 谁空了就领一个。" 谁分到哪个 SM、分几批，**硬件调度，程序员不用管**。

### 第三步：block 内把 256 个元素切给多个 warp（程序员做）

一个 block 算 $16\times16=256$ 个元素，配 256 个线程。硬件自动把 256 线程每 32 个打包成 warp → $256/32 = 8$ 个 warp。程序员在代码里决定每个 warp/线程算哪些元素（最简单：256 线程一一对应 256 个输出元素）：

```
        列0   列1   ...  列15
 行0   T0    T1    ...  T15    ┐
 行1   T16   T17   ...  T31    ┘ warp0（线程0~31）
 行2   T32   ...                 warp1（线程32~63）
 ...
 行15  T240  ...        T255    warp7（线程224~255）

每个线程 Tn 负责算一个 C[i][j] = A第i行 · B第j列 的点积
```

---

## 三、warp 的执行逻辑：谁执行、SIMT 怎么齐步走

### 代码由线程承载，但以 warp 为单位调度执行

- **编程视角（SPMD）**：你写 kernel 时站在**单个线程**视角（"我这个线程算哪个元素"），用 `threadIdx` 区分。
- **执行视角（SIMT）**：硬件不逐个线程跑，而是**以 warp 为单位调度**——32 个线程绑在一起，同一时刻执行同一条指令。

> **warp 是"执行/调度单位"，thread 是"逻辑/编程单位"。** 程序员定义的运算逻辑由线程承载，但线程以 warp（32 个齐步走）为单位被打包、调度、执行。

### warp 内 32 线程怎么齐步走

聚焦 warp0 算点积：

```cuda
float sum = 0;
for (int k = 0; k < 512; k++) {
    sum += A[i][k] * B[k][j];   // 32 个线程同时执行这一条
}
C[i][j] = sum;
```

在 $k=0$ 这一步：32 个线程**同时**执行 `sum += A[i][k]*B[k][j]`（同一条指令），但线程 0 的 $(i,j)=(0,0)$、线程 1 的 $(i,j)=(0,1)$……**同一指令，32 份不同数据，一拍算完**。这就是 SIMT。

**访存合并（coalescing）**：32 线程读连续地址时，硬件合并成**一次**大内存事务，效率极高。

### SM 靠多 warp 切换隐藏延迟（latency hiding）

某 warp 卡在等数据时，SM 立刻切去执行别的 warp：

```
t0: warp0 发起访存 → 等待（~几百周期）
   ↓ SM 立刻切换
t1: warp1 计算 → 也发起访存 → 等待
   ↓ 切换
t2: warp2 计算 ...
   ↓（warp0 数据回来）切回 warp0 继续
```

用"别的 warp 在算"填满"这个 warp 等数据"的空档。warp 越多越能藏延迟——这就是为何要切多 warp，也是 GPU "超额订阅"的原因。

### 各层级职责总结表

| 层级 | 这个例子里是什么 | 数量 | 谁决定 |
|---|---|---|---|
| **Grid** | 整个 $C=A\times B$ | 1 | kernel 启动 |
| **Thread Block** | 一个 $16\times16$ 的 C 子块 | 1024 个 | **程序员**切分 |
| **SM** | 物理车间，运行 block | 108 个 | **硬件**调度 |
| **Warp** | 32 线程齐步走，算 32 个元素 | 每 block 8 个 | 硬件打包 + **程序员**分工 |
| **Thread** | 算一个 $C[i][j]$ 点积 | 每 block 256 个 | **程序员**写逻辑 |

---

## 四、GPU 存储层级：寄存器 / Shared Memory / HBM

这是理解 GPU 性能的核心，也是 FlashAttention 一切优化的根基。

| 存储层级 | 位置 | 速度 | 大小 | 作用域 |
|---|---|---|---|---|
| **寄存器 Register** | 每线程私有 | **最快**（1 周期） | 极小（每线程几百个） | **单个线程私有** |
| **Shared Memory** | SM 片上（SRAM） | 很快（几十周期） | 小（每 SM ~100+ KB） | **同 block 内所有线程共享** |
| **L2 Cache** | 全芯片共享 | 中等 | 几十 MB | 所有 SM |
| **HBM（全局显存）** | 芯片外 | **最慢**（几百周期） | 大（几十 GB） | 所有线程 + CPU |

```
        ┌─────────────────────────────────────┐
        │          HBM（几十 GB，最慢）          │  ← Q,K,V,C 大矩阵住这里
        └─────────────────────────────────────┘
                        ↑↓  (慢，几百周期)
        ┌─────────────────────────────────────┐
        │           L2 Cache（全芯片共享）        │
        └─────────────────────────────────────┘
                        ↑↓
  ┌───────────── 一个 SM 内部 ──────────────────┐
  │   ┌───────────────────────────────────┐    │
  │   │  Shared Memory（~100KB，block 共享）│    │ ← tile 数据缓存这里
  │   └───────────────────────────────────┘    │
  │        ↑↓ (快，几十周期)                     │
  │   ┌──────────┐ ┌──────────┐ ┌──────────┐   │
  │   │ 寄存器    │ │ 寄存器    │ │ 寄存器    │   │ ← 计算结果/中间值住这里
  │   │(线程0私有)│ │(线程1私有)│ │(线程2私有)│   │
  │   └──────────┘ └──────────┘ └──────────┘   │
  └─────────────────────────────────────────────┘
```

**结果存哪、数据从哪取（结合点积例子）：**

- **运算时取数据**：原始 A、B 在 **HBM** → 先搬到 **Shared Memory**（一次搬运，反复复用）→ 计算时载入**寄存器** → 送计算单元。
- **中间结果**：累加值 `sum` 存在**线程私有寄存器**里（warp 内 32 线程各有各的 `sum`）。
- **最终结果**：写回 **HBM**（供后续 kernel / CPU 使用）。

---

## 五、kernel 内部如何从 HBM 逐级加载数据

**核心答案：不是直接从 HBM 读来算，而是逐级搬运。** 计算单元只能读寄存器：

```
HBM ──►（Shared Memory）──► 寄存器 ──► 计算单元(ALU/Tensor Core)
最慢         很快             最快
```

### 用 `S = Q @ Kᵀ` 这个 kernel 走一遍

**第 0 步**：$Q, K$ 整个大矩阵在 **HBM**。

**第 1 步：HBM → Shared Memory（分块搬运）**

block 内线程协作，把要用的 $Q, K$ 子块从 HBM 搬**一次**到 Shared Memory：

```cuda
__shared__ float Qs[16][16];
__shared__ float Ks[16][16];

Qs[ty][tx] = Q_global[row][col];   // HBM → Shared Memory
Ks[ty][tx] = K_global[row][col];
__syncthreads();                   // 等所有线程搬完再算
```

> **为什么搬到 Shared Memory？** tile 里数据被反复复用（每个 Q 元素要和 K 的 16 列各点积一次 → 用 16 次）。搬进 Shared Memory 一次，之后从快存储读 16 次，省掉重复的慢速 HBM 访问。**这就是分块（tiling）的核心价值：用片上快存储缓存被复用的数据。**

**第 2 步：Shared Memory → 寄存器（计算时载入）**

```cuda
float sum = 0;                     // sum 在寄存器
for (int k = 0; k < 16; k++) {
    float q  = Qs[ty][k];          // Shared Memory → 寄存器
    float kk = Ks[k][tx];          // Shared Memory → 寄存器
    sum += q * kk;                 // 在寄存器/计算单元里乘加
}
```

**第 3 步：计算单元执行（只认寄存器）**

`sum += q * kk` 时，ALU/Tensor Core 从**寄存器**取 `q, kk`，算完写回寄存器 `sum`。计算单元**够不着 Shared Memory 和 HBM**，只和寄存器打交道。

**第 4 步：结果写回 HBM**

```cuda
S_global[row][col] = sum;          // 寄存器 → HBM（供 softmax kernel 用）
```

### 数据流动全图

```
┌─────────── HBM（慢）───────────┐
│  Q, K 大矩阵        S 结果矩阵    │
└──┬────────────────────▲────────┘
   │ ① 分块搬运          │ ④ 写回结果
   ▼                     │
┌── SM 内部 ─────────────┼────────┐
│ ┌── Shared Memory ──┐  │        │
│ │ Qs[16][16]         │  │        │
│ │ Ks[16][16]（复用）  │  │        │
│ └────────┬──────────┘  │        │
│          │ ② 计算时载入  │        │
│          ▼              │        │
│    ┌─ 寄存器（私有）─┐    │        │
│    │ q, kk, sum     │────┘        │
│    └───────┬────────┘             │
│            │ ③ 乘加                │
│            ▼                      │
│      计算单元 ALU / Tensor Core    │
└───────────────────────────────────┘
```

### 两个关键补充

**补充 1：Shared Memory 不强制，但通常必经。**

- 会被**复用**的数据（矩阵乘输入）→ 先进 Shared Memory 缓存才划算。
- 只用一次、**不复用**的数据（如逐元素加法 `C[i]=A[i]+B[i]`）→ 可跳过 Shared Memory，直接 HBM → 寄存器 → 算 → 写回。
- 严格路径：**HBM →（可选 Shared Memory）→ 寄存器 → 计算单元**。

**补充 2：硬件还有 L1/L2 Cache 自动缓存。**

- **Cache（L1/L2）**：硬件**自动**管理，缓存什么由硬件决定。
- **Shared Memory**：程序员**手动**管理（自己写代码搬进搬出），能精确控制"把哪块数据留在片上复用"。这是高性能 kernel 都手动用 Shared Memory 而不完全依赖 cache 的原因。
- （L1 Cache 和 Shared Memory 在物理上常是同一块 SRAM。）

---

## 六、关键澄清：分块（tiling） ≠ 融合（fusion）

一个极易混淆的点，一定要分清。

### 澄清一：标准 Attention 也分块 —— 分的是"单个算子内部"

任何矩阵乘在 GPU 上都是分块（tiling）的，标准 Attention 也不例外。准确表述：

> 标准 Attention 是**一个算子一个算子**跑，每个算子（matmul、softmax）**各自内部分块**，但**算子之间**的中间结果 $S$、$P$ **完整落回 HBM**。

标准 Attention 的执行（三个独立 kernel）：

```
Kernel 1: S = Q @ Kᵀ
   - 内部：Q、K 分块搬进 Shared Memory 算（确实分块！）
   - 结束：把完整 N×N 矩阵 S 写回 HBM        ← 问题在这
Kernel 2: P = softmax(S)
   - 从 HBM 读回完整 S → 算 softmax → 完整 P 写回 HBM  ← 问题在这
Kernel 3: O = P @ V
   - 从 HBM 读回完整 P → 分块算 → O 写回 HBM
```

**关键**：每个 kernel 内部**都分块了**，问题不在"块"，而在于——**三个 kernel 分开，中间的 $N\times N$ 大矩阵在 HBM 里走了一个来回**。

### 澄清二：不是"数据太大加载不上"，而是"算子边界强制它落盘"

中间矩阵落 HBM 的真正原因是 **kernel 的边界**：

- 每个 kernel **独立启动、独立结束**。kernel 结束时，它在 Shared Memory / 寄存器里的中间结果**全部释放**（那是临时工作区）。
- 所以 kernel 1 算的 $S$，若要 kernel 2 用到，**只能写回 HBM**（唯一能跨 kernel 保存数据的地方）。
- 这跟数据大小**无关**——哪怕 $S$ 很小，只要跨 kernel 传递，就得经过 HBM。

> 中间矩阵落 HBM，**不是因为放不下 SRAM，而是因为它必须在两个独立 kernel 间"接力"，唯一的接力棒交接点就是 HBM。**

### FlashAttention 做了什么：Kernel Fusion（算子融合）

把三个 kernel **融合成一个**：

```
FlashAttention（一个 kernel 干完所有事）:
   for 每个 Q 块:
       for 每个 K/V 块:
           在 Shared Memory / 寄存器里:
               S_block = Q_block @ K_blockᵀ       ┐
               P_block = online_softmax(S_block)  ├─ 全程在片上，不落 HBM
               O_block += P_block @ V_block        ┘
       把 O_block 写回 HBM（只写最终结果）
```

同一个 kernel 内，中间的 $S$、$P$ 块**始终待在 Shared Memory / 寄存器**，用完立刻被下一步消费，**不写回 HBM**。整个过程 HBM 只读了 Q/K/V、写了最终 O。

### 对比表

| | 标准 Attention | FlashAttention |
|---|---|---|
| 每个 matmul 内部分块？ | ✅ 分块 | ✅ 分块 |
| 有几个 kernel？ | **3 个独立 kernel** | **1 个融合 kernel** |
| 中间 $S$、$P$ 存哪？ | **完整写回 HBM** | **只在 Shared Memory/寄存器** |
| $N\times N$ 大矩阵 HBM 读写 | 有（S 写+读，P 写+读） | **没有** |

### 分块 vs 融合是两个层面的事

1. **分块（tiling）**：解决"一个矩阵乘怎么在有限 Shared Memory 里高效算" —— **标准和 FlashAttention 都做**，是 GEMM 基本功。
2. **融合（fusion）**：解决"多个算子之间的中间结果要不要落 HBM" —— **这才是 FlashAttention 的创新点**。

> FlashAttention 的 tiling 除了"让块放进 Shared Memory"，还多一层意义：**块要足够小，才能让 $S_{block}$、$P_{block}$ 完全放进片上、在融合 kernel 内流转而不外溢**。这也是它必须配合 **online softmax** 的原因——softmax 要全局归一化，但融合后一次只有一个块，得用 online softmax 增量地算。
>
> **一句话：慢的根源是"没融合"，不是"没分块"。**

---

## 七、一切串起来：这些如何解释 FlashAttention 为什么快

把整条 GPU 执行链路和 FlashAttention 对应起来：

| GPU 执行/存储概念 | FlashAttention 对应 |
|---|---|
| 把输出切成 tile 分给 block | 把 Q 按行分块，每块给一个 block（V2 序列维并行） |
| block 内切给多个 warp | split-Q：block 内 Q 行切给不同 warp（V2） |
| 线程算 tile 元素点积（调 Tensor Core） | 算 $S=QK^\top$、$PV$ 的 tile 元素 |
| 多 warp 切换隐藏访存延迟 | 藏住从 HBM/SRAM 读 Q/K/V 的延迟 |
| 存储层级：寄存器/Shared Memory/HBM | 让中间 $S$、$P$ 只活在寄存器/Shared Memory |
| kernel 边界强制中间结果落 HBM | 标准 Attention 的 $N\times N$ 落 HBM 之因 |
| kernel fusion 消除边界 | FlashAttention 融合 kernel，中间矩阵不落 HBM |

**最终逻辑链：**

1. 计算单元只读寄存器；数据走 `HBM →(Shared Memory)→ 寄存器` 逐级搬运，HBM 最慢。
2. 标准 Attention 把 attention 拆成多个独立 kernel，中间 $N\times N$ 矩阵被 kernel 边界逼着**写回 HBM 再读回**（昂贵的往返）。
3. FlashAttention 用 **kernel fusion + tiling + online softmax**，让 $N\times N$ 中间矩阵**停在寄存器/Shared Memory，绝不落 HBM**。
4. 省掉对最慢 HBM 的 $O(N^2)$ 往返 → 这就是它快的物理本质（呼应 V1 的 IO-Awareness）。

> **一句话总结全篇**：GPU 的性能瓶颈在最慢的 HBM 访问；FlashAttention 通过算子融合，把本该在 HBM 间往返的 $N\times N$ 中间矩阵留在片上快存储，从而大幅提速。SM/warp 调度决定"算得并行不并行"，存储层级决定"数据搬得快不快"，二者共同构成 FlashAttention 优化的物理基础。

---

## 配套笔记

- 《FlashAttention V1 学习笔记》：算法原理、访存复杂度推导、softmax 算术强度等。
- 《FlashAttention V2 学习笔记》：V2 三大改进、并行维度、warp 划分、autograd 手写反向等。
- 本篇《GPU 执行与存储模型》：SM/warp 调度、存储层级、数据搬运、分块≠融合——为上面两篇提供物理基础。
