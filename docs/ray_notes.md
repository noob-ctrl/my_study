# Ray 学习笔记（结合 verl 实践）

> 本文档沉淀的是**纯 Ray 相关的知识**，可独立学习；同时穿插 verl 的实战例子帮助理解。
> 与 verl 业务架构相关的内容请看 [`verl_architecture_notes.md`](./verl_architecture_notes.md)。
>
> 适用版本：Ray ≥ 2.10（verl 主要在用），核心概念在 Ray 1.x 之后大体一致。

---

## 目录

- [1. Ray 是什么、为什么用它做 RL 训练](#1-ray-是什么为什么用它做-rl-训练)
- [2. 核心概念：Task / Actor / Object / Driver](#2-核心概念task--actor--object--driver)
- [3. Placement Group + Bundle](#3-placement-group--bundle)
- [4. 资源调度：`num_cpus` / `num_gpus` 与共置](#4-资源调度num_cpus--num_gpus-与共置)
- [5. ActorHandle 与 Ray RPC 通信](#5-actorhandle-与-ray-rpc-通信)
- [6. `runtime_env`：环境变量、依赖、镜像隔离](#6-runtime_env环境变量依赖镜像隔离)
- [7. Ray 集群：head / worker / dashboard / job](#7-ray-集群head--worker--dashboard--job)
- [8. 多机训练：Ray 替代 torchrun 的机制](#8-多机训练ray-替代-torchrun-的机制)
- [9. 调试与可观测性](#9-调试与可观测性)
- [10. Ray 在 verl 中的具体使用速查](#10-ray-在-verl-中的具体使用速查)
- [11. 常见误解与陷阱](#11-常见误解与陷阱)
- [12. 学习路径与延伸阅读](#12-学习路径与延伸阅读)

---

## 1. Ray 是什么、为什么用它做 RL 训练

### 1.1 Ray 的定位

Ray 是一个**分布式 Python 运行时**，核心目标是让 Python 函数 / 类的调用可以无缝跨进程、跨机器执行。它提供：

- **Task**：远程函数调用（无状态）
- **Actor**：远程类实例（有状态，独享进程）
- **Object Store**：进程间零拷贝共享内存（基于 plasma）
- **Placement Group**：精细资源调度
- **Runtime Env**：依赖隔离、环境变量注入
- **Dashboard**：作业 / actor / 资源监控

它不是单纯的"任务队列"或"RPC 框架"，而是一套统一的**分布式编程抽象**——你写起来像单机 Python，背后是一个集群。

### 1.2 为什么 LLM RL 训练几乎都选 Ray

LLM RL 是个很特殊的工作负载：

1. **多种角色（actor / critic / ref / rollout / reward）共存**，每种用不同后端（FSDP / Megatron / vLLM / SGLang / 规则函数）
2. **GPU 资源紧张**，希望让多种角色共置同一张卡（HybridEngine）
3. **rollout 阶段异步、长尾**，不同 prompt 长短差异大
4. **需要跨机训练 + 跨机推理 + 跨机权重同步**，三种通信拓扑要并存

`torchrun` 只能解决"一种角色铺满所有卡"的均匀模型，无法表达"同卡多角色 + 异步 rollout"。Ray 的 actor 模型 + placement group + runtime_env 恰好能把这套复杂性优雅地表达出来。

verl、OpenRLHF、TRL（部分）、ColossalChat 等主流 RL 框架都选 Ray，原因都是这个。

---

## 2. 核心概念：Task / Actor / Object / Driver

### 2.1 Driver

**Driver = 启动 Ray 集群（或 `ray.init()`）的那个进程**。它是控制面的入口，提交 Task / 创建 Actor / 等待结果都由它发起。

在 verl 里，`TaskRunner.run()` 这个 Ray actor 起到 driver 角色（虽然它本身也是 actor，但承担"主控"职责）。

### 2.2 Task

```python
@ray.remote
def add(a, b):
    return a + b

future = add.remote(1, 2)        # 提交到集群（不阻塞）
result = ray.get(future)          # 等结果
```

特点：
- **无状态**：每次调用是独立的
- **进程池复用**：Ray 用 worker 进程池跑 task，进程在多个 task 间复用
- **轻量**：调用开销小，适合短任务

### 2.3 Actor

```python
@ray.remote
class Counter:
    def __init__(self):
        self.n = 0
    def inc(self):
        self.n += 1
        return self.n

c = Counter.remote()              # 创建 actor → 拉起独立 OS 进程
ray.get(c.inc.remote())           # → 1
ray.get(c.inc.remote())           # → 2  状态保留
```

特点：
- **有状态**：实例属性长期保留
- **独享进程**：每个 actor 是一个独立 Python 解释器
- **顺序执行**：默认同一 actor 上的方法按调用顺序执行（除非用 `async def` 和 `max_concurrency`）

### 2.4 Object（ObjectRef）

`*.remote(...)` 立刻返回的不是结果，而是 `ObjectRef`（对结果的引用）：

```python
ref = add.remote(1, 2)            # ObjectRef
print(type(ref))                  # ray.ObjectRef
result = ray.get(ref)             # 此时才阻塞等待
```

**关键设计**：
- 大对象（>~100KB）会进入 **plasma 共享内存**（Object Store），跨进程零拷贝读
- ObjectRef 可以作为参数传给其他 task / actor，Ray 自动解析依赖

```python
ref1 = func1.remote(big_data)     # 输出在 plasma
ref2 = func2.remote(ref1)         # ref1 在 ref2 调度时被自动 ray.get
```

### 2.5 Task vs Actor 对比

| 维度 | Task | Actor |
|---|---|---|
| 状态 | 无状态 | 有状态 |
| 进程 | 池子复用 | 独享 |
| 适合 | 函数式计算、map-reduce | 长驻服务、模型 worker |
| RL 训练里的位置 | reward 计算（部分）、辅助函数 | trainer worker、rollout worker、reward worker |

verl 里几乎都是 actor（GPU worker 全是 actor）。

---

## 3. Placement Group + Bundle

这是分布式训练里最关键的 Ray 概念，也是初学者最容易绕晕的部分。

### 3.1 为什么需要 Placement Group

Ray 默认调度是「**逐 actor 独立调度**」：你 `Counter.remote()` 一次，Ray 找一台空闲机器塞进去。下次再起就可能是另一台。

但分布式训练里这是**致命问题**：
- 你需要 16 个 worker，刚好占满 2 台 8 卡机
- Ray 默认可能把它们散到 3 台 5/5/6 不均
- 剩下的位置又被别的任务占走，导致后续 worker 起不来
- NCCL 性能强烈依赖**同机走 NVLink、跨机走 RDMA**，乱放性能断崖

Placement Group 就是 Ray 提供的「**资源组**」抽象：
1. **一次性预留**一组资源，要么全成功、要么全失败（**gang scheduling**）
2. **约束放置策略**：要么挤同节点（PACK），要么散到不同节点（SPREAD）
3. **绑定 actor 到具体 bundle**：精确指定第 i 个 actor 落到 PG 的第 j 个 bundle

### 3.2 Bundle：最小资源单元

**Bundle = "一组资源，必须能塞到同一台节点上"**：

```python
{"CPU": 4, "GPU": 1}                          # 4 CPU + 1 GPU
{"CPU": 2, "GPU": 1, "memory": 8 * 1024**3}   # 加内存约束
{"NPU": 1}                                    # 华为 NPU
{"CPU": 1, "TPU": 1}                          # TPU
```

Ray 调度器把 bundle 整体当成调度单元：要么把这 1 张 GPU + 4 CPU 在某台节点上整体找到，要么调度失败。**bundle 永远不跨节点**。

### 3.3 Placement Group：一组 Bundle + 放置策略

```python
from ray.util.placement_group import placement_group

pg = placement_group(
    bundles=[
        {"CPU": 4, "GPU": 1},
        {"CPU": 4, "GPU": 1},
        {"CPU": 4, "GPU": 1},
        {"CPU": 4, "GPU": 1},
    ],
    strategy="PACK",                # 放置策略
    name="my_pg",
    lifetime=None,                  # 或 "detached"
)

ray.get(pg.ready())                 # 等待 gang scheduling 完成
```

四种 strategy：

| Strategy | 含义 | 典型用途 |
|---|---|---|
| `PACK` | **优先**塞同节点；不够才溢出 | 资源紧凑、减少跨机通信 |
| `STRICT_PACK` | **必须**全部 bundle 在同节点；不够直接失败 | 必须同机的训练任务 |
| `SPREAD` | **优先**分散到不同节点 | 容错、不抢同节点资源 |
| `STRICT_SPREAD` | **必须**每个 bundle 在不同节点 | 强制每节点一个 worker |

verl 的 `RayResourcePool.get_placement_groups` 用 `PACK`（默认）或 `STRICT_PACK`（开 `bin_pack=True`）：

```python
# verl/single_controller/ray/base.py:553-555
strategy = "PACK"
if bin_pack:
    strategy = "STRICT_PACK"
```

### 3.4 Gang Scheduling：要么全成、要么全失败

`placement_group(...)` 是异步预约。`ray.get(pg.ready())` 才阻塞等就绪。如果集群凑不齐资源（比如只剩 12 张 GPU 但你要 16 张），它会一直等到：
- 集群扩容到足够，或
- 你 cancel 它

这避免了「16 个 actor 起来 12 个开始跑、剩下 4 个起不来卡死」的悲剧。

### 3.5 把 Actor 绑到具体 Bundle

```python
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# 把这个 actor 绑到 pg 的第 3 个 bundle
WorkerCls.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=3,        # ★ 指定 bundle 编号
        placement_group_capture_child_tasks=False,
    ),
    num_cpus=1,
    num_gpus=1,
).remote(...)
```

verl 的用法（`_create_worker`）：

```python
worker = ray_cls_with_init(
    placement_group=pg,
    placement_group_bundle_idx=local_rank,    # 0..7
    use_gpu=self.use_gpu,
    num_gpus=num_gpus,
    device_name=self.device_name,
)
```

`placement_group_bundle_idx=local_rank` 把 actor 精确钉到本 PG 的某个 bundle，从而钉到具体 GPU。

### 3.6 资源记账规则（容易踩坑）

actor 绑到 bundle 时：

- `num_gpus=1` → 占用该 bundle 的整张 GPU
- `num_gpus=0.5` → 占该 bundle 的半张 GPU，剩 0.5 留给同 bundle 的其他 actor
- `num_gpus=2` → 失败（bundle 只有 1 张 GPU，超量）

verl 的 `num_gpus = 1 / max_colocate_count`（默认 1/3）正是利用这一点：让一个 bundle 上能塞下多种 worker（actor + critic + reward 共置同卡）。

**注意**：`num_gpus` 是 Ray 的**逻辑账本**，不是真的限制 CUDA 算力/显存。物理上每个 actor 都能用整张卡。

### 3.7 `placement_group_capture_child_tasks`

这个开关决定：**绑到 PG 的 actor 内部再起的子任务，要不要也自动绑到同一 PG**？

- `True`：子任务继承父 actor 的 PG
- `False`：子任务用全局调度

verl 的 vLLM rollout 内部还会再起子任务，这个开关决定它们要不要也挤在同一张 GPU 的 bundle 上。

### 3.8 生命周期

- **默认 lifetime**：跟着创建它的 driver 走，driver 退出 → PG 回收
- **`lifetime="detached"`**：脱离 driver，需 `remove_placement_group(pg)` 显式回收。适合常驻服务

verl 默认 `detached=False`：训练结束 PG 自动释放。

### 3.9 与节点资源的隔离作用

绑到 PG 的 actor 申请的资源**只从 bundle 中扣减**，不动节点上其他资源。这正是 verl 想要的——训练任务的 16 张卡被 PG 锁死，其他 Ray 任务（driver、AgentLoopWorker、reward worker）只能用 bundle 没申领走的剩余资源（CPU），不会偷训练用的 GPU。

### 3.10 verl 的实际形态：每节点一个 PG

```python
# verl/single_controller/ray/base.py:143-160
process_on_nodes = [n_gpus_per_node] * nnodes  # 比如 [8, 8]

bundle = {"CPU": max_colocate_count, "GPU": 1}

# 每节点一个 PG，PG 内 bundle 数 = 节点的 worker 数
pg_scheme = [[bundle] * 8, [bundle] * 8]

# strategy="PACK" → 每个 PG 的 8 个 bundle 挤在同一台机器
pgs = [placement_group(b, "PACK") for b in pg_scheme]   # len(pgs) = 2
```

效果：

```
PG_0 (Node A, STRICT_PACK)        PG_1 (Node B, STRICT_PACK)
┌────────────────────────────┐   ┌────────────────────────────┐
│ bundle 0 ─→ rank 0 → GPU0  │   │ bundle 0 ─→ rank 8  → GPU0 │
│ bundle 1 ─→ rank 1 → GPU1  │   │ bundle 1 ─→ rank 9  → GPU1 │
│ ...                        │   │ ...                        │
│ bundle 7 ─→ rank 7 → GPU7  │   │ bundle 7 ─→ rank 15 → GPU7 │
└────────────────────────────┘   └────────────────────────────┘
```

`len(pgs) == nnodes`，每个 PG 内 bundle 数 == 每节点 GPU 数。

---

## 4. 资源调度：`num_cpus` / `num_gpus` 与共置

### 4.1 `num_cpus` 和 `num_gpus` 的真实含义

`@ray.remote(num_cpus=1, num_gpus=1)` 或 `Cls.options(num_cpus=1, num_gpus=1)`：

- 这是**逻辑资源记账**，告诉调度器"这个 actor 需要预留 1 CPU + 1 GPU 的额度"
- **不是真的限制进程能用多少 CPU 核**——OS 层面进程仍可多核
- **不是真的限制进程能用多少显存**——CUDA 整张卡都能访问
- **真正起作用的是调度层面**：节点剩余资源不够就不调度，避免 over-subscription

### 4.2 设小数允许共置

```python
@ray.remote(num_gpus=0.5)
class Worker: ...

w1 = Worker.remote()   # 占 GPU 0 的 0.5 份
w2 = Worker.remote()   # 也占 GPU 0 的 0.5 份（共置！）
```

w1 和 w2 是**两个独立进程**，但 Ray 把它们都绑在 GPU 0 上（`CUDA_VISIBLE_DEVICES=0`）。它们物理上都能用整张卡，靠应用层（如 sleep/wake）协调显存。

### 4.3 verl 的 `num_gpus = 1 / max_colocate_count`

```python
# verl/single_controller/ray/base.py:627
num_gpus = 1 / resource_pool.max_colocate_count   # 默认 1/3
```

含义：每个 worker 进程只声明 1/3 张 GPU，Ray 调度器才允许在同一张卡（同一 bundle）上塞下多种 WorkerGroup 的进程。

但 verl 主流的 `ActorRolloutRefWorker` 模式下，actor / ref / rollout 是**同一进程内的三个对象**（不需要共置），`1/k` 主要为 actor + critic + reward 这种"多种 WorkerGroup colocate" 留余量。

### 4.4 Ray 自动设 `CUDA_VISIBLE_DEVICES`

actor 进程被拉起时，Ray 会根据它绑到的 bundle 自动设 `CUDA_VISIBLE_DEVICES`：
- 进程内 `torch.cuda.device_count()` = 分配到的 GPU 数（通常 1）
- 进程内 `torch.cuda.current_device()` = 0
- 但宿主机视角下，它用的是物理 GPU `local_rank`

**这个机制让应用代码不需要关心"我在哪张卡"**——直接 `tensor.cuda()` / `cuda:0` 就行。

### 4.5 自定义资源类型

Ray 支持自定义资源 key：

```bash
# NPU 集群
ray start --head --resources='{"NPU": 8}'
```

```python
@ray.remote(resources={"NPU": 1})
class NPUWorker: ...
```

verl 在华为 NPU 上就是这么用的（见 `docs/ascend_tutorial/...`）。

---

## 5. ActorHandle 与 Ray RPC 通信

### 5.1 ActorHandle 是什么

```python
c = Counter.remote()       # 这一行返回的不是 Counter 实例！
print(type(c))             # ray.actor.ActorHandle
```

`ActorHandle` 是 driver 端的**远程引用**：
- 它本身只是一个轻量的句柄对象，存在 driver 进程内存里
- 真正的 `Counter` 实例在另一个进程（GPU worker 进程）
- 调 `c.inc.remote()` 时通过 Ray RPC 把请求送过去

### 5.2 RPC 数据流

| 数据大小 | 传输方式 |
|---|---|
| 小（几 KB） | inline 嵌入 RPC 消息 |
| 中（>~100KB） | 自动放进 Object Store（plasma 共享内存） |
| 跨机大对象 | plasma 之间走零拷贝（基于 DMA） |
| GPU 张量 | **不走 Ray**，应用层用 NCCL 直接传 |

verl 的张量数据传输走两条路：
- **控制面 / 小元数据**（`DataProto.meta_info`、`KVBatchMeta`）→ Ray RPC
- **大张量**（梯度、激活、权重同步）→ NCCL

### 5.3 同步 vs 异步

```python
# 异步：立即返回 ObjectRef
ref = c.inc.remote()
# ... 做别的事 ...
result = ray.get(ref)         # 此时阻塞

# 也可以批量等
refs = [worker.method.remote() for worker in workers]
results = ray.get(refs)       # 等所有 ref 就绪
```

verl 的 `func_generator` 里就是这种模式：

```python
output = execute_fn(method_name, *args, **kwargs)  # 16 个 ObjectRef
if blocking:
    output = ray.get(output)                       # 阻塞等所有结果
```

### 5.4 同一 actor 内的并发

默认 actor 里的方法**串行执行**（同一个 actor 不会同时跑两个方法）。要并发：

```python
@ray.remote(max_concurrency=10)
class AsyncActor:
    async def fetch(self): ...
```

verl 的 `AgentLoopWorker` 就是这种模式：一个 actor 进程内用 asyncio 跑几百条样本的并发 rollout。

### 5.5 actor 死活管理

- actor 默认 `max_restarts=0`，挂了就挂了
- 设 `max_restarts=3` 可让 Ray 自动重启
- `max_task_retries` 控制单个方法调用失败后的重试次数

verl 训练任务一般不开 restart，因为模型状态丢失后重启意义不大，宁可让整个训练 fail 早发现。

---

## 6. `runtime_env`：环境变量、依赖、镜像隔离

### 6.1 用途

Runtime Env 让你为 actor / task 单独指定运行环境：
- 环境变量（最常用）
- pip / conda 依赖
- 工作目录、Python 路径
- Docker 镜像

### 6.2 verl 的核心用法：注入环境变量

```python
# verl/single_controller/ray/base.py:667
ray_cls_with_init.update_options({
    "runtime_env": {"env_vars": env_vars},
    "name": name,
})
```

注入的内容：

```python
env_vars = {
    "WORLD_SIZE": "16",
    "RANK": "5",
    "MASTER_ADDR": "192.168.1.10",
    "MASTER_PORT": "43891",
    "RAY_LOCAL_WORLD_SIZE": "8",
    ...
}
```

**这就是 Ray 替代 torchrun 的关键**——actor 启动前 Ray 把这些 env var 写进进程环境，业务代码 `os.environ["RANK"]` 就能读到。

### 6.3 全局 runtime_env：通过 `ray.init` 或 `ray job submit`

```bash
ray job submit \
    --address="http://NodeA:8265" \
    --runtime-env-json='{"working_dir": ".", "pip": ["transformers==4.45"]}' \
    -- python -m verl.trainer.main_ppo ...
```

或在 yaml 里（verl 用的 `verl/trainer/runtime_env.yaml`）：

```yaml
env_vars:
  TOKENIZERS_PARALLELISM: "true"
  NCCL_DEBUG: "WARN"
  VLLM_LOGGING_LEVEL: "WARN"
```

### 6.4 优先级

actor / task 级 > Job 级（`ray job submit`）> 集群级（`ray start`）

### 6.5 注意

- `env_vars` 是**字符串字典**，所有值要 `str(...)`
- `runtime_env` 内的 pip 依赖会在每个 worker 节点本地安装一份（首次运行较慢）
- 大的 `working_dir` 会被压缩上传到集群所有节点，注意 `.gitignore` 避免传无关文件

---

## 7. Ray 集群：head / worker / dashboard / job

### 7.1 角色划分

| 角色 | 命令 | 职责 |
|---|---|---|
| **Head 节点** | `ray start --head ...` | 跑 GCS（Global Control Service）、Dashboard，集群入口 |
| **Worker 节点** | `ray start --address=<head>:6379 ...` | 提供资源给 actor / task |
| **Dashboard** | head 节点 8265 端口 | Web UI，看 actor / task / 资源 |
| **Driver** | `ray.init()` 或在某 actor 内调 Ray API | 提交作业的入口 |

### 7.2 启动两节点集群（最小例子）

```bash
# Head (192.168.1.10)
ray start --head \
    --node-ip-address=192.168.1.10 \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8

# Worker (192.168.1.11)
ray start --address='192.168.1.10:6379' --num-gpus=8

# 验证（任一节点）
ray status
# 期望：2 nodes, 16 GPUs
```

### 7.3 Dashboard

Web UI 是 Ray 调试体验最大的提升点之一：
- **Cluster** 视图：所有节点资源占用
- **Jobs** 视图：每个 job 的状态、日志、stack trace
- **Actors** 视图：每个 actor 的调用历史、内存
- **Logs** 视图：log 实时滚动

打开方式：`http://<head_ip>:8265`

### 7.4 Ray Job vs 直接 driver 模式

两种提交方式：

**(1) 直接在 head 节点跑 driver**

```bash
python -m verl.trainer.main_ppo trainer.nnodes=2 ...
```

driver 进程就在你的 shell 里，挂了就挂了。

**(2) `ray job submit`（推荐）**

```bash
ray job submit \
    --address="http://192.168.1.10:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- python -m verl.trainer.main_ppo ...
```

driver 跑在集群里某个节点上，shell 关掉也不影响。还能在 dashboard 看状态。生产环境强推这个。

管理命令：

```bash
ray job list                              # 列所有 job
ray job logs <id> --follow                # 实时看日志
ray job status <id>                       # 看状态
ray job stop <id>                         # 停止
```

### 7.5 资源声明

`ray start` 时可以声明资源：

```bash
ray start --head --num-cpus=64 --num-gpus=8 \
    --resources='{"NPU": 0, "TPU": 0}'
```

注意 `--num-gpus` 报多了 → Ray 误以为有 GPU；报少了 → 资源浪费。**务必和实际硬件一致**。

---

## 8. 多机训练：Ray 替代 torchrun 的机制

### 8.1 为什么 Ray 能替代 torchrun

`torchrun` 解决的问题：
- 在每节点 fork N 个进程
- 给每个进程注入 `RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT`
- 进程退出码统一回收

Ray 用 actor 也能做到这些：
- placement group 决定每个 actor 拉到哪台机器哪张卡
- `runtime_env.env_vars` 注入分布式信息
- actor 生命周期由 raylet 管控
- **而且**：actor 是有状态长驻的，能持续接收 RPC（torchrun 起来的进程一般是 SPMD 跑完就退）

### 8.2 verl 启动流的精确对照

| 步骤 | torchrun | Ray + verl |
|---|---|---|
| 1. 拉起进程 | 在每节点 fork | placement group + actor |
| 2. 选 master | `--master_addr` 或 rdvz | `_get_master_addr_port` 在 PG_0 bundle 0 上动态选 |
| 3. 注入 env | torchrun 自动 | `runtime_env.env_vars` |
| 4. 进程绑卡 | `LOCAL_RANK` + `CUDA_VISIBLE_DEVICES` | placement_group_bundle_idx + Ray 自动 setenv |
| 5. 调 init_process_group | 业务代码 | 业务代码（`TrainingWorker.__init__`） |
| 6. NCCL 通信 | 走 PyTorch | 走 PyTorch（一样，因为底层就是 NCCL） |

### 8.3 拓扑对应关系

以 2 节点 × 8 GPU 为例：

```
ray start --num-gpus=8        ray start --address=...   ← 用户命令
   Node A (192.168.1.10)         Node B (192.168.1.11)
        │                              │
        └────────── PG_0 ──────────────┴────── PG_1
        (STRICT_PACK 锁同机)         (STRICT_PACK 锁同机)
        │                              │
        ▼                              ▼
  bundle 0..7                      bundle 0..7
  各占 1 张 GPU                    各占 1 张 GPU
        │                              │
        ▼                              ▼
  actor 0..7                      actor 8..15
  RANK=0..7                       RANK=8..15
  LOCAL_RANK=0..7                 LOCAL_RANK=0..7
        │                              │
        ▼                              ▼
  CUDA_VISIBLE_DEVICES=0..7       CUDA_VISIBLE_DEVICES=0..7
        │                              │
        └─────── PyTorch NCCL PG ──────┘
                 在 MASTER_ADDR=192.168.1.10:43891 rendezvous
                 WORLD_SIZE=16
```

---

## 9. 调试与可观测性

### 9.1 Dashboard

最直观的工具：`http://<head>:8265`，看 actor / task / 资源 / 日志。

### 9.2 actor 日志位置

每个 actor 的 stdout / stderr 自动收集：
- 集群级：`/tmp/ray/session_latest/logs/`
- driver 日志：`job-driver-raysubmit_<id>.log`
- actor 日志：`worker-<actor_id>-<job_id>-<pid>.{out,err}`

```bash
# 用 ray job submit 时
ray job logs <submission_id> --follow
```

### 9.3 Ray Distributed Debugger（VSCode 扩展）

Ray 2.39+ 支持 VSCode 远程断点调试：

```bash
export RAY_DEBUG_POST_MORTEM=1
```

在代码里 `breakpoint()` → 提交 job → VSCode 扩展自动检测到并 attach。

详见 `docs/start/multinode.rst:419-491`。

### 9.4 常见排查命令

```bash
ray status                          # 集群整体状况
ray nodes | head -50                # 列出所有节点
ray list actors --filter="state=ALIVE"
ray list tasks --limit=20
ray summary actors                  # actor 汇总
ray timeline                        # 时间线导出（chrome://tracing 看）
```

### 9.5 调试 NCCL 网络问题

```bash
export NCCL_DEBUG=INFO              # 排错时开 TRACE
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=eth0      # 没 IB 时绑网卡
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### 9.6 actor 启动失败的常见原因

| 症状 | 原因 |
|---|---|
| 卡在 `placement_group(...).ready()` | 资源不够，`ray status` 看 |
| actor 起来后立刻 die | `import` 阶段挂了，看 actor 日志 |
| `CUDA_VISIBLE_DEVICES` 不对 | `runtime_env.env_vars` 冲突，Ray 自动 setenv 被覆盖 |
| 跨机 RPC 慢 | gRPC 跨机走 TCP，大对象走 plasma；检查网卡 |

---

## 10. Ray 在 verl 中的具体使用速查

| Ray 特性 | verl 用在哪 |
|---|---|
| `@ray.remote` | 包装 `TaskRunner`、`ActorRolloutRefWorker`、`AgentLoopWorker`、`RewardLoopWorker` 等所有 worker 类 |
| `ray.init` | `verl/trainer/main_ppo.py:run_ppo`，driver 启动时一次性初始化 |
| Placement Group | `verl/single_controller/ray/base.py:RayResourcePool`，每节点一个 PG |
| `placement_group_bundle_index` | `_create_worker` 把 actor 绑到具体 GPU |
| `runtime_env.env_vars` | `_create_worker` 注入 `RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT` |
| `ActorHandle` | `RayWorkerGroup._workers` 里持有的 16 个 handle |
| `ray.get` | `func_generator` 里阻塞等所有 worker 返回结果 |
| `ObjectRef` | DataProto 在 driver↔worker 间传递 |
| `ray.remote(num_cpus=1)` | `TaskRunner`（driver actor），不抢 GPU |
| 自定义资源（NPU） | `ray start --resources='{"NPU": 8}'`（华为 NPU 路径） |
| `lifetime="detached"` | RolloutReplica 等长驻 actor 用 |
| `NodeAffinitySchedulingStrategy` | `RewardLoopWorker` / `AgentLoopWorker` 按节点亲和分布 |
| `Async actor`（`max_concurrency`） | `AgentLoopWorker` 内部 asyncio 并发跑多条 rollout |

---

## 11. 常见误解与陷阱

### 11.1 "`@ray.remote` 包装的类用 `Cls(...)` 创建实例"

❌ 错。`Cls(...)` 会报错（Ray actor 不能用普通构造函数）。
✅ 正确：`Cls.remote(...)` 创建 actor 进程；返回 `ActorHandle`。

### 11.2 "Ray actor 是线程，不是进程"

❌ 错。actor 是**独立 OS 进程**，每个有自己的 GIL、CUDA context、module-level state。

### 11.3 "`num_gpus=0.5` 真的限制只能用半张 GPU"

❌ 错。这是 Ray 的逻辑账本，不限制 CUDA。物理上整张卡都能用。

### 11.4 "`@ray.remote` 装饰函数和装饰类是一样的"

❌ 错：
- 装饰函数 → Task：进程池复用，无状态
- 装饰类 → Actor：独享进程，有状态

### 11.5 "ray.get 不阻塞"

❌ 错。`*.remote()` 不阻塞（返回 ObjectRef），`ray.get(ref)` 才阻塞等结果。

### 11.6 "PG 的 strategy 决定 actor 落点"

❌ 不完全。strategy 决定 PG 内 bundle 之间的关系；具体哪台节点是 Ray 调度器选的（受集群当时空闲资源影响）。verl 用 `sort_placement_group_by_node_ip` 做事后稳定排序。

### 11.7 "actor 死了能自动重启数据不丢"

❌ 默认不行。`max_restarts` 默认 0，要显式开。即使开了，actor 内部状态（模型权重）会丢，需要从 checkpoint 重建。

### 11.8 "Ray 自动跨机均衡 GPU 任务"

❌ 不会。Ray 调度是 best-effort，不保证均衡。要严格控制必须用 placement group。

### 11.9 "`runtime_env.env_vars` 等同于 export 环境变量"

✅ 对，但只对 actor 进程生效。父进程（driver）的环境变量不会自动传给子 actor，必须显式声明。

### 11.10 "把大张量直接传 actor 方法"

会触发 plasma store 写入，跨机会走零拷贝但仍有序列化开销。verl 在大张量路径上**不走 Ray**，而是用 NCCL 直接 GPU→GPU 传。

---

## 12. 学习路径与延伸阅读

### 12.1 推荐学习顺序（结合 verl）

1. **Ray 官方教程**：https://docs.ray.io/en/latest/ray-core/walkthrough.html （30 分钟跑通）
2. **本笔记 §2-§5**：把 Task / Actor / Object / Placement Group 概念落地
3. **verl `single_controller/ray/base.py`**：看 `RayResourcePool.get_placement_groups` 和 `RayWorkerGroup._init_with_resource_pool`
4. **跑一个本地 verl 例子**：单机就能跑（`trainer.nnodes=1`），观察 dashboard 上的 actor 拓扑
5. **verl `_create_worker`**：理解 `placement_group_bundle_idx + runtime_env.env_vars` 的组合
6. **多机起一个集群**：跟着 §7.2 起 2 节点，把 `ray status` 和 dashboard 玩熟
7. **进阶：自定义资源、Async actor、detached actor**

### 12.2 官方资源

- **核心文档**：https://docs.ray.io/en/latest/ray-core/
- **Placement Group**：https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
- **Runtime Env**：https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
- **多机部署**：https://docs.ray.io/en/latest/cluster/getting-started.html
- **Slurm 集成**：https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
- **Distributed Debugger**：https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html

### 12.3 可深入的 Ray 主题

| 主题 | 价值 |
|---|---|
| **Ray Serve** | LLM 推理服务化，vLLM 在用 |
| **Ray Train** | 分布式训练高层 API（用得少，verl 直接用 Ray Core） |
| **Ray Tune** | 超参搜索 |
| **Ray Data** | 分布式数据处理（数据集预处理可用） |
| **Async Actor + asyncio** | verl 的 `AgentLoopWorker` 核心模式 |
| **Custom Scheduling Strategy** | NodeAffinity / SpreadStrategy 等 |
| **Object Spilling** | 大对象自动落盘机制，OOM 时关键 |
| **Fault Tolerance** | actor 重启、checkpoint、lineage reconstruction |

### 12.4 一句话总结

> Ray 在 verl 里扮演的是**「分布式调度层」+「多语言 RPC 层」+「资源管家」**三重角色：
>
> - **调度层**：placement group + bundle 把 actor 进程精确摆到每张 GPU
> - **RPC 层**：actor handle + ObjectRef 让 driver 调用远程方法像调本地一样自然
> - **资源管家**：runtime_env 注入环境变量，replace `torchrun`；逻辑资源记账（`num_gpus=分数`）支持灵活共置
>
> 把这三件事吃透，你就理解了为什么 verl（以及几乎所有 LLM RL 框架）都选 Ray。
