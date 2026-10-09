# 分布式初始化与 Worker 环境变量学习笔记（结合 verl 实践）

> 本文档沉淀两块内容：
> 1. `torch.distributed.init_process_group` 的作用、参数与常见误区；
> 2. verl 中 Ray 如何拉起 Worker、如何给每个 Worker 注入 `RANK` / `WORLD_SIZE` 等环境变量，以及"没有 `@ray.remote` 装饰器的 Worker 类是怎么被 Ray 使用的"。
>
> 相关文档：Ray 通用知识见 [`ray_notes.md`](./ray_notes.md)，verl 业务架构见 [`verl_architecture_notes.md`](./verl_architecture_notes.md)。
>
> 3. 多进程 GPU 绑定的常见报错（`Duplicate GPU detected`）与修正。
>
> 后续可继续补充：device_mesh 构建、Ulysses SP、dispatch/collect 机制等。

---

## 目录

- [1. `torch.distributed.init_process_group` 是什么](#1-torchdistributedinit_process_group-是什么)
- [2. 参数逐个解读（结合 verl 代码）](#2-参数逐个解读结合-verl-代码)
- [3. 关键机制：会合（rendezvous）与阻塞](#3-关键机制会合rendezvous与阻塞)
- [4. 常见误区澄清](#4-常见误区澄清)
- [5. verl 里谁来设置 `RANK` / `WORLD_SIZE`？](#5-verl-里谁来设置-rank--world_size)
- [6. 为什么 `ActorRolloutRefWorker` 没有 `@ray.remote` 也能被 Ray 使用？](#6-为什么-actorrolloutrefworker-没有-rayremote-也能被-ray-使用)
- [7. 端到端串讲：从启动到 init_process_group](#7-端到端串讲从启动到-init_process_group)
- [8. 最小可运行示例（原生 torchrun）](#8-最小可运行示例原生-torchrun)
- [9. 多进程 GPU 绑定：`Duplicate GPU detected` 报错](#9-多进程-gpu-绑定duplicate-gpu-detected-报错)
- [10. 速查表](#10-速查表)

---

## 1. `torch.distributed.init_process_group` 是什么

它是 PyTorch 分布式训练的**初始化入口**：把当前进程加入一个"进程组（process group）"，建立进程之间的通信通道。只有初始化之后，后续的集合通信（`all_reduce`、`all_gather`、`broadcast`、`reduce_scatter` 等）才能工作。

多卡/多机训练的本质就是**多个进程各管一块 GPU**，它们需要频繁交换梯度、参数、数据。`init_process_group` 就相当于给这些进程"拉了一个群"，让它们能彼此通信。

verl 中的调用位置（`verl/workers/fsdp_workers.py`）：

```python
if not torch.distributed.is_initialized():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.distributed.init_process_group(
        backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
```

注意 `rank` / `world_size` 是**先从环境变量读出来**，再传给`init_process_group` 的——这点在第 4、5 节展开。

---

## 2. 参数逐个解读（结合 verl 代码）

| 参数 | 含义 | verl 里的取值 |
|------|------|--------------|
| `backend` | 通信后端 | `cpu:gloo,cuda:nccl`（混合写法）|
| `rank` | **当前进程**在进程组里的编号，从 0 开始 | 从环境变量 `RANK` 读取 |
| `world_size` | 进程组里**进程总数** | 从环境变量 `WORLD_SIZE` 读取 |
| `timeout` | 集合通信操作的超时时间 | 默认 600 秒（可由 `nccl_timeout` 配置）|
| `init_method` | 进程之间如何"找到彼此"完成初始握手 | 从 `DIST_INIT_METHOD` 读取，取不到为 `None`（默认 `env://`）|

### backend 详解

- **`nccl`**：NVIDIA 的 GPU 间通信库，GPU 训练首选，速度最快。
- **`gloo`**：CPU 通信后端，也支持一些 nccl 不方便做的操作。
- **`cpu:gloo,cuda:nccl`**：较新版本 PyTorch 支持的混合写法，含义是"CPU 上的张量走 gloo，GPU 上的张量走 nccl"。verl 里用 `get_device_name()` / `get_nccl_backend()` 动态拼接，以兼容 CUDA / NPU 等不同设备。

### init_method 详解

常见形式：

- `tcp://主节点IP:端口`：所有进程连到master 地址完成握手；
- `env://`：从环境变量 `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` 读取（PyTorch 默认）。

verl 里通常为 `None`，即走 `env://`，而这些环境变量由 Ray 侧设置（见第 5 节）。

---

## 3. 关键机制：会合（rendezvous）与阻塞

`init_process_group` 是一个**阻塞式的集合操作**：

1. 每个进程执行到这里时，会通过 `init_method` 指定的地址去"会合"（rendezvous）。
2. 先到达的进程会**阻塞等待**，直到 `world_size` 个进程全部到齐（或超时报错）。
3. 所有进程握手协商，最终建立起**同一个、全局唯一、所有进程共享**的进程组。

也就是说：**不是每个进程各自独立地建一个组**，而是所有进程共同建立同一个组。

---

## 4. 常见误区澄清

>❌ 误区一："每个进程执行到这里，就各自独立地构建自己的通信组。"

✅ 更准确：所有进程在这里**会合、阻塞等待、共同建立同一个全局进程组**。任何一个进程缺席都会导致其他进程一直等待直到超时。

> ❌ 误区二："`init_process_group` 会为每个进程分配对应的 rank 编号。"

✅ 更准确：**rank 不是这个函数分配的**。rank / world_size 必须在调用前就准备好（通过参数或环境变量传入）。这个函数只是"按你给定的 rank 把你登记进组"。真正决定"谁是rank 0、谁是 rank 1"的是**启动器**（torchrun / Ray）。

> ✅ 说对的部分："多进程执行同一份代码"（即 SPMD 模式）——完全正确。每个进程跑同一份代码，靠各自不同的 `RANK` 环境变量区分身份。

---

## 5. verl 里谁来设置 `RANK` / `WORLD_SIZE`？

在原生 PyTorch 里，这些变量由 `torchrun` 设置。**verl 用 Ray 取代了 torchrun 的角色**，由 `RayWorkerGroup` 在创建每个 Worker 时，通过 Ray 的 `runtime_env.env_vars` 注入。

核心代码在 `verl/single_controller/ray/base.py` 的 `RayWorkerGroup._create_worker`：

```python
# verl/single_controller/ray/base.py  _create_worker(...)
env_vars = {
    "WORLD_SIZE": str(world_size),
    "RANK": str(rank),
    "WG_PREFIX": self.name_prefix,
    "WG_BACKEND": "ray",
    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
    "MASTER_ADDR": self._master_addr,
    "MASTER_PORT": self._master_port,
}
...
ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})

# 真正创建 actor
worker = ray_cls_with_init(
    placement_group=pg,
    placement_group_bundle_idx=local_rank,
    use_gpu=self.use_gpu,
    num_gpus=num_gpus,
    device_name=self.device_name,
)
```

### rank 是怎么算出来的

在 `_init_with_resource_pool` 里，遍历 placement group 的每个 bundle，用一个全局自增计数器分配 rank：

```python
rank = -1
local_world_size = resource_pool.store[0]
for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
    if pg_idx == 0:
        # 第 0 个 bundle 所在节点作为 master，拿到 addr/port
        self._get_master_addr_port(pg, bundle_index=0, ...)
    for local_rank in range(local_world_size):
        rank += 1                       # 全局 rank 自增
        self._create_worker(rank=rank,...)
```

- **全局 `rank`**：跨节点从 0 连续自增，作为 `RANK`。
- **`local_rank`**：节点内的编号，用于绑定 GPU。
- **`MASTER_ADDR` / `MASTER_PORT`**：取第 0 个 bundle 所在节点的 IP 和一个空闲端口（`get_master_addr_port` 这个 `@ray.remote` 函数负责探测），保证所有 Worker 会合到同一个 master。

所以流程是：

```
RayWorkerGroup遍历所有 bundle
        │
        ├─ 给 bundle0 的 Worker 注入 RANK=0, WORLD_SIZE=N, MASTER_ADDR/PORT=...
        ├─ 给 bundle1 的 Worker 注入 RANK=1, WORLD_SIZE=N, MASTER_ADDR/PORT=... (相同)
        └─ ...
        │
每个 Worker 进程启动后，__init__ 读取自己的 RANK/WORLD_SIZE 写进 os.environ
        │
执行到 init_process_group → 走 env:// → 所有 Worker 会合到同一 MASTER → 建组
```

### Worker 侧如何消费这些环境变量

`verl/single_controller/base/worker.py` 的 `Worker.__init__` 会读取这些变量：

```python
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]
...
self._configure_with_store(store=store)  # 再写回 os.environ，供后续 init_process_group 用
```

`env_keys()` 列出了这套约定的环境变量：`WORLD_SIZE`、`RANK`、`LOCAL_WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT`、`*_VISIBLE_DEVICES`。

---

## 6. 为什么 `ActorRolloutRefWorker` 没有 `@ray.remote` 也能被 Ray 使用？

这是一个很好的观察。类定义确实是普通类：

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    ...
```

**关键点：`ray.remote` 既可以当装饰器用，也可以当普通函数调用。** verl 采用的是**后者**——在运行时用`ray.remote(cls)` 把普通类动态包装成 Ray Actor 类。

证据在 `verl/trainer/main_ppo.py`：

```python
# main_ppo.py
self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)   # actor_rollout_cls 就是 ActorRolloutRefWorker
self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
```

`ray.remote(ActorRolloutRefWorker)` 返回的是一个"Ray Actor 类"（等价于在类上面写 `@ray.remote`），之后就能 `.options(...).remote(...)` 来创建远程 actor 实例。

### 为什么不直接写 `@ray.remote` 装饰器？

因为 verl 需要**在运行时动态决定**：

1. **资源选项**：`num_gpus`、`scheduling_strategy`、`runtime_env`（含前面的 `env_vars`）等，都要根据 placement group、配置动态设置。用函数式 `ray.remote(cls)` + `.options(...)` 才能灵活注入；写死装饰器就固定了。
2. **同一个类多种用途**：`ActorRolloutRefWorker` 可以作为 actor、rollout、ref 或混合体，由 `role` 决定，装饰器无法表达这种运行时分支。
3. **共置 / Fused Worker**：`create_colocated_worker_cls_fused` 里会**动态构造一个新类**再`ray.remote(...)`：

   ```python
   # base.py
   remote_cls = ray.remote(raw_colocated_worker_cls)
   cia = RayClassWithInitArgs(cls=remote_cls)
   ```

   把多个业务 Worker 融进一个 Ray Actor（省进程/显存），这必须运行时构造，装饰器做不到。

### 完整包装链路

```
ActorRolloutRefWorker (普通类)
        │  ray.remote(cls)                 ← main_ppo.py，等价于加@ray.remote
        ▼
Ray Actor 类
        │  RayClassWithInitArgs(cls, *args, **kwargs)   ← 记住构造参数
        ▼
RayClassWithInitArgs实例（可调用）
        │  __call__(pg, bundle_idx, ...) 内部：
        │    self.cls.options(**options).remote(*args, **kwargs)
        ▼
Ray Actor 句柄（真正的远程进程实例）
        │  RayWorkerGroup 收集所有句柄，统一调度
        ▼
远程进程里执行 ActorRolloutRefWorker.__init__ → init_process_group
```

其中 `RayClassWithInitArgs.__call__`（`base.py`）负责把资源 options拼好并真正 `remote()`：

```python
options = {"scheduling_strategy": PlacementGroupSchedulingStrategy(...)}
options.update(self._options)          # 包含 runtime_env.env_vars
if use_gpu and device_name == "cuda":
    options["num_gpus"] = num_gpus
return self.cls.options(**options).remote(*self.args, **self.kwargs)
```

**一句话总结**：`ActorRolloutRefWorker` 不写 `@ray.remote`，是为了把"何时、以什么资源、注入哪些环境变量、以什么身份"变成运行时可编程的，由 `ray.remote(cls)` + `RayClassWithInitArgs` + `RayWorkerGroup` 这条链路动态完成。

---

## 7. 端到端串讲：从启动到 init_process_group

1. **入口** `main_ppo.py`：`ray.remote(ActorRolloutRefWorker)` 把普通类变成 Ray Actor 类，登记进 `role_worker_mapping`。
2. **资源** `ResourcePoolManager.create_resource_pool()`：按配置申请 placement group（每个 bundle ≈ 一块 GPU）。
3. **建组** `RayWorkerGroup._init_with_resource_pool()`：遍历 bundle，给每个 Worker 分配全局 `rank`、统一的 `MASTER_ADDR/PORT`，通过 `runtime_env.env_vars` 注入。
4. **创建** `RayClassWithInitArgs.__call__()`：`cls.options(...).remote(...)` 真正拉起远程进程。
5. **进程内** `Worker.__init__()`：从 `os.environ` 读取 `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` 并写回环境。
6. **通信初始化** `ActorRolloutRefWorker.__init__` → `torch.distributed.init_process_group(...)`：所有 Worker 走 `env://` 会合到同一 master，建立全局进程组。
7. 之后即可进行 FSDP 分片、集合通信等分布式训练。

---

## 8. 最小可运行示例（原生 torchrun）

用于理解"多进程 + rank + init_process_group"的最小闭环（不涉及 Ray）：

```python
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])               # 启动器设好，每个进程不同
world_size = int(os.environ["WORLD_SIZE"])   # 每个进程相同
local_rank = int(os.environ["LOCAL_RANK"])   # torchrun 自动设置，单机即 0..N-1

# 关键：把当前进程绑定到对应的 GPU（否则多进程会挤在 cuda:0，触发 Duplicate GPU 报错，见第 9 节）
torch.cuda.set_device(local_rank)

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
# 到这里，所有进程握手完成，进程组建好

device = torch.device(f"cuda:{local_rank}")   # 用local_rank 指定设备，而不是裸 .cuda()
tensor = torch.ones(1, device=device) * rank
dist.all_reduce(tensor)   # 结果 = 0+1+...+(world_size-1)
print(f"rank {rank} (local_rank {local_rank}): {tensor.item()}")

dist.destroy_process_group()
```

启动：`torchrun --nproc_per_node=4 script.py` → 拉起 4 个进程，rank 分别 0/1/2/3。
verl 中把 `torchrun` 这一角色换成了 Ray。

---

## 9. 多进程 GPU 绑定：`Duplicate GPU detected` 报错

### 报错现象

```
torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.cpp:77, invalid usage
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
Last error:
Duplicate GPU detected : rank 3 and rank 0 both on CUDA device 3000
```

### 根因

`torchrun` 只负责设置 `RANK` / `WORLD_SIZE` / `LOCAL_RANK` 等环境变量，**它不会自动把每个进程绑定到不同的 GPU**。如果直接用裸 `.cuda()`（不指定设备），默认都落在 `cuda:0`，于是 N 个进程全挤在同一块 GPU 上。而 NCCL 要求**一个 rank 独占一块 GPU**，因此报 `Duplicate GPU`。

修正办法就是第 8 节示例里的两行：`torch.cuda.set_device(local_rank)` + `torch.device(f"cuda:{local_rank}")`。

| 方案 | 做法 | 说明 |
|------|------|------|
| **推荐：`set_device(local_rank)`** | 代码里绑定设备 | 进程仍能看到所有 GPU，靠 local_rank 各自选一块。最通用，torchrun 官方推荐 |
| **`CUDA_VISIBLE_DEVICES`** | 让每个进程只看到一块卡 | 需在启动前给每个进程设不同值，`torchrun` 不会自动做，手动管理较麻烦 |

### verl 是怎么处理的

verl 在 `Worker._setup_env_cuda_visible_devices`（`verl/single_controller/base/worker.py`）里走的是 **`CUDA_VISIBLE_DEVICES` 方案**：

```python
if is_ray_noset_visible_devices:
    device_name = "NPU" if is_npu_available else "GPU"
    local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
    os.environ["LOCAL_RANK"] = local_rank
    get_torch_device().set_device(int(local_rank))
```

- Ray 默认会给每个 actor 自动设置 `CUDA_VISIBLE_DEVICES`（让它只看到分给它的那块卡）；
- 只有当关闭了该自动行为（`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`）时，verl 才手动 `set_device(local_rank)` 兜底。

两种方案verl 都覆盖了。

---

## 10. 速查表

| 问题 | 答案 |
|------|------|
| `init_process_group` 的作用 | 初始化进程组，建立进程间通信通道，是集合通信的前提 |
| rank 是谁分配的 | **不是**该函数分配；由启动器（torchrun / Ray）通过环境变量提前设好|
| verl 里谁设置 RANK/WORLD_SIZE | `RayWorkerGroup._create_worker` 通过 `runtime_env.env_vars` 注入 |
| MASTER_ADDR/PORT 从哪来 | 第 0 个 bundle 所在节点 IP + 探测到的空闲端口（`get_master_addr_port`）|
| Worker 侧在哪读取这些变量 | `Worker.__init__`（`worker.py`），再由 `_configure_with_store` 写回环境 |
| 为什么 Worker 类没 `@ray.remote` | 用 `ray.remote(cls)` 函数式包装，以便运行时动态注入资源/环境/身份，并支持 Fused Worker |
| `init_process_group` 会阻塞吗 | 会，等 `world_size` 个进程全部会合或超时 |
| backend `cpu:gloo,cuda:nccl` 含义 | CPU 张量走 gloo，GPU 张量走 nccl |
| `Duplicate GPU detected` 怎么解决 | 裸 `.cuda()` 导致多进程挤在同一卡；用 `torch.cuda.set_device(local_rank)` 绑定各自 GPU |
