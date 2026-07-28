"""方式 B：使用 torch.multiprocessing.spawn 在单机模拟多进程（多 rank）测试 DCP。

用途：
    - 演示 PyTorch Distributed Checkpoint (torch.distributed.checkpoint, 简称 DCP)
      的多 rank 并行保存 / 加载。
    - 每个 rank 写出自己的 ``__<rank>_<idx>.distcp`` 分片文件；
      coordinator rank（默认 rank 0）负责写出唯一的全局 ``.metadata`` 索引文件。
    - 纯 CPU（gloo 后端）即可运行，无需 GPU、无需多机。

运行方式：
    python tests/scripts/test_dcp_mp_spawn.py
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

CKPT_DIR = "./dcp_multi"
MASTER_ADDR = "localhost"
MASTER_PORT = "29500"


def worker(rank: int, world_size: int):
    # torch.multiprocessing.spawn 不会自动设置这些环境变量，需要手动指定
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    # 纯 CPU 用 gloo 后端；如需 GPU 改成 "nccl"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 每个 rank 造一份不同的数据，模拟分片场景
    state_dict = {"data": torch.ones(3) * rank}

    # 各 rank 并行写入自己的分片到同一目录
    dcp.save(state_dict, checkpoint_id=CKPT_DIR)

    if rank == 0:
        # 只有 coordinator rank（默认 0）负责写最终的 .metadata
        print(f"[rank {rank}] files in {CKPT_DIR}:", sorted(os.listdir(CKPT_DIR)))

    dist.barrier()

    # ---- 演示加载：先构造同结构的「空壳」模板，DCP 会 in-place 填充 ----
    load_sd = {"data": torch.zeros(3)}
    dcp.load(load_sd, checkpoint_id=CKPT_DIR)
    expected = torch.ones(3) * rank
    ok = torch.allclose(load_sd["data"], expected)
    print(f"[rank {rank}] load ok: {ok}, data={load_sd['data'].tolist()}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4

    # 清理上一次的产物，保证每次从干净目录开始
    if os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)

    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

    print("\nDone. 你会看到 4 个 __<rank>_0.distcp 分片 + 1 个 .metadata 全局索引。")

