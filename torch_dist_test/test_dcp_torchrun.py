"""方式 C：使用 torchrun 启动多进程测试 DCP（最接近真实分布式训练）。

用途：
    - 与方式 B 相同，演示 PyTorch Distributed Checkpoint (DCP) 的多 rank 保存 / 加载，
      但 rank / world_size / master 地址等由 torchrun 通过环境变量自动注入，
      脚本本身无需手动设置。
    - 额外演示 DCP 的杀手锏 —— 重分片（resharding）：
      保存时的 world_size 与加载时不同也能正确读回。
    - 纯 CPU（gloo 后端）即可运行，无需 GPU。

运行方式：
    # 4 个进程并行保存 + 加载
    torchrun --nproc_per_node=4 tests/scripts/test_dcp_torchrun.py

    # 观察重分片：先用 4 卡保存，再用 2 卡加载同一目录
    torchrun --nproc_per_node=4 tests/scripts/test_dcp_torchrun.py --mode save
    torchrun --nproc_per_node=2 tests/scripts/test_dcp_torchrun.py --mode load
"""

import argparse
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

CKPT_DIR = "./dcp_torchrun"


def build_model_and_optim():
    """构造一个简单可复现的模型 + 优化器，用于演示分布式 state_dict 的存取。"""
    torch.manual_seed(1234)
    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def do_save(rank: int, world_size: int):
    model, optimizer = build_model_and_optim()

    # 走一步，让优化器产生状态（Adam 的动量等）
    x = torch.randn(4, 8)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    model_sd, optim_sd = get_state_dict(model, optimizer)
    state_dict = {"model": model_sd, "optim": optim_sd}
    print(f"[rank {rank}] state_dict={state_dict}")

    dcp.save(state_dict, checkpoint_id=CKPT_DIR)
    if rank == 0:
        print(f"[rank {rank}] saved with world_size={world_size}, files:", sorted(os.listdir(CKPT_DIR)))


def do_load(rank: int, world_size: int):
    # 重新构造同结构的模型/优化器作为模板（此时权重是随机初始化的）
    model, optimizer = build_model_and_optim()
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state_dict = {"model": model_sd, "optim": optim_sd}

    # DCP 是 in-place / key-driven 加载：按模板 key 去读对应分片
    dcp.load(state_dict, checkpoint_id=CKPT_DIR)
    set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)

    # 打印一个权重指纹，验证不同 world_size 加载得到一致结果（重分片）
    fingerprint = next(model.parameters()).detach().flatten()[:4].tolist()
    print(f"[rank {rank}] loaded with world_size={world_size}, weight[:4]={fingerprint}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "save", "load"], default="all")
    args = parser.parse_args()

    # torchrun 会自动设置 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("gloo")

    if args.mode in ("all", "save"):
        if rank == 0 and os.path.isdir(CKPT_DIR):
            shutil.rmtree(CKPT_DIR)
        dist.barrier()
        do_save(rank, world_size)
        dist.barrier()

    if args.mode in ("all", "load"):
        dist.barrier()
        do_load(rank, world_size)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

