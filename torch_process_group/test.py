
'''
使用方法：
    torchrun --nproc_per_node=2 test.py
'''

import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])   # torchrun 自动设置，单机即等于 0..N-1
print(f'rank is {rank}')
# 关键：把当前进程绑定到对应的 GPU
torch.cuda.set_device(local_rank)

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# 用 local_rank 指定设备，而不是裸 .cuda()
device = torch.device(f"cuda:{local_rank}")
tensor = torch.ones(1, device=device) * rank
dist.all_reduce(tensor)
print(f"rank {rank} (local_rank {local_rank}): {tensor.item()}")

dist.destroy_process_group()


