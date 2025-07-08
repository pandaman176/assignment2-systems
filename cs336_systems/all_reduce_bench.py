import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import product, chain
import numpy as np
import timeit

MB = 1024 * 1024
DATA_SIZES = [int( i * MB // 4) for i in (1, 10, 100 ,1024 )] # 1MB, 10MB, 100MB, 1GB
WORLD_SIZES = [2, 4, 6]
BACKENDS = ["gloo", "nccl"] if torch.cuda.is_available() else ["gloo"]

WARMUP_ITERS = 5
TIMED_ITERS = 10


def distributed_demo(rank, world_size, data_size, backend, warmup, iters):
    # set up
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available() and backend == "nccl":
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # allocate tensor
    data = torch.rand(data_size, dtype=torch.float32, device=device)

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # warmup
    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
        _sync()
    dist.barrier()
    
    # benchmark
    durations: list[float] = []
    for _ in range(iters):
        _sync()
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        _sync()
        end_time = timeit.default_timer()
        duration = end_time - start_time
        durations.append(duration)

    # statistics
    gathered: list[list[float]] = [None] * world_size
    dist.all_gather_object(gathered, durations)

    if rank == 0:
        flat = list(chain.from_iterable(gathered))
        size_mb = data_size * 4 / MB
        print(
            f"{backend.upper():<4} | {device.type:<4} | {world_size} proc | "
            f"{size_mb:>4.0f} MB -> "
            f"min {min(flat) * 1e3:6.2f} mean {np.mean(flat) * 1e3:7.2f} "
            f"max {max(flat) * 1e3:6.2f} MB/s"
        )

    dist.destroy_process_group()

def benchmark():
    for data_size, world_size, backend in product(DATA_SIZES, WORLD_SIZES, BACKENDS):
        if backend == "nccl" and torch.cuda.device_count() < world_size:
            print(
                f"[WARNING] NCCL with {world_size=} requires "
                f">={world_size} GPUs, but only {torch.cuda.device_count()} GPUs are available."
            )
        mp.spawn(fn=distributed_demo, args=(world_size,data_size,backend, WARMUP_ITERS, TIMED_ITERS), nprocs=world_size, join=True)
        print("------------------------"*3)

if __name__ == "__main__":
    benchmark()
