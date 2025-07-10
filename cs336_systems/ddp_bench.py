from .flat_ddp import FlatDDPParameters
from .naive_ddp import NaiveDDPIndividualParameters
from .overlap_ddp import DDPBucketParameters, DDPIndividualParameters
from .common import _setup_process_group, _cleanup_process_group, ToyModel, _generate_all_data

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
from itertools import chain
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function

WORLD_SIZE = 2
BUCKET_SIZE_MBS = [1, 10, 100, 1000]
WARMUP_ITERS = 5
INNER_SIZE1 = 10
INNER_SIZE2 = 5 * INNER_SIZE1


def bench_ddp(rank, world_size, backend, model_class, bucket_size_mb):
    
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    dist.barrier()

    def _sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)

    baseline_model = model_class(INNER_SIZE1, INNER_SIZE2).to(device)
    naive_model = NaiveDDPIndividualParameters(baseline_model)
    flat_model = FlatDDPParameters(baseline_model)
    overlap_model = DDPIndividualParameters(baseline_model)
    overlap_bucketed_model = DDPBucketParameters(baseline_model, bucket_size_mb)

    models = [naive_model, flat_model, overlap_model, overlap_bucketed_model]

    # broadcast initial parameters so all ranks start identically
    for ddp_model in models:
        for p in ddp_model.parameters():
            dist.broadcast(p.data, src=0)

    loss_fn = torch.nn.MSELoss()

    num_epochs = 20
    all_x, all_y = _generate_all_data()
    batch_size = all_x.shape[0]
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    mini_batch_size = batch_size // world_size
    for ddp_model in models:
        ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
        ddp_model.train()
        
        durations = []
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities += [ProfilerActivity.CUDA]
        print(activities)
        with profile(
            activities=activities,
        ) as prof:
            for epoch in range(num_epochs):

                # train the DDP model
                local_x = all_x[rank * mini_batch_size : (rank + 1) * mini_batch_size].to(device)
                local_y = all_y[rank * mini_batch_size : (rank + 1) * mini_batch_size].to(device)


                ddp_optimizer.zero_grad()
                with record_function("forward"):
                    local_out = ddp_model(local_x)
                    loss = loss_fn(local_out, local_y)
                _sync()
                start = timeit.default_timer()
                loss.backward()
                ddp_model.finish_gradients_syncronization()
                _sync()
                end = timeit.default_timer()
                ddp_optimizer.step()


                if epoch >= WARMUP_ITERS:
                    duration = end - start
                    durations.append(duration)

                # shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
                torch.manual_seed(42 + epoch)
                shuffle_idxs = torch.randperm(all_x.size(0))
                all_x = all_x[shuffle_idxs]
                all_y = all_y[shuffle_idxs]


        # statistics
        gathered: list[list[float]] = [None] * world_size
        dist.all_gather_object(gathered, durations)

        if rank == 0:
            flat = list(chain.from_iterable(gathered))
            model_name = ddp_model.__class__.__name__
            if model_name == "DDPBucketParameters":
                model_name = f"{model_name}({bucket_size_mb} MB)"
            print(
                f"{model_name:<30} | bkw pass and communication time (ms) |"
                f"min {min(flat) * 1e3:4.2f} | mean {np.mean(flat) * 1e3:4.2f} | max {max(flat) * 1e3:4.2f}"
            )
        
        _sync()

    _cleanup_process_group()


def main():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    print(f"Benchmarking on {backend.upper()}")
    print("----------------------------"*2)
    for bucket_size_mb in BUCKET_SIZE_MBS:
        model_class = ToyModel

        mp.spawn(
            bench_ddp,
            args=(WORLD_SIZE, backend, model_class, bucket_size_mb),
            nprocs=WORLD_SIZE,
            join=True,
        )

        print("----------------------------"*2)


if __name__ == "__main__":
    main()