
from .common import _setup_process_group, _cleanup_process_group,ToyModelWithTiedWeights, ToyModel, _generate_all_data

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from copy import deepcopy
import timeit

WORLD_SIZE = 2
WARMUP = 5

def flat_ddp(rank, world_size, backend, model_class):
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    dist.barrier()

    def _sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)
    
    model = model_class().to(device)
    ddp_model = deepcopy(model)

    # broadcast initial parameters so all ranks start identically
    for p in ddp_model.parameters():
        dist.broadcast(p.data, src=0)

    loss_fn = torch.nn.MSELoss()
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    num_epochs = 15
    ddp_model.train()
    all_x, all_y = _generate_all_data()
    batch_size = all_x.shape[0]
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    mini_batch_size = batch_size // world_size
    for epoch in range(num_epochs):
        # train the DDP model on all the data
        local_x = all_x[rank * mini_batch_size : (rank + 1) * mini_batch_size].to(device)
        local_y = all_y[rank * mini_batch_size : (rank + 1) * mini_batch_size].to(device)
        ddp_optimizer.zero_grad()
        local_out = ddp_model(local_x)
        loss = loss_fn(local_out, local_y)
        loss.backward()

        dist.barrier()

        _sync()
        t0 = timeit.default_timer()

        for param in ddp_model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
                
        _sync()
        t1 = timeit.default_timer()

        # buffer this proceess
        grades = [param.grad for param in ddp_model.parameters() if param.requires_grad and param.grad is not None]
        if not grades:
            continue
        flat_grades = _flatten_dense_tensors(grades)
        dist.all_reduce(flat_grades, op=dist.ReduceOp.SUM)
        flat_grades.mul_(1.0 / world_size)
        restored_grads = _unflatten_dense_tensors(flat_grades, grades)
        i = 0 
        for param in ddp_model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.copy_(restored_grads[i])
                i += 1        

        _sync()
        t2 = timeit.default_timer()

        if epoch >= WARMUP:
            speedup = (t1 - t0) / (t2 - t1)
            print(f"| non_flat {t1-t0:.3f} | flat {t2-t1:.3f} | speedup {speedup:.3f}x")
        
        ddp_optimizer.step()

        # shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        torch.manual_seed(42 + epoch)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

        _sync()
    
    _cleanup_process_group()

def main():
    for model_class in [ToyModel, ToyModelWithTiedWeights]:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        mp.spawn(
            flat_ddp,
            args=(WORLD_SIZE, backend, model_class),
            nprocs=WORLD_SIZE,
            join=True,
        )

if __name__ == "__main__":
    main()