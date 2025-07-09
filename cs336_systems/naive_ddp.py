
from .common import _setup_process_group, _cleanup_process_group,ToyModelWithTiedWeights, ToyModel, _generate_all_data

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy

WORLD_SIZE = 2

def naive_ddp(rank, world_size, backend, model_class):
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    dist.barrier()

    model = model_class().to(device)
    ddp_model = deepcopy(model)

    # broadcast initial parameters so all ranks start identically
    for p in ddp_model.parameters():
        dist.broadcast(p.data, src=0)

    loss_fn = torch.nn.MSELoss()
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    non_parallel_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 10
    model.train()
    ddp_model.train()
    all_x, all_y = _generate_all_data()
    batch_size = all_x.shape[0]
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    mini_batch_size = batch_size // world_size
    for epoch in range(num_epochs):
        # train the non-parallel model on all the data
        if rank == 0:
            non_parallel_optimizer.zero_grad()
            all_out = model(all_x)
            non_parallel_loss = loss_fn(all_out, all_y)
            print(f"non-parallel loss: {non_parallel_loss.item():.3f}")
            non_parallel_loss.backward()
            non_parallel_optimizer.step()

        # train the DDP model on all the data
        local_x = all_x[rank * mini_batch_size : (rank + 1) * mini_batch_size].to(device)
        local_y = all_y[rank * mini_batch_size : (rank + 1) * mini_batch_size].to(device)
        ddp_optimizer.zero_grad()
        local_out = ddp_model(local_x)
        loss = loss_fn(local_out, local_y)
        loss.backward()
        for param in ddp_model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.mul_(1.0 / world_size)
        
        if rank == 0:
            print(f"ddp loss after all reduce: {loss.item():.3f}")
        ddp_optimizer.step()

        # shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        torch.manual_seed(42 + epoch)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

        if device.startswith("cuda"):
            torch.cuda.synchronize()
    
    if rank == 0:
        for(non_parallel_name, non_parallel_model_parameter), (ddp_name, ddp_model_parameter) in zip(
            model.named_parameters(), ddp_model.named_parameters()
        ):
            assert non_parallel_name == ddp_name
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        print("Rank0: Verification passed.")
    _cleanup_process_group()

def main():
    for model_class in [ToyModel, ToyModelWithTiedWeights]:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        mp.spawn(
            naive_ddp,
            args=(WORLD_SIZE, backend, model_class),
            nprocs=WORLD_SIZE,
            join=True,
        )

class NaiveDDPIndividualParameters(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # Broadcast gradients from rank 0 to all other ranks.
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, async_op=False)

        def _avg_grads(param):
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.div_(self.world_size)

        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_avg_grads)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradients_syncronization(self):
        """
        create for api compatibility
        """
        return


if __name__ == "__main__":
    main()