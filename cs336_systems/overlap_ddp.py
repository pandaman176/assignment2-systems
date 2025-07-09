import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):


    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        # Broadcast gradients from rank 0 to all other ranks.
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, async_op=False)

        def _avg_grads(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, param))

        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_avg_grads)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradients_syncronization(self):
        for handle, param in self.handles:
            handle.wait()
            param.grad.div_(self.world_size)
        self.handles.clear()

