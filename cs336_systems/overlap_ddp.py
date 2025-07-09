import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from dataclasses import dataclass, field


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

@dataclass
class Bucket:
    params: list[torch.nn.Parameter] = field(default_factory=list)
    num_params: int = 0
    num_params_ready: int = 0

class DDPBucketParameters(torch.nn.Module):


    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        # Determine parameters per bucket based on the data type used for the parameters
        param_dtype = next(module.parameters()).dtype
        bytes_per_param = param_dtype.itemsize
        self.bucket_size_params = int(bucket_size_mb * 1024**2 / bytes_per_param)

        self.world_size = dist.get_world_size()
        self.handles = []
        self.buckets: list[Bucket] = []
        self._init_buckets()

        # Broadcast gradients from rank 0 to all other ranks.
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, async_op=False)

        def _bucket(param):
            # add to bucket
            bucket_idx = param.bucket_idx
            bucket = self.buckets[bucket_idx]
            bucket.num_params_ready += param.numel()

            if bucket.num_params_ready == bucket.num_params:
                params_with_grads = [p for p in bucket.params if p.grad is not None]
                if not params_with_grads:
                    bucket.num_params_ready = 0
                    return
                
                grads = [p.grad for p in params_with_grads]
                flat_grads = _flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, flat_grads, grads, params_with_grads))
                bucket.num_params_ready = 0

        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_bucket)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradients_syncronization(self):
        for handle, flat_grads, grads, params_with_grads in self.handles:
            handle.wait()
            flat_grads.div_(self.world_size)
            restored_grads = _unflatten_dense_tensors(flat_grads, grads) 
            for param, restored_grad in zip(params_with_grads, restored_grads):
                param.grad.copy_(restored_grad)

        self.handles.clear()

    def _init_buckets(self):
        curr_bucket = Bucket()
        curr_idx = 0

        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            if curr_bucket.num_params + p.numel() > self.bucket_size_params:
                self.buckets.append(curr_bucket)
                curr_bucket = Bucket()
                curr_idx += 1

            p.bucket_idx = curr_idx
            curr_bucket.params.append(p)
            curr_bucket.num_params += p.numel()
        
        if curr_bucket.num_params > 0:
            self.buckets.append(curr_bucket)

