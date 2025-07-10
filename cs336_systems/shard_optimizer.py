import torch
from torch.optim import Optimizer
from typing import Any
import torch.distributed as dist

class ShardedOptimizer(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls
        self.param2rank = {}
        self.rank2num_params = [0] * self.world_size
        self.kwargs = kwargs
        # lazy initialization
        self.optimizer = None
        # we mannually set hyperparameters into the optimizer at 'add_param_group'
        super().__init__(params, defaults={})


    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        for p in self.param2rank:
            dist.broadcast(p.data, src=self.param2rank[p])

    def add_param_group(self, param_group: dict[str, Any]):
        param_to_add_locally: list[torch.nn.Parameter] = []

        # assign the parameters to the local rank
        for p in param_group["params"]:
            num_params = p.numel()

            min_load_rank = self.rank2num_params.index(min(self.rank2num_params))
            self.param2rank[p] = min_load_rank
            self.rank2num_params[min_load_rank] += num_params

            if self.rank == min_load_rank:
                param_to_add_locally.append(p)

        # If this rank is responsible for any parameters in the new group, update the local optimizer
        if param_to_add_locally:
            # append non-param attributes e.g. hyperparameters
            new_local_param_group = {k: v for k, v in param_group.items() if k != "params"}
            #new_local_param_group.update({"params": param_to_add_locally})
            new_local_param_group = {**new_local_param_group, "params": param_to_add_locally}

            if self.optimizer is None:
                # optimizer_cls(param_groups_list, **kwargs)
                self.optimizer = self.optimizer_cls([new_local_param_group], **self.kwargs)
            else:
                self.optimizer.add_param_group(new_local_param_group)

        # maintain the whole param_group in super
        super().add_param_group(param_group)
            

