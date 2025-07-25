
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

FEATURE_IN = 10
FEATURE_OUT = 5

def validate_ddp_net_equivalence(net):
    # Helper to validate synchronization of nets across ranks.
    net_module_states = list(net.module.state_dict().values())
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor in tensor_list:
            assert torch.allclose(tensor, t)


class _FC2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x

class ToyModel(nn.Module):
    def __init__(self, inner_size1=10, inner_size2=50):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_IN, inner_size1, bias=False)
        self.fc2 = _FC2(inner_size1, inner_size2)
        self.fc3 = nn.Linear(inner_size2, FEATURE_OUT, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BigToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_IN, 1000, bias=False)
        self.fc2 = _FC2(1000, 5000)
        self.fc3 = nn.Linear(5000, FEATURE_OUT, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ToyModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_IN, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, FEATURE_OUT, bias=False)
        self.fc4.weight = self.fc2.weight
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def _generate_all_data(total_samples=100, feature_dim=FEATURE_IN):
    torch.manual_seed(42)
    x = torch.randn(total_samples, feature_dim)
    y = torch.randn(total_samples, FEATURE_OUT)
    return x, y


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()
