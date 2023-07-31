"""Includes split aggregate function useful for implementating skip connections."""
import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Dict


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Lambda(nn.Module):
    def __init__(self, f: Callable[[Tensor], Tensor]):
        super().__init__()
        self.f = f

    def forward(self, x: Tensor):
        return self.f(x)


class SplitAggregate(nn.Module):
    """
    Allows for specifying a "split" in a sequential model where the input passes through multiple paths and is
    aggregated at the output (by default: added).

    Useful for specifying a residual connection in a model.
    """
    def __init__(self, path1: nn.Module, path2: nn.Module, aggregate_func: Callable[[Tensor, Tensor], Tensor] = torch.add) -> None:
        super().__init__()
        self.path1 = path1
        self.path2 = path2
        self.aggregate_func = aggregate_func
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate_func(
            self.path1(x),
            self.path2(x),
        )


class Residual(SplitAggregate):
    def __init__(self, path: nn.Module, aggregate_func: Callable[[Tensor, Tensor], Tensor] = torch.add) -> None:
        super().__init__(path1=path, path2=Identity(), aggregate_func=aggregate_func)
        

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

def get_submodules(model) -> Dict[str, str]:
    """Find the names of the layers in the model for feature extractor in feature_match.py"""
    submodules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            submodules[name] = f"conv_{name.split('.')[-1]}"
        elif isinstance(module, nn.BatchNorm2d):
            submodules[name] = f"bn_{name.split('.')[-1]}"
        elif isinstance(module, nn.Linear):
            submodules[name] = f"fc_{name.split('.')[-1]}"
        else:
            submodules[name] = name
    print(submodules)