import torch
from .tree_conv import TreeConvFunction
from .tree_ssm import SSMStepFunction

def tree_conv(x: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    indices = (indices - 1).to(torch.int32)
    return TreeConvFunction.apply(x, indices, weight, bias)

def tree_ssm(
    x: torch.Tensor,
    z: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    weight: torch.Tensor,
    indices_list: list[torch.Tensor],
    state_indices: list[torch.Tensor]
):
    indices_list = [indices.to(torch.int32) for indices in reversed(indices_list)]
    state_indices = [(states - 1).to(torch.int32) for states in reversed(state_indices)]
    y, _ = SSMStepFunction.apply(x, z, dt, dt_bias, A, B, C, D, weight, indices_list, state_indices)
    return y

__all__ = ['tree_conv', 'tree_ssm']