import torch
from tree_mamba_cuda import tree_conv_fwd, tree_conv_bwd

class TreeConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, indices, weight, bias):
        return tree_conv_fwd(x, indices, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, indices, weight, bias = inputs
        ctx.save_for_backward(x, indices, weight, bias)

    @staticmethod
    def backward(ctx, y_grad):
        x, indices, weight, bias = ctx.saved_tensors
        x_grad, weight_grad, bias_grad = tree_conv_bwd(y_grad, x, indices, weight, bias)
        return x_grad, None, weight_grad, bias_grad