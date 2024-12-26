import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl
from tree_mamba_cuda import tree_conv_fwd, tree_conv_bwd

group_in_channels = 1
group_out_channels = 1
num_groups = 64
in_channels = num_groups * group_in_channels
out_channels = num_groups * group_out_channels
batch_size = 512
seq_len = 256

conv = nn.Conv1d(in_channels, out_channels, 4, 4, groups=num_groups, device='cuda')
x = torch.randn(batch_size, seq_len, in_channels, device='cuda', requires_grad=True)
indices = torch.randint(-1, seq_len, (batch_size, 4 * seq_len), dtype=torch.int32, device='cuda')

def ref_tree_conv(x: torch.Tensor, indices: torch.Tensor, conv):
    x = x.transpose(1, 2)
    _, dim, _ = x.shape
    padded_x = F.pad(x, (1, 0, 0, 0, 0, 0))
    selected_x = torch.gather(padded_x, 2, (indices + 1).unsqueeze(1).expand(-1, dim, -1).to(torch.int64))
    y = conv(selected_x)
    y = y.transpose(1, 2)
    # y = F.silu(y)
    return y

def tree_conv(x, indices, conv):
    return tree_conv_fwd(x, indices, conv.weight, conv.bias)

# y_0 = F.silu(x[64, indices[64,0]] * conv.weight[:,:,0].view(-1) + x[64, indices[64,1]] * conv.weight[:,:,1].view(-1) + x[64, indices[64,2]] * conv.weight[:,:,2].view(-1) + x[64, indices[64,3]] * conv.weight[:,:,3].view(-1) + conv.bias)
y = ref_tree_conv(x, indices, conv)
random_grad = torch.randn_like(y)
l = torch.sum(random_grad * y)
l.backward()
y_triton = tree_conv(x, indices, conv)
x_grad, weight_grad, bias_grad = tree_conv_bwd(random_grad, x, indices, conv.weight, conv.bias)
max_diff = torch.abs(y - y_triton).max()
print(max_diff)
print(x.grad.dtype)
print(x_grad.dtype)
print(torch.abs(x_grad - x.grad).max())
index_max = torch.abs(x_grad - x.grad).argmax().item()
dim = index_max % num_groups
seq_idx = (index_max // num_groups) % seq_len
batch_idx = (index_max // num_groups) // seq_len
print((batch_idx, seq_idx, dim))
print(x_grad[batch_idx, seq_idx, dim] - x.grad[batch_idx, seq_idx, dim])
y_idx = (indices[batch_idx] == seq_idx).nonzero().view(-1) // 4
w_idx = (indices[batch_idx] == seq_idx).nonzero().view(-1) % 4
y_grads = random_grad[batch_idx, y_idx, dim]
ref_grad = torch.sum(y_grads * conv.weight[dim, 0, w_idx])
print(ref_grad - x_grad[batch_idx, seq_idx, dim])
print(ref_grad - x.grad[batch_idx, seq_idx, dim])
ref_bias_grad = torch.sum(random_grad, dim=(0, 1))
print(ref_bias_grad)
print(random_grad.shape)
random_grad = random_grad.reshape(-1, out_channels)
shuffled_indices = torch.randperm(random_grad.shape[0])
random_grad = random_grad[shuffled_indices,:]
ref_bias_grad2 = torch.sum(random_grad, dim=0)
print(torch.abs(ref_bias_grad - ref_bias_grad2).max())
print(torch.abs(ref_bias_grad - bias_grad).max())
print(torch.abs(ref_bias_grad - conv.bias.grad).max())

print(torch.abs(weight_grad - conv.weight.grad).max())
print(torch.max(torch.abs(bias_grad - conv.bias.grad) / torch.maximum(bias_grad.abs(), conv.bias.abs())))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(1, 20)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['cuda', 'torch'],  # possible values for `line_arg``
        line_names=[
            "CUDA",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="tree-conv-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'batch_size': 512},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(batch_size, seq_len, provider):
    conv = torch.nn.Conv1d(128, 128, 4, 4, groups=1, device='cuda')
    x = torch.randn(batch_size, seq_len, 128, device='cuda', dtype=torch.float32)
    indices = torch.randint(-1, seq_len, (batch_size, 4 * seq_len), dtype=torch.int32, device='cuda')
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: ref_tree_conv(x, indices, conv))
    if provider == 'cuda':
        ms = triton.testing.do_bench(lambda: tree_conv(x, indices, conv))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


# benchmark.run(show_plots=True, print_data=True)