import torch
from torch import nn
from torch.autograd import gradcheck
from tree_mamba import tree_conv, tree_ssm

group_in_channels = 1
group_out_channels = 1
num_groups = 16
in_channels = num_groups * group_in_channels
out_channels = num_groups * group_out_channels
batch_size = 8
seq_len = 32

conv = nn.Conv1d(in_channels, out_channels, 4, 4, groups=num_groups, device='cuda')
x = torch.randn(batch_size, seq_len, in_channels, device='cuda', requires_grad=True)
indices = torch.randint(-1, seq_len, (batch_size, 4 * seq_len), dtype=torch.int32, device='cuda')

test = gradcheck(lambda x, w, b: tree_conv(x, indices, w, b), (x, conv.weight, conv.bias), eps=1e-6, atol=1e-2)
print(test)