
import triton
import torch
import torch.nn.functional as F
from tree_mamba_cuda import ssm_step_fwd, ssm_step_bwd

def ref_ssm_step(ssm_state, x, y, z, dt, dt_bias, A, B, C, D, weight, indices, states):
    ssm_state = F.pad(ssm_state, (0, 0, 0, 0, 1, 0))
    ssm_left = ssm_state[states[:, 0] + 1, :, :]
    ssm_right = ssm_state[states[:, 1] + 1, :, :]
    ssm = weight * ssm_left + (1 - weight) * ssm_right
    level_x = x[indices[:, 0], indices[:, 1], :]
    level_z = z[indices[:, 0], indices[:, 1], :]
    level_dt = dt[indices[:, 0], indices[:, 1], : ]
    level_B = B[indices[:, 0], indices[:, 1], :]
    level_C = C[indices[:, 0], indices[:, 1], :]

    level_dt = F.softplus(level_dt + dt_bias)
    dA = torch.exp(torch.einsum("bd,dn->bdn", level_dt, A))
    dB = torch.einsum("bd,bn->bdn", level_dt, level_B)
    ssm = ssm * dA + level_x.unsqueeze(2) * dB
    level_y = torch.einsum("bdn,bn->bd", ssm, level_C) + level_x * D
    level_y = level_y * F.silu(level_z)

    y[indices[:, 0], indices[:, 1], :] = level_y
    return ssm

batch_size = 10
seq_len = 128
d_inner = 128
d_state = 16
level_batch_size = 20
prev_batch_size = 30
ssm_state = torch.randn(prev_batch_size, d_inner, d_state, dtype=torch.float32, device='cuda', requires_grad=True)
x = torch.randn(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda', requires_grad=True)
z = torch.randn(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda', requires_grad=True)
dt = torch.randn(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda', requires_grad=True)
dt_bias = torch.randn(d_inner, dtype=torch.float32, device='cuda', requires_grad=True)
A = torch.randn(d_inner, d_state, dtype=torch.float32, device='cuda', requires_grad=True)
B = torch.randn(batch_size, seq_len, d_state, dtype=torch.float32, device='cuda', requires_grad=True)
C = torch.randn(batch_size, seq_len, d_state, dtype=torch.float32, device='cuda', requires_grad=True)
D = torch.randn(d_inner, dtype=torch.float32, device='cuda', requires_grad=True)
weight = torch.randn(d_inner, d_state, dtype=torch.float32, device='cuda', requires_grad=True)
indices_0 = torch.randint(0, batch_size, (level_batch_size, 1), dtype=torch.int, device='cuda')
indices_1 = torch.randint(0, seq_len, (level_batch_size, 1), dtype=torch.int, device='cuda')
indices = torch.concat([indices_0, indices_1], dim=-1).contiguous()
states = torch.randperm(2 * prev_batch_size, dtype=torch.int, device='cuda').view(prev_batch_size, 2)[:level_batch_size]
states[states >= prev_batch_size] = -1
# states = torch.randint(0, prev_batch_size // 2 * 3, (level_batch_size, 2), dtype=torch.int, device='cuda')

# new_dt = step1_dt(dt, dt_bias, indices)
# ref_new_dt = ref_step1_dt(dt, dt_bias, indices)
# print(torch.max(torch.abs(new_dt - ref_new_dt)))

# dA = step2_da(dt, dt_bias, A, indices)
# ref_dA = ref_step2_da(dt, dt_bias, A, indices)
# print(torch.max(torch.abs(dA - ref_dA)))

# dB = step3_db(dt, dt_bias, A, B, indices)
# ref_dB = ref_step3_db(dt, dt_bias, A, B, indices)
# print(torch.max(torch.abs(dB - ref_dB)))

# print(states[0,0])
# print(indices[0,0])
# print(ssm_state[states[0,0], 0])
# ssm = step4_ssm(ssm_state, dt, dt_bias, A, B, weight, indices, states)
# ref_ssm = ref_step4_ssm(ssm_state, dt, dt_bias, A, B, weight, indices, states)
# print(torch.max(torch.abs(ssm - ref_ssm)))
# print(ssm[0,0])
# print(ref_ssm[0,0])
# print((ssm_state == ssm[0,0,0]).nonzero(as_tuple=False))
# print((ssm_state == ssm[0,1,0]).nonzero(as_tuple=False))
# print((ssm_state == ssm[1,0,0]).nonzero(as_tuple=False))
# print((ssm_state == ssm[1,1,0]).nonzero(as_tuple=False))

y = torch.zeros_like(x)
y_ref = torch.zeros_like(x)
ssm_ref = ref_ssm_step(ssm_state, x, y_ref, z, dt, dt_bias, A, B, C, D, weight, indices, states)
random_y_grad = torch.randn_like(y_ref)
random_ssm_grad = torch.randn_like(ssm_ref)
l = torch.sum(y_ref * random_y_grad) + torch.sum(ssm_ref * random_ssm_grad)
l.backward()
# print(z.grad[indices[:,0], indices[:,1], :5])
ssm = ssm_step_fwd(ssm_state, x, y, z, dt, dt_bias, A, B, C, D, weight, indices, states)

x_grad = torch.zeros_like(x)
z_grad = torch.zeros_like(z)
dt_grad = torch.zeros_like(dt)
dt_bias_grad = torch.zeros_like(dt_bias)
A_grad = torch.zeros_like(A)
B_grad = torch.zeros_like(B)
C_grad = torch.zeros_like(C)
D_grad = torch.zeros_like(D)
weight_grad = torch.zeros_like(weight)

ssm_in_grad = ssm_step_bwd(random_y_grad, random_ssm_grad, ssm_state, x, z, dt, dt_bias, A, B, C, D, weight, indices, states, x_grad, z_grad, dt_grad, dt_bias_grad, A_grad, B_grad, C_grad, D_grad, weight_grad)
# print(z_grad[indices[:,0], indices[:,1], :5])
print("ssm", torch.max(torch.abs(ssm_state.grad - ssm_in_grad)))
print("x", torch.max(torch.abs(x.grad - x_grad)))
print("z", torch.max(torch.abs(z.grad - z_grad)))
print("dt", torch.max(torch.abs(dt.grad - dt_grad)))
print("dt_bias", torch.max(torch.abs(dt_bias.grad - dt_bias_grad)))
print("A", torch.max(torch.abs(A.grad - A_grad)))
print("B", torch.max(torch.abs(B.grad - B_grad)))
print("C", torch.max(torch.abs(C.grad - C_grad)))
print("D", torch.max(torch.abs(D.grad - D_grad)))
print("weight", torch.max(torch.abs(weight.grad - weight_grad)))

print(torch.max(torch.abs(y - y_ref)))
print(torch.max(torch.abs(ssm - ssm_ref)))
# print(y[indices[:,0], indices[:,1], :5])
# print(y_ref[indices[:,0], indices[:,1], :5])

# print(indices_0.view(-1))
# print(indices_1.view(-1))
# print(states[:,0])
# print(states[:,1])

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['level_batch_size'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(1, 20)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['cuda', 'torch'],  # possible values for `line_arg``
        line_names=[
            "CUDA",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="tree-ssm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'batch_size': 512, 'seq_len': 512, 'd_inner': 128, 'd_state': 16},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(batch_size, seq_len, d_inner, d_state, level_batch_size, provider):
    ssm_state = torch.randn(level_batch_size, d_inner, d_state, dtype=torch.float32, device='cuda')
    x = torch.randn(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda')
    y = torch.zeros(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda')
    z = torch.randn(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda')
    dt = torch.randn(batch_size, seq_len, d_inner, dtype=torch.float32, device='cuda')
    dt_bias = torch.randn(d_inner, dtype=torch.float32, device='cuda')
    A = torch.randn(d_inner, d_state, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, seq_len, d_state, dtype=torch.float32, device='cuda')
    C = torch.randn(batch_size, seq_len, d_state, dtype=torch.float32, device='cuda')
    D = torch.randn(d_inner, dtype=torch.float32, device='cuda')
    weight = torch.randn(d_inner, d_state, dtype=torch.float32, device='cuda')
    indices_0 = torch.randint(0, batch_size, (level_batch_size, 1), dtype=torch.int, device='cuda')
    indices_1 = torch.randint(0, seq_len, (level_batch_size, 1), dtype=torch.int, device='cuda')
    indices = torch.concat([indices_0, indices_1], dim=-1).contiguous()
    states = torch.randint(-1, level_batch_size, (level_batch_size, 2), dtype=torch.int, device='cuda')
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: ref_ssm_step(ssm_state, x, y, z, dt, dt_bias, A, B, C, D, weight, indices, states))
    if provider == 'cuda':
        ms = triton.testing.do_bench(lambda: ssm_step_fwd(ssm_state, x, y, z, dt, dt_bias, A, B, C, D, weight, indices, states))
    bytes_read = 0
    bytes_read += level_batch_size * d_inner * x.element_size()
    bytes_read += level_batch_size * d_inner * z.element_size()
    bytes_read += level_batch_size * d_inner * dt.element_size()
    bytes_read += d_inner * dt_bias.element_size()
    bytes_read += d_inner * d_state * A.element_size()
    bytes_read += level_batch_size * d_state * B.element_size()
    bytes_read += level_batch_size * d_state * C.element_size()
    bytes_read += d_inner * D.element_size()
    bytes_read += d_inner * d_state * weight.element_size()

    bytes_written = 0
    bytes_read += level_batch_size * d_inner * y.element_size()
    gbps = lambda ms: (bytes_read + bytes_written) * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)