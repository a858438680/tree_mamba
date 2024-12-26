import torch
from tree_mamba_cuda import ssm_step_fwd, ssm_step_bwd

class SSMStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, z, dt, dt_bias, A, B, C, D, weight, indices_list: list[torch.Tensor], state_indices: list[torch.Tensor]):
        d_inner, d_state = A.shape
        ssm_state = torch.empty((0, d_inner, d_state))
        y = torch.zeros_like(x)
        ssms = []
        for indices, states in zip(indices_list, state_indices):
            ssms.append(ssm_state)
            ssm_state = ssm_step_fwd(ssm_state, x, y, z, dt, dt_bias, A, B, C, D, weight, indices, states)
        return y, tuple(ssms)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, z, dt, dt_bias, A, B, C, D, weight, indices_list, state_indices = inputs
        _, ssms = outputs
        ctx.save_for_backward(x, z, dt, dt_bias, A, B, C, D, weight)
        ctx.indices_list = indices_list
        ctx.state_indices = state_indices
        ctx.ssms = ssms

    @staticmethod
    def backward(ctx, *grad_outputs):
        y_grad, _ = grad_outputs
        x, z, dt, dt_bias, A, B, C, D, weight = ctx.saved_tensors
        indices_list = ctx.indices_list
        state_indices = ctx.state_indices
        ssms = ctx.ssms
        x_grad = torch.zeros_like(x)
        z_grad = torch.zeros_like(z)
        dt_grad = torch.zeros_like(dt)
        dt_bias_grad = torch.zeros_like(dt_bias)
        A_grad = torch.zeros_like(A)
        B_grad = torch.zeros_like(B)
        C_grad = torch.zeros_like(C)
        D_grad = torch.zeros_like(D)
        weight_grad = torch.zeros_like(weight)
        d_inner, d_state = A.shape
        last_level_batch, _ = indices_list[-1].shape
        ssm_grad = torch.zeros(last_level_batch, d_inner, d_state, dtype=torch.float32, device=x.device)
        for ssm_state, indices, states in zip(reversed(ssms), reversed(indices_list), reversed(state_indices)):
            ssm_grad = ssm_step_bwd(y_grad, ssm_grad, ssm_state, x, z, dt, dt_bias, A, B, C, D, weight, indices, states, x_grad, z_grad, dt_grad, dt_bias_grad, A_grad, B_grad, C_grad, D_grad, weight_grad)
        return x_grad, z_grad, dt_grad, dt_bias_grad, A_grad, B_grad, C_grad, D_grad, weight_grad, None, None