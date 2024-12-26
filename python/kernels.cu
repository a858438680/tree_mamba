#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <utility>

__device__ __forceinline__ float softplus(float x) {
    if (x <= 20.f) {
        x = std::log1p(std::exp(x));
    }
    return x;
}

__device__ __forceinline__ float softplus_backward(float x) {
    if (x <= 20.f) {
        float z = std::exp(x);
        return z / (1.f + z);
    }
    return 1.f;
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.f + std::exp(-x));
}

__device__ __forceinline__ float silu_backward(float x) {
    float s = sigmoid(x);
    return s * (1 + x * (1 - s));
}

__global__ void tree_conv_fwd_kernel(
    const float* __restrict__ x,
    const int*   __restrict__ indices,
    const float* __restrict__ weight,
    const float* __restrict__ bias,

    int stride_xn, int stride_xs, int stride_xc,
    int stride_in, int stride_is,
    int stride_wc, int stride_ws,
    int stride_bc,

    float* __restrict__ y,
    int stride_yn, int stride_ys, int stride_yc,

    int batch_size,
    int seq_len,
    int channels
) {
    int vector_id = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (vector_id >= batch_size * seq_len || channel_id >= channels) {
        return;
    }

    int idx_n = vector_id / seq_len;
    int idx_y_s = vector_id % seq_len;
    int idx_y_c = channel_id;
    int idx_x_c = channel_id;

    float result = bias[idx_y_c * stride_bc];

    for (int i = 0; i < 4; ++i) {
        float w = weight[idx_y_c * stride_wc + i * stride_ws];
        int idx_i_s = idx_y_s * 4 + i;
        int idx_x_s = indices[idx_n * stride_in + idx_i_s * stride_is];
        float f_x = idx_x_s < 0 ? 0.f : x[idx_n * stride_xn + idx_x_s * stride_xs + idx_x_c * stride_xc];
        result += w * f_x;
    }

    result = silu(result);
    y[idx_n * stride_yn + idx_y_s * stride_ys + idx_y_c * stride_yc] = result;
}

torch::Tensor tree_conv_fwd(
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int channels = x.size(2);

    auto y = torch::empty_like(x);

    int num_vecs_per_block = 4;
    int num_channels_per_block = 64;

    dim3 dimBlock((batch_size * seq_len + num_vecs_per_block - 1) / num_vecs_per_block, (channels + num_channels_per_block - 1) / num_channels_per_block);
    dim3 dimThread(num_vecs_per_block, num_channels_per_block);
    tree_conv_fwd_kernel<<<dimBlock, dimThread>>>(
        x.data<float>(),
        indices.data<int>(),
        weight.data<float>(),
        bias.data<float>(),

        x.stride(0), x.stride(1), x.stride(2),
        indices.stride(0), indices.stride(1),
        weight.stride(0), weight.stride(2),
        bias.stride(0),
        
        y.data<float>(),
        y.stride(0), y.stride(1), y.stride(2),

        batch_size,
        seq_len,
        channels
    );

    return y;
}

__global__ void tree_conv_bwd_kernel(
    const float* __restrict__ y_grad,
    const float* __restrict__ x,
    const int*   __restrict__ indices,
    const float* __restrict__ weight,
    const float* __restrict__ bias,

    int stride_y_grad_n, int stride_y_grad_s, int stride_y_grad_c,
    int stride_xn,       int stride_xs,       int stride_xc,
    int stride_in,       int stride_is,
    int stride_wc,       int stride_ws,
    int stride_bc,

    float* __restrict__ x_grad,
    float* __restrict__ weight_grad,
    float* __restrict__ bias_grad,

    int stride_x_grad_n, int stride_x_grad_s, int stride_x_grad_c,
    int stride_w_grad_c, int stride_w_grad_s,
    int stride_b_grad_c,

    int batch_size,
    int seq_len,
    int channels
) {
    int vector_id = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (vector_id >= batch_size * seq_len || channel_id >= channels) {
        return;
    }

    int idx_n = vector_id / seq_len;
    int idx_y_s = vector_id % seq_len;
    int idx_y_c = channel_id;
    int idx_x_c = channel_id;

    double grad = y_grad[idx_n * stride_y_grad_n + idx_y_s * stride_y_grad_s + idx_y_c * stride_y_grad_c];
    float result = bias[idx_y_c * stride_bc];

    float w_arr[4];
    float x_arr[4];
    int idx_arr[4];
    for (int i = 0; i < 4; ++i) {
        float w = weight[idx_y_c * stride_wc + i * stride_ws];
        w_arr[i] = w;
        int idx_i_s = idx_y_s * 4 + i;
        int idx_x_s = indices[idx_n * stride_in + idx_i_s * stride_is];
        idx_arr[i] = idx_x_s;
        float f_x = idx_x_s < 0 ? 0.f : x[idx_n * stride_xn + idx_x_s * stride_xs + idx_x_c * stride_xc];
        x_arr[i] = f_x;
        result += w * f_x;
    }

    grad *= silu_backward(result);
    atomicAdd(bias_grad + idx_y_c * stride_b_grad_c, grad);
    for (int i = 0; i < 4; ++i) {
        atomicAdd(weight_grad + idx_y_c * stride_w_grad_c + i * stride_w_grad_s, grad * x_arr[i]);
        int idx_x_s = idx_arr[i];
        if (idx_x_s >= 0) {
            atomicAdd(x_grad + idx_n * stride_x_grad_n + idx_x_s * stride_x_grad_s + idx_x_c * stride_x_grad_c, grad * w_arr[i]);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tree_conv_bwd(
    torch::Tensor y_grad,
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int channels = x.size(2);

    auto x_grad = torch::zeros_like(x);
    auto weight_grad = torch::zeros_like(weight);
    auto bias_grad = torch::zeros_like(bias);

    int num_vecs_per_block = 4;
    int num_channels_per_block = 64;

    dim3 dimBlock((batch_size * seq_len + num_vecs_per_block - 1) / num_vecs_per_block, (channels + num_channels_per_block - 1) / num_channels_per_block);
    dim3 dimThread(num_vecs_per_block, num_channels_per_block);
    tree_conv_bwd_kernel<<<dimBlock, dimThread>>>(
        y_grad.data<float>(),
        x.data<float>(),
        indices.data<int>(),
        weight.data<float>(),
        bias.data<float>(),

        y_grad.stride(0), y_grad.stride(1), y_grad.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        indices.stride(0), indices.stride(1),
        weight.stride(0), weight.stride(2),
        bias.stride(0),
        
        x_grad.data<float>(),
        weight_grad.data<float>(),
        bias_grad.data<float>(),

        x_grad.stride(0), x_grad.stride(1), x_grad.stride(2),
        weight_grad.stride(0), weight_grad.stride(2),
        bias_grad.stride(0),

        batch_size,
        seq_len,
        channels
    );

    return {x_grad, weight_grad, bias_grad};
}

template <int d_inner, int d_model>
__global__ void ssm_step_fwd_kernel(
    // in
    const float* __restrict__ ssm_state,
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ dt,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    const float* __restrict__ weight,
    const int*   __restrict__ indices,
    const int*   __restrict__ states,

    int stride_state_b,   int stride_state_d, int stride_state_n,
    int stride_xb,        int stride_xs,      int stride_xd,
    int stride_zb,        int stride_zs,      int stride_zd,
    int stride_dtb,       int stride_dts,     int stride_dtd,
    int stride_dt_bias_d,
    int stride_ad,        int stride_an,
    int stride_bb,        int stride_bs,      int stride_bn,
    int stride_cb,        int stride_cs,      int stride_cn,
    int stride_dd,
    int stride_wd,        int stride_wn,
    int stride_ib,        int stride_ii,
    int stride_sb,        int stride_si,

    // out
    float* __restrict__ ssm_state_update,
    float* __restrict__ y,

    int stride_update_b, int stride_update_d, int stride_update_n,
    int stride_yb,       int stride_ys,       int stride_yd
) {
    int level_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    int batch_idx = indices[level_idx * stride_ib];
    int seq_idx = indices[level_idx * stride_ib + stride_ii];
    int left_idx = states[level_idx * stride_sb];
    int right_idx = states[level_idx * stride_sb + stride_si];

    float f_dt = dt[batch_idx * stride_dtb + seq_idx * stride_dts + dim_idx * stride_dtd];
    float f_x = x[batch_idx * stride_xb + seq_idx * stride_xs + dim_idx * stride_xd];
    float f_z = z[batch_idx * stride_zb + seq_idx * stride_zs + dim_idx * stride_zd];
    float f_dt_bias = dt_bias[dim_idx * stride_dt_bias_d];
    float f_D = D[dim_idx * stride_dd];

    f_dt = softplus(f_dt + f_dt_bias);

    float f_y = f_D * f_x;
    for (int state_idx = 0; state_idx < d_model; ++state_idx) {
        float f_dA = std::exp(f_dt * A[dim_idx * stride_ad + state_idx * stride_an]);
        float f_dB = f_dt * B[batch_idx * stride_bb + seq_idx * stride_bs + state_idx * stride_bn];
        float f_C = C[batch_idx * stride_cb + seq_idx * stride_cs + state_idx * stride_cn];
        float left_state = left_idx < 0 ? 0.f : ssm_state[left_idx * stride_state_b + dim_idx * stride_state_d + state_idx * stride_state_n];
        float right_state = right_idx < 0 ? 0.f : ssm_state[right_idx * stride_state_b + dim_idx * stride_state_d + state_idx * stride_state_n];
        float f_weight = weight[dim_idx * stride_wd + state_idx * stride_wn];
        float merged_state = f_weight * left_state + (1 - f_weight) * right_state;
        float new_state = f_dA * merged_state + f_dB * f_x;
        ssm_state_update[level_idx * stride_update_b + dim_idx * stride_update_d + state_idx * stride_update_n] = new_state;
        f_y += f_C * new_state;
    }
    f_y = f_y * silu(f_z);
    y[batch_idx * stride_yb + seq_idx * stride_ys + dim_idx * stride_yd] = f_y;
}

torch::Tensor ssm_step_fwd(
    torch::Tensor ssm_state,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor z,
    torch::Tensor dt,
    torch::Tensor dt_bias,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor weight,
    torch::Tensor indices,
    torch::Tensor states
) {
    const int batch_size = dt.size(0);
    const int seq_len = dt.size(1);
    const int d_inner = dt.size(2);
    const int d_state = A.size(1);
    const int level_batch_size = indices.size(0);
    auto ssm = torch::empty({level_batch_size, d_inner, d_state}, torch::TensorOptions(c10::kFloat).device(c10::kCUDA));
    const int threads = d_inner;
    const int blocks = level_batch_size;
    ssm_step_fwd_kernel<128, 16><<<blocks, threads>>>(
        ssm_state.data<float>(),
        x.data<float>(),
        z.data<float>(),
        dt.data<float>(),
        dt_bias.data<float>(),
        A.data<float>(),
        B.data<float>(),
        C.data<float>(),
        D.data<float>(),
        weight.data<float>(),
        indices.data<int>(),
        states.data<int>(),

        ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0),
        weight.stride(0), weight.stride(1),
        indices.stride(0), indices.stride(1),
        states.stride(0), states.stride(1),

        ssm.data<float>(),
        y.data<float>(),

        ssm.stride(0), ssm.stride(1), ssm.stride(2),
        y.stride(0), y.stride(1), y.stride(2)
    );
    return ssm;
}

template <int d_inner, int d_model>
__global__ void ssm_step_bwd_kernel(
    // in
    const float* __restrict__ y_grad,
    const float* __restrict__ ssm_out_grad,
    const float* __restrict__ ssm_state,
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ dt,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    const float* __restrict__ weight,
    const int*   __restrict__ indices,
    const int*   __restrict__ states,

    int stride_y_grad_b,   int stride_y_grad_s,   int stride_y_grad_d,
    int stride_out_grad_b, int stride_out_grad_d, int stride_out_grad_n,
    int stride_state_b,    int stride_state_d,    int stride_state_n,
    int stride_xb,         int stride_xs,         int stride_xd,
    int stride_zb,         int stride_zs,         int stride_zd,
    int stride_dtb,        int stride_dts,        int stride_dtd,
    int stride_dt_bias_d,
    int stride_ad,         int stride_an,
    int stride_bb,         int stride_bs,         int stride_bn,
    int stride_cb,         int stride_cs,         int stride_cn,
    int stride_dd,
    int stride_wd,         int stride_wn,
    int stride_ib,         int stride_ii,
    int stride_sb,         int stride_si,

    float* __restrict__ ssm_in_grad,
    float* __restrict__ x_grad,
    float* __restrict__ z_grad,
    float* __restrict__ dt_grad,
    float* __restrict__ dt_bias_grad,
    float* __restrict__ A_grad,
    float* __restrict__ B_grad,
    float* __restrict__ C_grad,
    float* __restrict__ D_grad,
    float* __restrict__ weight_grad,

    int stride_in_grad_b,    int stride_in_grad_d,    int stride_in_grad_n,
    int stride_x_grad_b,     int stride_x_grad_s,     int stride_x_grad_d,
    int stride_z_grad_b,     int stride_z_grad_s,     int stride_z_grad_d,
    int stride_dt_grad_b,    int stride_dt_grad_s,    int stride_dt_grad_d,
    int stride_dt_bias_grad_d,                                             // need reduce
    int stride_a_grad_d,     int stride_a_grad_n,                          // need reduce
    int stride_b_grad_b,     int stride_b_grad_s,     int stride_b_grad_n, // need reduce
    int stride_c_grad_b,     int stride_c_grad_s,     int stride_c_grad_n, // need reduce
    int stride_d_grad_d,                                                   // need reduce
    int stride_w_grad_d,     int stride_w_grad_n                           // need reduce
) {
    int level_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    int batch_idx = indices[level_idx * stride_ib];
    int seq_idx = indices[level_idx * stride_ib + stride_ii];
    int left_idx = states[level_idx * stride_sb];
    int right_idx = states[level_idx * stride_sb + stride_si];

    float f_y_grad = y_grad[batch_idx * stride_y_grad_b + seq_idx * stride_y_grad_s + dim_idx * stride_y_grad_d];
    float f_dt = dt[batch_idx * stride_dtb + seq_idx * stride_dts + dim_idx * stride_dtd];
    float f_x = x[batch_idx * stride_xb + seq_idx * stride_xs + dim_idx * stride_xd];
    float f_z = z[batch_idx * stride_zb + seq_idx * stride_zs + dim_idx * stride_zd];
    float f_dt_bias = dt_bias[dim_idx * stride_dt_bias_d];
    float f_D = D[dim_idx * stride_dd];

    float dt1 = f_dt + f_dt_bias;
    float dt2 = softplus(dt1);

    float nA[d_model];
    float nB[d_model];
    float nC[d_model];
    float nSl[d_model];
    float nSr[d_model];
    float nw[d_model];
    float y1 = f_D * f_x;
    for (int state_idx = 0; state_idx < d_model; ++state_idx) {
        float f_A = A[dim_idx * stride_ad + state_idx * stride_an];
        nA[state_idx] = f_A;
        float f_dA = std::exp(dt2 * f_A);
        float f_B = B[batch_idx * stride_bb + seq_idx * stride_bs + state_idx * stride_bn];
        nB[state_idx] = f_B;
        float f_dB = dt2 * f_B;
        float f_C = C[batch_idx * stride_cb + seq_idx * stride_cs + state_idx * stride_cn];
        nC[state_idx] = f_C;
        float left_state = left_idx < 0 ? 0.f : ssm_state[left_idx * stride_state_b + dim_idx * stride_state_d + state_idx * stride_state_n];
        nSl[state_idx] = left_state;
        float right_state = right_idx < 0 ? 0.f : ssm_state[right_idx * stride_state_b + dim_idx * stride_state_d + state_idx * stride_state_n];
        nSr[state_idx] = right_state;
        float f_weight = weight[dim_idx * stride_wd + state_idx * stride_wn];
        nw[state_idx] = f_weight;
        float merged_state = f_weight * left_state + (1 - f_weight) * right_state;
        float new_state = f_dA * merged_state + f_dB * f_x;
        y1 += f_C * new_state;
    }
    float z1 = silu(f_z);
    float y1_grad = f_y_grad * z1;
    float z1_grad = f_y_grad * y1;
    float f_z_grad = z1_grad * silu_backward(f_z);
    z_grad[batch_idx * stride_z_grad_b + seq_idx * stride_z_grad_s + dim_idx * stride_z_grad_d] = f_z_grad;
    float f_D_grad = y1_grad * f_x;
    atomicAdd(D_grad + dim_idx * stride_d_grad_d, f_D_grad);
    float f_x_grad = y1_grad * f_D;
    float dt2_grad = 0.f;
    for (int i = 0; i < d_model; ++i) {
        float dAi = std::exp(dt2 * nA[i]);
        float dBi = dt2 * nB[i];
        float Smi = nw[i] * nSl[i] + (1 - nw[i]) * nSr[i];
        float Si = dAi * Smi + dBi * f_x;
        float Ci_grad = y1_grad * Si;
        atomicAdd(C_grad + batch_idx * stride_c_grad_b + seq_idx * stride_c_grad_s + i * stride_c_grad_n, Ci_grad);
        float Si_grad = y1_grad * nC[i] + ssm_out_grad[level_idx * stride_out_grad_b + dim_idx * stride_out_grad_d + i * stride_out_grad_n];
        float dAi_grad = Si_grad * Smi;
        float Smi_grad = Si_grad * dAi;
        float dBi_grad = Si_grad * f_x;
        f_x_grad += Si_grad * dBi;
        float wi_grad = Smi_grad * (nSl[i] - nSr[i]);
        atomicAdd(weight_grad + dim_idx * stride_w_grad_d + i * stride_w_grad_n, wi_grad);
        float Sli_grad = Smi_grad * nw[i];
        if (left_idx >= 0) {
            ssm_in_grad[left_idx * stride_in_grad_b + dim_idx * stride_in_grad_d + i * stride_in_grad_n] = Sli_grad;
        }
        float Sri_grad = Smi_grad * (1.f - nw[i]);
        if (right_idx >= 0) {
            ssm_in_grad[right_idx * stride_in_grad_b + dim_idx * stride_in_grad_d + i * stride_in_grad_n] = Sri_grad;
        }
        float Bi_grad = dBi_grad * dt2;
        atomicAdd(B_grad + batch_idx * stride_b_grad_b + seq_idx * stride_b_grad_s + i * stride_b_grad_n, Bi_grad);
        float dA1i_grad = dAi_grad * dAi;
        dt2_grad += dA1i_grad * nA[i] + dBi_grad * nB[i];
        float Ai_grad = dA1i_grad * dt2;
        atomicAdd(A_grad + dim_idx * stride_a_grad_d + i * stride_a_grad_n, Ai_grad);
    }
    x_grad[batch_idx * stride_x_grad_b + seq_idx * stride_x_grad_s + dim_idx * stride_x_grad_d] = f_x_grad;
    float dt1_grad = dt2_grad * softplus_backward(dt1);
    dt_grad[batch_idx * stride_dt_grad_b + seq_idx * stride_dt_grad_s + dim_idx * stride_dt_grad_d] = dt1_grad;
    atomicAdd(dt_bias_grad + dim_idx * stride_dt_bias_d, dt1_grad);
}

torch::Tensor ssm_step_bwd(
    torch::Tensor y_grad,
    torch::Tensor ssm_out_grad,
    torch::Tensor ssm_state,
    torch::Tensor x,
    torch::Tensor z,
    torch::Tensor dt,
    torch::Tensor dt_bias,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor weight,
    torch::Tensor indices,
    torch::Tensor states,

    torch::Tensor x_grad,
    torch::Tensor z_grad,
    torch::Tensor dt_grad,
    torch::Tensor dt_bias_grad,
    torch::Tensor A_grad,
    torch::Tensor B_grad,
    torch::Tensor C_grad,
    torch::Tensor D_grad,
    torch::Tensor weight_grad
) {
    const int batch_size = dt.size(0);
    const int seq_len = dt.size(1);
    const int d_inner = dt.size(2);
    const int d_state = A.size(1);
    const int level_batch_size = indices.size(0);

    auto ssm_in_grad = torch::zeros_like(ssm_state);

    const int threads = d_inner;
    const int blocks = level_batch_size;
    ssm_step_bwd_kernel<128, 16><<<blocks, threads>>>(
        y_grad.data<float>(),
        ssm_out_grad.data<float>(),
        ssm_state.data<float>(),
        x.data<float>(),
        z.data<float>(),
        dt.data<float>(),
        dt_bias.data<float>(),
        A.data<float>(),
        B.data<float>(),
        C.data<float>(),
        D.data<float>(),
        weight.data<float>(),
        indices.data<int>(),
        states.data<int>(),

        y_grad.stride(0), y_grad.stride(1), y_grad.stride(2),
        ssm_out_grad.stride(0), ssm_out_grad.stride(1), ssm_out_grad.stride(2),
        ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0),
        weight.stride(0), weight.stride(1),
        indices.stride(0), indices.stride(1),
        states.stride(0), states.stride(1),

        ssm_in_grad.data<float>(),
        x_grad.data<float>(),
        z_grad.data<float>(),
        dt_grad.data<float>(),
        dt_bias_grad.data<float>(),
        A_grad.data<float>(),
        B_grad.data<float>(),
        C_grad.data<float>(),
        D_grad.data<float>(),
        weight_grad.data<float>(),

        ssm_in_grad.stride(0), ssm_in_grad.stride(1), ssm_in_grad.stride(2),
        x_grad.stride(0), x_grad.stride(1), x_grad.stride(2),
        z_grad.stride(0), z_grad.stride(1), z_grad.stride(2),
        dt_grad.stride(0), dt_grad.stride(1), dt_grad.stride(2),
        dt_bias_grad.stride(0),
        A_grad.stride(0), A_grad.stride(1),
        B_grad.stride(0), B_grad.stride(1), B_grad.stride(2),
        C_grad.stride(0), C_grad.stride(1), C_grad.stride(2),
        D_grad.stride(0),
        weight_grad.stride(0), weight_grad.stride(1)
    );

    return ssm_in_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tree_conv_fwd", &tree_conv_fwd, "forward pass of tree convolution");
  m.def("tree_conv_bwd", &tree_conv_bwd, "backward pass of tree convolution");
  m.def("ssm_step_fwd", &ssm_step_fwd, "forward pass of ssm step");
  m.def("ssm_step_bwd", &ssm_step_bwd, "backward pass of ssm step");
}