/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *
 *  CUDA kernels for the forward pass of the neural network.
 */

#include <cuda.h>
#include <math.h>
#include "kernels.h"

__global__ void matvec_relu(const float* in, const float* W, const float* b,
                            float* out, int in_dim, int out_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    float sum = b[j];
    for (int i = 0; i < in_dim; i++) {
        sum += in[i] * W[i * out_dim + j];
    }
    out[j] = sum > 0.0f ? sum : 0.0f;
}

__global__ void matvec_softmax(const float* in, const float* W, const float* b,
                               float* out, int in_dim, int out_dim) {
    extern __shared__ float s[];
    int j = threadIdx.x;
    if (j < out_dim) {
        float sum = b[j];
        for (int i = 0; i < in_dim; i++) {
            sum += in[i] * W[i * out_dim + j];
        }
        s[j] = sum;
    }
    __syncthreads();

    if (j == 0) {
        float max = s[0];
        for (int k = 1; k < out_dim; k++) if (s[k] > max) max = s[k];
        float total = 0.0f;
        for (int k = 0; k < out_dim; k++) {
            s[k] = expf(s[k] - max);
            total += s[k];
        }
        for (int k = 0; k < out_dim; k++) s[k] /= total;
    }
    __syncthreads();

    if (j < out_dim) out[j] = s[j];
}

__global__ void compute_delta3(const float* outa, const float* label,
                               float* delta, int len) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= len) return;
    delta[k] = label[k] - outa[k];
}

__global__ void compute_delta_hidden(const float* W_next, const float* delta_next,
                                     const float* h_act, float* delta_out,
                                     int dim, int next_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    float err = 0.0f;
    for (int k = 0; k < next_dim; k++) {
        err += delta_next[k] * W_next[j * next_dim + k];
    }
    delta_out[j] = h_act[j] > 0.0f ? err : 0.0f;
}

__global__ void weight_update(float* W, const float* input, const float* delta,
                              int in_dim, int out_dim, float lr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= in_dim || j >= out_dim) return;
    W[i * out_dim + j] += lr * input[i] * delta[j];
}

__global__ void bias_update(float* b, const float* delta, int dim, float lr) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;
    b[k] += lr * delta[k];
}

__global__ void accumulate_loss(const float* outa, const float* label,
                                float* loss_accum, int len) {
    extern __shared__ float s[];
    int k = threadIdx.x;
    s[k] = (k < len) ? -label[k] * logf(outa[k] + 1e-8f) : 0.0f;
    __syncthreads();
    if (k == 0) {
        float total = 0.0f;
        for (int i = 0; i < len; i++) total += s[i];
        atomicAdd(loss_accum, total);
    }
}
