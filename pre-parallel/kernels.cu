/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *
 *  Non-GEMM CUDA kernels (bias-broadcast, activations, ReLU mask,
 *  bias-reduce, loss reduce). The three matmul-shaped operations live
 *  in nnp.cu as cuBLAS sgemm calls.
 *
 *  Layout: activations row-major [batch, dim], indexed t[n*dim + j].
 */

#include <cuda.h>
#include <math.h>
#include "kernels.h"

// In-place bias-add + ReLU. Grid: (ceil(dim/threads), batch).
__global__ void bias_relu_batched(float* x, const float* b, int dim, int batch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (j >= dim || n >= batch) return;
    int idx = n * dim + j;
    float v = x[idx] + b[j];
    x[idx] = v > 0.0f ? v : 0.0f;
}

// In-place bias-add + per-row softmax. Grid: (batch). Block: dim threads.
// Shared mem: dim floats. dim is small (CLASSES=10), so the per-row
// max/exp/normalize reductions stay simple.
__global__ void bias_softmax_batched(float* x, const float* b, int dim, int batch) {
    extern __shared__ float s[];
    int j = threadIdx.x;
    int n = blockIdx.x;
    if (n >= batch) return;

    if (j < dim) s[j] = x[n * dim + j] + b[j];
    __syncthreads();

    if (j == 0) {
        float max = s[0];
        for (int k = 1; k < dim; k++) if (s[k] > max) max = s[k];
        float total = 0.0f;
        for (int k = 0; k < dim; k++) {
            s[k] = expf(s[k] - max);
            total += s[k];
        }
        for (int k = 0; k < dim; k++) s[k] /= total;
    }
    __syncthreads();

    if (j < dim) x[n * dim + j] = s[j];
}

__global__ void compute_delta3_batched(const float* outa, const float* label,
                                       float* delta, int len, int batch) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (k >= len || n >= batch) return;
    int idx = n * len + k;
    delta[idx] = label[idx] - outa[idx];
}

__global__ void relu_mask_batched(float* delta, const float* h_act, int dim, int batch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (j >= dim || n >= batch) return;
    int idx = n * dim + j;
    if (h_act[idx] <= 0.0f) delta[idx] = 0.0f;
}

__global__ void bias_update_batched(float* b, const float* delta,
                                    int dim, int batch, float lr_scaled) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;
    float acc = 0.0f;
    for (int n = 0; n < batch; n++) acc += delta[n * dim + k];
    b[k] += lr_scaled * acc;
}

__global__ void accumulate_loss_batched(const float* outa, const float* label,
                                        float* loss_accum, int len, int batch) {
    extern __shared__ float s[];
    int k = threadIdx.x;
    int n = blockIdx.x;
    if (n >= batch) return;
    int idx = n * len + k;
    s[k] = (k < len) ? -label[idx] * logf(outa[idx] + 1e-8f) : 0.0f;
    __syncthreads();
    if (k == 0) {
        float total = 0.0f;
        for (int i = 0; i < len; i++) total += s[i];
        atomicAdd(loss_accum, total);
    }
}
