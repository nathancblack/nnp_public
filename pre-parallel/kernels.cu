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
