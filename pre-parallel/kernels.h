/*
 * kernels.h
 *
 *  Worked in a pair: Nathaniel Black and Marcos Diaz Vazquez
 *
 *  Created on: Nov 9, 2025
 *
 *  Header file for CUDA kernel functions.
 *
 *  All kernels operate on mini-batches. Activation tensors are row-major,
 *  sample-major: shape [batch, dim], indexed as t[n*dim + j]. Weights are
 *  row-major shape [in_dim, out_dim], indexed as W[i*out_dim + j].
 *
 *  Matmul-shaped operations (forward, delta @ W^T, In^T @ delta weight
 *  gradient) are done via cuBLAS in nnp.cu. The kernels here cover the
 *  remaining bias-broadcast, activation, ReLU-mask, bias-reduce, and loss
 *  reduction steps.
*/

#ifndef KERNELS_H
#define KERNELS_H

// In-place: x[n,j] = relu(x[n,j] + b[j]).
__global__ void bias_relu_batched(float* x, const float* b, int dim, int batch);

// In-place: x[n,j] = softmax_row(x[n,j] + b[j]). One block per sample.
__global__ void bias_softmax_batched(float* x, const float* b, int dim, int batch);

// Output-layer delta = label - outa.
__global__ void compute_delta3_batched(const float* outa, const float* label,
                                       float* delta, int len, int batch);

// In-place: delta[n,j] *= (h_act[n,j] > 0) (ReLU derivative mask).
__global__ void relu_mask_batched(float* delta, const float* h_act, int dim, int batch);

// b += lr_scaled * sum_n delta[n, :].
__global__ void bias_update_batched(float* b, const float* delta,
                                    int dim, int batch, float lr_scaled);

// loss_accum += sum over the batch of CE(outa[n,:], label[n,:]).
__global__ void accumulate_loss_batched(const float* outa, const float* label,
                                        float* loss_accum, int len, int batch);

#endif
