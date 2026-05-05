/*
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *
 *  Header file for CUDA kernel functions
*/

#ifndef KERNELS_H
#define KERNELS_H

__global__ void matvec_relu(const float* in, const float* W, const float* b,
                            float* out, int in_dim, int out_dim);

__global__ void matvec_softmax(const float* in, const float* W, const float* b,
                               float* out, int in_dim, int out_dim);

__global__ void compute_delta3(const float* outa, const float* label,
                               float* delta, int len);

__global__ void compute_delta_hidden(const float* W_next, const float* delta_next,
                                     const float* h_act, float* delta_out,
                                     int dim, int next_dim);

__global__ void weight_update(float* W, const float* input, const float* delta,
                              int in_dim, int out_dim, float lr);

__global__ void bias_update(float* b, const float* delta, int dim, float lr);

__global__ void accumulate_loss(const float* outa, const float* label,
                                float* loss_accum, int len);

#endif
