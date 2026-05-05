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

#endif
