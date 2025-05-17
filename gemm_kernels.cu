// gemm_kernels.cu
#include <cuda_fp16.h>

#include <cmath>

#include "gemm_kernels.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifndef INNER_TILE_SIZE_1
#define INNER_TILE_SIZE_1 8
#endif

#ifndef INNER_TILE_SIZE_f
#define INNER_TILE_SIZE_f 2
#endif

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float accum = 0.0f;
        const float *a_row_ptr = A + row * K;  // A[row, 0]
        const float *b_col_ptr = B + col;      // B[0, col]
        float *c_ptr = C + row * N + col;

        for (int k_idx = 0; k_idx < K; k_idx++) {
            accum += a_row_ptr[k_idx] * b_col_ptr[k_idx * N];
        }
        *c_ptr = accum;
    }
}

__global__ void matmul_tiled_16_kernel(const float *A, const float *B, float *C,
                                       int M, int K, int N) {
#define TILE_SIZE 16
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float accum = 0.0f;
    for (int idx = 0; idx < K; idx += TILE_SIZE) {
        int k = idx + tx;
        if (row < M && k < K) {
            a_shared[ty][tx] = A[row * K + k];
        } else {
            a_shared[ty][tx] = 0.0f;
        }
        k = idx + ty;
        if (k < K && col < N) {
            b_shared[ty][tx] = B[k * N + col];
        } else {
            b_shared[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int tk = 0; tk < TILE_SIZE; tk++) {
            accum += a_shared[ty][tk] * b_shared[tk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
#undef TILE_SIZE
}

// 1x2 => 2.5x
// __global__ void matmul_tiled_op_kernel(const float *__restrict__ A,
//                                        const float *__restrict__ B, float *C,
//                                        int M, int K, int N) {
// #define TILE_SIZE 32
//     __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
//     __shared__ float b_shared[TILE_SIZE][TILE_SIZE];
//     int tx_pair = threadIdx.x * INNER_TILE_SIZE_1, ty = threadIdx.y;
//     int bx = blockIdx.x, by = blockIdx.y;
//     int row = by * TILE_SIZE + ty;
//     int col_base = bx * TILE_SIZE + tx_pair;

//     float accum[INNER_TILE_SIZE_1] = {0.0f, 0.0f};
// #pragma unroll
//     for (int idx = 0; idx < K; idx += TILE_SIZE) {
//         int k = idx + tx_pair;
// #pragma unroll
//         for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//             if (row < M && k + i < K) {
//                 a_shared[ty][tx_pair + i] = A[row * K + k + i];
//             } else {
//                 a_shared[ty][tx_pair + i] = 0.0f;
//             }
//         }
//         k = idx + ty;
// #pragma unroll
//         for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//             if (k < K && col_base + i < N) {
//                 b_shared[ty][tx_pair + i] = B[k * N + col_base + i];
//             } else {
//                 b_shared[ty][tx_pair + i] = 0.0f;
//             }
//         }
//         __syncthreads();
// #pragma unroll
//         for (int tk = 0; tk < TILE_SIZE; tk++) {
//             float val_a = a_shared[ty][tk];
// #pragma unroll
//             for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//                 accum[i] += val_a * b_shared[tk][tx_pair + i];
//             }
//         }
//         __syncthreads();
//     }
//     if (row < M) {
// #pragma unroll
//         for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//             if (col_base + i < N) {
//                 C[row * N + col_base + i] = accum[i];
//             }
//         }
//     }
// #undef TILE_SIZE
// }

// 1x2 => 2.2 ~ 2.5x  256 (TILE_SIZE 32)
// 2x2 => 3.2 ~ 3.5x
// 4x2 => 4.0 ~ 4.4x
// 8x2 => 4.8 ~ 4.9x

// 8x2 => 4.5 ~ 4.8x  256   (TILE_SIZE 64)
// 8x2 => 6.1x        1024
__global__ void matmul_tiled_op_kernel(const float *__restrict__ A,
                                       const float *__restrict__ B, float *C,
                                       int M, int K, int N) {
#define TILE_SIZE 64
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];
    int tx_f = threadIdx.x * INNER_TILE_SIZE_f,
        ty_pair = threadIdx.y * INNER_TILE_SIZE_1;
    int bx = blockIdx.x, by = blockIdx.y;
    int row_base = by * TILE_SIZE + ty_pair;
    int col_base = bx * TILE_SIZE + tx_f;
    float2 accum[INNER_TILE_SIZE_1];
#pragma unroll
    for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
        accum[i] = make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int idx = 0; idx < K; idx += TILE_SIZE) {
        int k = idx + tx_f;
#pragma unroll
        for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
#pragma unroll
            for (int j = 0; j < INNER_TILE_SIZE_f; j++) {
                if (row_base + i < M && k + j < K) {
                    a_shared[ty_pair + i][tx_f + j] =
                        A[(row_base + i) * K + k + j];
                } else {
                    a_shared[ty_pair + i][tx_f + j] = 0.0f;
                }
            }
        }
        k = idx + ty_pair;
#pragma unroll
        for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
#pragma unroll
            for (int j = 0; j < INNER_TILE_SIZE_f; j++) {
                if (k + i < K && col_base + j < N) {
                    b_shared[ty_pair + i][tx_f + j] =
                        B[(k + i) * N + col_base + j];
                } else {
                    b_shared[ty_pair + i][tx_f + j] = 0.0f;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int tk = 0; tk < TILE_SIZE; tk++) {
#pragma unroll
            for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
                float val_a = a_shared[ty_pair + i][tk];
                accum[i].x += val_a * b_shared[tk][tx_f];
                accum[i].y += val_a * b_shared[tk][tx_f + 1];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
        if (row_base + i < M) {
            if (col_base < N) {
                C[(row_base + i) * N + col_base] = accum[i].x;
            }
            if (col_base + 1 < N) {
                C[(row_base + i) * N + col_base + 1] = accum[i].y;
            }
        }
    }
#undef TILE_SIZE
}

// 1x4 => 2.0~2.2x  256
// 2x4 => 2.6~2.9x
// 4x4 => 3.2~3.5x
// __global__ void matmul_tiled_op_kernel(const float *__restrict__ A,
//                                        const float *__restrict__ B, float *C,
//                                        int M, int K, int N) {
// #define TILE_SIZE 32
//     __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
//     __shared__ float b_shared[TILE_SIZE][TILE_SIZE];
//     int tx_f = threadIdx.x * INNER_TILE_SIZE_f,
//         ty_pair = threadIdx.y * INNER_TILE_SIZE_1;
//     int bx = blockIdx.x, by = blockIdx.y;
//     int row_base = by * TILE_SIZE + ty_pair;
//     int col_base = bx * TILE_SIZE + tx_f;

//     float4 accum[INNER_TILE_SIZE_1];
// #pragma unroll
//     for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//         accum[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
//     }
// #pragma unroll
//     for (int idx = 0; idx < K; idx += TILE_SIZE) {
//         int k = idx + tx_f;
// #pragma unroll
//         for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
// #pragma unroll
//             for (int j = 0; j < INNER_TILE_SIZE_f; j++) {
//                 if (row_base + i < M && k + j < K) {
//                     a_shared[ty_pair + i][tx_f + j] =
//                         A[(row_base + i) * K + k + j];
//                 } else {
//                     a_shared[ty_pair + i][tx_f + j] = 0.0f;
//                 }
//             }
//         }
//         k = idx + ty_pair;
// #pragma unroll
//         for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
// #pragma unroll
//             for (int j = 0; j < INNER_TILE_SIZE_f; j++) {
//                 if (k + i < K && col_base + j < N) {
//                     b_shared[ty_pair + i][tx_f + j] =
//                         B[(k + i) * N + col_base + j];
//                 } else {
//                     b_shared[ty_pair + i][tx_f + j] = 0.0f;
//                 }
//             }
//         }
//         __syncthreads();
// #pragma unroll
//         for (int tk = 0; tk < TILE_SIZE; tk++) {
// #pragma unroll
//             for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//                 float val_a = a_shared[ty_pair + i][tk];
//                 accum[i].x += val_a * b_shared[tk][tx_f];
//                 accum[i].y += val_a * b_shared[tk][tx_f + 1];
//                 accum[i].z += val_a * b_shared[tk][tx_f + 2];
//                 accum[i].w += val_a * b_shared[tk][tx_f + 3];
//             }
//         }
//         __syncthreads();
//     }
// #pragma unroll
//     for (int i = 0; i < INNER_TILE_SIZE_1; i++) {
//         if (row_base + i < M) {
//             if (col_base < N) {
//                 C[(row_base + i) * N + col_base] = accum[i].x;
//             }
//             if (col_base + 1 < N) {
//                 C[(row_base + i) * N + col_base + 1] = accum[i].y;
//             }
//             if (col_base + 2 < N) {
//                 C[(row_base + i) * N + col_base + 2] = accum[i].z;
//             }
//             if (col_base + 3 < N) {
//                 C[(row_base + i) * N + col_base + 3] = accum[i].w;
//             }
//         }
//     }
// #undef TILE_SIZE
// }

void launch_naive_gemm_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N, cudaStream_t stream) {
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_naive_kernel<<<numBlocks, threadPerBlock, 0, stream>>>(A, B, C, M, K,
                                                                  N);
}

void launch_tiled_16_gemm_kernel(const float *A, const float *B, float *C,
                                 int M, int K, int N, cudaStream_t stream) {
#define TILE_SIZE 16
    dim3 threadPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_16_kernel<<<numBlocks, threadPerBlock, 0, stream>>>(A, B, C, M,
                                                                     K, N);
#undef TILE_SIZE
}

void launch_tiled_op_gemm_kernel(const float *__restrict__ A,
                                 const float *__restrict__ B, float *C, int M,
                                 int K, int N, cudaStream_t stream) {
#define TILE_SIZE 64
    dim3 threadPerBlock(TILE_SIZE / INNER_TILE_SIZE_f,
                        TILE_SIZE / INNER_TILE_SIZE_1);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_op_kernel<<<numBlocks, threadPerBlock, 0, stream>>>(A, B, C, M,
                                                                     K, N);
#undef TILE_SIZE
}