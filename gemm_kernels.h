// gemm_kernels.h
#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H
// 防止重复声明

// #include <cuda_fp16.h>
#include <cuda_runtime.h>

// nvcc -O 编译，下面这种形式的格式化，加上注释 字母 + 文字
// 会报错，可能是windows的问题 void launch_naive_gemm_kernel(
//     const float *A, const float *B, float *C, int M,int K, int N,
//     cudaStream_t stream  // a就
// );

void launch_naive_gemm_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N, cudaStream_t stream);

void launch_tiled_16_gemm_kernel(const float *__restrict__A,
                                 const float *__restrict__B, float *C, int M,
                                 int K, int N, cudaStream_t stream);

void launch_tiled_op_gemm_kernel(const float *__restrict__ A,
                                 const float *__restrict__ B, float *C, int M,
                                 int K, int N, cudaStream_t stream);

#endif
