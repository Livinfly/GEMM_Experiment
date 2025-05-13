// gemm_kernels.h
#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H
// 防止重复声明

#include <cuda_runtime.h>


void launch_naive_gemm_kernel(
	const float *A,
	const float *B,
	float *C,
	int M,
	int K,
	int N,
	cudaStream_t stream  // 为了集成到pytorch中，传入pytorch中的cuda流控制 
);

void launch_tiled_16_gemm_kernel(
	const float *A,
	const float *B,
	float *C,
	int M,
	int K,
	int N,
	cudaStream_t stream	
);

void launch_tiled_op_gemm_kernel(
	const float * __restrict__ A,
	const float * __restrict__ B,
	float *C,
	int M,
	int K,
	int N,
	cudaStream_t stream	
);

#endif
