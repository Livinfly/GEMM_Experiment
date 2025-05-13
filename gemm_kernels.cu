// gemm_kernels.cu
#include "gemm_kernels.h"
#include <cmath>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void matmul_naive_kernel(
	const float *A,
	const float *B,
	float *C,
	int M,
	int K,
	int N
) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < M && col < N) {
		float accum = 0.0f;
		const float *a_row_ptr = A + row * K;  // A[row, 0]
		const float *b_col_ptr = B + col;	   // B[0, col]
		float *c_ptr = C + row * N + col;
		for (int k_idx = 0; k_idx < K; k_idx++) {
			accum += a_row_ptr[k_idx] * b_col_ptr[k_idx * N];
		}
		*c_ptr = accum;
	}
}

__global__ void matmul_tiled_16_kernel(
	const float *A,
	const float *B,
	float *C,
	int M,
	int K,
	int N
) {
	#define TILE_SIZE 16
	__shared__ float a_shared[TILE_SIZE][TILE_SIZE];
	__shared__ float b_shared[TILE_SIZE][TILE_SIZE];
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x,  by = blockIdx.y;
	int row = by * TILE_SIZE + ty;
	int col = bx * TILE_SIZE + tx;
	
	float accum = 0.0f;

	for (int idx = 0; idx < K; idx += TILE_SIZE) {
		int k = idx + tx;
		if (row < M && k < K) {
			a_shared[ty][tx] = A[row * K + k];
		}
		else {
			a_shared[ty][tx] = 0.0f;
		}
		k = idx + ty;
		if (k < K && col < N) {
			b_shared[ty][tx] = B[k * N + col];
		}
		else {
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

__global__ void matmul_tiled_op_kernel(
	const float * __restrict__ A,
	const float * __restrict__ B,
	float *C,
	int M,
	int K,
	int N
) {
	#define TILE_SIZE 16
	__shared__ float a_shared[TILE_SIZE][TILE_SIZE];
	__shared__ float b_shared[TILE_SIZE][TILE_SIZE];
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x,  by = blockIdx.y;
	int row = by * TILE_SIZE + ty;
	int col = bx * TILE_SIZE + tx;
	
	float accum = 0.0f;

	for (int idx = 0; idx < K; idx += TILE_SIZE) {
		int k = idx + tx;
		if (row < M && k < K) {
			a_shared[ty][tx] = A[row * K + k];
		}
		else {
			a_shared[ty][tx] = 0.0f;
		}
		k = idx + ty;
		if (k < K && col < N) {
			b_shared[ty][tx] = B[k * N + col];
		}
		else {
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

void launch_naive_gemm_kernel(
	const float *A,
	const float *B,
	float *C,
	int M,
	int K,
	int N,
	cudaStream_t stream
) {
	dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(M + BLOCK_SIZE - 1) / BLOCK_SIZE
	);
	matmul_naive_kernel<<<numBlocks, threadPerBlock, 0, stream>>>(A, B, C, M, K, N);
}

void launch_tiled_16_gemm_kernel(
	const float *A,
	const float *B,
	float *C,
	int M,
	int K,
	int N,
	cudaStream_t stream
) {
	#define TILE_SIZE 16
	dim3 threadPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 numBlocks(
		(N + TILE_SIZE - 1) / TILE_SIZE,
		(M + TILE_SIZE - 1) / TILE_SIZE
	);
	matmul_tiled_16_kernel<<<numBlocks, threadPerBlock, 0, stream>>>(A, B, C, M, K, N);
	#undef TILE_SIZE
}

void launch_tiled_op_gemm_kernel(
	const float * __restrict__ A,
	const float * __restrict__ B,
	float *C,
	int M,
	int K,
	int N,
	cudaStream_t stream
) {
	#define TILE_SIZE 16
	dim3 threadPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 numBlocks(
		(N + TILE_SIZE - 1) / TILE_SIZE,
		(M + TILE_SIZE - 1) / TILE_SIZE
	);
	matmul_tiled_op_kernel<<<numBlocks, threadPerBlock, 0, stream>>>(A, B, C, M, K, N);
	#undef TILE_SIZE
}