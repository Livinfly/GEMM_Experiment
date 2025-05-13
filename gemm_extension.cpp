// gemm_extension.cpp
#include <torch/extension.h>
#include <vector>
#include "gemm_kernels.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

torch::Tensor naive_gemm_pytorch(
	torch::Tensor A,
	torch::Tensor B
) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	TORCH_CHECK(A.dim() == 2, "INPUT A must be 2-dimensional (M, K)");
	TORCH_CHECK(B.dim() == 2, "INPUT B must be 2-dimensional (K, N)");
	TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions K must match");

	int M = A.size(0), K = A.size(1), N = B.size(1);
	
	auto C = torch::empty({M, N}, A.options());

	const float *A_ptr = A.data_ptr<float>();
	const float *B_ptr = B.data_ptr<float>();
	float *C_ptr = C.data_ptr<float>();

	// cudaStream_t stream = torch::cuda::current_stream();  // 不知道为什么不能用
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index());
	
	launch_naive_gemm_kernel(A_ptr,	B_ptr, C_ptr, M, K, N, stream);

	return C;
}

torch::Tensor tiled_16_gemm_pytorch(
	torch::Tensor A,
	torch::Tensor B
) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	TORCH_CHECK(A.dim() == 2, "INPUT A must be 2-dimensional (M, K)");
	TORCH_CHECK(B.dim() == 2, "INPUT B must be 2-dimensional (K, N)");
	TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions K must match");

	int M = A.size(0), K = A.size(1), N = B.size(1);
	
	auto C = torch::empty({M, N}, A.options());

	const float *A_ptr = A.data_ptr<float>();
	const float *B_ptr = B.data_ptr<float>();
	float *C_ptr = C.data_ptr<float>();

	// cudaStream_t stream = torch::cuda::current_stream();  // 不知道为什么不能用
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index());
	

	launch_tiled_16_gemm_kernel(A_ptr,	B_ptr, C_ptr, M, K, N, stream);

	return C;
}

torch::Tensor tiled_op_gemm_pytorch(
	torch::Tensor A,
	torch::Tensor B
) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	TORCH_CHECK(A.dim() == 2, "INPUT A must be 2-dimensional (M, K)");
	TORCH_CHECK(B.dim() == 2, "INPUT B must be 2-dimensional (K, N)");
	TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions K must match");

	int M = A.size(0), K = A.size(1), N = B.size(1);
	
	auto C = torch::empty({M, N}, A.options());

	const float *A_ptr = A.data_ptr<float>();
	const float *B_ptr = B.data_ptr<float>();
	float *C_ptr = C.data_ptr<float>();

	// cudaStream_t stream = torch::cuda::current_stream();  // 不知道为什么不能用
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index());
	

	launch_tiled_op_gemm_kernel(A_ptr,	B_ptr, C_ptr, M, K, N, stream);

	return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("naive_gemm", &naive_gemm_pytorch, "Naive Matrix Multiplication (CUDA C++)");	
	m.def("tiled_16_gemm", &tiled_16_gemm_pytorch, "Tiled-16 Matrix Multiplication (CUDA C++)");
	m.def("tiled_op_gemm", &tiled_op_gemm_pytorch, "Tiled-op Matrix Multiplication (CUDA C++)");
}
