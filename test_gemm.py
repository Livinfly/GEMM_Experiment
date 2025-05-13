# test_gemm.py
import unittest
import torch
import numpy as np
import os

try:
    import gemm_cuda_kernels
    EXTENSION_AVAILABLE = True
    print("Successfully imported 'gemm_cuda_kernels' C++ extension for testing.")
except ImportError as e:
    print(f"Failed to import 'gemm_cuda_kernels' for testing: {e}")
    print("Ensure the extension is built and installed (e.g., 'python setup.py install' or 'develop').")
    EXTENSION_AVAILABLE = False

skip_reason = "PyTorch C++ extension 'gemm_cuda_kernels' not built, CUDA not available, or functions not found."

def skip_if_unavailable(func_name_str):
    def decorator(test_func):
        if not EXTENSION_AVAILABLE or not torch.cuda.is_available():
            return unittest.skip(skip_reason)(test_func)
        if not hasattr(gemm_cuda_kernels, func_name_str):
            return unittest.skip(f"Function '{func_name_str}' not found in gemm_cuda_kernels. Skipping test.")(test_func)
        return test_func
    return decorator

class TestGemmCorrectness(unittest.TestCase):

    def _run_test(self, M, K, N, custom_func_name, custom_func, ref_func=torch.matmul, atol=0, rtol=0):
        """ Helper function to run a correctness test for GEMM """
        print(f"\nTesting {custom_func_name} with M={M}, K={K}, N={N}")
        try:
            A_torch = torch.randn(M, K, dtype=torch.float32, device='cuda')
            B_torch = torch.randn(K, N, dtype=torch.float32, device='cuda')
        except Exception as e:
            self.fail(f"Failed to create tensors on CUDA for {custom_func_name}: {e}")
            return

        print(f"  Calculating reference (torch.matmul) for {custom_func_name}...")
        # 用 torch.matmul 放 cuda 上算，应该是为了效率，存在小误差
        C_ref = ref_func(A_torch.to('cpu'), B_torch.to('cpu')).to('cuda')

        print(f"  Running custom function {custom_func_name}...")
        try:
            C_custom = custom_func(A_torch, B_torch)
            torch.cuda.synchronize()
        except Exception as e:
            self.fail(f"Custom function {custom_func_name} raised an exception: {e}")
            return

        self.assertEqual(C_custom.shape, (M, N), f"{custom_func_name} output shape mismatch")
        self.assertEqual(C_custom.device, C_ref.device, f"{custom_func_name} output device mismatch")
        self.assertEqual(C_custom.dtype, C_ref.dtype, f"{custom_func_name} output dtype mismatch")

        print(f"  Comparing values for {custom_func_name}...")
        all_close = torch.allclose(C_ref, C_custom, atol=atol, rtol=rtol)
        if not all_close:
            max_diff = (C_ref - C_custom).abs().max()
            print(f"    WARNING: Results for {custom_func_name} differ. Max diff: {max_diff}")

        self.assertTrue(
            all_close,
            f"{custom_func_name} results differ significantly from torch.matmul. Max diff: {(C_ref - C_custom).abs().max()}"
        )
        print(f"  {custom_func_name} results match torch.matmul (within tolerance).")

    # Tests for naive_gemm
    @skip_if_unavailable("naive_gemm")
    def test_naive_gemm_small_square(self):
        self._run_test(64, 64, 64, "naive_gemm", gemm_cuda_kernels.naive_gemm)

    @skip_if_unavailable("naive_gemm")
    def test_naive_gemm_large_square(self):
        self._run_test(256, 256, 256, "naive_gemm", gemm_cuda_kernels.naive_gemm)

    @skip_if_unavailable("naive_gemm")
    def test_naive_gemm_rect1(self):
        self._run_test(100, 200, 300, "naive_gemm", gemm_cuda_kernels.naive_gemm)

    @skip_if_unavailable("naive_gemm")
    def test_naive_gemm_rect2(self):
        self._run_test(300, 100, 200, "naive_gemm", gemm_cuda_kernels.naive_gemm)

    # Tests for tiled_16_gemm
    @skip_if_unavailable("tiled_16_gemm")
    def test_tiled_16_gemm_small_square(self):
        self._run_test(64, 64, 64, "tiled_16_gemm", gemm_cuda_kernels.tiled_16_gemm)

    @skip_if_unavailable("tiled_16_gemm")
    def test_tiled_16_gemm_large_square(self):
        self._run_test(256, 256, 256, "tiled_16_gemm", gemm_cuda_kernels.tiled_16_gemm)

    @skip_if_unavailable("tiled_16_gemm")
    def test_tiled_16_gemm_non_multiple(self):
        self._run_test(70, 90, 110, "tiled_16_gemm", gemm_cuda_kernels.tiled_16_gemm)

    @skip_if_unavailable("tiled_16_gemm")
    def test_tiled_16_gemm_very_small(self):
        self._run_test(5, 6, 7, "tiled_16_gemm", gemm_cuda_kernels.tiled_16_gemm)


    # Tests for tiled_op_gemm
    @skip_if_unavailable("tiled_op_gemm")
    def test_tiled_op_gemm_small_square(self):
        self._run_test(64, 64, 64, "tiled_op_gemm", gemm_cuda_kernels.tiled_op_gemm)

    @skip_if_unavailable("tiled_op_gemm")
    def test_tiled_op_gemm_large_square(self):
        self._run_test(256, 256, 256, "tiled_op_gemm", gemm_cuda_kernels.tiled_op_gemm)

    @skip_if_unavailable("tiled_op_gemm")
    def test_tiled_op_gemm_non_multiple(self):
        self._run_test(70, 90, 110, "tiled_op_gemm", gemm_cuda_kernels.tiled_op_gemm)

    @skip_if_unavailable("tiled_op_gemm")
    def test_tiled_op_gemm_very_small(self):
        self._run_test(5, 6, 7, "tiled_op_gemm", gemm_cuda_kernels.tiled_op_gemm)

if __name__ == '__main__':
    if not EXTENSION_AVAILABLE or not torch.cuda.is_available():
        print(f"Skipping tests: {skip_reason}")
    else:
        print("Running GEMM Correctness Tests using PyTorch C++ Extension...")
        unittest.main()