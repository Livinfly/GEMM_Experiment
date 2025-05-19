# benchmark_pytorch.py
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils import CustomLinearGEMM, create_mlp, benchmark_gemm_op, benchmark_model

try:
	import gemm_cuda_kernels
	EXTENSION_AVAILABLE = True
	print("Successfully import 'gemm_cuda_kernels' C++ extension.")
except ImportError as e:
	print(f"Failed to import 'gemm_cuda_kernels': {e}")
	print(f"Build the extension using 'python setup.py install.")
	EXTENSION_AVAILABLE = False

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run GEMM and MLP PyTorch C++ extension benchmarks")

	parser.add_argument('--mode', type=str, default='all', choices=['gemm', 'mlp', 'all'], help="Benchmark mode: 'gemm' for raw ops, 'mlp' for MLP models, 'all' for both.")
	
	# argument for GEMM
	parser.add_argument('--size', type=int, default=256, help="Matrix size for GEMM (M=K=N if others not set)")
	parser.add_argument("--M", type=int, default=0, help="M dimension for GEMM (overrides size if > 0)")
	parser.add_argument("--K", type=int, default=0, help="K dimension for GEMM (overrides size if > 0)")
	parser.add_argument("--N", type=int, default=0, help="N dimension for GEMM (overrides size if > 0)")
	
	# argument for MLP
	parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for MLP input")
	parser.add_argument("--input_dim", type=int, default=1024, help="MLP input dimension")
	parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 256, 64], help="MLP hidden dimensions list")
	parser.add_argument("--output_dim", type=int, default=10, help="MLP output dimension")
	
	# argument for all
	parser.add_argument("--runs", type=int, default=5, help="Number of timing runs")
	parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")

	args = parser.parse_args()

	if not EXTENSION_AVAILABLE:
		print("Error: C++ extension not loaded. Please build first.")
		exit()
	if not torch.cuda.is_available():
		print("Error: CUDA device not available.")
		exit()
	
	gemm_results = {}
	mlp_results = {}

	# GEMM op benchmark
	if args.mode == 'gemm' or args.mode == 'all':
		M = args.M if args.M > 0 else args.size
		K = args.K if args.K > 0 else args.size
		N = args.N if args.N > 0 else args.size

		print(f"\n===== Benchmarking Raw GEMM Operations =====")
		print(f"Dimensions: M={M}, K={K}, N={N}")

		try:
			A_torch = torch.randn(M, K, dtype=torch.float32, device='cuda')
			B_torch = torch.randn(K, N, dtype=torch.float32, device='cuda')
			print("  Created GEMM tensors on GPU.")
		except Exception as e:
			print(f"  Error creating GEMM tensors: {e}")
			exit()

		gemm_results['pytorch'] = benchmark_gemm_op("torch.matmul", torch.matmul, A_torch, B_torch, args.warmup, args.runs)
		gemm_results['naive_cuda'] = benchmark_gemm_op("my_naive_gemm", gemm_cuda_kernels.naive_gemm, A_torch, B_torch, args.warmup, args.runs)
		gemm_results['tiled_16_cuda'] = benchmark_gemm_op("my_tiled_16_gemm", gemm_cuda_kernels.tiled_16_gemm, A_torch, B_torch, args.warmup, args.runs)
		gemm_results['tiled_op_cuda'] = benchmark_gemm_op("my_tiled_op_gemm", gemm_cuda_kernels.tiled_op_gemm, A_torch, B_torch, args.warmup, args.runs)
	
	# MLP model benchmark
	if args.mode == 'mlp' or args.mode == 'all':
		print(f"\n===== Benchmarking MLP Models =====")
		print(f"Batch={args.batch_size}, Inp={args.input_dim}, Hidden={args.hidden_dims}, Out={args.output_dim}")
		mlp_input_tensor = torch.randn(args.batch_size, args.input_dim, dtype=torch.float32)

		# PyTorch Linear MLP
		# model_torch = create_mlp(args.input_dim, args.hidden_dims, args.output_dim)
		# mlp_results['pytorch_linear_mlp'] = benchmark_model("PyTorch Linear MLP", model_torch, mlp_input_tensor, args.warmup, args.runs)

		# Custom Naive MLP
		# try:
		# 	model_custom_naive = create_mlp(args.input_dim, args.hidden_dims, args.output_dim,
		# 									use_custom_linear=True, custom_gemm_func=gemm_cuda_kernels.naive_gemm)
		# 	mlp_results['custom_naive_mlp'] = benchmark_model("Custom Naive MLP", model_custom_naive, mlp_input_tensor, args.warmup, args.runs)
		# except Exception as e:
		# 	print(f"Error setting up/benchmarking Custom Naive MLP: {e}")
		# 	mlp_results['custom_naive_mlp'] = float('nan')

		# Custom Tiled_16 MLP
		# try:
		# 	model_custom_tiled_16 = create_mlp(args.input_dim, args.hidden_dims, args.output_dim,
		# 									use_custom_linear=True, custom_gemm_func=gemm_cuda_kernels.tiled_16_gemm)
		# 	mlp_results['custom_tiled_16_mlp'] = benchmark_model("Custom Tiled_16 MLP", model_custom_tiled_16, mlp_input_tensor, args.warmup, args.runs)
		# except Exception as e:
		# 	print(f"Error setting up/benchmarking Custom Tiled_16 MLP: {e}")
		# 	mlp_results['custom_tiled_16_mlp'] = float('nan')

		# Custom Tiled_op MLP
		try:
			model_custom_tiled_op = create_mlp(args.input_dim, args.hidden_dims, args.output_dim,
											use_custom_linear=True, custom_gemm_func=gemm_cuda_kernels.tiled_op_gemm)
			mlp_results['custom_tiled_op_mlp'] = benchmark_model("Custom Tiled_op MLP", model_custom_tiled_op, mlp_input_tensor, args.warmup, args.runs)
		except Exception as e:
			print(f"Error setting up/benchmarking Custom Tiled_op MLP: {e}")
			mlp_results['custom_tiled_op_mlp'] = float('nan')
		
	# Summary
	if args.mode == 'gemm' or args.mode == 'all':
		print("\n\n--- GEMM Operation Benchmark Summary (Avg Time ms, baseline Naive GEMM) ---")
		M = args.M if args.M > 0 else args.size
		K = args.K if args.K > 0 else args.size
		N = args.N if args.N > 0 else args.size
		header = "| M      | K    | N    | torch.matmul | naive_cuda | tiled_16_cuda | tiled_op_cuda | Tiled_16 | Tiled_op | Torch |"
		print("-" * len(header))
		print(header)
		print("-" * len(header))

		torch_time = gemm_results.get('pytorch', float('nan'))
		naive_time = gemm_results.get('naive_cuda', float('nan'))
		tiled_16_time = gemm_results.get('tiled_16_cuda', float('nan'))
		tiled_op_time = gemm_results.get('tiled_op_cuda', float('nan'))

		tiled_16_vs_naive = naive_time / tiled_16_time if not np.isnan(tiled_16_time) and tiled_16_time > 0 and not np.isnan(naive_time) and naive_time > 0 else float('nan')
		tiled_op_vs_naive = naive_time / tiled_op_time if not np.isnan(tiled_op_time) and tiled_op_time > 0 and not np.isnan(naive_time) and naive_time > 0 else float('nan')
		torch_vs_naive = naive_time / torch_time if not np.isnan(torch_time) and torch_time > 0 and not np.isnan(naive_time) and naive_time > 0 else float('nan')

		flops = 2 * M * K * N
		gflops_torch = (flops / (torch_time / 1000)) / 1e9 if not np.isnan(torch_time) and torch_time > 0 else 0
		gflops_naive = (flops / (naive_time / 1000)) / 1e9 if not np.isnan(naive_time) and naive_time > 0 else 0
		gflops_tiled_16 = (flops / (tiled_16_time / 1000)) / 1e9 if not np.isnan(tiled_16_time) and tiled_16_time > 0 else 0
		gflops_tiled_op = (flops / (tiled_op_time / 1000)) / 1e9 if not np.isnan(tiled_op_time) and tiled_op_time > 0 else 0

		print(f"| {M:<6} | {K:<4} | {N:<4} | {torch_time:>12.4f} | {naive_time:>10.4f} | {tiled_16_time:>13.4f} | {tiled_op_time:>13.4f} | {tiled_16_vs_naive:>7.2f}x | {tiled_op_vs_naive:>7.2f}x | {torch_vs_naive:>4.2f}x |")
		print(f"| GFLOPs |      |      | {gflops_torch:>12.1f} | {gflops_naive:>10.1f} | {gflops_tiled_16:>13.1f} | {gflops_tiled_op:>13.1f} | {'-':>8} | {'-':>8} | {'-':>5} |")
		print("-" * len(header))

	if args.mode == 'mlp' or args.mode == 'all':
		print("\n\n--- MLP End-to-End Forward Benchmark Summary (Avg Time ms) ---")
		header_mlp = "| Model Type            | Avg Time (ms) | Speedup vs Custom Naive MLP |"
		print("-" * len(header_mlp))
		print(header_mlp)
		print("-" * len(header_mlp))

		torch_mlp_time = mlp_results.get('pytorch_linear_mlp', float('nan'))
		naive_mlp_time = mlp_results.get('custom_naive_mlp', float('nan'))
		tiled_16_mlp_time = mlp_results.get('custom_tiled_16_mlp', float('nan'))
		tiled_op_mlp_time = mlp_results.get('custom_tiled_op_mlp', float('nan'))

		print(f"| Custom Naive MLP      | {naive_mlp_time:>13.4f} | {'(Baseline) 1.00x':>27} |")

		if not np.isnan(tiled_16_mlp_time) and tiled_16_mlp_time > 0 and not np.isnan(naive_mlp_time) and naive_mlp_time > 0:
			print(f"| Custom Tiled_16 MLP   | {tiled_16_mlp_time:>13.4f} | {naive_mlp_time / tiled_16_mlp_time:>26.2f}x |")
		else:
			print(f"| Custom Tiled_16 MLP   | {tiled_16_mlp_time:>13.4f} | {'N/A':>27} |")
		
		if not np.isnan(tiled_op_mlp_time) and tiled_op_mlp_time > 0 and not np.isnan(naive_mlp_time) and naive_mlp_time > 0:
			print(f"| Custom Tiled_op MLP   | {tiled_op_mlp_time:>13.4f} | {naive_mlp_time / tiled_op_mlp_time:>26.2f}x |")
		else:
			print(f"| Custom Tiled_op MLP   | {tiled_op_mlp_time:>13.4f} | {'N/A':>27} |")

		if not np.isnan(torch_mlp_time) and torch_mlp_time > 0 and not np.isnan(naive_mlp_time) and naive_mlp_time > 0:
			print(f"| PyTorch MLP           | {torch_mlp_time:>13.4f} | {naive_mlp_time / torch_mlp_time:>26.2f}x |")
		else:
			print(f"| PyTorch MLP           | {torch_mlp_time:>13.4f} | {'N/A':>27} |")
		
		print("-" * len(header_mlp))