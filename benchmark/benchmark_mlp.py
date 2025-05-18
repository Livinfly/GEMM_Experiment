# benchmark_mlp.py
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils import create_mlp, benchmark_model
import matplotlib.pyplot as plt

try:
	import gemm_cuda_kernels
	EXTENSION_AVAILABLE = True
	print("Successfully import 'gemm_cuda_kernels' C++ extension.")
except ImportError as e:
	print(f"Failed to import 'gemm_cuda_kernels': {e}")
	print(f"Build the extension using 'python setup.py install.")
	EXTENSION_AVAILABLE = False

if __name__ == "__main__":
    if not EXTENSION_AVAILABLE:
        print("Error: C++ extension not loaded. Please build first.")
        exit()
    if not torch.cuda.is_available():
        print("Error: CUDA device not available.")
        exit()

    gemm_funcs = {
        "naive": gemm_cuda_kernels.naive_gemm,
        "tiled_16": gemm_cuda_kernels.tiled_16_gemm,
        "tiled_op": gemm_cuda_kernels.tiled_op_gemm,
        "pytorch": torch.matmul,
    }

    mlp_results_time = {
        "naive": {},
        "tiled_16": {},
        "tiled_op": {},
        "pytorch": {},
    }

    mlp_results_speedup = {
        "naive": {},
        "tiled_16": {},
        "tiled_op": {},
        "pytorch": {},
    }

    mlp_results_gflops = {
        "naive": {},
        "tiled_16": {},
        "tiled_op": {},
        "pytorch": {},
    }

    batch_sizes = [64, 256, 1024, 4096]
    input_dim = 1024
    hidden_dims = [512, 256, 64]
    output_dim = 10

    kernel_names = ["naive", "tiled_16", "tiled_op", "pytorch"]

    for batch_size in batch_sizes:
        A_torch = torch.randn(batch_size, input_dim, dtype=torch.float32, device='cuda')
        flops = 2 * batch_size * (1024 * 512 + 512 * 256 + 256 * 64 + 64 * 10)
        for gemm_op_name, gemm_op_func in gemm_funcs.items():
            print(gemm_op_name, batch_size)
            model = create_mlp(input_dim, hidden_dims, output_dim, 
                use_custom_linear=(gemm_op_name!="pytorch"), custom_gemm_func=gemm_op_func)
            avg_time_ms = benchmark_model(gemm_op_name, model, A_torch, 20, 1000, verbose=False)
            avg_gflops = (flops / avg_time_ms) * 1e-6
            mlp_results_time[gemm_op_name][batch_size] = avg_time_ms
            mlp_results_gflops[gemm_op_name][batch_size] = avg_gflops


    num_batch_sizes = len(batch_sizes)
    num_kernels = len(kernel_names)

    for batch_size in batch_sizes:
        baseline_time_naive = mlp_results_time["naive"].get(batch_size)
        for kernel_name in kernel_names:
            kernel_time = mlp_results_time[kernel_name].get(batch_size)
            mlp_results_speedup[kernel_name][batch_size] = baseline_time_naive / kernel_time


    x_indices = np.arange(num_batch_sizes)
    total_bar_width_for_group = 0.8
    bar_width = total_bar_width_for_group / num_kernels

    colors = plt.colormaps['viridis'].resampled(num_kernels).colors

    # Speedup
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    for i, kernel_name in enumerate(kernel_names):
        kernel_speedups = [mlp_results_speedup[kernel_name].get(s, np.nan) for s in batch_sizes]
        positions = x_indices + (i - num_kernels / 2 + 0.5) * bar_width
        bar_container = ax1.bar(positions, kernel_speedups, width=bar_width, label=kernel_name, color=colors[i], edgecolor='grey')
        ax1.bar_label(bar_container, fmt='%.2fx', fontsize=8, padding=3, fontweight='bold')

    ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.0, label='naive baseline (1.0x)')
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Speedup", fontsize=12)
    ax1.set_title("Speedup Comparison of Different GEMM Kernels (MLP)", fontsize=16)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(title="Kernel Implementations", loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig1.tight_layout()

    # Gflops
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    for i, kernel_name in enumerate(kernel_names):
        kernel_gflops = [mlp_results_gflops[kernel_name].get(s, 0) for s in batch_sizes]
        positions = x_indices + (i - num_kernels / 2 + 0.5) * bar_width
        bar_container = ax2.bar(positions, kernel_gflops, width=bar_width, label=kernel_name, color=colors[i], edgecolor='grey')
        ax2.bar_label(bar_container, fmt='%.1f', fontsize=8, padding=3, fontweight='bold')

    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("GFLOPS", fontsize=12)
    ax2.set_title("GFLOPS Comparison of Different GEMM Kernels (MLP)", fontsize=16)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(title="Kernel Implementations", loc='upper left')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig2.tight_layout()

    # plt.show()
    fig1.savefig('result/mlp_speedup.png')
    fig2.savefig('result/mlp_gflops.png')