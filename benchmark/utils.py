import torch
import torch.nn as nn
import numpy as np

class CustomLinearGEMM(nn.Module):
	def __init__(self, in_features, out_features, custom_gemm_func, bias=True):
		super().__init__()
		self.in_features  = in_features
		self.out_features = out_features
		self.custom_gemm_func = custom_gemm_func

		self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32))
		if bias:
			self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
		else:
			self.bias = nn.register_parameter('bias', None)

	def forward(self, x):
		if x.dim() != 2 or x.size(-1) != self.in_features:
			raise ValueError(f"Input shape mismatch. Expected (*, {self.in_features}), but got {x.shape}.")
		
		output_gemm = self.custom_gemm_func(x.to(self.weight.device), self.weight)

		if self.bias is not None:
			output = output_gemm + self.bias
		else:
			output = output_gemm
		
		return output

def create_mlp(input_dim, hidden_dims, output_dim, use_custom_linear=False, custom_gemm_func=None, activation_fn=nn.ReLU):
	if use_custom_linear and custom_gemm_func is None:
		raise ValueError("Must provide custom_gemm_func when use_custom_linear is True.")
	
	layers = []
	current_dim = input_dim
	
	all_dims = [input_dim] + hidden_dims + [output_dim]

	for i in range(len(all_dims) - 1):
		in_dim  = all_dims[i]
		out_dim = all_dims[i+1]
	
		linear = CustomLinearGEMM(in_dim, out_dim, custom_gemm_func) if use_custom_linear else nn.Linear(in_dim, out_dim)
		
		layers.append(linear)

		if i < len(all_dims) - 2 and activation_fn is not None:
			layers.append(activation_fn())
	
	model = nn.Sequential(*layers)
	return model

def benchmark_gemm_op(gemm_op_name, gemm_op_func, A_torch, B_torch, num_warmup=5, num_runs=20, verbose=True):
	if not torch.cuda.is_available():
		print("CUDA is not available, skipping GEMM op benchmark.")
		return float('nan')
	if gemm_op_func is None:
		print(f"GEMM op function {gemm_op_name} is None, skipping.")
		return float('nan')
	
	if verbose:
		print(f"  Benchmarking GEMM op: {gemm_op_name}...")

	A_torch, B_torch = A_torch.to('cuda'), B_torch.to('cuda')
	
	if verbose:
		print(f"    Warmup ({num_warmup} runs)...")
	for _ in range(num_warmup):
		_ = gemm_op_func(A_torch, B_torch)
		torch.cuda.synchronize()  # 保证GPU运行完
	
	if verbose:
		print(f"    Timing ({num_runs} runs)...")
	start_event = torch.cuda.Event(enable_timing=True)
	end_event   = torch.cuda.Event(enable_timing=True)
	timings     = []
	
	torch.cuda.synchronize()  # 先同步
	for _ in range(num_runs):
		start_event.record()
		_ = gemm_op_func(A_torch, B_torch)
		end_event.record()  # 跟着流记录

		torch.cuda.synchronize()
		timings.append(start_event.elapsed_time(end_event))
	
	avg_time_ms = np.mean(timings)
	if verbose:
		print(f"    Avg time per GEMM op run: {avg_time_ms:.4f} ms")
	return avg_time_ms

def benchmark_model(model_name, model, input_tensor, num_warmup=5, num_runs=20, verbose=True):
	if not torch.cuda.is_available():
		print("CUDA is not available, skipping model benchmark.")
		return float('nan')
	if model is None:
		print(f"Model {model_name} is None, skipping.")
		return float('nan')
	
	if verbose:
		print(f"  Benchmarking Model: {model_name}...")
	model.to('cuda').eval()

	input_tensor = input_tensor.to('cuda')
	
	if verbose:
		print(f"    Warmup ({num_warmup} runs)...")
	for _ in range(num_warmup):
		_ = model(input_tensor)
		torch.cuda.synchronize()  # 保证GPU运行完
	
	if verbose:
		print(f"    Timing ({num_runs} runs)...")
	start_event = torch.cuda.Event(enable_timing=True)
	end_event   = torch.cuda.Event(enable_timing=True)
	timings     = []
	
	with torch.no_grad():
		torch.cuda.synchronize()  # 先同步
		for _ in range(num_runs):
			start_event.record()
			_ = model(input_tensor)
			end_event.record()  # 跟着流记录

			torch.cuda.synchronize()
			timings.append(start_event.elapsed_time(end_event))
	
	avg_time_ms = np.mean(timings)
	if verbose:
		print(f"    Avg time per GEMM op run: {avg_time_ms:.4f} ms")
	return avg_time_ms