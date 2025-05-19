# GEMM_EXP

## Environment


- **CPU**: AMD Ryzen 5 3600 6-Core Processor
- **RAM**: 32 GB 
- **GPU**: NVIDIA GeForce GTX 1660 SUPER
- **OS** : Windows 10 (22H2) - 19045
- **Display Driver Version**: 576.02
- **CUDA Version**: 12.9
- **Python**: 3.10.13

- **Warmup (default)**: 20
- **Runs (default)**:   1000

```bash
# Python 3.12.9 / Python 3.10.13
# CUDA 12.4     / CUDA 12.9
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Test & Benchmarks

```bash
python setup.py install
python test/test_gemm.py
python benchmark/benchmark.py --mode gemm --size 256 --warmup 20 --runs 1000


nsys profile -t cuda,nvtx -o gemm_profile --stats=true --force-overwrite true python benchmark/benchmark.py --mode gemm --size 256
nsys profile -t cuda,nvtx -o gemm_profile_mlp_256 --stats=true --force-overwrite true python benchmark/benchmark.py --mode mlp --batch_size 256
nsys-ui gemm_profile.nsys-rep
nsys stats gemm_profile.nsys-rep

ncu --kernel-name matmul_tiled_op_kernel -o matmul_tiled_op_profile --set full --force-overwrite python benchmark/benchmark.py --mode gemm --size 256
ncu-ui matmul_tiled_op_profile.ncu-rep
ncu --import matmul_tiled_op_profile.ncu-rep --page details > ncu_cli.txt
```

```python
# nvtx range example
torch.cuda.nvtx.range_push("my_gemm_op")

# xxx

torch.cuda.nvtx.range_pop()
```