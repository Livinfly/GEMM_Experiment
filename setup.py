# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name='gemm_cuda_kernels',
	ext_modules=[
		CUDAExtension(
			name='gemm_cuda_kernels',
			sources=[
				'gemm_extension.cpp',
				'gemm_kernels.cu',
			],
			extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}  # '-DTILE_SIZE=16'
		)
	],
	cmdclass={
		'build_ext': BuildExtension
	}
)
