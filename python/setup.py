from setuptools import setup
from torch.utils import cpp_extension

setup(
    packages=['tree_mamba'],
    name='tree_mamba',
    ext_modules=[cpp_extension.CUDAExtension('tree_mamba_cuda', ['kernels.cu'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)