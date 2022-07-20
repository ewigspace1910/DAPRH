from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="onnx_trt",
    version="0.0.1",
    author = "tnt",
    ext_modules=[CUDAExtension('onnx_trt._C',
        ['extensions.cpp', 'engine.cpp'],
        extra_compile_args={
            'cxx': ['-std=c++14', '-O2', '-Wall'],
            'nvcc': [
                '-g','-G',
                '-std=c++14', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall',
                '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
                '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
                '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_75,code=compute_75'
            ],
        },
        libraries=['nvinfer', 'nvinfer_plugin', 'nvonnxparser'])
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        'torch>=1.0.0a0',
        'torchvision'
    ],
)