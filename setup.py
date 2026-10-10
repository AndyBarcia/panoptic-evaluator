from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="panoptic_evaluator",
    version="0.1.0",
    packages=["panoptic_evaluator"],
    install_requires=["torch>=2.0"],
    extras_require={"test": ["pycocotools>=2.0"]},
    ext_modules=[
        CUDAExtension(
            name="panoptic_evaluator._cuda_evaluator",
            sources=[
                "panoptic_evaluator/csrc/bindings.cpp",
                "panoptic_evaluator/csrc/histogram_kernel.cu",
                "panoptic_evaluator/csrc/compute_kernel.cu",
                "panoptic_evaluator/csrc/instance_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
