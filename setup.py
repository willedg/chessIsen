from setuptools import setup, Extension
import platform
import os

# Try to use Pybind11 if available, otherwise fallback to basic Extension logic
try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    HAS_PYBIND11 = True
except ImportError:
    from setuptools.command.build_ext import build_ext
    HAS_PYBIND11 = False

extra_compile_args = []
if platform.system() == "Windows":
    extra_compile_args = ["/EHsc", "/bigobj", "/std:c++latest", "/D_USE_MATH_DEFINES", "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-march=native"]

if HAS_PYBIND11:
    ext_modules = [
        Pybind11Extension(
            "wp3.cpp_mcts",
            ["wp3/mcts.cpp"],
            extra_compile_args=extra_compile_args,
        ),
    ]
else:
    ext_modules = [
        Extension(
            "wp3.cpp_mcts",
            ["wp3/mcts.cpp"],
            extra_compile_args=extra_compile_args,
            # You might need to add pybind11 include paths manually here if Pybind11Extension is missing
        ),
    ]

setup(
    name="wp3.cpp_mcts",
    version="0.1",
    packages=["wp3"],
    package_dir={"wp3": "wp3"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext} if HAS_PYBIND11 else {},
)
