from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Pybind11Extension(
        "trading_strategies",
        ["trading_strategies.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
        ],
        language='c++'
    ),
]

setup(
    name="trading_strategies",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)