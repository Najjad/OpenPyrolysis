from setuptools import setup, Extension
import sys
import sysconfig
import pybind11

from pathlib import Path

ext_modules = [
    Extension(
        "openpyrolysis.pykinetics",
        sources=[
            "openpyrolysis/cpp_backend/kinetics.cpp",
            "openpyrolysis/cpp_backend/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "openpyrolysis/cpp_backend",
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall"],
    )
]


setup(
    name="openpyrolysis",
    version="0.1.0",
    description="A pyrolysis simulation library",
    author="Your Name",
    packages=["openpyrolysis"],
    ext_modules=ext_modules,
    zip_safe=False,
)
