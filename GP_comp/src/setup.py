from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "GPlib",  # Name of the module
        ["GPlib.cpp", "hermite_polynomial.cpp"],  # C++ source files
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language="c++",
        extra_compile_args=["-std=c++11"]
    ),
]

setup(
    name="GPlib",
    ext_modules=ext_modules,
)