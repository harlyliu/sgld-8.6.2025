from setuptools import setup, Extension
import pybind11
import os
import subprocess

def get_system_include_paths():
    """Try to get system include paths using xcrun."""
    try:
        result = subprocess.run(['xcrun', '--show-sdk-path', '--sdk', 'macosx'], capture_output=True, text=True, check=True)
        sdk_path = result.stdout.strip()
        return [os.path.join(sdk_path, 'usr', 'include'),
                os.path.join(sdk_path, 'usr', 'include', 'c++', 'v1')]
    except Exception:
        return []

ext_modules = [
    Extension(
        "GPlib",  # Name of the module
        ["GPlib.cpp", "hermite_polynomial.cpp"],  # C++ source files
        include_dirs=[pybind11.get_include()] + get_system_include_paths(),  # Include pybind11 and system headers
        language="c++",
        extra_compile_args=["-std=c++11", "-stdlib=libc++"]
    ),
]

setup(
    name="GPlib",
    ext_modules=ext_modules,
)
