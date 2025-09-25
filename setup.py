"""
Setup script for Symmetric Non-Negative Matrix Factorization (SymNMF) Python C extension.
Usage:
    python setup.py build_ext --inplace

This will compile the C extension and make it available for import as 'symnmf'.
"""

from setuptools import setup, Extension

# Define the C extension module.
# Module name (import symnmf), the source files.
# Note: symnmf.h is included via #include in the source files.
module = Extension(
    'symnmf',
    sources=['symnmfmodule.c', 'symnmf.c'],
)

setup(
    name='symnmf',
    version='0.1.0',
    description='Symmetric Non-Negative Matrix Factorization C extension',
    ext_modules=[module],
)
