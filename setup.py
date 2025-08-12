from setuptools import setup, Extension

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
