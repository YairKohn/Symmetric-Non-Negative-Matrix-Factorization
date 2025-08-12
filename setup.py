from setuptools import setup, Extension

module = Extension(
    'symnmf',
    sources=['symnmfmodule.c', 'symnmf.c'],
    extra_compile_args=['-ansi', '-Wall', '-Wextra', '-Werror', '-pedantic-errors'],
    extra_link_args=['-lm']
)

setup(
    name='symnmf',
    version='0.1.0',
    description='Symmetric Non-Negative Matrix Factorization C extension',
    ext_modules=[module],
)
