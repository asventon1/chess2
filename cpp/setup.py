from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

example_module = Pybind11Extension(
    'cpp_part',
    [str(fname) for fname in Path('src').glob('*.cpp')],
    libraries = ["/home/adam/stuff/python/chess2/env/lib/python3.12/site-packages/torch/lib/libtorch.so"],
    include_dirs=['include', 'chess-library/include', '../env/lib/python3.12/site-packages/torch/include/', '../env/lib/python3.12/site-packages/torch/include/torch/csrc/api/include'],
    extra_compile_args=['-O3']
)

setup(
    name='cpp_part',
    ext_modules=[example_module],
    cmdclass={"build_ext": build_ext},
)
