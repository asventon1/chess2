from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sysconfig

example_module = Pybind11Extension(
    'cpp_part',
    [str(fname) for fname in Path('src').glob('*.cpp')],
    #libraries = ["../env/lib/python3.12/site-packages/torch/lib/libtorch.so", "../env/lib/python3.12/site-packages/torch/_C.cpython-312-x86_64-linux-gnu.so"],
    #library_dirs = ["../env/lib/python3.12/site-packages/torch/lib"],
    libraries = [
        "../env/lib/python3.12/site-packages/torch/lib/libtorch.so",
        "/home/adam/stuff/python/chess2/env/lib/python3.12/site-packages/torch/lib/libc10.so/usr/lib/libcuda.so",
        "/opt/cuda/lib/libnvrtc.so",
        "/opt/cuda/lib64/libnvToolsExt.so",
        "/opt/cuda/lib64/libcudart.so",
        "/home/adam/stuff/python/chess2/env/lib/python3.12/site-packages/torch/lib/libc10_cuda.so"
    ],

    include_dirs=['include', 'chess-library/include', '../env/lib/python3.12/site-packages/torch/include/', '../env/lib/python3.12/site-packages/torch/include/torch/csrc/api/include'],
    #runtime_library_dirs=[sysconfig.get_config_var("LIBDIR")],
    extra_compile_args=['-O3']
)

setup(
    name='cpp_part',
    ext_modules=[example_module],
    cmdclass={"build_ext": build_ext},
)
