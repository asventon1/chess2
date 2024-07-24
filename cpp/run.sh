#! /usr/bin/sh

source ../chess2env/bin/activate
#rm -r build cpp_part.cpython-312-x86_64-linux-gnu.so  cpp_part.egg-info/
#pip install -e .
#python main.py
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build . --config release && cp cpp_part.cpython-312-x86_64-linux-gnu.so ../ && time python main.py
