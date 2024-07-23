#! /usr/bin/bash

source ../env/bin/activate
rm -r build cpp_part.cpython-312-x86_64-linux-gnu.so  cpp_part.egg-info/
pip install -e .
python main.py
