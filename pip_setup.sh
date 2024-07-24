#! /bin/sh

python -m venv chess2env
source chess2env/bin/activate
pip install numpy torch python-chess flask
