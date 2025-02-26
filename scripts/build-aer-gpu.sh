#!/bin/bash

##
# Script for building the qiskit-aer from source for gpu support.
# Following script provides version compatible with Pascal architecture, so it does NOT support cuStateVec capabilities.
##

# load pyenv into the shell
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init - bash)"

# set up the environment (for now 3.12.x version is required)
pyenv deactivate
pyenv install 3.12.9
pyenv virtualenv 3.12.9 aer-build && pyenv activate aer-build

# ensure all the dependencies are in place
pip install qiskit-aer
pip uninstall -y qiskit-aer
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cuquantum-cu12 pybind11

# install the dev dependencies from the repository
git clone https://github.com/Qiskit/qiskit-aer
pip install -r qiskit-aer/requirements-dev.txt

# ensure no build files are in the repository
rm -rf qiskit-aer/_skbuild

# via
# export AER_CUDA_ARCH="6.1"
# you can set the what architecture your GPU is compatible with

cd qiskit-aer && python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH="6.1" -DCMAKE_VERBOSE_MAKEFILE=true  -DAER_DEBUG=false -DAER_MPI=false -DCMAKE_CUDA_FLAGS=-std=c++14 -DAER_PYTHON_CUDA_ROOT=$VIRTUAL_ENV --

# move the wheel for easy access and remove the repo
cd ../ && mkdir static && mv ./qiskit-aer/dist/* ./static
rm -rf qiskit-aer

# remove the build environment
pyenv uninstall -f aer-build
