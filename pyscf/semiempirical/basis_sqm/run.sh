#!/bin/bash

eval "$(conda shell.bash hook)"
source /Applications/anaconda3/etc/profile.d/conda.sh

export LIBSEMI=/Users/chancelander/Documents/Shao/semiempirical/code/semiempirical/build/lib.macosx-10.9-x86_64-cpython-38/pyscf/semiempirical/lib/libsemiempirical.so

conda activate pysemi
python3 --version
python3 make_sto6g.py
conda deactivate 
