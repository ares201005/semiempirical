#!/bin/bash

eval "$(conda shell.bash hook)"
source /Applications/anaconda3/etc/profile.d/conda.sh

conda activate pysemi
python --version
python3 --version
#python test_mindo3.py
#python test_am1.py
#python am1_energy.py
python3 test_om2.py
conda deactivate 
