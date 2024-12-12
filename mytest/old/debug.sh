#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=8
##SBATCH --output=%J_stdout.txt
##SBATCH --error=%J_stderr.txt
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=SEQM

module purge

eval "$(conda shell.bash hook)"
source ~/mambaforge/etc/profile.d/mamba.sh

mamba activate pybase
#python test_mindo3.py
#python test_am1.py
#python am1_energy.py
python test_om2.py
mamba deactivate 
