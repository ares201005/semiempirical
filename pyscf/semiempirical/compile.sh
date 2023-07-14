#!/bin/bash
#SBATCH --partition=express
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=8
##SBATCH --output=%J_stdout.txt
##SBATCH --error=%J_stderr.txt
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=SEQM

module purge

eval "$(conda shell.bash hook)"
source ~/mambaforge/etc/profile.d/mamba.sh

mamba activate pybase
python __init__.py 
mamba deactivate 
