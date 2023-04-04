#!/bin/sh
#SBATCH -p gpu.q
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gpus=1
#SBATCH -J csa2
#SBATCH -o 5P9H_result/csatest.log
#SBATCH -e csa.log

python libcsa4.py
