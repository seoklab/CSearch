#!/bin/sh
#SBATCH -p gpu.q
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gpus=1
#SBATCH -J csa2
#SBATCH -o 6M0K_result/csatest.log
#SBATCH -e csa.log

python libcsa3.py
