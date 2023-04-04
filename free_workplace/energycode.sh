#!/bin/sh
#SBATCH -p gpu.q
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -J initial
#SBATCH -o csatest.log
#SBATCH -e csa.log

python energy_calc.py
