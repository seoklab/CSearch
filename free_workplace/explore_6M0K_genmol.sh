#!/bin/sh
#SBATCH -p gpu.q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --nodelist=nova013
#SBATCH --gpus=1
#SBATCH -J gen6M0K
#SBATCH -o 6M0K_result/csa_6M0K_033116.log
python libfragcsa.py -p 6M0K -f True -t False