#!/bin/sh
#SBATCH -p gpu.q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --nodelist=nova013
#SBATCH --gpus=1
#SBATCH -J gen5P9H
#SBATCH -o 5P9H_result/csa_5P9H_033116.log
python libfragcsa.py -p 5P9H -f True -t False