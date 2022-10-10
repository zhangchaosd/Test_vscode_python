#!/bin/bash
#SBATCH --constraint=v100
#SBATCH -o ./t1.out # STDOUT
#SBATCH -e ./e1.err # STDERR
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node 2
#SBATCH -J name11111
#SBATCH --mem 40GB
#SBATCH -N 1
#SBATCH -p fvl
#SBATCH -q high
#SBATCH -t 0-00:01:00


conda source activate py36
python t.py
