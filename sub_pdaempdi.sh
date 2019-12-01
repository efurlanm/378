#!/bin/bash
#SBATCH --ntasks=1

cd $SLURM_SUBMIT_DIR

mpiexec -n $SLURM_NTASKS python3 padempdi.py
