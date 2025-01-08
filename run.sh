#!/bin/tcsh
#SBATCH -A sooty2
#SBATCH -p shared
#SBATCH -t 09:30:00
#SBATCH -N 1
#SBATCH -o ppe.out
#SBATCH -e ppe.err
#SBATCH -J ppe.exe.default

module load gcc/5.2.0
module load netcdf
module load python/3.7.2

python main_run_ppe_batchit.py
