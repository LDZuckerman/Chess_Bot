#!/bin/bash
   
#SBATCH --account=ucb520_asc2 # To use additional resources
#SBATCH --time=24:00:00 # 24:00:00
#SBATCH --output=../../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=160G   #160G          # memory to request
#SBATCH --qos=normal
#SBATCH --partition=amilan  # amilan for cpu, ami100 for gpu

gpu=True

module purge
module load rocm/6.1
module load miniforge # module load mambaforge/23.1.0-1
conda activate pytorch241_rocm61 #mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1

while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;
 esac
done

echo "Running data prep"
python run_dataprep.py

#####
# run from ../ with 'sbatch run_dataprep.sh'
######

