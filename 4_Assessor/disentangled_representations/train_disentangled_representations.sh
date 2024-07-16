#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --time=02:00:00
#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

module load devel/cuda/11.0
export CXX=g++
source /pfs/work7/workspace/scratch/tu_zxmav84-thesis/miniconda3/etc/profile.d/conda.sh
conda activate disentangled_representations
python cluster_train.py