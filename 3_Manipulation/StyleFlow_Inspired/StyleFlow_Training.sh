#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=6000
#SBATCH --time=18:00:00
#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

module load devel/cuda/11.0
export CXX=g++
source /pfs/work7/workspace/scratch/tu_zxmav84-thesis/miniconda3/etc/profile.d/conda.sh
conda activate styleflow
#echo "setup complete"

cd /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis/StyleFlow/

python train_flow.py \
        --latent_path=/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/StyleFlow/Inputs/category/latents.npy \
        --attributes_path=/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/StyleFlow/Inputs/category/targets.npy \
        --epochs=30 \
        --cond_size=8 \
        --out_dir=/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/StyleFlow/Outputs/category/ \
        
