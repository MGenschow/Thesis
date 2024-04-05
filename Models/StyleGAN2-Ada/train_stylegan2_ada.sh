#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

module load devel/cuda/11.0
export CXX=g++
source /pfs/work7/workspace/scratch/tu_zxmav84-thesis/miniconda3/etc/profile.d/conda.sh
conda activate thesis
#echo "setup complete"
python /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis/stylegan2-ada-pytorch/train.py \
    --snap 10 \
    --resume ffhq512 \
    --outdir /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/Stylegan2_Ada/Experiments \
    --data /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Zalando_Germany_Dataset/dresses/images/stylegan2_ada_images \
    --kimg=1000 \
    --gpus=2 \
    --cfg=auto \
    --mirror=1 \
#    --dry-run
