#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=6000
#SBATCH --time=35:00:00
#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

module load devel/cuda/11.0
export CXX=g++
source /pfs/work7/workspace/scratch/tu_zxmav84-thesis/miniconda3/etc/profile.d/conda.sh
conda activate thesis
#echo "setup complete"

cd /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis/restyle-encoder/


python scripts/train_restyle_e4e.py \
    --dataset_type=zalando_germany_encode \
    --encoder_type=ResNetProgressiveBackboneEncoder \
    --exp_dir=/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/restyle/00005_snapshot_1200/ \
    --stylegan_weights /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/hyperstyle/pretrained/00005-stylegan2_ada_images-mirror-auto2-kimg5000-resumeffhq512_network-snapshot-001200.pt \
    --workers=8 \
    --batch_size=8 \
    --test_batch_size=8 \
    --test_workers=8 \
    --val_interval=2000 \
    --image_interval=500 \
    --save_interval=2000 \
    --start_from_latent_avg \
    --lpips_lambda=0.8 \
    --l2_lambda=1 \
    --w_norm_lambda=0 \
    --id_lambda=0 \
    --moco_lambda=0.5 \
    --input_nc=6 \
    --n_iters_per_batch=5 \
    --output_size=512 \
    --save_training_data \