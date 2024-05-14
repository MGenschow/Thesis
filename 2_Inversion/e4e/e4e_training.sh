#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=6000
#SBATCH --time=30:00:00
#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

module load devel/cuda/11.0
export CXX=g++
source /pfs/work7/workspace/scratch/tu_zxmav84-thesis/miniconda3/etc/profile.d/conda.sh
conda activate thesis
#echo "setup complete"

cd /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis/encoder4editing/

python scripts/train.py \
    --exp_dir=/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/e4e/resume/ \
    --sub_exp_dir=resume \
    --dataset_type zalando_germany_encode \
    --workers 8 \
    --batch_size 8 \
    --test_batch_size 4 \
    --test_workers 4 \
    --stylegan_weights /pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/e4e/pretrained/00005-stylegan2_ada_images-mirror-auto2-kimg5000-resumeffhq512_network-snapshot-001200.pt \
    --stylegan_size 512 \
    --learning_rate=0.0001 \
    --start_from_latent_avg \
    --use_w_pool \
    --w_discriminator_lambda 0.1 \
    --progressive_start 20000 \
    --id_lambda 0.5 \
    --max_steps 200000 \
    --save_interval 2000 \
    --val_interval 2000 \
    --image_interval 500 \
    --save_training_data \
    --resume_training_from_ckpt=/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync/Models/e4e/00005_snapshot_1200/checkpoints/iteration_54000.pt