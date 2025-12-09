#!/bin/bash

# SBATCH --partition=gpu_a100
# SBATCH --gpus=1
# SBATCH --job-name=Train
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=9
# SBATCH --time=10:00:00
# SBATCH --output=%A.out


port=$(gshuf -i 29500-29599 -n 1)

config=disco_topiocqa_mistral_n_positives_debug.yaml
ckpt_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_noce/
runpath=DATA/topiocqa_distil/distil_run_top_mistral.json

torchrun --nproc_per_node 1 --master_port $port -m splade.hf_train \
    --config-name=$config  \
    data.TRAIN.DATASET_PATH=$runpath \
    config.checkpoint_dir=$ckpt_dir