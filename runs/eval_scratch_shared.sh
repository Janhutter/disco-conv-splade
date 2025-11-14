#!/bin/bash


seeds=(123)

dataset='topiocqa'
model='mistral' # options are 'mistral', 'mistral_improved', 'mistral_rewrites', 'splade_vanilla'
checkpoint_ids=(4544 9088 13632 18176 22720) # specify checkpoint ids or leave empty for final model

for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoint_ids[@]}; do
        sbatch runs/eval_scratch_shared.job $seed $dataset $model $checkpoint
    done
done