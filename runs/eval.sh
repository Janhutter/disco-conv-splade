#!/bin/bash


seeds=(123)

dataset='topiocqa'
model='human_rewrites' # options are 'mistral', 'mistral_improved', 'mistral_rewrites', 'splade_vanilla', 'human_rewrites'

for seed in ${seeds[@]}; do
    sbatch runs/eval.job $seed $dataset $model
done
