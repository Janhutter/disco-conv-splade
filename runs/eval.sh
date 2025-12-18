#!/bin/bash


seeds=(123)

dataset='ikat23'
model='mistral' # options are 'mistral', 'mistral_rewrites', 'splade_vanilla', 't5_rewrites', 'human_rewrites'

for seed in ${seeds[@]}; do
    sbatch runs/eval.job $seed $dataset $model
done
