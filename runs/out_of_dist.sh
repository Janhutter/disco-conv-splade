#!/bin/bash


seeds=(123)

dataset='ikat24'
model='t5_rewrites' # options are 'mistral', 'mistral_improved', 'mistral_rewrites', 'splade_vanilla', 't5_rewrites', 'human_rewrites'

for seed in ${seeds[@]}; do
    sbatch runs/out_of_dist.job $seed $dataset $model
done
