#!/bin/bash


seeds=(123)

dataset='topiocqa'
model='mistral_rewrites' # options are 'mistral', 'mistral_improved', 'mistral_rewrites', 'splade_vanilla'

for seed in ${seeds[@]}; do
    sbatch runs/eval.job $seed $dataset $model
done
