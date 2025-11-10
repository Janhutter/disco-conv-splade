#!/bin/bash


seeds=(123 456 789)

dataset='topiocqa'
model='mistral'

for seed in ${seeds[@]}; do
    sbatch runs/eval.job $seed $dataset $model
done
