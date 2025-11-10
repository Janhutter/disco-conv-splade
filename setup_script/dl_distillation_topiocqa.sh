#!/bin/bash
mkdir -p DATA/topiocqa_distil

wget -O DATA/topiocqa_distil/distil_run_top_mistral.json https://huggingface.co/datasets/slupart/topiocqa-distillation-mistral-splade/resolve/main/run.json

wget -O DATA/topiocqa_distil/distil_run_top_llama_mistral.json https://huggingface.co/datasets/slupart/topiocqa-distillation-llama-mistral-splade/resolve/main/run.json

wget -O DATA/topiocqa_distil/distil_run_top_human.json https://huggingface.co/datasets/slupart/qrecc-distillation-human-mistral-splade/resolve/main/run.json
