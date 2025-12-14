#!/bin/bash
mkdir -p DATA/topiocqa_distil

wget -O DATA/topiocqa_distil/distil_run_top_mistral.json https://huggingface.co/datasets/slupart/topiocqa-distillation-mistral-splade/resolve/main/run.json

wget -O DATA/topiocqa_distil/distil_run_top_llama_mistral.json https://huggingface.co/datasets/slupart/topiocqa-distillation-llama-mistral-splade/resolve/main/run.json


mkdir -p DATA/qrecc_distil
wget -O DATA/qrecc_distil/distil_run_top_human.json https://huggingface.co/datasets/slupart/qrecc-distillation-human-mistral-splade/resolve/main/run.json
wget -O DATA/qrecc_distil/distil_run_top_mistral.json https://huggingface.co/datasets/slupart/qrecc-distillation-mistral-splade/resolve/main/run.json


mkdir -p DATA/topiocqa_rewrites
wget -O DATA/topiocqa_rewrites/rewrites_mistral_test.parquet https://huggingface.co/datasets/slupart/topiocqa-rewrite-mistral/resolve/main/data/test-00000-of-00001.parquet

mkdir -p DATA/cast22_rewrites
wget -O DATA/cast22_rewrites/rewrites_mistral_test.parquet https://huggingface.co/datasets/slupart/cast22-rewrite-mistral/resolve/main/data/test-00000-of-00001.parquet

mkdir -p DATA/ikat23_rewrites
wget -O DATA/ikat23_rewrites/rewrites_mistral_test.parquet https://huggingface.co/datasets/slupart/ikat23-rewrite-mistral/resolve/main/data/test-00000-of-00001.parquet

mkdir -p DATA/cast20_rewrites
wget -O DATA/cast20_rewrites/rewrites_mistral_test.parquet https://huggingface.co/datasets/slupart/cast20-rewrite-mistral/resolve/main/data/test-00000-of-00001.parquet

mkdir -p DATA/qrecc_rewrites
wget -O DATA/qrecc_rewrites/rewrites_mistral_test.parquet https://huggingface.co/datasets/slupart/qrecc-rewrite-mistral/resolve/main/data/test-00000-of-00001.parquet

