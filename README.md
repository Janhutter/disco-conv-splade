# A Systematic Reproducibility Study of DiSCo for Conversational Search

This repository contains the code and resources for our reproducibility study of DiSCo for Conversational Search by Lupart et al.
## 1. Installation and Dataset Download

### Conda Environment

```bash
conda env create -f environment.yml
conda activate disco
```

### Setup for all utilized datasets
We reproduce the experiments on TopiOCQA, CAsT 2020, iKAT 2023, and iKAT 2024. Please follow the instructions in the respective folders to download and preprocess these datasets.

```bash
bash setup_script/dl_data.sh
```

### Distilation files
```bash
bash setup_script/dl_distillation.sh
```

### Preprocessing

We provide scripts to preprocess the datasets into a format suitable for indexing and retrieval.

```bash
python setup_script/parse_topiocqa.py
python setup_script/parse_cast.py
python setup_script/parse_ikat23.py
```

### Generating rewrites

We provide scripts to generate query rewrites using T5 and Mistral models.

```bash
python setup_script/generate_rewrites.py
```

### Huggingface 🤗
Our preprocessed datasets, distillation files, and rewritten queries, that were not available from the original authors, have been uploaded to Huggingface for easier access and future work. Please check it out at: https://huggingface.co/collections/JanHutter/conversational-search

## 2. Indexing

The index can be created with the provided SPLADE configuration files. This is an example of indexing the TopiOCQA collection with SPLADE.

```bash
config=disco_topiocqa_mistral.yaml
collection_path=DATA/full_wiki_segments_topiocqa.tsv
index_dir=DATA/topiocqa_index_self

bash -m splade.index --config-name=$config \
    init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    data.COLLECTION_PATH="$collection_path" \
    config.index_retrieve_batch_size=2048 \
```

Do note that you would have to change the config, collection_path, and index_dir for other datasets.

## 3. Training
Training DiSCo can be done using the provided configuration files. Below is an example of training DiSCo on TopiOCQA using Mistral Teacher.

```bash
port=$(shuf -i 29500-29599 -n 1)

config=disco_topiocqa_mistral.yaml
runpath=/scratch-shared/disco/DATA/topiocqa/distil_run_top_mistral.json
ckpt_dir=/scratch-shared/disco/disco_TOPIOCQA_mistral/

torchrun --nproc_per_node 1 --master_port $port -m splade.hf_train \
    --config-name=$config  \
    data.TRAIN.DATASET_PATH=$runpath \
    config.checkpoint_dir=$ckpt_dir

```

For the distillation, it is necessary to have the distil_run_top_mistral.json file for the specific dataset. This can be obtained by performing evaluation on the training set with a specific query rewriting method. This outputs a run.json file which contains the similarity score for that specific teacher model. 

To see how this is done, please refer to the `conf/disco_topiocqa_t5_distil.yaml` file.


## 4. Evaluation

To evaluate the trained DiSCo model, you can use the following command:

```bash
bash eval.sh
```

Where this bash file specificies the specific run.

```bash


seeds=(123)

dataset='ikat23'
model='mistral' # options are 'mistral', 'mistral_improved', 'mistral_rewrites', 'splade_vanilla', 't5_rewrites', 'human_rewrites'

for seed in ${seeds[@]}; do
    sbatch runs/eval.job $seed $dataset $model
done

```

`eval.job` contains the SLURM job configuration for evaluation.

```bash


seed=$1

dataset=$2
capitalized_dataset=${dataset^^}
model=$3

config=disco_${dataset}_${model}.yaml

if [ "$dataset" == "ikat23" ]; then
    index_dir=/scratch-shared/disco/DATA/ikat23/splade_index/splade_index/
elif [ "$dataset" == "ikat24" ]; then
    index_dir=/scratch-shared/disco/DATA/ikat23/splade_index/splade_index/
else
    index_dir=/scratch-shared/disco/DATA/${dataset}/${dataset}_index_self/
fi

out_dir=/scratch-shared/disco/disco_${capitalized_dataset}_${model}_out_${seed}/
ckpt_dir=/scratch-shared/disco/disco_${capitalized_dataset}_${model}/

export SPLADE_CONFIG_NAME=$config

python -m splade.retrieve \
    --config-name=$config \
    config.checkpoint_dir=$ckpt_dir \
    config.index_dir=$index_dir \
    config.out_dir=$out_dir \
    config.seed=$seed
```


For these runs, and the config files, the following naming scheme is utilized:


| Key               | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `mistral`         | The trained DiSCo model distilled from the Mistral teacher using in-domain queries. |
| `t5`              | The trained DiSCo model distilled from the T5 teacher using in-domain queries. |
| `multi`           | The trained DiSCo model distilled from multiple teachers using in-domain queries. |
| `mistral_rewrites`| The Mistral query rewrite baseline.                                          |
| `splade_vanilla`  | The SPLADE model without any distillation or rewriting; it takes the whole conversation as input. |
| `t5_rewrites`     | The T5 query rewrite baseline.                                               |
| `human_rewrites`  | The human query rewrite baseline.                                            |
| `distil`          | The configuration used to obtain distillation files from the training set.  |
| `fusion`          | The configuration used to obtain DiSCo fusion.                              |


Do note that is necessary to set `hf_training: false` in the config files for query rewrite methods, as it utilizes the pretrained SPLADE model without any further training.

### multi-teacher distillation
To perform multi-teacher distillation, you would need to create a distillation run file that combines the outputs from multiple teachers. This can be done by merging the run files obtained from evaluating each teacher on the training set with the following script:

```bash
setup_script/combine_distils.py
```

## 5. Objective extensions
The following extensions to the original training were made in our experiments. These changes can be made, and the DiSCo model can be trained with these extensions by modifying the config files accordingly.

### Contrastive loss
To enable training with a contrastive loss, you can simply set the following parameters in the config file:

```yaml

config:
  contrastive_weight: 0.05

hf:
  training:
    training_loss: kldiv_InfoNCE_with_weights

```
This means that 5% of the loss will be InfoNCE, and 95% will be the original KL-Divergence loss.

standard splade utilizes `kldiv_contrastive_with_weights` as the training loss and a contrastive weight of 1%.

### Different number of negatives
To change the number of negatives used during training, you can modify the following parameter in the config

```yaml
hf:
  training:
    n_negatives: 16
```

## 6. Lambda regularization
To modify the lambda regularization parameter used in SPLADE, you can change the following parameter in the config file:
```yaml
config:
  regularizer:
    FLOPS:
      lambda_d: 0.0005
      targeted_rep: rep
      reg: FLOPS
    L1:
      lambda_q: 0.001
      targeted_rep: rep
      reg: L1
```
By changing the values of `lambda_d` and `lambda_q`, you can adjust the regularization strength for document and query representations, respectively.

We utilized the following lambda values in our experiments:
| Reg. Setting     | λ_d        | λ_q        |
|------------------|------------|------------|
| DiSCo            | 0          | 0          |
| Splade Setting   | 5 × 10^-4  | 1 × 10^-3  |
| High           | 1 × 10^-3  | 5 × 10^-3  |
| Higher             | 1 × 10^-2  | 5 × 10^-2  |
| Highest          | 5 × 10^-2  | 1 × 10^-1  |