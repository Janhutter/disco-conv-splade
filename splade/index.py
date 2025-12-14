import hydra
from omegaconf import DictConfig
import os
import torch

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .index_merge_shards import merge_index_shards
from .utils.utils import get_initialize_config


def _get_dist_info():
    """Return (rank, world_size, local_rank) from torchrun environment.

    Defaults to single-process values when not running under torchrun.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def index(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    rank, world_size, local_rank = _get_dist_info()
    if world_size > 1:
        # Ensure each process uses the correct GPU when launched with torchrun.
        torch.cuda.set_device(local_rank)

    #if HF: need to udate config.
    if "hf_training" in config and config["hf_training"]: # and not config.pretrained_no_yamlconfig
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None
       print('HF model')

    model = get_model(config, init_dict)

    d_collection = CollectionDatasetPreLoad(
        data_dir=exp_dict["data"]["COLLECTION_PATH"],
        id_style="row_id",
        topiocqa=("topiocqa" in exp_dict["data"]["COLLECTION_PATH"]),
        rank=rank,
        world_size=world_size,
    )
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"],
                                    batch_size=config["index_retrieve_batch_size"],
                                    shuffle=False, num_workers=8, prefetch_factor=4)

    evaluator = SparseIndexing(model=model, config=config, compute_stats=True, rank=rank)
    evaluator.index(d_loader)

    # If running in distributed mode, let rank 0 merge all shard indexes
    # into a single index_dir with rank-free naming (array_index.h5py, doc_ids.pkl, ...).
    if rank == 0 and config.get("index_dir", None) is not None:
        try:
            merge_index_shards(config["index_dir"])
        except Exception as e:
            # Do not crash the whole job if merging fails; just report.
            print(f"[rank 0] Warning: failed to merge index shards: {e}")


if __name__ == "__main__":
    index()
