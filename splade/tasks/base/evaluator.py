import os

import torch

from splade.utils.utils import restore_model


def _get_world_size_from_env() -> int:
    """Return WORLD_SIZE from torchrun/torch.distributed env if present.

    This is used to avoid wrapping models in ``DataParallel`` when we are
    already running in a multi-process distributed setting (e.g. launched
    via ``torchrun --nproc_per_node>1``). In such a case each process is
    expected to drive a single GPU, and adding ``DataParallel`` on top
    would oversubscribe devices and hurt performance without changing
    numerical results.
    """

    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


class Evaluator:
    def __init__(self, model, config=None, restore=True):
        """Generic wrapper for evaluation-style tasks (indexing, retrieval).

        ``Evaluator`` is used both in single-process jobs and in
        multi-process jobs launched with ``torchrun``. In the latter case
        we avoid ``torch.nn.DataParallel`` and instead rely on one process
        per GPU, which is more efficient and keeps outputs identical.
        """

        self.model = model
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Detect whether we are in a multi-process (DDP/torchrun) setup.
        world_size = _get_world_size_from_env()
        use_dataparallel = torch.cuda.device_count() > 1 and world_size == 1

        if restore:
            # if config.get("adapter_name", None):                   
            #     adapter_path = config.get("adapter_path", os.path.join(config["checkpoint_dir"],"model"))
            #     if config.get("is_reranker", False):
            #         self.model.transformer.load_adapter(os.path.join(adapter_path,f"{config['adapter_name']}_rep"))
            #         self.model.transformer.set_active_adapters(f"{config['adapter_name']}_rep")
            #     else:
            #         self.model.transformer_rep.transformer.load_adapter(os.path.join(adapter_path,f"{config['adapter_name']}_rep"))
            #         self.model.transformer_rep.transformer.set_active_adapters(f"{config['adapter_name']}_rep")

            #     # load query adapter if it exists
            #     adapter_path_query=os.path.join(adapter_path,"query")
            #     if os.path.exists(os.path.join(adapter_path_query,f"{config['adapter_name']}_rep_q")):
            #         self.model.transformer_rep_q.transformer.load_adapter(os.path.join(adapter_path_query, f"{config['adapter_name']}_rep_q"))
            #         self.model.transformer_rep_q.transformer.set_active_adapters(f"{config['adapter_name']}_rep_q")
               
            #     self.model.eval()
            #     if torch.cuda.device_count() > 1:
            #         print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
            #         self.model = torch.nn.DataParallel(self.model)
            #     self.model.to(self.device)
            #else:            
            if self.device == torch.device("cuda"):
                if 'hf_training'  in config and config["hf_training"]:
                    ## model already loaded
                    pass
                elif ("pretrained_no_yamlconfig" not in config or not config["pretrained_no_yamlconfig"] ):
                    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "model/model.tar"))
                    restore_model(model, checkpoint["model_state_dict"])

                self.model.eval()
                if use_dataparallel:
                    print(" --- use {} GPUs (DataParallel) --- ".format(torch.cuda.device_count()))
                    self.model = torch.nn.DataParallel(self.model)
                self.model.to(self.device)
#                    print("restore model on GPU at {}".format(os.path.join(config["checkpoint_dir"], "model")),
#                        flush=True)
            else:  # CPU
                if 'hf_training' in config and config["hf_training"]:
                    ## model already loaded
                    pass                    
                elif ("pretrained_no_yamlconfig" not in config or not config["pretrained_no_yamlconfig"] ):
                    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "model/model.tar"),
                                            map_location=self.device)
                    restore_model(model, checkpoint["model_state_dict"])
        else:
            print("WARNING: init evaluator, NOT restoring the model")
            self.model.to(self.device)
        self.model.eval()  # => put in eval mode
