from typing import Any, List
import torch
from lightning import LightningModule
from src import utils
import wandb
from src.evaluate import evaluate
import importlib
import numpy as np

log = utils.get_pylogger(__name__)

class BaselineModule(LightningModule):
    """LightningModule.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        split: int,
        dataset: Any, 
        setting: int,
        video_path: str,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.predictions = {}
        
        self.split = split
        self.dataset = dataset 
        self.setting = setting
        self.video_path = video_path
        
        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
        elif self.dataset == "anet":
            dict_test_name = (
                f"t2_dict_test_{split}"
                if self.setting == 50
                else f"t1_dict_test_{split}" if self.setting == 75 else None
            )
        else:
            raise ValueError("Dataset not implemented")
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = list(self.dict_test.keys())


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        pass

    def model_step(self, batch: Any):
        return self.forward(batch)

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_start(self):
        print("Start testing...")
        pass
    
    def test_step(self, batch: Any, batch_idx: int):
        video_name, output = self.model_step(batch)
        self.predictions[video_name] = output

    def on_test_epoch_end(self):
        aps, tious = evaluate(self.dataset, self.predictions, self.split, self.setting, self.video_path)
        for i, ap in enumerate(aps.tolist()):
            wandb.log({f"AP_{i}": ap})
        self.log("avg_AP", np.mean(aps))
        self.log("AP_0", aps[0])
        
        return

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.get("optimizer") is None:
            return None
        for param in self.parameters():
            param.requires_grad = False
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}
