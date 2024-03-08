from typing import Any
import torch
from lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from src import utils
import numpy as np
import wandb
from src.evaluate import evaluate
import importlib

log = utils.get_pylogger(__name__)


class T3ALModule(LightningModule):
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
        optimizer: torch.optim.Adam,
        dataset: Any,
        setting: int,
        video_path: str,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.optimizer = optimizer
        self.dataset = dataset
        self.setting = setting
        self.video_path = video_path

        self.predictions = {}
        self.binary_pred, self.binary_gt = [], []
        self.binary_acc = Accuracy("binary")
        self.label_gt, self.label_pred = [], []

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
        self.lr = optimizer.keywords["lr"]
        self.scaling_factor = 0.001
        self.split = split

    def forward(self, x: torch.Tensor, opt):
        return self.net(x, opt)

    def on_train_start(self):
        pass

    def model_step(self, batch: Any, opt):
        return self.forward(batch, opt)

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

    @torch.inference_mode(False)
    @torch.enable_grad()
    def test_step(self, batch: Any, batch_idx: int):
        video_name, output, pred_mask, gt_mask, unique_labels, sim_plt = (
            self.model_step(
                batch,
                self.optimizer,
            )
        )
        self.predictions[video_name] = output
        self.binary_gt.append(gt_mask)
        self.binary_pred.append(pred_mask)

        for ulabel in list(unique_labels):
            if ulabel in self.dict_test.keys():
                self.label_gt.append(self.dict_test[ulabel])
                break
        self.label_pred.append(output[0]["label"])

        if sim_plt != None:
            wandb.log({f"Similarity/{video_name}": wandb.Image(sim_plt)})
            sim_plt.close()

    def on_test_epoch_end(self):

        aps, tious = evaluate(self.dataset, self.predictions, self.split, self.setting, self.video_path)
        binary_pred = torch.cat(self.binary_pred)
        binary_gt = torch.cat(self.binary_gt)
        self.binary_acc(binary_pred, binary_gt)
        tot = binary_gt.shape[0]
        TP = (torch.sum((binary_gt == 1) & (binary_pred == 1)).item() / tot) * 100
        FP = (torch.sum((binary_gt == 0) & (binary_pred == 1)).item() / tot) * 100
        FN = (torch.sum((binary_gt == 1) & (binary_pred == 0)).item() / tot) * 100
        TN = (torch.sum((binary_gt == 0) & (binary_pred == 0)).item() / tot) * 100

        for i, ap in enumerate(aps.tolist()):
            wandb.log({f"AP_{i}": ap})
        wandb.log({f"Localization/IOU": tious})
        wandb.log({"Localization/Binary Accuracy": self.binary_acc.compute()})
        wandb.log({"Localization/TP": TP})
        wandb.log({"Localization/FP": FP})
        wandb.log({"Localization/FN": FN})
        wandb.log({"Localization/TN": TN})

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

        params_list = []
        if self.net.text_projection:
            self.net.model.text.text_projection.requires_grad = True
            params_list.append(
                {"params": self.net.model.text.text_projection, "lr": self.lr}
            )

        if self.net.image_projection:
            self.net.model.visual.proj.requires_grad = True
            params_list.append(
                {
                    "params": self.net.model.visual.proj,
                    "lr": (self.lr * self.scaling_factor),
                }
            )

        if self.net.text_encoder:
            for param in self.net.model.text.parameters():
                param.requires_grad = True
            params_list.append(
                {"params": self.net.model.text.parameters(), "lr": self.lr}
            )

        if self.net.logit_scale:
            self.net.model.logit_scale.requires_grad = True
            params_list.append({"params": self.net.model.logit_scale, "lr": self.lr})

        optimizer = self.hparams.optimizer(params=params_list)
        self.optimizer = optimizer

        return {"optimizer": optimizer}
