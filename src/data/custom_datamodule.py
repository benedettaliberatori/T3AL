from typing import Any, Dict, Optional
import yaml
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.dataset import T3ALDataset

class T3ALDataModule(LightningDataModule):
    """LightningDataModule:

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = False,
        nsplit: int = 0,
        config: str = "", 
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.nsplit = nsplit 
        self.config = config 
        
    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        
        with open(self.config, "r", encoding="utf-8") as f:
            tmp = f.read()
            config = yaml.load(tmp, Loader=yaml.FullLoader)
    
        if not self.data_test and not self.data_test:
            self.data_train = T3ALDataset(subset="train", nsplit=self.nsplit, config=config)
            self.data_test = T3ALDataset(subset="validation", nsplit=self.nsplit, config=config)
                    
        self.subset_mask_list = self.data_test.subset_mask_list
        


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=False,
        )   

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def val_dataloader(self):
        pass

    def subset_mask_list(self):
        return self.subset_mask_list

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
