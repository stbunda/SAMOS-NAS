import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from numpy.random import RandomState
from torch.utils.data import DataLoader, TensorDataset, random_split

class SurrogateLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }

class DataModule(pl.LightningDataModule):
    def __init__(self, x, y, batch_size=256, train_split=0.8, random_state: RandomState=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_split = train_split
        self.random_state = random_state
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)


    def setup(self, stage=None):
        dataset = TensorDataset(self.x, self.y)
        n_total = len(dataset)
        n_train = int(self.train_split * n_total)
        n_val = n_total - n_train

        g = torch.Generator().manual_seed(self.random_state.randint(0, 100))
        self.train_ds, self.val_ds = random_split(dataset, [n_train, n_val], generator=g)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)