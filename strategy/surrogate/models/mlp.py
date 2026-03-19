import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from strategy.surrogate.models.lightning_model import SurrogateLightningModule, DataModule



class MLPnet(nn.Module):
    # N-layer MLP
    def __init__(self, n_feature, n_layers=2, n_hidden=300, n_output=1, drop=0.2, **kwargs):
        super().__init__()

        self.stem = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU())
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_features = n_feature

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        self.regressor = nn.Linear(n_hidden, n_output)  # output layer
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regressor(x)  # linear output
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class NeuralNet:
    def __init__(self, random_state, device='cpu', **kwargs):
        self.random_state = random_state
        self.model = None          # set by subclass
        self.net = None
        self.name = 'NN'
        self.device = device
        self.epochs = 100
        self.batch_size = 32

    def __str__(self):
        return self.name

    def fit(self, x, y, **train_params):
        self.train(x, y, epochs=self.epochs, batch_size=self.batch_size, **train_params)

    def predict(self, x):
        # shape handling
        data = torch.from_numpy(x).float().unsqueeze(0)

        with torch.no_grad():
            pred = self.model(data.to(self.device))

        return pred.cpu().numpy()

    def train(self, x, y,
              train_split=1.0,
              pretrained=None,
              lr=0.001,
              epochs=100,
              batch_size=32):

        # DataModule
        dm = DataModule(
            x, y,
            train_split=train_split,
            batch_size=batch_size,
            random_state=self.random_state
        )

        # LightningModule
        self.model = SurrogateLightningModule(self.net, lr=lr)

        trainer = pl.Trainer(
            max_epochs=epochs,
            enable_progress_bar=False,
            accelerator='auto'
        )

        trainer.fit(self.model, datamodule=dm)
        return self.model

class MLP(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = MLPnet(**kwargs)
        self.name = f'Multi Layer Perceptron ({self.net.n_layers}x{self.net.n_hidden})(e{self.epochs}-b{self.batch_size})'
        self.str = f'mlp_{self.net.n_layers}x{self.net.n_hidden}_e{self.epochs}_b{self.batch_size}'
        self.metadata = {}

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "n_layers": self.net.n_layers,
            "n_hidden": self.net.n_hidden,
            "n_features": self.net.n_features,
            "epochs": self.net.epochs,
            "batch_size": self.net.batch_size,
        }
        return self.metadata