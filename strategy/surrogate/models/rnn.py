import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from strategy.surrogate.models.lightning_model import SurrogateLightningModule, DataModule, \
    RNNDataModule
import pytorch_lightning as pl



class RNNnet(nn.Module):
    def __init__(self, rnn_type, n_feature, n_layers=10, n_hidden=50, output_dim=1, **kwargs):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=n_feature,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=n_feature,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=n_feature,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True
            )

        self.regressor = nn.Linear(n_hidden, output_dim)

        self.apply(self.init_weights)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_x)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        return self.regressor(output[:, -1, :])

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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
        self.batch_size = 16

    def __str__(self):
        return self.name

    def fit(self, x, y, **train_params):
        self.train(x, y, epochs=self.epochs, batch_size=self.batch_size, **train_params)

    def predict(self, query):
        # shape handling
        if query.ndim < 2:
            data = torch.from_numpy(query).float().unsqueeze(0)
        else:
            data = torch.from_numpy(query.astype(np.float32)).float()

        with torch.no_grad():
            pred = self.model(data.to(self.device))

        return pred.cpu().numpy()

    def train(self, x, y,
              train_split=1.0,
              pretrained=None,
              lr=0.001,
              epochs=10,
              batch_size=16):

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

class RNN(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn_type = 'RNN'
        self.net = RNNnet(self.rnn_type, **kwargs)
        self.name = f'Recurrent Neural Network ({self.net.n_layers}x{self.net.n_hidden})(e{self.epochs}-b{self.batch_size})'
        self.str = f'rnn_{self.net.n_layers}x{self.net.n_hidden}_e{self.epochs}_b{self.batch_size}'

    def train(self, x, y,
              train_split=1.0,
              pretrained=None,
              lr=0.001,
              epochs=100,
              batch_size=16):

        sequences = []
        for seq in x:
            t = torch.tensor(np.array(seq, dtype=float), dtype=torch.float32)

            # If the tensor is 1D, make it (1, L)
            if t.dim() == 1:
                t = t.unsqueeze(-1)

            sequences.append(t)

        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        sequence_lengths = torch.tensor([len(seq) for seq in sequences])

        # DataModule
        dm = RNNDataModule(
            x=(padded_sequences, sequence_lengths),  # wrap as a single structure
            y=y,
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

    def predict(self, x):
        self.model.eval()

        sequences = []
        for seq in x:
            t = torch.tensor(np.array(seq, dtype=float), dtype=torch.float32)

            # If the tensor is 1D, make it (1, L)
            if t.dim() == 1:
                t = t.unsqueeze(-1)

            sequences.append(t)
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        sequence_lengths = torch.tensor([len(seq) for seq in sequences])

        with torch.no_grad():
            preds = self.model(padded_sequences.float(), sequence_lengths)

        return preds.cpu().numpy()

class GRU(RNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = RNNnet(rnn_type='GRU', **kwargs)
        self.name = f'GRU ({self.net.n_layers}x{self.net.n_hidden})(e{self.epochs}-b{self.batch_size})'
        self.str = f'rnn_{self.net.n_layers}x{self.net.n_hidden}_e{self.epochs}_b{self.batch_size}'

class LSTM(RNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = RNNnet(rnn_type='LSTM', **kwargs)
        self.name = f'LSTM ({self.net.n_layers}x{self.net.n_hidden})(e{self.epochs}-b{self.batch_size})'
        self.str = f'rnn_{self.net.n_layers}x{self.net.n_hidden}_e{self.epochs}_b{self.batch_size}'