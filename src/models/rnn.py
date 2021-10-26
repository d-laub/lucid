from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as tl
import einops


class RNN(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass


class LSTM(nn.Module):
    """An LSTM for binary prediction from a sequence.

    The high level architecture is:
    Embedding -> LSTM -> FC -> logit for binary prediction

    Parameters
    ----------
    num_emb : int
        Size of dictionary of embeddings.
    emb_dim : int
        Embedding dimension.
    hidden_dim : int
        Hidden & cell state dimension.
    fc_hidden_dims : list[int]
        List of hidden dimensions for final FC layers.
    num_layers : int
        Number of LSTMs to stack.
    dropout : float, default 0
        Whether to use dropout in the LSTM or not.
    bidirectional : bool, default False
        Whether to use a bidirectional LSTM or not.
    """
    def __init__(
        self,
        num_emb: int,
        emb_dim: int,
        hidden_dim: int,
        fc_hidden_dims: List[int]=None,
        num_layers: int=1,
        dropout: float=0,
        bidirectional: bool=False,
        batch_first: bool=False
    ):
        super().__init__()
        emb = nn.Embedding(num_emb, emb_dim) # only supports SGD and SparseAdam
        self.has_embedding = True

        lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

        # output is concatenation of final hidden and cell states
        # hidden contains short term, cell contains long term
        lstm_out_dim = 2*num_layers*hidden_dim
        if bidirectional: lstm_out_dim *= 2

        if fc_hidden_dims is not None:
            fc_dims = [lstm_out_dim, *fc_hidden_dims, 1]
            fc_list = nn.ModuleList()
            for i in range(len(fc_dims) - 1):
                fc_list.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
                if i != len(fc_dims) - 2: fc_list.append(nn.ReLU())
            fc = nn.Sequential(*fc_list)
        else:
            fc = nn.Linear(lstm_out_dim, 1)

        self.layers = nn.ModuleDict({
            'emb': emb,
            'lstm': lstm,
            'fc': fc
        })

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.LSTM):
        #         torch.nn.init.orthogonal_(m)

    def forward(self, x):
        x = self.layers['emb'](x) # (L N) -> (L N Hin)
        _, (hn, cn) = self.layers['lstm'](x) # (D*num_l N Hout), (D*num_l N Hout)
        x = einops.rearrange([hn, cn], 'b d n h -> n (b d h)') # (N 2(D*num_l*Hout))
        x = self.layers['fc'](x) # (N 1)
        return x


class LSTM_No_Embedding(nn.Module):
    """An LSTM for binary prediction from a sequence.

    The high level architecture is:
    LSTM -> FC -> logit for binary prediction

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        Hidden & cell state dimension.
    num_layers : int
        Number of LSTMs to stack.
    dropout : float, default 0
        Whether to use dropout in the LSTM or not.
    bidirectional : bool, default False
        Whether to use a bidirectional LSTM or not.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        fc_hidden_dims: List[int]=None,
        num_layers: int=1,
        dropout: float=0,
        bidirectional: bool=False
    ):
        super().__init__()
        lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        # output is concatenation of final hidden and cell states
        # hidden contains short term, cell contains long term
        lstm_out_dim = 2*num_layers*hidden_dim
        if bidirectional: lstm_out_dim *= 2

        if fc_hidden_dims is not None:
            fc_dims = [lstm_out_dim, *fc_hidden_dims, 1]
            fc_list = nn.ModuleList([nn.Linear(fc_dims[i], fc_dims[i+1]) for i in range(len(fc_dims)-1)])
            fc = nn.Sequential(*fc_list)
        else:
            fc = nn.Linear(lstm_out_dim, 1)

        self.model = nn.ModuleDict({
            'lstm': lstm,
            'fc': fc
        })
    
    def forward(self, x):
        _, (hn, cn) = self.model['lstm'](x)
        x = einops.rearrange([hn, cn], 'b n d h -> n (b d h)')
        x = self.model['fc'](x)
        return x


class GRU(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class tl_RNN(tl.LightningModule):
    """Pytorch Lightning module for training RNNs that outputs binary predictions.
    
    Parameters
    ----------
    model : torch.nn.Module
        An instance of an RNN that outputs binary predictions.
    lr : float, default 1e-3
        Learning rate.
    wd : float, default 0
        Weight decay.
    """
    def __init__(self, model, lr=1e-3, wd=0, optimizer=torch.optim.Adam):
        self.model = model
        self.save_hyperparameters()
        self.optimizer = optimizer

    def forward(self, x):
        x = self.model(x) # (L N D*Hout)
        class_x = F.sigmoid(x)
        return class_x
    
    def training_step(self, x, y):
        x = self.model(x) # (L N D*Hout)
        loss = F.binary_cross_entropy_with_logits(x, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        if self.hparams.wd != 0:
            optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        return optimizer