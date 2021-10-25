import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as tl
from torch.optim import optimizer

class LSTM(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: bool=False,
        bidirectional: bool=False
    ):
        enc = nn.Embedding(542, emb_dim)
        lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_out_dim = 2*hidden_dim if bidirectional else hidden_dim
        fc = nn.Linear(lstm_out_dim, 1)
        self.model = nn.Sequential(
            enc,
            lstm,
            fc
        )
    
    def forward(self, x):
        return self.model(x)


class tl_LSTM(tl.LightningModule):
    def __init__(self, model, lr=None, weight_decay=None):
        self.model = model

    def forward(self, x):
        x = self.model(x)
        class_x = F.sigmoid(x[-1])
        return class_x
    
    def training_step(self, x, y):
        x = self.model(x)
        pred_x = x[-1]
        loss = F.binary_cross_entropy_with_logits(pred_x, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
