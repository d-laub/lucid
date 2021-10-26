import torch
import torch.nn.functional as F
import einops
from src.models import rnn

torch.manual_seed(0)
seqlen, batch_n, x_dim = 3, 10, 5
num_emb = x_dim
emb_dim = 5
x_shape = (seqlen, batch_n, x_dim)
hidden_dim = 3

def test_forward_rnn():
    pass

def test_forward_gru():
    pass

def test_forward_lstm():
    x = torch.randint(0, x_dim, (batch_n,)) # (N), index based
    x = einops.repeat(x, 'n -> l n', l=seqlen) # (L N)
    lstm = rnn.LSTM(num_emb=num_emb, emb_dim=emb_dim, hidden_dim=hidden_dim)
    x = lstm(x) # (N 1)
    assert x.shape == (batch_n, 1)