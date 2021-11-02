import torch
from src.models import rnn
import pytest

seqlen, batch_n, x_dim = 3, 10, 5
num_emb = x_dim
emb_dim = 5
hidden_dim = 3

@pytest.fixture
def x():
    torch.manual_seed(0)
    x = torch.randint(0, x_dim, (seqlen, batch_n)) # (N), index based
    return x

def test_forward_rnn():
    pass

def test_forward_gru():
    pass

def test_forward_lstm(x):
    lstm = rnn.LSTM(num_emb=num_emb, emb_dim=emb_dim, hidden_dim=hidden_dim)
    x = lstm(x) # (N 1)
    assert x.shape == (batch_n, 1)
    print('yay')