import torch
from ..models import rnn

# TODO: test models' forward()

L, N, Hin = 3, 5, 4
inpt_shape = (L, N, Hin)

def test_forward_rnn():
    pass

def test_forward_lstm():
    inpt = torch.zeros([]) # (L N Hin)

def test_forward_gru():
    pass
