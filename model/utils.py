# utils.py
#
# Utility functions for models.
#
# @author: Dung Tran
# @date: September 8, 2025

import inspect
from torch import nn
import torch
import torch.nn.functional as F


class PositionWiseFFN(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, ffn_num_hiddens, num_hiddens, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        save_hyperparams(self)
        # Linear map with 2 layers and ReLU. Note the output shape is unchanged.
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, num_hiddens)

    def forward(self, X):
        return self.dense2(F.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """
    Residual connection followed by layer normalization.
    """
    def __init__(self, norm_shape, dropout, **kwargs):
        save_hyperparams(self)
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape=norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def save_hyperparams(self: object):
    """
    Save the hyperparameters to the class instance.
    """
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    params = {k: v for k, v in local_vars.items() if k != 'self'}
    
    for k, v in params.items():
        if not self.__dict__.get(k, None):
            self.__setattr__(k, v)


class Test():
    """Unit test for save_hyperparams"""
    def __init__(self, a, b, c=3):
        save_hyperparams(self)
    
    def show(self):
        print(self.a, self.b, self.c)
    
if __name__ == "__main__":
    t = Test(1, 2)
    t.show()  # Output: 1 2 3
    
    t = Test(4, 5, 2)
    t.show()  # Output: 4 5 2