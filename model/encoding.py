# encoding.py
#
# Contain encoding-related classes and functions.
#
# @author: Dung Tran
# @date: September 8, 2025

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=5000):
        """
        Positional encoding module.
        
        # Params
        
        - num_hiddens: Dimension of the embedding layer in the model.
        - max_len: Maximum length in # of tokens of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        
        # Pre allocate the positional encoding matrix
        self.P = torch.zeros((1, max_len, num_hiddens))
        
        # Compute the positional encodings with sin/cos functions
        X = torch.arange(1, max_len + 1, dtype=torch.float32).unsqueeze(1)  # Shape (max_len, 1)
        # Even column indices
        self.P[:, :, 0::2] = torch.sin(X / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))
        # Odd column indices
        self.P[:, :, 1::2] = torch.cos(X / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))
        
    def forward(self, X):
        X = X + self.P[:, X.shape[1], :].to(X.device)
        return X


# Unit test for positional encoding
if __name__ == "__main__":
    num_hiddens = 32
    pos_encoding = PositionalEncoding(num_hiddens)
    print(pos_encoding.P.shape)
    print(pos_encoding.P)