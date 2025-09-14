# attention.py
#
# Contain modules for different type of attention mechanisms
#
# @author: Dung Tran
# @date: August 30, 2025

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import math


class Attention(nn.Module):
    """
    Base class for attention mechanism
    """
    def __init__(self, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
    
    def masked_softmax(self, scores: torch.Tensor, valid_lengths: torch.Tensor=None):
        """
        Use masked softmax to normalize the attention scores.
        
        # Parameters:
        
        - scores: the attention scores that need normalizing. Expect of shape (batch_size, num_queries, num_keys)
        - valid_lenghts: the lengths of the sequences of which we are calculating the scores (batch_size) or (batch_size, num_queries).
        The values in valid_lengths must be less than or equal to num_keys.
        
        # Returns:
        
        - the normalized attention scores with 0 score at the end of the sequences.
        """
        # Create the mask to apply values
        def _sequence_mask(X, valid_len, value=0):
            """Note that X must be shaped (batch_size * num_queries, num_keys)"""
            maxlen = X.size(1)  # num_keys
            # The slices are for broadcasting
            mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
            X[~mask] = value
            return X
        
        # Compute usual if no valid length is provided
        if valid_lengths is None:
            return F.softmax(scores, dim=-1)
        else:
            shape = scores.shape 
            if valid_lengths.dim() == 1:
                valid_lengths = torch.repeat_interleave(valid_lengths, repeats=scores.shape[1])  # Repeat num_queires time
            else:
                valid_lengths = valid_lengths.reshape(-1)

            scores = _sequence_mask(scores.reshape(-1, shape[-1]), valid_lengths, value=-1e6)
            return F.softmax(scores.reshape(shape), dim=-1)  # Output is shaped (batch_size, num_queries, num_keys)


class AdditiveAttention(Attention):
    """
    Additive attention mechanism.
    """
    def __init__(self, hidden_size, drop_out=0.2, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)

        # Linear layers for queries, keys, and values
        self.W_q = nn.LazyLinear(hidden_size, bias=False)
        self.W_k = nn.LazyLinear(hidden_size, bias=False)
        self.W_v = nn.LazyLinear(1, bias=False)
        
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, queries, keys, values, valid_lengths=None):
        """
        Compute the normalized attention scores for the given queires, keys, and values.
        Optionally, one can provide the valid_lengths for the samples.
        """
        out_q = self.W_q(queries)  # (batch_size, num_queries, hidden_size)
        out_k = self.W_k(keys)      # (batch_size, num_keys, hidden_size)

        # Combine the queries and keys
        features = torch.tanh(out_q.unsqueeze(2) + out_k.unsqueeze(1))  # (batch_size, num_queries, num_keys, hidden_size)
        scores = self.W_v(features).squeeze(-1)  # (batch_size, num_queries, num_keys)
        
        # Calculate attention weights
        self.attention_weights = self.masked_softmax(scores, valid_lengths)  # (batch_size * num_queries, num_keys)

        return torch.bmm(self.dropout(self.attention_weights.reshape(queries.shape[0], queries.shape[1], -1)), values)


class DotProductAttention(Attention):
    """
    Dot-product attention mechanism.
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lengths=None):
        """
        Compute the normalized attention scores for the given queries, keys, and values.
        Optionally, one can provide the valid_lengths for the samples.
        """

        d = queries.shape[-1]
        
        # Compute dot-product attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.tensor(math.sqrt(d))  # (batch_size, num_queries, num_keys)
        self.attention_weights = self.masked_softmax(scores, valid_lengths)
        
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(Attention):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, num_heads, hidden_size, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        # Shared attention layer
        self.attention = DotProductAttention(dropout=dropout)

        # Linear transformation of query, key-value subspaces
        self.Wq = nn.LazyLinear(hidden_size, bias=bias)
        self.Wk = nn.LazyLinear(hidden_size, bias=bias)
        self.Wv = nn.LazyLinear(hidden_size, bias=bias)
        
        # Combine the result of all the heads
        self.Wo = nn.LazyLinear(hidden_size)
    
    def transpose_qkv(self, x: torch.Tensor):
        """
        Given a tensor of shape (batch_size, time_step, hidden_size), return that same tensor with shape
        (batch_size * num_heads, time_step, hidden_size / num_heads). In essence split the last dim into num_heads
        subspaces for parallel computing.
        """
        # Make it 4 dimensional
        assert x.dim() == 3
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[-1])

    def transpose_output(self, x: torch.Tensor):
        """
        Given a tensor of shape (batch_size * num_heads, timestep, hidden_size / num_heads), return the same tensor with
        shape (batch_size, time_step, hidden_size)"""
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens=None):
        # Reshape the queries, keys, and values for parallel computing
        queries = self.transpose_qkv(self.Wq(queries))
        keys = self.transpose_qkv(self.Wk(keys))
        values = self.transpose_qkv(self.Wv(values))
        
        if valid_lens is not None:
            # Repeat the valid lengths to match the number of attention heads.
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)  # batch_size -> batch_size * num_heads

        # Compute attention weights & output
        outputs = self.attention(queries, keys, values, valid_lens)  # Output with shape (batch_size * num_heads, timestep, hidden_size / num_heads)
        
        # Reshape output
        outputs = self.transpose_output(outputs)
        return self.Wo(outputs)