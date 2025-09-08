# encoder.py
#
# The transformer encoder module.
#
# @author: Dung Tran
# @date: September 8, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.encoding import PositionalEncoding
from model.utils import save_hyperparams
from model.utils import PositionWiseFFN, AddNorm


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block.
    """
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        save_hyperparams(self)

        # Self attention layer
        self.attention = MultiHeadAttention(num_heads, num_hiddens, dropout)
        # Position-wise feed-forward network
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens, dropout)
        
        # Add & Norm layers
        self.ln1 = AddNorm(dropout)
        self.ln2 = AddNorm(dropout)

    def forward(self, X, valid_lens=None):
        # Self-attention + Add & Norm
        Y = self.ln1(X, self.attention(X, X, X, valid_lens))
        return self.ln2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder blocks.
    """
    def __init__(self, vocab_size, num_hiddens, num_heads, num_layers, ffn_num_hiddens, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        save_hyperparams(self)
        
        # Embeddings & Positional Encoding
        self.embed = nn.Embedding(vocab_size, num_hiddens)
        self.pos_enc = PositionalEncoding(num_hiddens, dropout)
        
        # Encoder blocks
        self.encoder_blks = nn.Sequential()
        for i in range(num_layers):
            self.encoder_blks.add_module(f"blk_{i}", TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout))

    def forward(self, X, valid_lens=None):
        # Attention weights for encoder self-attention
        self.attention_weights = [None] * self.num_layers
        
        # Embedding + Positional Encoding
        X = self.pos_enc(self.embed(X) * (self.num_hiddens ** 0.5))
        for i, layer in enumerate(self.encoder_blks):
            X = layer(X, valid_lens)
            # Record the attention weights of this block
            self.attention_weights[i] = layer.attention.attention.attention_weights
        
        return X, valid_lens