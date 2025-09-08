# decoder.py
# 
# The transformer decoder module.
#
# @author: Dung Tran
# @date: September 8, 2025

import torch
from torch import device, nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.encoding import PositionalEncoding
from model.utils import PositionWiseFFN, AddNorm, save_hyperparams


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block.
    """
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, index, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        save_hyperparams(self)
        # The index of this block in the decoder
        self.i = index

        # Self-attention layer
        self.attention1 = MultiHeadAttention(num_heads, num_hiddens, dropout)
        # Encoder-Decoder attention layer
        self.attention2 = MultiHeadAttention(num_heads, num_hiddens, dropout)
        
        # Position-wise feed-forward network
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens, dropout)
        
        # Add & Norm layers
        self.ln1 = AddNorm(dropout)
        self.ln2 = AddNorm(dropout)
        self.ln3 = AddNorm(dropout)
        
    def forward(self, X, state):
        """
        Forward function for the decoder block.
        Note that the state will hold the enc_outputs, valid_lens, and the current output sequence of all decoder blocks.
        """
        # Extract the enc_outputs and valid_lens from the state
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        # Get the current output sequence for self-attention
        if state[2][self.i] is None:
            key_values = X  # Initialize the output if needed
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        
        # If this is training, we want the masked self-attention to ensure the autoregressive property of the model
        if self.training:
            batch_size, num_steps, _ = X.shape
            # For the i^th query, we just attend to the first i keys
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)  # [[1, 2, ..., num_steps] * batch_size]
        else:
            dec_valid_lens = None
        
        # Self-attention + Add & Norm
        X2 = self.ln1(X, self.attention1(X, key_values, key_values, dec_valid_lens))
        # Encoder-Decoder attention + Add & Norm
        X3 = self.ln2(X2, self.attention2(X2, enc_outputs, enc_outputs, enc_valid_lens))
        # Position-wise FFN + Add & Norm
        Y = self.ln3(X3, self.ffn(X3))
        return Y, state


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder blocks.
    """
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        save_hyperparams(self)
        
        # Embeddings & Positional Encoding
        self.embed = nn.Embedding(vocab_size, num_hiddens)
        self.pos_enc = PositionalEncoding(num_hiddens, dropout)
        
        # Decoder blocks
        self.decoder_blks = nn.Sequential()
        for i in range(num_layers):
            self.decoder_blks.add_module(f"blk_{i}", TransformerDecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        
        # Output layer
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, enc_valid_lens):
        """
        Initialize the state for the decoder.
        The state will hold the enc_outputs, valid_lens, and the current output sequence of all decoder blocks.
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, X, state):
        # Attention weights for decoder self-attention and encoder-decoder attention
        self.attention_weights = [[None] * self.num_layers, [None] * self.num_layers]
        
        # Embedding + Positional Encoding
        X = self.pos_enc(self.embed(X) * (self.num_hiddens ** 0.5))
        
        for layer in self.decoder_blks:
            X, state = layer(X, state)
            # Record the attention weights of this block
            # Self-attention
            self.attention_weights[0][layer.i] = layer.attention1.attention.attention_weights
            # Encoder-Decoder attention
            self.attention_weights[1][layer.i] = layer.attention2.attention.attention_weights

        # Output layer
        output = self.dense(X)
        return output, state