# transformer.py
#
# The transformer module.
#
# @author: Dung Tran
# @date: September 8, 2025

import torch
import torch.nn as nn
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Transformer model.
    """
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, X, targets, enc_valid_lens):
        """
        Forward function for the transformer model.

        #@ Will be used in the training loop. The integration of sampling and beam search will be implemented.
        """
        enc_outputs, enc_valid_lens = self.encoder(X, enc_valid_lens)
        dec_outputs = self.decoder(targets, enc_outputs, enc_valid_lens)
        return dec_outputs