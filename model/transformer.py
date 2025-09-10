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
from model.utils import save_hyperparams


class TransformerSeq2Seq(nn.Module):
    """
    Transformer model.
    """
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, target_pad, **kwargs):
        super(TransformerSeq2Seq, self).__init__(**kwargs)
        save_hyperparams(self)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, enc_valid_lens):
        """
        Forward function for the transformer model.

        """
        enc_outputs, enc_valid_lens = self.encoder(enc_X, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        dec_outputs = self.decoder(dec_X, state)
        return dec_outputs[0]

    def predict(self, enc_X, enc_valid_lens, device, start_token, num_steps=5000, beam_size=1, keep_attention_weights=False):
        """
        Predict function for the transformer model.
        """
        # Set the model to evaluation mode
        self.eval()

        batch_size, _ = enc_X.shape
        outputs = [start_token.to(device).unsqueeze(0).repeat(batch_size)]
        attention_weights = [] if keep_attention_weights else None
        
        with torch.no_grad():
            # Get the encoder outputs
            enc_outputs, enc_valid_lens = self.encoder(enc_X, enc_valid_lens)
            
            # Prepare the initial states
            dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
            
            # Beam search for the rest steps
            for _ in range(num_steps):
                dec_X = outputs[-1]
                dec_outputs = self.decoder(dec_X, dec_state)  # dec_outputs[0] shape: (batch_size, 1, vocab_size)
                
                # Record the outputs and attention weights
                if keep_attention_weights:
                    attention_weights.append(dec_outputs[2])
                outputs.append(dec_outputs[0].argmax(dim=-1))

        return torch.cat(outputs, dim=1), attention_weights
    