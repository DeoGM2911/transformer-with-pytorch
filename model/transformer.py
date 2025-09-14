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
        self.tgt_pad = target_pad
    
    def forward(self, enc_X, dec_X, enc_valid_lens):
        """
        Forward function for the transformer model.

        """
        enc_outputs, enc_valid_lens = self.encoder(enc_X, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        dec_outputs = self.decoder(dec_X, state)
        return dec_outputs[0]

    def predict(self, enc_X, enc_valid_lens, device, num_steps=1000, beam_size=1, keep_attention_weights=False):
        """
        Predict function for the transformer model.
        """
        # Set the model to evaluation mode
        self.eval()

        batch_size, _ = enc_X.shape
        outputs = [torch.tensor(self.tgt_pad).to(device).repeat(batch_size).unsqueeze(1)]
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

        return torch.cat(outputs[1:], dim=1), attention_weights


# Unit test for the transformer model
if __name__ == "__main__":
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 4, 5
    num_heads, ffn_num_hiddens = 8, 64
    vocab_size = 10
    encoder = TransformerEncoder(vocab_size, num_hiddens, num_heads, num_layers, ffn_num_hiddens, dropout)
    decoder = TransformerDecoder(vocab_size, num_hiddens, num_heads, num_layers, ffn_num_hiddens, dropout)
    model = TransformerSeq2Seq(encoder, decoder, target_pad=0)
    dummy_enc_input = torch.ones((batch_size, num_steps), dtype=torch.long)
    dummy_dec_input = torch.ones((batch_size, num_steps), dtype=torch.long)
    dummy_enc_valid_lens = torch.tensor([3, 2, 0, 4])
    output = model(dummy_enc_input, dummy_dec_input, dummy_enc_valid_lens)
    print(output.shape)
    
    # Test prediction
    dummy_predict_input = torch.tensor([[1, 1, 1, 2, 2]])
    dummy_enc_valid_lens_predict = torch.tensor([3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps_pred = 100
    output, attention_weights = model.predict(dummy_predict_input, dummy_enc_valid_lens_predict, device, num_steps_pred)
    print(output.shape)
