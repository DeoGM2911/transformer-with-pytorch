# bahdanau.py
#
# Implement the Bahdanau attention mechanism for RNN encoder-decoder architecture
#
# @author: Dung Tran
# @date: September 1, 2025


import torch
import torch.nn as nn
from attention import AdditiveAttention


class Seq2SeqEncoder(nn.Module):
    """
    The sequence to sequence encoder using GRU.
    # Returns
    
    - enc_outputs: Tensor of shape (batch_size, seq_len, hidden_dim)
    - hidden_states: Tensor of shape (n_layers, batch_size, hidden_dim)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = emb_dim
        self.hidden_size = hidden_dim
        self.rnn_num_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, X):
        embedded = self.embedding(X)
        enc_outputs, hidden_states = self.rnn(embedded, torch.zeros((self.rnn_num_layers, embedded.size(0), self.hidden_size)))
        return enc_outputs, hidden_states


class Seq2SeqDecoder(nn.Module):
    """
    The sequence to sequence decoder using GRU and the Bahdanau attention mechanism.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_dim
        self.rnn_num_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.attention = AdditiveAttention(hidden_dim)
        self.fc = nn.LazyLinear(vocab_size)
    
    @property
    def get_attention_weights(self):
        return self.attention_weights
    
    def init_states(self, states, valid_lens):
        """Initialize the states for the first time step. Expect states to be the encoder's output"""
        enc_output, hidden_states = states
        return enc_output, hidden_states, valid_lens

    def forward(self, X, states):
        # Outputs
        self.outputs = []
        self.attention_weights = []
        
        # Embed the input
        X = self.embedding(X)
        # Init the states
        enc_output, hidden_states, valid_lens = states
        
        # Decoding steps
        for x in X.permute(1, 0, 2):  # Permute to shape (timestep, batch_size, embed_dim)
            # Prepare the queries and the key-value pairs
            queries = hidden_states[-1].unsqueeze(1)  # Add a dim to reshape (num_batch, 1, hidden_dim)
            
            # Compute the context
            context = self.attention(queries, enc_output, enc_output, valid_lens)
            self.attention_weights.append(self.attention.attention_weights)
            
            # Compute the output of this timestep
            new_context = torch.cat([context, x.unsqueeze(1)], dim=-1)  # Unsqueeze x to be shaped (batch_size, 1, embed_dim)
            output, hidden_states = self.rnn(new_context, hidden_states)
            self.outputs.append(output)

        # Concat the outputs to recreate the sequence
        self.outputs = self.fc(torch.cat(self.outputs, dim=1))  # Shape (batch_size, time_step, num_hiddens) 
        return self.outputs, [enc_output, hidden_states, valid_lens]


class Seq2Seq(nn.Module):
    ...
