from model.bahdanau_seq2seq import Seq2SeqEncoder, Seq2SeqDecoder
import torch


def check_shape(tensor, expected_shape):
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {tensor.shape}"


vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)

X = torch.zeros((batch_size, num_steps), dtype=torch.long)
state = decoder.init_states(encoder(X), None)
output, state = decoder(X, state)
check_shape(output, (batch_size, num_steps, vocab_size))
check_shape(state[0], (batch_size, num_steps, num_hiddens))
check_shape(state[1][0], (batch_size, num_hiddens))