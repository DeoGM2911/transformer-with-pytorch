# test.py
#
# Unit test for the attention mechanisms
#
# @author: Dung Tran
# @date: August 31, 2025

from attention import DotProductAttention, AdditiveAttention, MultiHeadAttention
from test_bahdanau import check_shape
import torch


def test_additive_attention_random():
    # Random test for AdditiveAttention
    attention = AdditiveAttention(hidden_size=8)
    attention.eval()  # Set to evaluation mode
    queries = torch.rand(2, 3, 8)  # (batch_size=2, num_queries=3, hidden_size=8)
    keys = torch.rand(2, 4, 8)     # (batch_size=2, num_keys=4, hidden_size=8)
    values = torch.rand(2, 4, 6)   # (batch_size=2, num_keys=4, value_dim=6)
    valid_lengths = torch.tensor([4, 3])

    output = attention(queries, keys, values, valid_lengths)
    print("AdditiveAttention Random Test Output:", output)
    assert output.shape == (2, 3, 6)


def test_additive_attention_known():
    # Known result test for AdditiveAttention
    attention = AdditiveAttention(hidden_size=2)
    attention.eval()
    queries = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    values = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (1, 2, 2)
    valid_lengths = torch.tensor([2])

    output = attention(queries, keys, values, valid_lengths)
    print("AdditiveAttention Known Test Output:", output)
    assert output.shape == (1, 1, 2)


def test_dot_product_attention_random():
    # Random test for DotProductAttention
    attention = DotProductAttention()
    queries = torch.rand(2, 3, 8)  # (batch_size=2, num_queries=3, hidden_size=8)
    keys = torch.rand(2, 4, 8)     # (batch_size=2, num_keys=4, hidden_size=8)
    values = torch.rand(2, 4, 6)   # (batch_size=2, num_keys=4, value_dim=6)
    valid_lengths = torch.tensor([4, 3])

    output = attention(queries, keys, values, valid_lengths)
    print("DotProductAttention Random Test Output:", output)
    assert output.shape == (2, 3, 6)


def test_dot_product_attention_known():
    # Known result test for DotProductAttention
    attention = DotProductAttention()
    queries = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    values = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (1, 2, 2)
    valid_lengths = torch.tensor([2])

    output = attention(queries, keys, values, valid_lengths)
    print("DotProductAttention Known Test Output:", output)
    assert output.shape == (1, 1, 2)


def test_additive_attention_none_valid_lengths():
    # Test AdditiveAttention with None valid_lengths
    attention = AdditiveAttention(hidden_size=8)
    attention.eval()
    queries = torch.rand(2, 3, 8)  # (batch_size=2, num_queries=3, hidden_size=8)
    keys = torch.rand(2, 4, 8)     # (batch_size=2, num_keys=4, hidden_size=8)
    values = torch.rand(2, 4, 6)   # (batch_size=2, num_keys=4, value_dim=6)

    output = attention(queries, keys, values, valid_lengths=None)
    print("AdditiveAttention None valid_lengths Test Output:", output)
    assert output.shape == (2, 3, 6)


def test_dot_product_attention_none_valid_lengths():
    # Test DotProductAttention with None valid_lengths
    attention = DotProductAttention()
    queries = torch.rand(2, 3, 8)  # (batch_size=2, num_queries=3, hidden_size=8)
    keys = torch.rand(2, 4, 8)     # (batch_size=2, num_keys=4, hidden_size=8)
    values = torch.rand(2, 4, 6)   # (batch_size=2, num_keys=4, value_dim=6)

    output = attention(queries, keys, values, valid_lengths=None)
    print("DotProductAttention None valid_lengths Test Output:", output)
    assert output.shape == (2, 3, 6)


if __name__ == "__main__":
    # test_additive_attention_random()
    # test_additive_attention_known()
    # test_dot_product_attention_random()
    # test_dot_product_attention_known()
    # test_additive_attention_none_valid_lengths()
    # test_dot_product_attention_none_valid_lengths()
    # att = AdditiveAttention(hidden_size=8)
    # print(att.masked_softmax(torch.randn(2, 3, 4), torch.tensor([[2, 3, 1], [1, 4, 2]])))
    
    # Test multi head attention
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_heads, num_hiddens, 0.5)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    query = torch.ones((batch_size, num_queries, num_hiddens))
    kv_pair = torch.ones((batch_size, num_kvpairs, num_hiddens))
    check_shape(attention(query, kv_pair, kv_pair, valid_lens),
                    (batch_size, num_queries, num_hiddens))
    print("Done!")
