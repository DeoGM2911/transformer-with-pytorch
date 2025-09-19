# ViT.py
#
# Contains the Vision Transformer module
#
# @author: Dung Tran
# @date: Sep 18, 2025


from utils import PatchEmbedding
import torch
from torch import nn
from torch.nn import functional as F
from model.utils import save_hyperparams
from model.attention import MultiHeadAttention


class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout, **kwargs):
        super(ViTMLP, self).__init__(**kwargs)
        
        # Dense & norm layers
        self.fc1 = nn.LazyLinear(mlp_num_hiddens)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, X):
        out = F.gelu(self.dropout1(self.fc1(X)))
        out = F.gelu(self.dropout2(self.fc2(out)))
        return out


class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, mlp_num_hiddens, norm_shape, num_heads, dropout, **kwargs):
        super(ViTBlock, self).__init__(**kwargs)
        save_hyperparams(self)
        
        # Norm & Attention layer
        self.ln1 = nn.LayerNorm(normalized_shape=self.norm_shape)
        self.attention = MultiHeadAttention(self.num_heads, self.num_hiddens, self.dropout)
        
        # MLP & Norm layer
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
        self.ln2 = nn.LayerNorm(norm_shape)
    
    def forward(self, X):
        out = self.attention(self.ln1(X))
        out = self.ln2(self.mlp(out))
        return out


class ViT(nn.Module):
    """Vision Transformer"""
    def __init__(self, num_output, num_hiddens, mlp_num_hiddens, max_patches, num_layers, num_heads, dropout, img_size, patch_size, **kwargs):
        super(ViT, self).__init__(**kwargs)
        save_hyperparams(self)
        
        # Ensure img_size is a tuple
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        # Patch embedding and learnable positional encoding
        self.patch_embed = PatchEmbedding(num_hiddens, patch_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.max_patches, num_hiddens))
        
        # learnable cls token
        self.cls = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        
        # Attentions layers
        self.attention = nn.Sequential()
        for i in range(num_layers):
            self.attention.add_module(f"blk_{i}", ViTBlock(num_hiddens, mlp_num_hiddens, num_hiddens, num_heads,
                                                        dropout))
        
        # Linear layer for prediction
        self.ln = nn.LayerNorm(num_hiddens)
        self.fc = nn.LazyLinear(num_output)
    
    def forward(self, X):
        # attention weights
        self.attention_weights = [None] * self.num_layers
        
        # Patch embedding and prepend the cls token
        X = torch.concat([self.cls.expand(X.shape[0], -1, -1), self.patch_embed(X)], dim=1)
        
        # Pos. Encoding
        X = X + self.pos_enc[:, :X.shape[1], :]
        
        # Attention layers
        out = self.attention(X)
        for i in range(self.num_layers):
            self.attention_weights[i] = self.attention.get_submodule(f"blk_{i}").attention.attention.attention_weights
        
        # Linear layer with the cls token
        out = self.fc(self.ln(out[:, 0, :]))
        return out
