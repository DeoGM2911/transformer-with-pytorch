# utils.py
#
# Contain the utility classes for the ViT
# 
# @author: Dung Tran
# @date: Sep. 18, 2025

import torch
from torch import nn
from torch.nn import functional as F
from model.utils import save_hyperparams


class PatchEmbedding(nn.Module):
    """Convert images into patch vectors for attention"""
    def __init__(self, num_hiddens, img_size, patch_size, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        # Convert patch_size to a tuple
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        save_hyperparams(self)
        self.num_patches = img_size[0] // patch_size[0] * img_size[1] // patch_size[1]
        
        # Use convolution to embed the patches
        # batch_size, num_hiddens, new_height, new_width
        self.conv = nn.LazyConv2d(self.num_hiddens, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, X):
        return self.conv(X).flatten(2).transpose(2, 1)
