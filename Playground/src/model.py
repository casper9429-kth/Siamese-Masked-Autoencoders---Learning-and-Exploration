# Inspired by: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html
# A very general training script for jax consitent over all the UVADLC notebooks

import os
import time
from tqdm.auto import tqdm
from typing import Sequence, Any
from collections import defaultdict
from utils.get_obj_from_str import get_obj_from_str
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.example_libraries import stax, optimizers
# from functools import partial
import omegaconf
from jax.config import config
import flax
import flax.core
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.datasets import STL10




class Attention(nn.Module):
    """
        (Multi-head) self-attention layer.    (JAX)
    """
    embed_dim : int   # Dimensionality of input and attention feature vectors
    num_heads : int   # Number of heads to use in the Multi-Head Attention block

    def setup(self):
        self.qkv = nn.linear(self.embed_dim, self.embed_dim*3)
        self.num_heads = self.num_heads
        # self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,embed_dim=self.embed_dim)


    def __call__(self, x, train=True):

        B,N,D = x.shape
        proj_dim = D // self.num_heads
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4)
        qkv = jnp.reshape(self.qkv(x), (B, N, 3, self.num_heads, proj_dim))
        
        
        # 
        q, k, v = qkv[0], qkv[1], qkv[2] # shape: B x num_heads x N x proj_dim


        att_scores = q @ k.transpose(2,3) # B x num_heads x N x N
        att_scores_sm = nn.softmax(att_scores, -1)
        weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x N x N x proj_dim
        sum = jnp.sum(weighted_vals, axis=2) # B x num_heads x N x proj_dim
        
        out = jnp.reshape(sum, (B, N, D))

        return out



class CrossAttention(nn.Module):
    """
        Cross-attention layer.    
    """
    embed_dim : int   # Dimensionality of input and attention feature vectors
    hidden_dim : int  # Dimensionality of hidden layer in feed-forward network
    num_heads : int   # Number of heads to use in the Multi-Head Attention block
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,embed_dim=self.embed_dim)
        

        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim)
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, y, train=True):
        inp_x = self.layer_norm_1(x)
        inp_y = self.layer_norm_1(y)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_y)
        x = x + self.dropout(attn_out, deterministic=not train)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        return x