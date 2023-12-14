"""
Recreate figure 5 from the paper.
"""
from test_loader import TestLoader
from model import SiamMAE
from util.patchify import unpatchify, patchify

import jax.numpy as jnp
import jax
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import orbax.checkpoint

class ReproduceF5():
    def __init__(self, dataloader, model, checkpoint_path):
        self.dataloader = dataloader
        self.model = model
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(checkpoint_path)
        self.model_params = restored['model']['params']
        self.patch_size = 16

        self.cls_token = self.model_params['cls_token']
        self.cls_token = jnp.tile(self.cls_token, (1, 196, 1))

        self.attention = nn.MultiHeadDotProductAttention(num_heads=1)
        self.att_vars = self.attention.init(jax.random.PRNGKey(0), jnp.zeros_like(self.cls_token))

    def get_attention_map(self, x):
        attention_map = self.attention.apply(self.att_vars, x, self.cls_token)

        return attention_map
    
    def run(self):
        pred, mask = self.model.apply(self.model_params, self.dataloader.get_test_batch(), mutable=False)
