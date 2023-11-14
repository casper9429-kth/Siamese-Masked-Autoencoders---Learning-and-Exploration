from jax import random
import flax.linen as nn
import jax
import jax.numpy as jnp
import torch

from model import SiamMAE


def test_me(x):
    mask_token = torch.zeros(1, 1, 512)
    ids_restore = torch.ones((2, 196), dtype=torch.int64)
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat((x[:, 1:, :], mask_tokens), dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    
if __name__=="__main__":
    x = random.uniform(random.key(9), (2,3,224,224))
    y = random.uniform(random.key(123), (2,3,224,224))


    z = torch.ones((2,196,512))
    
    model = SiamMAE(depth=1)
    vars = model.init(random.key(0), x, y, 0.9)
    pred, mask = model.apply(vars, x, y, 0.9)
    print(pred.shape)
    loss = model.loss(x, pred, mask)
    print(loss)

    
