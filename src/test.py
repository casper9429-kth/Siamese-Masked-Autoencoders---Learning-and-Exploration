from jax import random
import flax.linen as nn
import jax
import jax.numpy as jnp
import torch

from model import SiamMAE


def test():
    x = random.uniform(random.key(9), (2,3,224,224))
    y = random.uniform(random.key(123), (2,3,224,224))


    z = torch.ones((2,196,512))
    
    model = SiamMAE(depth=1)
    vars = model.init(random.key(0), x, y)
    
    pred, mask = model.apply(vars, x, y)
    print(pred.shape)
    loss = model.loss(x, pred, mask)
    print(loss)

    
if __name__=="__main__":
    test()