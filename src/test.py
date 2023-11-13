from jax import random
import flax.linen as nn
import jax
import jax.numpy as jnp

from model import SiamMAE




    
if __name__=="__main__":
    x = random.uniform(random.key(9), (2,3,224,224))
    y = random.uniform(random.key(123), (2,3,224,224))
    
    model = SiamMAE()
    vars = model.init(random.key(0), x, y, 0.9)
    pred, mask = model.apply(vars, x, y, 0.9, train=True)
    print(pred.shape)
    loss = model.loss(x, pred, mask)
    print(loss)

    