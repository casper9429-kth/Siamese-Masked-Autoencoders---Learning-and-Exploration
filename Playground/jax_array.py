import jax.numpy as jnp
import numpy as np

# Create a random array of size 10x100x100x3
x = jnp.array(np.random.rand(10, 100, 100, 3))
y = jnp.array(np.random.rand(10, 100, 100, 3))

for x_mini,y_mini in zip(x,y):
    print(x_mini.shape)
    print(y_mini.shape)
    print(x_mini)
    print(y_mini)
    break
