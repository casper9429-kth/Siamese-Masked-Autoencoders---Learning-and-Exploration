# Tutorial 2 (JAX): Introduction to JAX+Flax
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html#JAX-as-NumPy-on-accelerators


## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.auto import tqdm

# Import JAX
import jax
import jax.numpy as jnp
print("Using jax", jax.__version__)

# Create a array of zeros
a = jnp.zeros((2, 5), dtype=jnp.float32)
print(a)

# Create a array 
b = jnp.arange(10, dtype=jnp.float32)
print(b)
# Print the type of b and the device it is stored on
print(b.__class__)
print(b.device())

# Transfer to CPU
b = jax.device_get(b)
print(b.__class__)

# Transfer to GPU
b = jax.device_put(b, jax.devices("cpu")[0])
print(b.device())

# Print the available devices
print(jax.devices())

# JAX tensors are immutable
try:
    b[0] = 1.0 # This will raise an error
except Exception as e:
    print(e)

print(b)
b = b.at[0].set(1.0) # This is the correct way to set a value
print(b)

rng = jax.random.PRNGKey(42)
# A non-desirable way of generating pseudo-random numbers...
jax_random_number_1 = jax.random.normal(rng)
jax_random_number_2 = jax.random.normal(rng)
print('JAX - Random number 1:', jax_random_number_1)
print('JAX - Random number 2:', jax_random_number_2)

# Typical random numbers in NumPy
np.random.seed(42)
np_random_number_1 = np.random.normal()
np_random_number_2 = np.random.normal()
print('NumPy - Random number 1:', np_random_number_1)
print('NumPy - Random number 2:', np_random_number_2)


# Illustration of the difference between JAX and NumPy random numbers
rng = jax.random.PRNGKey(42)
rng_1, rng_2 = jax.random.split(rng, num=2)
for i in range(3):
    rng_1, rng_2 = jax.random.split(rng_1, num=2)
    print('JAX - Random number:', jax.random.normal(rng))
    print('NumPy - Random number:', np.random.normal())
    print('JA - Random number:', jax.random.normal(rng_2))
    print('---')

