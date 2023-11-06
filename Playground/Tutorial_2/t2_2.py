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


def simple_graph(x):
    """
    Add 2 to the input, square it, add 3, and take the mean
    """
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y

inp = jnp.arange(3, dtype=jnp.float32)
print('Input', inp)
print('Output', simple_graph(inp))
# Visualize the computation graph
from jax import make_jaxpr
simple_graph_jaxpr = make_jaxpr(simple_graph)(inp)
print(simple_graph_jaxpr)


# Autodiff
from jax import grad
grad_simple_graph = grad(simple_graph)
print('Gradient', grad_simple_graph(inp))
# Visualize the computation graph
from jax import make_jaxpr
grad_simple_graph_jaxpr = make_jaxpr(grad_simple_graph)(inp)
print(grad_simple_graph_jaxpr)

# Value and gradient
from jax import value_and_grad
val_grad_simple_graph = value_and_grad(simple_graph)
print('Value and gradient', val_grad_simple_graph(inp))
# Visualize the computation graph
from jax import make_jaxpr
val_grad_simple_graph_jaxpr = make_jaxpr(val_grad_simple_graph)(inp)
print(val_grad_simple_graph_jaxpr)

# Hessian
from jax import hessian
hess_simple_graph = hessian(simple_graph)
print('Hessian', hess_simple_graph(inp))
# Visualize the computation graph
from jax import make_jaxpr
hess_simple_graph_jaxpr = make_jaxpr(hess_simple_graph)(inp)
print(hess_simple_graph_jaxpr)

# Jit
from jax import jit
jit_simple_graph = jit(simple_graph)
print('Jit', jit_simple_graph(inp))
# Visualize the computation graph
from jax import make_jaxpr
jit_simple_graph_jaxpr = make_jaxpr(jit_simple_graph)(inp)


# Create a new random subkey for generating new random values
rng = jax.random.PRNGKey(42)
rng, normal_rng = jax.random.split(rng)
large_input = jax.random.normal(normal_rng, (1000,))
# Run the jitted function once to start compilation
jitted_function = jax.jit(simple_graph)
_ = jitted_function(large_input).block_until_ready()



