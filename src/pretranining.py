## Standard libraries
import os
import numpy as np
import math
import json
from functools import partial
from PIL import Image
from collections import defaultdict


## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for progress bars
from tqdm.auto import tqdm

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
from jax import random
# Seeding for random operations
main_rng = random.PRNGKey(42)

## Flax (NN in JAX)
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data/MovingMNIST"


import torch
from torchvision.datasets import MovingMNIST

dataset = MovingMNIST(root=DATASET_PATH,download=True)


print(dataset.data.shape)

# split into train and test, random split
## Create a random permutation of indices
dataset_data = dataset.data[torch.randperm(dataset.data.shape[0])]
# Split the indices into train and test
train_data = dataset_data[:8000]
test_data = dataset_data[8000:]

# Transform to JAX arrays
train_data = jnp.array(train_data)
test_data = jnp.array(test_data)

# Find mean and std of the train dataset
mean = jnp.mean(train_data,axis=(0,1,2,3,4))
std = jnp.std(train_data,axis=(0,1,2,3,4))

# Normalize the train and test dataset
train_data = (train_data - mean)/std
test_data = (test_data - mean)/std

# print the mean and std of the train dataset
print("Train data mean: ",train_data.mean())
print("Train data std: ",train_data.std())

# For each frame, sample one value in range 0.0-0.5
t1 = random.uniform(main_rng,shape=(train_data.shape[0],1),minval=0.0,maxval=0.5)
t1 = t1*20

# For each frame, sample one value in range 0.5-1.0
t2 = random.uniform(main_rng,shape=(train_data.shape[0],1),minval=0.0,maxval=0.5)
t2 = t2*20 +t1

# Make both t1 and t2 into integers
t1 = t1.astype(int)
t2 = t2.astype(int)

# # Plot train data at index t1 and t2 for 10 random samples
# fig,ax = plt.subplots(10,2,figsize=(10,20))
# for i in range(10):
#     ax[i,0].imshow(train_data[i,t1[i,0],0])
#     ax[i,1].imshow(train_data[i,t2[i,0],0])
#     ax[i,0].axis("off")
#     ax[i,1].axis("off")
# plt.show()



