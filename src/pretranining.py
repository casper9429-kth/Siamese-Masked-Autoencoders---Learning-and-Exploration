"""
import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.example_libraries import stax, optimizers
from torch.utils import data
# from functools import partial
import omegaconf
from jax.config import config
# IMPORTANT NOTE:
# if you have got a NaN loss and/or have trouble debugging. Then, set
# jax_disable_jit to True. This will help you print out the variables.
import flax
import flax.core

config.update('jax_disable_jit', False)


# Get the parameters as a omegaconf 
hparams = omegaconf.OmegaConf.load("hparams.yaml")

# Get the dataset 
from dataset import PreTrainingDataset # pytorch dataset
from dataset import PreTrainingTransform # pytorch transforms
dataset = PreTrainingDataset(transform=PreTrainingTransform())

# Load dataset into dataloader
from dataset import PreTrainingDataLoader
dataloader = PreTrainingDataLoader()

# Initialize model 
from model import SiamMAE
model = SiamMAE() # Contains: forward
forward = model.forward

# Initialize loss function, operates on the entire batch at once
def siamMAELoss():
    pass

# Initalize a scheduler
def scheduler():
    pass

# Initialize a loss scaler (Skip for now)
def lossScaler():
    pass

# Initialize a logger (Skip for now)
def logger():
    pass

# Initialize a epoch trainer, iter over all batches in the dataset, reset all gradients and optimizer at the start of each epoch 
def epoch_step():
    pass


# Initialize a eval function (can be merged with loss function later)
def eval():
    pass

# Initialize a trainer, iter over all epochs, save model at the end of each epoch

# initialize optimizer
# opt_state = opt_init(params)
opt_init, opt_update, get_params = optimizers.adam(step_size=0.0001)  # this time we use a smaller learning rate due to numerical stability
opt_state = opt_init(params)

"""
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
print('Device:', jax.devices())

class TrainState(train_state.TrainState):
    # Batch statistics from BatchNorm
    batch_stats : Any
    # PRNGKey for augmentations
    rng : Any

class TrainerSiamMAE:

    def __init__(self,params,exmp_imgs):
        """

        """
        super().__init__()
        self.hparams = params
        self.model_name = params.model_name
        self.model_class = get_obj_from_str(params.model_class)
        self.eval_key = "MSE" # hard coded for now
        self.lr = params.learning_rate
        self.num_epochs = params.num_epochs
        self.min_lr = params.min_learning_rate
        self.blr = params.base_learning_rate
        self.optimizer_b1 = params.optimizer_momentum.beta1
        self.optimizer_b2 = params.optimizer_momentum.beta2
        self.weight_decay = params.weight_decay
        self.seed = params.seed
        self.warmup_epochs = params.warmup_epochs
        self.rng = jax.random.PRNGKey(self.seed)
        self.check_val_every_n_epoch = params.check_val_every_n_epoch
        self.CHECKPOINT_PATH = params.CHECKPOINT_PATH
        self.exmp_imgs = exmp_imgs

        # Prepare logging
        self.log_dir = os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}/')
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, batch_stats, rng, batch,train=True): # TODO: Fix feeding model with the correct input (batch) and params
            # Feed model with batch, random, params and batch_stats
            outs = self.model_class.apply({'params': params, 'batch_stats': batch_stats},batch=batch,rng=rng,train=train)
            # TODO: If model class doesn't return a loss, then we need to calculate it here
            (loss, metrics), new_model_state = outs if train else (outs, None)
            return loss, (metrics, new_model_state)


        # Training function
        def train_step(state, batch):
            rng, forward_rng = random.split(state.rng)
            loss_fn = lambda params: calculate_loss(params,
                                                    state.batch_stats,
                                                    forward_rng,
                                                    batch,
                                                    train=True)
            (_, (metrics, new_model_state)), grads = jax.value_and_grad(loss_fn,
                                                                        has_aux=True)(state.params)
            # Update parameters, batch statistics and PRNG key
            state = state.apply_gradients(grads=grads,
                                          batch_stats=new_model_state['batch_stats'],
                                          rng=rng)
            return state, metrics

        # Eval function
        def eval_step(state, rng, batch):
            _, (metrics, _) = calculate_loss(state.params,
                                             state.batch_stats,
                                             rng,
                                             batch,
                                             train=False)
            return metrics

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)


    def init_model(self, exmp_imgs):
        """Initialize model"""
        # Initialize model
        rng = random.PRNGKey(self.seed)
        rng, init_rng = random.split(rng)

        variables = self.model_class.init(init_rng, exmp_imgs,self.hparams.model_param) # TODO: This is 100% wrong, but I don't have a model so I can't test it
        self.state = TrainState(step=0,
            apply_fn=self.model_class.apply,
            params=variables['params'],
            tx=None,
            batch_stats=variables.get('batch_stats'),
            rng=rng,
            opt_state=None)



    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        """
        Initialize the optimizer and the learning rate scheduler.
        
        Inputs:
            num_epochs - Number of epochs to train for
            num_steps_per_epoch - Number of steps per epoch        
        """
        # By default, we decrease the learning rate with cosine annealing
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=self.warmup_epochs * num_steps_per_epoch,
            decay_steps=num_epochs * num_steps_per_epoch,
            end_value=self.blr
        )
        optimizer = optax.adamw(lr_schedule, weight_decay=self.weight_decay,b1=self.optimizer_b1,b2=self.optimizer_b2)
        self.create_train_state(optimizer)

    def create_train_state(self, optimizer):
        """Update self.state with a new optimizer"""
        # Initialize training state
        self.state = TrainState.create(step=self.state.step,
                                       apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       tx=optimizer,
                                       batch_stats=self.state.batch_stats,
                                       rng=self.state.rng)
                                       

    def train_model(self, train_loader, val_loader):
        num_epochs = self.num_epochs
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval metric
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader)
                for key in eval_metrics:
                    self.logger.add_scalar(f'val/{key}', eval_metrics[key], global_step=epoch_idx)
                if eval_metrics[self.eval_key] >= best_eval:
                    best_eval = eval_metrics[self.eval_key]
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, data_loader, epoch):
        # Train model for one epoch, and log avg metrics
        metrics = defaultdict(float)
        num_train_steps = len(data_loader)
        for batch in tqdm(data_loader, desc='Training', leave=False):
            self.state, batch_metrics = self.train_step(self.state, batch,num_train_steps)
            for key in batch_metrics:
                metrics[key] += batch_metrics[key]
        for key in metrics:
            avg_val = metrics[key].item() / num_train_steps
            self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)



    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg metrics
        metrics = defaultdict(float)
        count = 0
        for batch_idx, batch in enumerate(data_loader):
            batch_metrics = self.eval_step(self.state, random.PRNGKey(batch_idx), batch)
            batch_size = (batch[0] if isinstance(batch, (tuple, list)) else batch).shape[0]
            count += batch_size
            for key in batch_metrics:
                metrics[key] += batch_metrics[key] * batch_size
        metrics = {key: metrics[key].item() / count for key in metrics}
        return metrics

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        num_params = sum([np.prod(p.shape) for p in jax.tree_leaves(state_dict)])
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       rng=self.state.rng,
                                       tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist
        return os.path.isfile(os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}.ckpt'))

def main():
    # Get the parameters as a omegaconf 
    hparams = omegaconf.OmegaConf.load("src/pretraining_params.yaml")

    print(hparams)
    # Enable or disable JIT
    config.update('jax_disable_jit', hparams.jax_disable_jit)


if __name__ == "__main__":
    main()




# Question: 
# 1. No gradient clipping?
# 2. 