# Inspired by: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html
# A very general training script for jax consitent over all the UVADLC notebooks

import os
import time
from tqdm.auto import tqdm
from typing import Sequence, Any
from collections import defaultdict
from util.get_obj_from_str import get_obj_from_str
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.example_libraries import stax, optimizers
# from functools import partial
import omegaconf
from omegaconf import OmegaConf
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




# **Integrate with magnus code**
# How to iterate through the problems
# 1. Init of magnus model
# 2. Trainstate 
# 3. Backprop
# 






class TrainerSiamMAE:

    def __init__(self,params):
        """

        """
        super().__init__()
        self.hparams = params
        self.model_name = params.model_name
        self.model_class = get_obj_from_str(params.model_class)(params.model_param, hparams=params)
        self.eval_key = "MSE" # hard coded for now
        self.lr = params.learning_rate
        self.num_epochs = params.epochs
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
        self.mask_ratio = params.mask_ratio
        self.batch_size = params.batch_size
        self.effective_batch_size = self.batch_size * self.repeted_sampling
        self.repeted_sampling = params.repeted_sampling
        self.rng, self.init_rng = random.split(self.rng)

        # Create an example
        # (batch_size*repeted_sampling, in_chans, img_size, img_size)
        # (effective_batch_size, in_chans, img_size, img_size)
        self.example_x = random.uniform(self.init_rng, (self.effective_batch_size,params.in_chans,params.img_size,params.img_size))
        self.example_y = random.uniform(self.init_rng, (self.effective_batch_size,params.in_chans,params.img_size,params.img_size))


        # Prepare logging
        self.log_dir = os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}/')
        self.logger = SummaryWriter(log_dir=self.log_dir)

        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model

        def calculate_loss(params,state,x,y,mask_ratio):# params, batch_stats, rng, batch,train=True): 
            """
                Calculate loss for a batch
            """

            # Feed model with batch, random, params and batch_stats
            
            # Get predictions
            outs = self.model_class.apply({'params': params, 'batch_stats': batch_stats},batch=batch,rng=rng,train=train)
            (pred, mask), new_model_state = outs if train else (outs, None)
            # Calculate loss
            loss = self.model_class.loss(batch, pred, mask)
            # Calculate metrics



            # TODO: If model class doesn't return a loss, then we need to calculate it here
            (loss, metrics), new_model_state = outs if train else (outs, None)
            return loss, (metrics, new_model_state)


        # Training function
        def train_step(state, batch):
            """
            Train one step
            """

            # Get PRNG key for random augmentations
            rng, forward_rng = random.split(state.rng)

            # Get loss function
            loss_fn = lambda params: calculate_loss(params,
                                                    state.batch_stats,
                                                    forward_rng,
                                                    batch,
                                                    train=True)
            
            # Evaluate loss function and calculate gradients
            (_, (metrics, new_model_state)), grads = jax.value_and_grad(loss_fn,
                                                                        has_aux=True)(state.params)
            # Update parameters, batch statistics and PRNG key
            state = state.apply_gradients(grads=grads,
                                          batch_stats=new_model_state['batch_stats'],
                                          rng=rng)
            return state, metrics

        # Eval function
        def eval_step(state, rng, batch):
            """
            Calculate metrics on batch
            """
            # Calculate metrics for batch 
            _, (metrics, _) = calculate_loss(state.params,
                                             state.batch_stats,
                                             rng,
                                             batch,
                                             train=False)
            return metrics

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)


    def init_model(self):
        """
        Initialize model
        """
        # Initialize model
        #self.rng, init_rng = random.split(self.rng)
        rng = random.PRNGKey(self.seed)
        rng, init_rng = random.split(rng)

        variables = self.model_class.init(init_rng, self.example_x,self.example_y,self.mask_ratio) #  rng, same args as __call__ in model.py
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
        
        # Will linearly increase learning rate from 0 to base learning rate over the first warmup_epochs
        # Will Go from peak learning rate to end learning rate over the remaining epochs as a cosine annealing schedule
        #      *    *         peak
        #     *          *    
        #    *             *
        #   *                   *    end_value
        #  *  0.0
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.blr,
            warmup_steps=self.warmup_epochs * num_steps_per_epoch,
            decay_steps=num_epochs * num_steps_per_epoch,
            end_value=self.lr
        )
        optimizer = optax.adamw(lr_schedule, weight_decay=self.weight_decay,b1=self.optimizer_b1,b2=self.optimizer_b2)
        self.create_train_state(optimizer)

    def create_train_state(self, optimizer):
        """
            Update self.state with a new optimizer
        """
        # Initialize training state, we use flax's train_state.TrainState class
        self.state = TrainState.create(step=self.state.step, 
                                       apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       tx=optimizer,
                                       batch_stats=self.state.batch_stats,
                                       rng=self.state.rng)
                                       

    def train_model(self, train_loader, val_loader):
        """
            Train model for a certain number of epochs, evaluate on validation set and save best performing model.
        """
        num_epochs = self.num_epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval metric

        best_eval = 0.0
        # Iterate over epochs
        for epoch_idx in tqdm(range(1, num_epochs+1)):

            # Train model for one epoch
            self.train_epoch(train_loader, epoch=epoch_idx)

            # Check if we should save model 
            if epoch_idx % self.check_val_every_n_epoch == 0:

                # Evaluate model
                eval_metrics = self.eval_model(val_loader)

                # Log metrics
                for key in eval_metrics:
                    self.logger.add_scalar(f'val/{key}', eval_metrics[key], global_step=epoch_idx)

                # Save model if it's the best yet
                if eval_metrics[self.eval_key] >= best_eval:
                    best_eval = eval_metrics[self.eval_key]
                    self.save_model(step=epoch_idx)

                # Flush logger
                self.logger.flush()

    def train_epoch(self, data_loader, epoch):
        """
        Train model for one epoch, and log avg metrics
        """
        # Initialize metrics
        metrics = defaultdict(float)

        # Get number of training steps aka number of batches
        num_train_steps = len(data_loader)

        # Iterate over batches
        for batch in tqdm(data_loader, desc='Training', leave=False):

            # Train model on batch and update metrics and state
            self.state, batch_metrics = self.train_step(self.state, batch,num_train_steps)
            for key in batch_metrics:
                metrics[key] += batch_metrics[key]

        # Log metrics
        for key in metrics:

            # Average metrics over all batches
            avg_val = metrics[key].item() / num_train_steps

            # Log to tensorboard
            self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)



    def eval_model(self, data_loader):
        """
        Evaluate model on a dataset and return avg metrics
        """

        # Test model on all images of a data loader and return avg metrics
        metrics = defaultdict(float)

        # Iterate over batches
        count = 0
        for batch_idx, batch in enumerate(data_loader):
            # Evaluate model on batch
            batch_metrics = self.eval_step(self.state, random.PRNGKey(batch_idx), batch)

            # Get batch size
            batch_size = (batch[0] if isinstance(batch, (tuple, list)) else batch).shape[0]

            # Update metrics
            count += batch_size
            for key in batch_metrics:
                metrics[key] += batch_metrics[key] * batch_size

        # Average metrics over all batches
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

def train_siamMAE(hparams):
    """
        Train a model with the given hyperparameters.
    """

    # Get datasets from hparams using get_obj_from_str
    dataset_train = None
    dataset_val = None
    # Create dataloaders
    train_loader = None
    val_loader = None

    # Create a trainer module with specified hyperparameters
    trainer = TrainerSiamMAE(params=hparams) # Feed trainer with example images from one batch of the dataset and the hyperparameters
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(train_loader, val_loader)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)

    return trainer



def main():
    # Get the parameters as a omegaconf 
    hparams = omegaconf.OmegaConf.load("Playground/src/pretraining_params.yaml")


    print(hparams)

    # Enable or disable JIT
    config.update('jax_disable_jit', hparams.jax_disable_jit)

    # train the model
    trainer = train_siamMAE(hparams)



if __name__ == "__main__":
    main()
