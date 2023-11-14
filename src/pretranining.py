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
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.training.train_state import TrainState
import optax

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.datasets import STL10
print('Device:', jax.devices())



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

        # TODO: import data loader and dataset and get
        self.num_epochs = 0
        self.num_steps_per_epoch = 0


        # Prepare logging
        self.log_dir = os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}/')
        self.logger = SummaryWriter(log_dir=self.log_dir)

        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model_optimizer_scheduler_trainstate()

    def create_functions(self):

        def calculate_loss(params,state,x,y,mask_ratio): 
            """
                Calculate loss for a batch
            """
            # Get predictions
            pred, mask = state.apply_fn(params, x, y, mask_ratio) # TODO: Might need to add rng

            # Get loss
            loss = self.model_class.loss(y, pred, mask)

            return loss


        def train_step(state,x,y,mask_ratio):
            """
            Train one step
            """
            # Define a grad and loss function # TODO: Move it to save computations
            val_grad_fn = jax.value_and_grad(calculate_loss,argnums=0)
            loss,grads = val_grad_fn(state.params,state,x,y,mask_ratio)
            state = state.apply_gradients(grads=grads)
            return state, loss
        

        def eval_step(state, x, y,mask_ratio):
            """
            Calculate metrics on batch
            """
            
            # Calculate metrics for batch 
            loss = calculate_loss(state.params,state,x,y,mask_ratio)

            return loss

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def create_mask(self,params,label_fn):
        """
        Takes in a params dict and freezes the layers in layer
        """
        def _map(params, mask, label_fn):
            for k in params:
                if label_fn(k):
                    mask[k] = 'zero'
                else:
                    if isinstance(params[k], FrozenDict):
                        mask[k] = {}
                        _map(params[k], mask[k], label_fn)
                    else:
                        mask[k] = 'adam'
        mask = {}
        _map(params, mask, label_fn)
        return frozen_dict.freeze(mask)
    
    def zero_grads(self):
        """
        Zero gradient optimizer
        """
        def init_fn(_):
            return ()
        def update_fn(updates, state, params=None):
            return jax.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)


    def init_model_optimizer_scheduler_trainstate(self):
        """
        Initialize model, optimizer,learning rate scheduler and training state.
        """
        # Get random key
        self.rng, init_rng = random.split(self.rng)

        # Initialize model
        params = self.model_class.init(init_rng, self.example_x,self.example_y,self.mask_ratio) #  rng, same args as __call__ in model.py

        # Initialize Optimizer scheduler
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.blr,
            warmup_steps=self.warmup_epochs * self.num_steps_per_epoch,
            decay_steps=self.num_epochs * self.num_steps_per_epoch,
            end_value=self.lr
        )

        # Initialize optimizer
        # optimizer = optax.adamw(lr_schedule, weight_decay=self.weight_decay,b1=self.optimizer_b1,b2=self.optimizer_b2)
        optimizers = optax.multi_transform({'adamw': optax.adamw(learning_rate=lr_schedule, weight_decay=self.weight_decay,b1=self.optimizer_b1,b2=self.optimizer_b2),
                                            'zero':self.zero_grads()},
                                            self.create_mask(params, lambda s: s.startswith("frozen")))

        self.opt_state = optimizers.init(params)

        # Initialize training state
        self.model_state = TrainState.create(apply_fn=self.model_class.apply,params=params,tx=optimizers)

    def train_model(self, train_loader, val_loader):
        """
            Train model for a certain number of epochs, evaluate on validation set and save best performing model.
        """
        num_epochs = self.num_epochs
        metrics = defaultdict(list)

        # Iterate over epochs
        for epoch_idx in tqdm(range(1, num_epochs+1)):

            # Train model for one epoch
            avg_loss = self.train_epoch(train_loader, epoch=epoch_idx)
            metrics['train_loss'].append(avg_loss)
            print(f"Epoch {epoch_idx} | Train Loss: {avg_loss:.3f}")

        return metrics


    def train_epoch(self, data_loader, epoch):
        """
        Train model for one epoch, and log avg metrics
        """

        losses = []
        # Iterate over batches
        for (batch_x,batch_y) in tqdm(data_loader, desc='Training', leave=False):

            # Train model on batch
            self.model_state, loss = self.train_step(self.model_state,batch_x,batch_y,self.mask_ratio)

            # Log metrics
            losses.append(loss)
        
        # Log average metrics for epoch
        avg_loss = sum(losses) / len(losses)
        return avg_loss



    def eval_model(self, data_loader):
        """
        Evaluate model on a dataset and return avg metrics
        """

        # Test model on all images of a data loader and return avg metrics
        losses = []

        # Iterate over batches
        for (batch_x,batch_y) in (data_loader):
            # Evaluate model on batch
            loss = self.eval_step(self.model_state, self.model_class, batch_x, batch_y,self.mask_ratio)

            # Log metrics
            losses.append(loss)

        # Log average metrics for epoch
        avg_loss = sum(losses) / len(losses)
        return avg_loss


    def save_model(self, step=0): # TODO: Copied and needs adaptation
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.model_state.params},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False): # TODO: Copied and needs adaptation
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        num_params = sum([np.prod(p.shape) for p in jax.tree_leaves(state_dict)])
        self.model_state = TrainState.create(apply_fn=self.model_state.apply_fn,
                                       params=state_dict['params'],
                                       tx=self.model_state.tx)

    def checkpoint_exists(self): # TODO: Copied and needs adaptation
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
    metrics = trainer.train_model(train_loader, val_loader)

    # if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
    #     trainer.train_model(train_loader, val_loader)
    #     trainer.load_model()
    # else:
    #     trainer.load_model(pretrained=True)

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
