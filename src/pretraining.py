import os
import shutil
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" # uncomment to see real memory usage 
#This disables the preallocation behavior. JAX will instead allocate GPU memory as needed, potentially decreasing the overall memory usage. 
#However, this behavior is more prone to GPU memory fragmentation, 
#meaning a JAX program that uses most of the available GPU memory may OOM with preallocation disabled.

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# If preallocation is enabled, this makes JAX preallocate XX% of the total GPU memory, 
# instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" # Needed to not run out of memory on GPU after a while of training, but reduces performance a little bit, go down in batch size is also a solution
# This makes JAX allocate exactly what is needed on demand, 
# and deallocate memory that is no longer needed (note that this is the only configuration that will deallocate GPU memory, instead of reusing it). 
# This is very slow, so is not recommended for general use, 
# but may be useful for running with the minimal possible GPU memory footprint or debugging OOM failures.
import matplotlib.pyplot as plt
import time
import datetime
from tqdm.auto import tqdm
from typing import Sequence, Any
from collections import defaultdict
from util.get_obj_from_str import get_obj_from_str
import numpy as np
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
from flax.training import train_state, checkpoints, orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint
import optax
from jax.sharding import PositionalSharding

from util.patchify import unpatchify, patchify
from PIL import Image

## PyTorch
import torch
#import torch.utils.data as data
from data import PreTrainingDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
# import DataLoader module
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
from data_loader import SiamMAEloader
import glob
print('Device:', jax.devices())
sharding = PositionalSharding(jax.devices())

# https://github.com/google/flax/discussions/1690

class TrainerSiamMAE:

    def __init__(self,params,data_loader,remove_checkpoints=True):
        """
        Initialize trainer module for pretraining of siamMAE model.
        """
        super().__init__()
        self.hparams = params
        self.remove_checkpoints = remove_checkpoints
        self.model_name = params.model_name
        self.model_class = get_obj_from_str(params.model_class)(**params.model_param, hparams=params)
        self.eval_key = "MSE" # hard coded for now
        self.lr = params.learning_rate
        self.num_epochs = params.epochs
        self.min_lr = params.min_learning_rate
        self.blr = params.base_learning_rate
        self.optimizer_b1 = params.optimizer_momentum.beta1
        self.optimizer_b2 = params.optimizer_momentum.beta2
        self.weight_decay = params.weight_decay
        self.seed = params.random_seed
        self.warmup_epochs = params.warmup_epochs
        self.rng = jax.random.PRNGKey(self.seed)
        self.check_val_every_n_epoch = params.check_val_every_n_epoch
        self.CHECKPOINT_PATH = params.CHECKPOINT_PATH
        self.mask_ratio = self.hparams.mask_ratio
        self.batch_size = params.batch_size
        self.repeted_sampling = params.repeted_sampling
        self.effective_batch_size = self.batch_size * self.repeted_sampling
        self.rng, self.init_rng = random.split(self.rng)
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.params = None

        # Create an example
        # (batch_size*repeted_sampling, in_chans, img_size, img_size)
        # (effective_batch_size, in_chans, img_size, img_size)
        example_batch = jnp.zeros((self.effective_batch_size,params.model_param.in_chans,params.model_param.img_size,params.model_param.img_size))

        # TODO: import data loader and dataset and get
        self.num_epochs = self.num_epochs
        self.num_steps_per_epoch = len(data_loader)
        assert self.num_steps_per_epoch != 0, "Dataloader is empty"

        # Remove all files in ./checkpoints folder
        if self.remove_checkpoints:
            if os.path.exists(self.CHECKPOINT_PATH):
                shutil.rmtree(self.CHECKPOINT_PATH)
            os.makedirs(self.CHECKPOINT_PATH)
        
        # Prepare logging
        self.log_dir = os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}/')
        self.logger = SummaryWriter()

        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model_optimizer_scheduler_trainstate(example_batch,example_batch)

    def create_functions(self):

        def calculate_loss(params,state,x,y,mask_ratio): 
            """
            Calculate loss for a batch
            """
            # Get predictions
            pred, mask = state.apply_fn(params, x, y) # TODO: Might need to add rng
            # save img
            # if True:
            #     t = datetime.datetime.now()
            #     save_name = "pred_img_{}.png".format(t.strftime("%H: %M:"))
            #     self.save_pred_img(pred, save_name)

            # Get loss
            loss = self.model_class.loss(y, pred, mask)

            return loss

        def train_step(state,x,y,mask_ratio):
            """
            Train one step
            """
            # grads = self.grad_fn(state.params,state,x,y,mask_ratio) # Uncomment to save a little bit of gpu memory
            loss,grads = self.val_grad_fn(state.params,state,x,y,mask_ratio)
            state = state.apply_gradients(grads=grads)
            return state, loss
        

        def eval_step(state, x, y,mask_ratio): # TODO: Check that it works
            """
            Calculate metrics on batch
            """
            
            # Calculate metrics for batch 
            loss = calculate_loss(state.params,state,x,y,mask_ratio)

            return loss

        # jit for efficiency
        self.val_grad_fn = jax.value_and_grad(calculate_loss,argnums=0)
        self.grad_fn = jax.grad(calculate_loss,argnums=0)
        self.train_step = jax.jit(train_step) 


    def create_mask(self,params,label_fn,optimizer_key='adamw',freeze_optimizer_key='zero'):
        """
        Input:
            params: parameter dict
            label_fn: function that takes in a string and returns a boolean
            optimizer_key: optimizer key
            freeze_optimizer_key: freeze optimizer key
        Output:
            mask: mask dict

        Takes in a label function and maps the parameters to the optimizer keys.
        Example:
        params =
        {
        layer1: {
            }
        layer2: {
            layer2.1: {
                }
            layer2.2: {
                }
            }
        layer3: {
            weight: {}
            bias: {}
            }

        }
        label_fn = lambda s: s.startswith("layer2")
        optimizer_key = 'adamw'
        freeze_optimizer_key = 'zero'
        mask = create_mask(params, label_fn, optimizer_key, freeze_optimizer_key)
        print(mask) 
        {
        layer1: 'adamw'
        layer2: {
            layer2.1: 'zero'
            layer2.2: 'zero'
            }
        layer3: {
            weight: 'adamw'
            bias: 'adamw'
            }
        }
        """
        def _map(params, mask, label_fn):
            for k in params:
                if label_fn(k):
                    mask[k] = freeze_optimizer_key
                else:
                    if isinstance(params[k], dict):
                        mask[k] = {}
                        _map(params[k], mask[k], label_fn)
                    else:
                        mask[k] = optimizer_key
        mask = {}
        _map(params, mask, label_fn)
        return mask


    def zero_grads(self):
        """
        Zero gradient optimizer
        """
        def init_fn(_):
            return ()
        def update_fn(updates, state, params=None):
            return jax.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)


    def init_model_optimizer_scheduler_trainstate(self,example_x,example_y):
        """
        Initialize model, optimizer,learning rate scheduler and training state.
        """
        # Get random key
        self.rng, init_rng = random.split(self.rng)

        # Initialize model
        #params = jax.jit(self.model_class.init,backend='cpu')(init_rng, example_x,example_y,self.mask_ratio) #  rng, same args as __call__ in model.py
        self.params = self.model_class.init(init_rng, example_x,example_y) #  rng, same args as __call__ in model.py
        # params = jax.device_put(params, jax.devices("gpu")[0])
        # Initialize Optimizer scheduler
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.blr,
            warmup_steps=self.warmup_epochs * self.num_steps_per_epoch,
            decay_steps=self.num_epochs * self.num_steps_per_epoch,
            end_value=self.lr
        )
        # Cosine schedule
        # lambda_fn = lambda i: jnp.sin(jnp.pi * i / ((self.num_epochs//3) * self.num_steps_per_epoch))*(1.0e-2 - 5e-4)/2 + (1.0e-2 + 5e-4)/2
        # lr_schedule = lambda i: jnp.array(lambda_fn(i),dtype=jnp.float32)

        
        
        # Test the lr_schedule and plot it
        # import matplotlib.pyplot as plt
        # lr = []
        # for i in range(self.num_epochs * self.num_steps_per_epoch):
        #     lr.append(lr_schedule(i))
        # plt.plot(lr)
        # plt.savefig('lr_schedule.png')
        

        # Depending on layer name use different optimizer
        # * If layer name starts with frozen use zero_grads optimizer
        # * If layer name does not start with frozen use adamw optimizer
        # create_mask will take in parameter dict and with the same structure map: layer names to optimizer names: IF the function returns true - map to freeze_optimizer_key ELSE optimizer else map to optimizer_key
        #optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=self.weight_decay,b1=self.optimizer_b1,b2=self.optimizer_b2)
        optimizer = optax.multi_transform({'adamw': optax.adamw(learning_rate=lr_schedule, weight_decay=self.weight_decay,b1=self.optimizer_b1,b2=self.optimizer_b2),
                                             'zero':self.zero_grads()},
                                             self.create_mask(self.params, lambda s: s.startswith("frozen"),optimizer_key='adamw',freeze_optimizer_key='zero'))

        # Initialize training state
        self.model_state = TrainState.create(apply_fn=self.model_class.apply,params=self.params,tx=optimizer)
        self.model_state = jax.device_put(self.model_state, jax.devices("cpu")[0])

    def train_model(self, train_loader, val_loader):
        """
            Train model for a certain number of epochs, evaluate on validation set and save best performing model.
        """
        num_epochs = self.num_epochs
        metrics = defaultdict(list)
        model_state = self.model_state
        model_state = jax.device_put(model_state, sharding.replicate())
            
        
        # Iterate over epochs
        for epoch_idx in tqdm(range(1, num_epochs+1)):

            if epoch_idx % self.hparams.save_model_interval:
                save_model = True
            else:
                save_model = False
            # Train model for one epoch
            time_to_train_epoch = time.time()
            avg_loss,model_state = self.train_epoch(train_loader, epoch=epoch_idx,model_state=model_state, save_model=save_model)
            self.logger.add_scalar(f"Time/train epoch", time.time() - time_to_train_epoch, epoch_idx)
            avg_loss = float(avg_loss)
            self.logger.add_scalar(f"Loss/train [epoch]", avg_loss, epoch_idx)
            metrics['train_loss'].append(avg_loss)
            print(f"Epoch {epoch_idx} | Train Loss: {avg_loss:.3f}")

        return metrics



    def train_epoch(self, data_loader, epoch,model_state, save_model=False):
        """
        Train model for one epoch, and log avg metrics
        """

        losses = []
        # Iterate over batches
        #model_state = self.model_state
        mask_ratio = self.mask_ratio
        time_to_load_batch = time.time()
        for i,batch in enumerate(tqdm(data_loader, desc='Training', leave=False)):

            # Transform batch_x and batch_y to jnp arrays (here the batches are moved to gpu)
            batch_x = batch[:,:,0,:,:,:]
            batch_y = batch[:,:,1,:,:,:]
            batch_x = jnp.array(batch_x)
            batch_y = jnp.array(batch_y)
            
            # If batch size is wrong skip batch
            if batch_x.shape[0] != self.batch_size or batch_y.shape[0] != self.batch_size:
                print(f"Batch: {i} Epoch: {epoch} has wrong batch size. Skipping batch")
                continue

            # BxNxCxHxW --> (B*N)xCxHxW
            batch_x = jnp.reshape(batch_x,(self.effective_batch_size,self.hparams.model_param.in_chans,self.hparams.model_param.img_size,self.hparams.model_param.img_size))
            batch_y = jnp.reshape(batch_y,(self.effective_batch_size,self.hparams.model_param.in_chans,self.hparams.model_param.img_size,self.hparams.model_param.img_size))

            # Log time to load batch
            self.logger.add_scalar(f"Time/load batch", time.time() - time_to_load_batch, epoch * self.num_steps_per_epoch + i)

            # Log time to train batch
            time_to_train_batch = time.time()
            
            # Parallelize batch on multiple devices
            # https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
            # Define sharding, it is a structure that defines how to split data across devices

            # Replicate model state on all devices
            # Put half of the batch on each device
            batch_x = jax.device_put(batch_x, sharding.reshape((len(jax.devices()),1,1,1)))
            batch_y = jax.device_put(batch_y, sharding.reshape((len(jax.devices()),1,1,1)))
            # Put mask ratio on all devices
            mask_ratio = jax.device_put(mask_ratio, sharding.replicate())        
            
            # if i == int(len(data_loader)/self.batch_size):
            #     save_pred = True
            # else:
            #     save_pred = False
            model_state, loss = self.train_step(model_state,batch_x,batch_y,mask_ratio)
            self.logger.add_scalar(f"Time/train batch", time.time() - time_to_train_batch, epoch * self.num_steps_per_epoch + i)
            # Log metrics
            losses.append(loss)

            # Publish metrics to tensorboard
            self.logger.add_scalar(f"Loss/train [batch]", float(loss), epoch * self.num_steps_per_epoch + i)

            # Log time to load batch
            time_to_load_batch = time.time()

        if save_model or epoch == self.num_epochs:
            self.save_model(model_state, epoch, batch_x, batch_y, save_img=False)
        
        # Log average metrics for epoch
        avg_loss = sum(losses) / len(losses)
        return avg_loss,model_state
    

    def eval_model(self, data_loader): # TODO: Might need adaptation
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


    def save_model(self, state,epoch, batch_x, batch_y, save_img=False): # TODO: Copied and needs adaptation
        # Save current model at certain training iteration
        # checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
        #                             target={'params': self.model_state.params},
        #                             step=step,
        #                             overwrite=True)
        # following documentation on: https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
        checkpoint = {"model": state}
        # predict 
        # f1 = jnp.expand_dims(batch_x[0], axis=0)
        # f2 = jnp.expand_dims(batch_y[0], axis=0)
        # pred, loss = state.apply_fn(state.params, batch_x, batch_y)
        # cpus = jax.devices("cpu")
        # params = jax.device_put(state.params,cpus[0])
        # batch_x = jax.device_put(batch_x,cpus[0])
        # batch_y = jax.device_put(batch_y,cpus[0])
        if save_img:
            pred, loss = self.model_class.apply(state.params, batch_x, batch_y)

            t = datetime.datetime.now()
            save_name = "pred_img_{}.png".format(t.strftime("%H: %M:"))
            self.save_pred_img(pred, save_name)

        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.orbax_checkpointer.save(self.CHECKPOINT_PATH + "_epoch_" + str(epoch), checkpoint, save_args=save_args)



    def load_model(self, params, optimizer, chkp_path,  pretrained=False): # TODO: Copied and needs adaptation
        # Load model. We use different checkpoint for pretrained models
        # if not pretrained:
        #     state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        # else:
        #     state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        # num_params = sum([np.prod(p.shape) for p in jax.tree_leaves(state_dict)])
        # self.model_state = TrainState.create(apply_fn=self.model_state.apply_fn,
        #                                params=state_dict['params'],
        #                                tx=self.model_state.tx)
        
        # following documentation on: https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html

        # when loading the parameters for the finetuning, we will only need the encoder part of the model
        empty_state = TrainState.create(apply_fn=self.model_class.apply, params=jax.tree_map(np.zeros_like, params), tx=optimizer)
        target = {"model": empty_state}
        restored = self.orbax_checkpointer.restore(chkp_path, item=target)

        return restored

    def save_pred_img(self, pred, name):
        if pred.shape[0] > 1:
            pred = jnp.array([pred[0]])
        out_img = unpatchify(pred)
        print(out_img.shape)
        out_img = jnp.einsum('ijkl->klj', out_img)
        # Minmax normalize to range 0-255
        out_img = (out_img - out_img.min()) * (255/(out_img.max() - out_img.min()))
        # Convert to uint8
        out_img = out_img.astype(np.uint8)
        out_img = np.array(out_img)
        # Save output image
        plt.imsave('./reproduction/{}'.format(name), out_img)
        print("Saved {}!".format(name))

    def test_model(self, input1, input2, idx, checkpoint_path):
        print("Loading checkpoint: {}".format(checkpoint_path))
        restored = self.orbax_checkpointer.restore(checkpoint_path)
        pred, mask = self.model_class.apply(restored['model']['params'], input1, input2)
        ckp = checkpoint_path.split("/")[-2]
        print(ckp)
        save_name = "{}_output{}.png".format(ckp,idx)
        self.save_pred_img(pred, save_name)
        
        


    def checkpoint_exists(self): # TODO: Copied and needs adaptation
        # Check whether a pretrained model exist
        return os.path.isfile(os.path.join(self.CHECKPOINT_PATH, f'{self.model_name}.ckpt'))

def train_siamMAE(hparams):
    """
        Train a model with the given hyperparameters.
    """

    # Get datasets from hparams using get_obj_from_str
    # dataset_train = get_obj_from_str(hparams.dataset)(data_dir="./data/Kinetics/train_jpg/*")
    # dataset_val = None
    # Create dataloaders
    train_loader = SiamMAEloader(num_samples_per_video=hparams.repeted_sampling,batch_size=hparams.batch_size)
    # train_loader = DataLoader(dataset_train, batch_size=hparams.batch_size, shuffle=False)
    #assert len(train_loader) == 0, "Dataloader is empty"
    print(len(train_loader))
    # Create a trainer module with specified hyperparameters
    trainer = TrainerSiamMAE(params=hparams,data_loader=train_loader) # Feed trainer with example images from one batch of the dataset and the hyperparameters
    metrics = trainer.train_model(train_loader,val_loader=None)

    # if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
    #     trainer.train_model(train_loader, val_loader)
    #     trainer.load_model()
    # else:
    #     trainer.load_model(pretrained=True)

    return metrics


def test_checkpoints(hparams):
    test_loader = SiamMAEloader(num_samples_per_video=1,batch_size=hparams.test_batch_size)
    trainer = TrainerSiamMAE(params=hparams, data_loader=test_loader,remove_checkpoints=False)
    # Load all checkpoints in folder ./checkpoints using glob
    checkpoints = glob.glob("./checkpoints/*")
    # Take the checkpoint with the highest epoch number
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    # Load the checkpoint
    checkpointlast_path = checkpoints[-1] + "/"
    checkpointfirst_path = checkpoints[0] + "/"
    checkpointmiddle_path = checkpoints[int(len(checkpoints)/2)] + "/"
    checkpoint_lst = [checkpointfirst_path, checkpointmiddle_path, checkpointlast_path]
    for checkpoint in checkpoint_lst:
        for i, frames in enumerate(test_loader):
            f1 = frames.squeeze(1)[:,0]
            f2 = frames.squeeze(1)[:,1]

            trainer.test_model(f1, f2, i, checkpoint)



def main():
    # Get the parameters as a omegaconf 
    hparams = omegaconf.OmegaConf.load("src/pretraining_params.yaml")
    print(hparams)

    # Enable or disable JIT
    config.update('jax_disable_jit', hparams.jax_disable_jit)

    # train the model
    # metrics = train_siamMAE(hparams)

    # test model
    test_checkpoints(hparams)


if __name__ == "__main__":
    main()
