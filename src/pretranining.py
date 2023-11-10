import os
import sys
# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import math
from typing import Iterable
import torch
import utils.misc as misc
import utils.lr_sched as lr_sched
import argparse
import uuid
from pathlib import Path
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import model
from torchvision.datasets import Kinetics,MovingMNIST


def get_params_dict(path=SCRIPT_DIR+"/pretraining_params.json"):
    try:
        with open(path, "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("No params file found at {}".format(path))

    return params



class Pretraining:
    def __init__(self, params,debug=True):
        """
        Initializes a Pretraining object for a Siamese Masked Autoencoder.

        Args:
            params (dict): A dictionary containing the parameters for pretraining.
            debug (bool, optional): Whether to print debug information. Defaults to True.
        """
        self.params = params
        self.debug = debug

        # Print Info
        self.print_debug("Initializing pretraining...")
        self.print_debug("Debug mode: {}".format(self.debug))
        self.print_debug("Params:")
        for key, value in params.items():
            self.print_debug("\t{}: {}".format(key, value))
            #setattr(self, key, value) 
        
        # Set device
        try:
            self.device = torch.device(self.params["device"])
        except Exception as e:
            self.print_debug("Error setting device: {}".format(e))
            self.print_debug("Setting device to cuda if available, else cpu")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # fix the seed for reproducibility
        seed = self.params["seed"] #+ misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set cudnn benchmarking
        if self.device.type == "cuda":
            cudnn.benchmark = True

        # Load loss scaler
        self.loss_scaler = NativeScaler()
        


        # Model parameters
        self.model = None
        self.optimizer = None
        self.transform = None
        self.dataset = None

        ## Variables for state management
        self.data_loaded = False
        self.transform_loaded = False        
        self.model_loaded = False
        self.optimizer_loaded = False
        self.ready_for_training = False

    def load_transform(self, transform):
        """
        Loads a transform for the dataset.

        Args:
            transform (torchvision.transforms): A torchvision transform.
        """
        self.transform = transform
        self.transform_loaded = True

    def load_dataset(self, dataset):
        """
        Loads a dataset for pretraining.

        Args:
            dataset (torchvision.datasets): A torchvision dataset.
        """
        self.dataset = dataset
        self.data_loaded = True

    def load_model(self, model):
        """
        Loads a model for pretraining.

        Args:
            model (torch.nn.Module): A torch model.
        """
        self.model = model
        self.model_loaded = True

    def load_optimizer(self, optimizer):
        """
        Loads an optimizer for pretraining.

        Args:
            optimizer (torch.optim): A torch optimizer.
        """
        self.optimizer = optimizer
        self.optimizer_loaded = True
        
    def prepare_for_training(self):
        """
        Prepares the pretraining for training. This function must be called after loading a dataset, model, and optimizer.
        
        """
        if not self.transform_loaded:
            self.print_debug("No new dataset transform loaded. Using already loaded transform in dataset if available.")
        if not self.data_loaded:
            raise ValueError("Dataset not loaded. Please load a dataset before preparing for training.")
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please load a model before preparing for training. ")
        if not self.optimizer_loaded:
            self.print_debug("No new optimizer loaded. Using default optimizer.")

        # Add transform to dataset if not already present
        if self.transform is not None:
            self.dataset.transform = self.transform

        # Set up a sampler
        self.sampler = torch.utils.data.RandomSampler(self.dataset)

        # Set up a data loader
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, sampler=self.sampler,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
            pin_memory=self.params["pin_mem"],
            drop_last=True,
        )

        # Set up a model
        self.model.to(self.device)
        self.model_without_ddp = self.model

        # Get efficient batch size
        self.eff_batch_size = self.params["batch_size"] * self.params["accum_iter"]

        # Set up an optimizer 
        self.param_groups = optim_factory.add_weight_decay(self.model_without_ddp,self.params["weight_decay"])
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.param_groups, lr=self.params["lr"], betas=(0.9, 0.95))

        # set ready for training
        self.ready_for_training = True
        self.print_debug("Pretraining ready for training.")


    def train(self):
        if not self.ready_for_training:
            raise ValueError("Pretraining not ready for training. Please call prepare_for_training() before training.")
        self.print_debug("Starting pretraining...")
        print(f"Start training for {self.params['epochs']} epochs")
        start_time = time.time()
        for epoch in range(self.params["start_epoch"], self.params["epochs"]):
            train_stats = self.train_one_epoch(
                model, self.data_loader,
                self.optimizer, self.device, epoch, self.loss_scaler,
                log_writer=self.log_writer,
                params=self.params
            )


            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #                 'epoch': epoch,}


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))



    def print_debug(self, msg):
        if self.debug:
            print(msg)

    def train_one_epoch(self,model: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, loss_scaler,
                        log_writer=None,
                        params=None):
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20


        accum_iter = params["accum_iter"]

        optimizer.zero_grad()

        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

        for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, params)

            samples = samples.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=params["mask_ratio"])

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main():
    params = get_params_dict()
    pretraining = Pretraining(params)
    dataset = Kinetics(root="./data/kinetics",frames_per_clip=16,step_between_clips=1)
    model = model.SiamMAE()
    pretraining.load_dataset(dataset)
    pretraining.load_model(model)
    pretraining.prepare_for_training()
    pretraining.train()

if __name__ == "__main__":
    main()