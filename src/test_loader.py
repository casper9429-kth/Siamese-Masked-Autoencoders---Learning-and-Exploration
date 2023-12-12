import glob
import decord
from decord import VideoReader,cpu,gpu
import jax
import torch
import random
import numpy as np
import jax.numpy as jnp
from patchify import patchify
import matplotlib.pyplot as plt 
from torchvision.datasets import Kinetics
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomResizedCrop,RandomHorizontalFlip,ToTensor
import os
from PIL import Image
import time
import concurrent.futures
from multiprocessing import Pool
import cv2


class KineticsDataset(Dataset):
    def __init__(self, data_dir = './data/Kinetics/train_jpg/*',n_per_video = 8,step_size = 4,target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5):
        self.data_paths = glob.glob(data_dir)
        self.root = data_dir
        self.n_per_video = n_per_video
        self.step_size = step_size
        self.target_size = target_size
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.transform = Compose([ToTensor(),
                                 RandomResizedCrop(size=target_size,scale = scale, antialias=True),
                                  RandomHorizontalFlip(p=horizontal_flip_prob)])

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        frame_list = os.listdir(self.data_paths[idx])

        # random sample index of first frame
        id1 = random.randint(0,len(frame_list)-self.n_per_video*self.step_size)

        # get n_per_video-1 frame with step size
        frame_idx = [id1+i*self.step_size for i in range(self.n_per_video)]

        # get frames
        frames = [Image.open(self.data_paths[idx]+'/'+frame_list[i]) for i in frame_idx]
        frames = torch.stack(frames)
        frames = self.transform(frames).numpy()
        return frames
    


