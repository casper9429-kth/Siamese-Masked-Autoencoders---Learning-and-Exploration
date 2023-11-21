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


class PreTrainingDataset(Dataset):
    # [test_dataset]: data_dir = ./test_dataset/* 
    # [kinetics]: data_dir = ./data/Kinetics/train/*/*
    def __init__(self, data_dir = "./test_dataset/*",n_per_video = 2,frame_range = (4,48),patch_size = (16,16,3),target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5):
        self.data_paths = glob.glob(data_dir)
        self.n_per_video = n_per_video
        self.frame_range = frame_range
        self.patch_size = patch_size
        self.target_size = target_size
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.transform = Compose([RandomResizedCrop(size=target_size,scale = scale, antialias=True),RandomHorizontalFlip(p=horizontal_flip_prob)])
        decord.bridge.set_bridge('torch')
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Open video
        with open(self.data_paths[idx], 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))

        # Get length of video
        nr_frames = len(vr)

        # Make sure video is long enough or not to short
        # If nr_frames is 0, then video is corrupted and we skip it
        if nr_frames < self.frame_range[1]+1:
            return self.__getitem__(idx + 1) 

        # Choose random frames
        idx_f1 = np.random.choice(np.arange(0,nr_frames-self.frame_range[1]), size=self.n_per_video, replace=False)
        idx_f2 = np.random.choice(np.arange(self.frame_range[0],self.frame_range[1] + 1), size=self.n_per_video, replace=True) + idx_f1
        frames = vr.get_batch(np.concatenate([idx_f1,idx_f2],axis = 0))
        frames = torch.moveaxis(frames,-1,1)
        if self.transform:
            frames_t = self.transform(frames).float()
            frames_mean = torch.mean(frames_t, dim=(2, 3))
            frames_std = torch.std(frames_t, dim=(2, 3))
            frames_norm = (frames_t- frames_mean.view(2*self.n_per_video,3,1,1))/frames_std.view(2*self.n_per_video,3,1,1)
        f1s = frames_norm[:self.n_per_video]
        f2s = frames_norm[self.n_per_video:]
        # Shape f1s, f2s is [n_per_video,C,H,W] 
        return f1s,f2s


def main():
    dataset = PreTrainingDataset()
    dataloader = DataLoader(dataset,batch_size =4,shuffle=True)
    print("Length dataset: ",dataset.__len__())
    print("Length dataloader: ",dataloader.__len__())
    for i, samples in enumerate(dataloader):
        f1s,f2s = samples
        print(f1s.shape)
        print(f2s.shape)
        if i  == 9:
            break


if __name__ == '__main__':
    main()