import cv2
import glob
import jax
import torch
import random
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics
from skimage.transform import rescale,resize
from torchvision.transforms import Compose, RandomResizedCrop,RandomHorizontalFlip,ToTensor



class PreTrainingDataset(Dataset):
    def __init__(self, data_dir = "../test_dataset/*",n_per_video = 2,frame_range = (4,48),patch_size = (16,16,3),target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5):
        self.data_paths = sorted(glob.glob(data_dir))
        self.n_per_video = n_per_video
        self.frame_range = frame_range
        self.patch_size = patch_size
        self.target_size = target_size
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.transform = Compose([RandomResizedCrop(size=target_size,scale = scale, antialias=True),RandomHorizontalFlip(p=horizontal_flip_prob)])
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        vid_capture = cv2.VideoCapture(self.data_paths[idx])
        nr_frames = int(vid_capture.get(7))
        idx_f1 = np.random.choice(np.arange(0,nr_frames-self.frame_range[1]), size=self.n_per_video, replace=False)
        idx_f2 = np.random.choice(np.arange(self.frame_range[0],self.frame_range[1] + 1), size=self.n_per_video, replace=True) + idx_f1
        f1s = []
        f2s = []
        for i in range(self.n_per_video):
            if idx_f2[i] >= nr_frames:
                idx_f2[i] = np.random.choice(np.arange(idx_f1+4,nr_frames))
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_f1[i])
            _, f1 = vid_capture.read()
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_f2[i])
            _, f2 = vid_capture.read()
            f1 = np.moveaxis(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB),-1,0)
            f2 = np.moveaxis(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB),-1,0)
            if self.transform:
                f1 = torch.from_numpy(f1)
                f1t = self.transform(f1)
                f1t = f1t.float()
                f1t_mean = torch.mean(f1t, dim=(1, 2))
                f1t_std = torch.std(f1t, dim=(1, 2))
                f1_norm = (f1t- f1t_mean.view(3, 1, 1)) / f1t_std.view(3, 1, 1)
                f1s.append(f1_norm.unsqueeze(0).numpy())
                f2 = torch.from_numpy(f2).float()
                f2t = self.transform(f2)
                f2t_mean = torch.mean(f2t, dim=(1, 2))
                f2t_std = torch.std(f2t, dim=(1, 2))
                f2_norm = (f2t - f2t_mean.view(3, 1, 1)) / f2t_std.view(3, 1, 1)
                f2s.append(f2_norm.unsqueeze(0).numpy())
        f1s = np.concatenate(f1s,axis=0)
        f2s = np.concatenate(f2s,axis=0)
        # Shape f1s, f2s is [n_per_video,H,W,C] 
        return f1s,f2s


def main():
    dataset = PreTrainingDataset()
    print(dataset.__len__())
    for i, samples in enumerate(dataset):
        f1s,f2s = samples
        break


if __name__ == '__main__':
    main()