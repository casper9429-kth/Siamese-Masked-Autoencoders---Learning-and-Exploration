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



CORES = os.cpu_count()

class PreTrainingDataset(Dataset):
    # [test_dataset]: data_dir = ./test_dataset/* 
    # [kinetics]: data_dir = ./data/Kinetics/train/*/*
    def __init__(self, data_dir = "./test_dataset/*",n_per_video = 2,frame_range = (4,10),patch_size = (16,16,3),target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5):
        self.data_paths = glob.glob(data_dir)
        self.root = data_dir
        self.n_per_video = n_per_video
        self.frame_range = frame_range
        self.patch_size = patch_size
        self.target_size = target_size
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.transform = Compose([ToTensor(),
                                 RandomResizedCrop(size=target_size,scale = scale, antialias=True),
                                  RandomHorizontalFlip(p=horizontal_flip_prob)])
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Open video
        dir = self.data_paths[idx]
        frames_full = os.listdir(dir)



        if len(frames_full) < 300 and idx < len(self.data_paths):
            return self.__getitem__(idx + 1)
        
        
        if len(self.data_paths) == idx -1:
            raise StopIteration

        # # Get length of video
        nr_frames = len(frames_full)

        # # Make sure video is long enough or not to short
        # # If nr_frames is 0, then video is corrupted and we skip it
        # if nr_frames < self.frame_range[1]+1:
        #     return self.__getitem__(idx + 1) 

        # Choose random frames
        # frames1 = random.sample(frames[:nr_frames-self.frame_range[1]], self.n_per_video)
        # frames2 = random.sample(frames[self.frame_range[0], self.frame_range[1]+1])
        idx_f1 = np.random.choice(np.arange(0,nr_frames-self.frame_range[1]), size=self.n_per_video, replace=False)
        idx_f2 = np.random.choice(np.arange(self.frame_range[0],self.frame_range[1] + 1), size=self.n_per_video, replace=True) + idx_f1
        frames1 = [frames_full[i] for i in idx_f1]
        frames2 = [frames_full[i] for i in idx_f2]
        frames1_lst = []
        frames2_lst = []
        for i in range(self.n_per_video):
            # frame1 = cv2.imread(os.path.join(dir, frames1[i]))
            # frame2 = cv2.imread(os.path.join(dir, frames2[i]))
            frame1 = Image.open(os.path.join(dir, frames1[i])).convert('RGB')
            frame2 = Image.open(os.path.join(dir, frames2[i])).convert('RGB')
            frame1 = self.transform(frame1).unsqueeze(0)
            frame2 = self.transform(frame2).unsqueeze(0)
            frames1_lst.append(frame1)
            frames2_lst.append(frame2)

        frames1_tensor = torch.cat(frames1_lst, dim=0)
        frames2_tensor = torch.cat(frames2_lst, dim=0)
        frames_t = torch.cat((frames1_tensor, frames2_tensor), dim=0)
            
        # frames = vr.get_batch(np.concatenate([idx_f1,idx_f2],axis = 0))
        # frames = torch.moveaxis(frames,-1,1)
        # if self.transform:
        # frames_t = self.transform(frames).float()

        frames_mean = torch.mean(frames_t, dim=(2, 3))
        frames_std = torch.std(frames_t, dim=(2, 3))
        frames_norm = (frames_t- frames_mean.view(2*self.n_per_video,3,1,1))/frames_std.view(2*self.n_per_video,3,1,1)
        f1s = frames_norm[:self.n_per_video]
        f2s = frames_norm[self.n_per_video:]

        # Shape f1s, f2s is [n_per_video,C,H,W] 
        return f1s,f2s

def process_index_wrapper(obj, idx):
    return obj.__getitem__(idx)
class homebrew_dataloader():
    def __init__(self, data_dir = "./test_dataset/*",n_per_video = 2,frame_range = (4,48),patch_size = (16,16,3),target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5,batch_size = 160):
        self.data_paths = glob.glob(data_dir)
        # shuffle data_paths
        random.shuffle(self.data_paths)
        self.root = data_dir
        self.batch_size = batch_size
        self.n_per_video = n_per_video
        self.frame_range = frame_range
        self.patch_size = patch_size
        self.target_size = target_size
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.transform = Compose([ToTensor(),
                                 RandomResizedCrop(size=target_size,scale = scale, antialias=True),
                                  RandomHorizontalFlip(p=horizontal_flip_prob)])
        self.batch_idx = 0
        
    def __len__(self):
        return len(self.data_paths)


    def get_batch_parallel(self):
        if self.batch_idx + self.batch_size >= len(self.data_paths):
            self.batch_idx = 0
            random.shuffle(self.data_paths)
        idxs = np.arange(self.batch_idx, self.batch_idx + self.batch_size)
        self.batch_idx += self.batch_size

        # Use torch.utils.data.DataLoader for parallel loading
        loader = DataLoader(self, batch_size=self.batch_size, sampler=idxs)
        x_batch, y_batch = next(iter(loader))

        return x_batch, y_batch





    def get_batch(self):
        # Get batch of data
        if self.batch_idx + self.batch_size >= len(self.data_paths):
            self.batch_idx = 0
            random.shuffle(self.data_paths)
        idxs = np.arange(self.batch_idx,self.batch_idx+self.batch_size)
        self.batch_idx += self.batch_size
        f1s_batch = np.zeros((self.batch_size,self.n_per_video,3,self.target_size[0],self.target_size[1]))
        f2s_batch = np.zeros((self.batch_size,self.n_per_video,3,self.target_size[0],self.target_size[1]))        


        for i,idx in enumerate(idxs):
            f1s,f2s = self.__getitem__(idx)
            f1s_batch[i] = f1s
            f2s_batch[i] = f2s
        return f1s_batch,f2s_batch        


    def get_sub_batch(self,idxs):
        f1s = []
        f2s = []
        for idx in idxs:
            f1,f2 = self.__getitem__(idx)
            f1s.append(f1)
            f2s.append(f2)
        
        print("Done with batch")
        return f1s,f2s
            
            

    
    def __getitem__(self, idx):
        # Open video
        dir = self.data_paths[idx]
        frames_full = os.listdir(dir)



        if len(frames_full) < 300 and idx < len(self.data_paths):
            return self.__getitem__(idx + 1)
        
        
        if len(self.data_paths) == idx -1:
            raise StopIteration

        # # Get length of video
        nr_frames = len(frames_full)

        # # Make sure video is long enough or not to short
        # # If nr_frames is 0, then video is corrupted and we skip it
        # if nr_frames < self.frame_range[1]+1:
        #     return self.__getitem__(idx + 1) 

        # Choose random frames
        # frames1 = random.sample(frames[:nr_frames-self.frame_range[1]], self.n_per_video)
        # frames2 = random.sample(frames[self.frame_range[0], self.frame_range[1]+1])
        idx_f1 = np.random.choice(np.arange(0,nr_frames-self.frame_range[1]), size=self.n_per_video, replace=False)
        idx_f2 = np.random.choice(np.arange(self.frame_range[0],self.frame_range[1] + 1), size=self.n_per_video, replace=True) + idx_f1
        frames1 = [frames_full[i] for i in idx_f1]
        frames2 = [frames_full[i] for i in idx_f2]
        frames1_lst = []
        frames2_lst = []
        for i in range(self.n_per_video):
            # frame1 = cv2.imread(os.path.join(dir, frames1[i]))
            # frame2 = cv2.imread(os.path.join(dir, frames2[i]))


            frame1 = Image.open(os.path.join(dir, frames1[i])).convert('RGB')
            frame2 = Image.open(os.path.join(dir, frames2[i])).convert('RGB')
            frame1 = self.transform(frame1).unsqueeze(0)
            frame2 = self.transform(frame2).unsqueeze(0)
            frames1_lst.append(frame1)
            frames2_lst.append(frame2)

        frames1_tensor = torch.cat(frames1_lst, dim=0)
        frames2_tensor = torch.cat(frames2_lst, dim=0)
        frames_t = torch.cat((frames1_tensor, frames2_tensor), dim=0)
            
        # frames = vr.get_batch(np.concatenate([idx_f1,idx_f2],axis = 0))
        # frames = torch.moveaxis(frames,-1,1)
        # if self.transform:
        # frames_t = self.transform(frames).float()

        frames_mean = torch.mean(frames_t, dim=(2, 3))
        frames_std = torch.std(frames_t, dim=(2, 3))
        frames_norm = (frames_t- frames_mean.view(2*self.n_per_video,3,1,1))/frames_std.view(2*self.n_per_video,3,1,1)
        f1s = frames_norm[:self.n_per_video]
        f2s = frames_norm[self.n_per_video:]
        # Shape f1s, f2s is [n_per_video,C,H,W] 
        return f1s,f2s




def test_homebrew(_):
    dataset = homebrew_dataloader("./data/Kinetics/train_jpg/*",batch_size=500)
    t1 = time.time()    
    x_batch,y_batch = dataset.get_batch()
    print(time.time() - t1)

def test_torch():
    # test PreTrainingDataset with torch.utils.data.DataLoader
    dataset = PreTrainingDataset("./data/Kinetics/train_jpg/*")
    t1 = time.time()
    loader = DataLoader(dataset, batch_size=500, sampler=None)
    x_batch, y_batch = next(iter(loader))
    print(time.time() - t1)

def main():
    # run both tests in parallel
    # test_homebrew(None)
    test_torch()
    test_homebrew(None)
    test_torch()
    test_homebrew(None)

    # with Pool(processes=4) as executor:
    #     t1 = time.time()
    #     executor.map(test_homebrew,[None,None,None,None])
    #     # executor.map(test_homebrew,[None])
    #     # executor.map(test_homebrew,[None])
    #     # executor.map(test_homebrew,[None])
    # print(time.time() - t1)


if __name__ == '__main__':
    main()