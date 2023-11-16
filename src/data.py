import cv2
import glob
import torch
import random
# import jax.numpy as np
# import jax.random as random
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics
#from skimage.transform import rescale,resize
from torchvision.transforms import Compose, RandomResizedCrop,RandomHorizontalFlip,ToTensor
from torch.utils.data import DataLoader



class PreTrainingDataset(Dataset):
    def __init__(self, data_dir = "./test_dataset/*",n_per_video = 2,frame_range = (4,48),patch_size = (16,16,3),target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5):
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
        # Open video
        vid_capture = cv2.VideoCapture(self.data_paths[idx])

        # Get length of video
        nr_frames = int(vid_capture.get(7))

        # Make sure video is long enough or not to short
        # If nr_frames is 0, then video is corrupted and we skip it
        if nr_frames == 0 or nr_frames < self.frame_range[1]+1:
            # Check that idx is not the last video
            if idx == self.__len__()-1:
                raise StopIteration
            else:
                return self.__getitem__(idx + 1) # TODO: Check that we don't double count videos

        # Choose random frames
        idx_f1 = np.random.choice(np.arange(0,nr_frames-self.frame_range[1]), size=self.n_per_video, replace=False)
        idx_f2 = np.random.choice(np.arange(self.frame_range[0],self.frame_range[1] + 1), size=self.n_per_video, replace=True) + idx_f1

        # Create empty lists to store frames
        f1s = []
        f2s = []

        # Loop over number of frames per video
        for i in range(self.n_per_video):
            # Read frames
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_f1[i])
            _, f1 = vid_capture.read()
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_f2[i])
            _, f2 = vid_capture.read()

            # Swap channels from BGR to RGB
            f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
            f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
            
            # Resize and normalize
            if self.transform:
                # Transform
                f1t = torch.from_numpy(np.einsum('hwc->chw',f1))
                f2t = torch.from_numpy(np.einsum('hwc->chw',f2))
                f1t = self.transform(f1t)
                f2t = self.transform(f2t)
                f1 = f1t.numpy()
                f2 = f2t.numpy()
                f1 = np.einsum('chw->hwc',f1)
                f2 = np.einsum('chw->hwc',f2)

                # Normalize
                f1_mean = np.mean(f1, axis=(0, 1,2))
                f1_std = np.std(f1, axis=(0, 1,2))
                f1_norm = (f1 - f1_mean) / f1_std

                f2_mean = np.mean(f2, axis=(0, 1,2))
                f2_std = np.std(f2, axis=(0, 1,2))
                f2_norm = (f2 - f2_mean) / f2_std

                # Append
                f1s.append(f1_norm)
                f2s.append(f2_norm)
        
        # Stack
        f1s = np.stack(f1s,axis=0)
        f2s = np.stack(f2s,axis=0)

        # n x h x w x c -> n x c x h x w
        f1s = np.einsum('nhwc->nchw',f1s)
        f2s = np.einsum('nhwc->nchw',f2s)

        # Shape f1s, f2s is [n_per_video,c,H,W] 
        return f1s,f2s


def main():
    dataset = PreTrainingDataset()
    print(dataset.__len__())

    for a,b in dataset:
        print(a.shape)
        print(b.shape)
        break

    # test data loader
    train_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    print(len(train_loader))
    assert len(train_loader) > 0, "train_loader is empty"


if __name__ == '__main__':
    main()