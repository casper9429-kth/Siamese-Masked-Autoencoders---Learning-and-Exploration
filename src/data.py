import cv2
import glob
import torch
import random
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics
from skimage.transform import rescale,resize


def random_resized_crop(img,scale,target_size):
    ratio = (3.0 / 4.0, 4.0 / 3.0)

    def get_params(img,scale,ratio):
        height, width,_ = img.shape
        area = height * width
        log_ratio = np.log(np.array(ratio))
        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
                return i, j, h, w
        
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else: 
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    i, j, h, w = get_params(img,scale,ratio)
    cropped_image = img[i:i + h, j:j + w]
    print(cropped_image.shape)
    plt.imshow(cropped_image)
    plt.show()
    resized_image = resize(cropped_image, target_size)
    
    print(resized_image.shape)

    return resized_image


class PreTrainingDataset(Dataset):
    def __init__(self, data_dir = "../data_test/*",n_per_video = 2,frame_range = (4,48),patch_size = (16,16,3),target_size = (224,224,3),scale = (0.5,1)):
        self.data_paths = sorted(glob.glob(data_dir))
        self.n_per_video = n_per_video
        self.frame_range = frame_range
        self.patch_size = patch_size
        self.target_size = target_size
        self.scale = scale
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        vid_capture = cv2.VideoCapture(self.data_paths[idx])
        nr_frames = int(vid_capture.get(7))
        idx_f1 = np.random.choice(np.arange(0,nr_frames-self.frame_range[0]), size=self.n_per_video, replace=False)
        idx_f2 = np.random.choice(np.arange(self.frame_range[0],self.frame_range[1] + 1), size=self.n_per_video, replace=True) + idx_f1
        rnd_var = np.random.uniform(0,1,self.n_per_video)
        frames = []
        for i in range(self.n_per_video):
            if idx_f2[i] > nr_frames - 1:
                idx_f2[i] = np.random.choice(np.arange(idx_f1[i] + 4, nr_frames), size=1, replace=True)
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_f1[i])
            _, f1 = vid_capture.read()
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_f2[i])
            _, f2 = vid_capture.read()
            # QUESTION: Should this be the same or seperate like it is now? Similar decision to fliplr which is the same for both images.
            f1t = random_resized_crop(f1,scale=self.scale,target_size=self.target_size)
            f2t = random_resized_crop(f2,scale=self.scale,target_size=self.target_size)
            if rnd_var[i]>0.5:
                f1t = np.fliplr(f1t)
                f2t = np.fliplr(f2t)
            f1t = patchify(f1t,self.patch_size,step=self.patch_size[0])
            f2t = patchify(f2t,self.patch_size,step=self.patch_size[0])
            frames.append([f1t,f2t])

        return frames


if __name__ == '__main__':
    dataset = PreTrainingDataset()
    print(dataset.__len__())
    for i, data in enumerate(dataset):
        frames = data