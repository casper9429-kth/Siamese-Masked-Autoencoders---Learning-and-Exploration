import glob
import concurrent.futures
import numpy as np
import time
import os
import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize
from osgeo import gdal

def load_sample(file_path, num_samples_per_video=1, under_limit_sample=2,upper_limit_sample=10,scale=(0.5,1.0),horizontal_flip_prob=0.5):

    sample = []
    for i in range(num_samples_per_video):
        idx1 = np.random.randint(0, 300-upper_limit_sample)
        idx2 = np.random.randint(idx1 + under_limit_sample, idx1 + upper_limit_sample)

        img1 = gdal.Open(file_path + f"/frame_{idx1}.jpg").ReadAsArray()
        img2 = gdal.Open(file_path + f"/frame_{idx2}.jpg").ReadAsArray()
        img_sample = [img1, img2]
        sample.append(img_sample)

    sample = np.array(sample, dtype=np.float32)
    sample = transforms(sample, scale=scale, horizontal_flip_prob=horizontal_flip_prob)
    return sample

def transforms(imgs, target_size=(224, 224), scale=(0.5, 1.0), horizontal_flip_prob=0.5):
    imgs_tensor = torch.from_numpy(imgs)
    transform = Compose([
        RandomResizedCrop(size=target_size, scale=scale, antialias=True),
        RandomHorizontalFlip(p=horizontal_flip_prob),
        Normalize(mean=[94.58919054671311, 101.76960119823667, 109.7119184903159], std=[60.4976600980992, 61.531615689196876, 62.836912383122076])
    ])

    # Make sure that transformations are identical for both images
    cropped_imgs = torch.stack([transform(imgs_tensor[i]) for i in range(imgs_tensor.shape[0])])
    
    cropped_imgs_numpy = cropped_imgs.numpy()
    # Normalize
    
    
    return cropped_imgs_numpy


class SiamMAEloader:
    def __init__(self, image_directory='./data/Kinetics/train_jpg/*', num_samples_per_video=20, batch_size=10,under_limit_sample=2,upper_limit_sample=10,scale=(0.5,1.0),horizontal_flip_prob=0.5):
        self.image_directory = image_directory
        self.num_samples_per_video = num_samples_per_video
        self.batch_size = batch_size
        self.under_limit_sample = under_limit_sample
        self.upper_limit_sample = upper_limit_sample
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.cores = os.cpu_count()
        self.file_paths = glob.glob(self.image_directory)
        self.current_batch = 0

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __iter__(self):
        return self

    def reset_iterator(self):
        self.current_batch = 0

    def __next__(self):
        start_idx = self.current_batch * self.batch_size
        end_idx = (self.current_batch + 1) * self.batch_size
        batch_paths = self.file_paths[start_idx:end_idx]

        if end_idx > len(self.file_paths):
            self.current_batch = 0
            raise StopIteration

        self.current_batch += 1

        images = self.load_samples_parallel(batch_paths)

        return images


    def load_samples_parallel(self, file_paths, num_workers=None):
        if num_workers is None:
            num_workers = self.cores - 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # BXNUM_SAMPER_PER_VIDEOX2XHxWX3
            # samples = list(executor.map(load_sample, file_paths, [self.num_samples_per_video] * len(file_paths)))
            # samples = list(executor.map(load_sample, file_paths, [self.num_samples_per_video,self.under_limit_sample,self.upper_limit_sample] * len(file_paths)))
            # add load sample to executor, for each executor give it the next file path, but keep num_samples_per_video, under_limit_sample, upper_limit_sample the same for all
            samples = list(executor.map(load_sample, file_paths, [self.num_samples_per_video] * len(file_paths),[self.under_limit_sample] * len(file_paths),[self.upper_limit_sample] * len(file_paths),[self.scale]*len(file_paths),[self.horizontal_flip_prob]*len(file_paths)))
            
            
        return np.array(samples)



if __name__ == '__main__':
    # Test loading multiple batches
    num_batches_to_test = 5
    file_path = "./data/Kinetics/train_jpg/*"
    num_samples_per_video = 5
    batch_size = 10
    under_limit_sample = 2
    upper_limit_sample = 10

    loader = SiamMAEloader(image_directory=file_path, num_samples_per_video=num_samples_per_video, batch_size=batch_size,under_limit_sample=under_limit_sample,upper_limit_sample=upper_limit_sample)


    for i in range(num_batches_to_test):
        print(i)
        start_time = time.time()
        batch = next(loader)
        print(f'Batch shape: {batch.shape}')
        elapsed_time = time.time() - start_time

        print(f'Time to load batch: {elapsed_time:.2f} seconds')
        # Add any additional processing or analysis of the loaded batch here
        # Print mean and std of batch
        print(batch.shape)
        # For each batch take a random sample at a random video(axis 0) and plot all num_samples_per_video images side by side
        import matplotlib.pyplot as plt
        import random
        random.seed(0)
        random_video = random.randint(0,batch_size-1)
        random_video_sample = batch[random_video]
        # Plot all images side by side using matplotlib
        # f11, f12
        # f21, f22
        # f31, f32
        # .  , .
        # .  , .
        # fnum_samples_per_video1, fnum_samples_per_video2

        # Create a figure with 2 columns and num_samples_per_video rows
        fig, axs = plt.subplots(num_samples_per_video, 2)
        fig.suptitle('Vertically stacked subplots')
        # For each image in the random_video_sample
        for i in range(num_samples_per_video):
            # Plot the first image in the left column
            axs[i,0].imshow(np.einsum('ijk->jki',random_video_sample[i,0]))
            # Plot the second image in the right column
            axs[i,1].imshow(np.einsum('ijk->jki',random_video_sample[i,1]))
        plt.show()
        
    
        
        

    print("Testing complete.")


    
        
