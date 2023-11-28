import glob
import concurrent.futures
import numpy as np
import time
import os
import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip
from osgeo import gdal

def load_sample(file_path, num_samples_per_video=1):

    sample = []
    for i in range(num_samples_per_video):
        idx1 = np.random.randint(0, 252)
        idx2 = np.random.randint(idx1 + 2, idx1 + 49)

        img1 = gdal.Open(file_path + f"/frame_{idx1}.jpg").ReadAsArray()
        img2 = gdal.Open(file_path + f"/frame_{idx2}.jpg").ReadAsArray()
        sample.append(img1)
        sample.append(img2)

    sample = np.array(sample)
    sample = transforms(sample)
    # Fold it to Num_samples_per_video x 2 x 3 x H x W
    sample = sample.reshape((num_samples_per_video, 2, *sample.shape[1:]))
    # Normalize
    mean = np.mean(sample, axis=(0, 1, 3, 4))
    std = np.std(sample, axis=(0, 1, 3, 4))
    sample = (sample - mean[None,None,:,None,None]) / std[None,None,:,None,None]
    
    
    return sample

def transforms(imgs, target_size=(224, 224), scale=(0.5, 1.0), horizontal_flip_prob=0.5):
    imgs_tensor = torch.from_numpy(imgs)

    transform = Compose([
        RandomResizedCrop(size=target_size, scale=scale, antialias=True),
        RandomHorizontalFlip(p=horizontal_flip_prob)
    ])

    cropped_imgs = torch.stack([transform(img) for img in imgs_tensor])

    cropped_imgs_numpy = cropped_imgs.numpy()
    # Normalize
    
    
    return cropped_imgs_numpy


class SiamMAEloader:
    def __init__(self, image_directory='./data/Kinetics/train_jpg/*', num_samples_per_video=1, batch_size=10):
        self.image_directory = image_directory
        self.num_samples_per_video = num_samples_per_video
        self.batch_size = batch_size
        self.cores = os.cpu_count()
        self.file_paths = glob.glob(self.image_directory)
        self.current_batch = 0

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __iter__(self):
        return self

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
            samples = list(executor.map(load_sample, file_paths, [self.num_samples_per_video] * len(file_paths)))

        return np.array(samples)


if __name__ == '__main__':
    loader = SiamMAEloader()

    # Test loading multiple batches
    num_batches_to_test = 5

    for _ in range(num_batches_to_test):
        start_time = time.time()
        batch = next(loader)
        print(f'Batch shape: {batch.shape}')
        elapsed_time = time.time() - start_time

        print(f'Time to load batch: {elapsed_time:.2f} seconds')
        # Add any additional processing or analysis of the loaded batch here

    print("Testing complete.")
