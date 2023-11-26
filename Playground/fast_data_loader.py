import cv2
import glob
import concurrent.futures
import numpy as np
import time
import os
import torch
from torchvision.datasets import Kinetics
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomResizedCrop,RandomHorizontalFlip,ToTensor
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst

cores = os.cpu_count() # 4

def load_sample(folder_path,num_samples_per_video=1):
    # Sample 2 random indices
    # Load the images
    sample = []
    for i in range(num_samples_per_video):
        idx1 = np.random.randint(0, 252)
        idx2 = np.random.randint(idx1+2,idx1+49)
        # load image using gdal
        img1 = gdal.Open(folder_path + "/frame_" + str(idx1) +".jpg")
        img2 = gdal.Open(folder_path + "/frame_" + str(idx2) +".jpg")
        img1 = img1.ReadAsArray()
        img2 = img2.ReadAsArray()
        sample.append(img1)
        sample.append(img2)
        
    
    sample = np.array(sample)
    sample = transforms(sample)
    return sample


def transforms(imgs, target_size=(224, 224),scale=(0.5,1.0), horizontal_flip_prob=0.5):  
    """
    Input: imgs - numpy array of shape (N, W, H, 3)
    Output: Randomly cropped and resized images of shape (N, 256, 256, 3), all images in the batch are cropped in the same way
    """

    # Convert numpy array to torch tensor
    imgs_tensor = torch.from_numpy(imgs)  # Convert to (N, 3, W, H)

    # Create RandomResizedCrop transformation
    transform = Compose([
                            RandomResizedCrop(size=target_size,scale = scale, antialias=True),
                            RandomHorizontalFlip(p=horizontal_flip_prob)])
    # Apply the same random crop to all images in the batch
    cropped_imgs = torch.stack([transform(img) for img in imgs_tensor])

    # Convert torch tensor back to numpy array
    cropped_imgs_numpy = cropped_imgs.numpy().transpose((0, 2, 3, 1))  # Convert back to (N, W, H, 3)

    return cropped_imgs_numpy

    
    
def load_samples_parallel(file_paths, num_workers=cores-1):
    # Using multithreading to parallelize image loading
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        samples = list(executor.map(load_sample, file_paths))

    return samples



def main():
    # Replace 'your_image_directory/*.jpg' with the path to your directory containing the JPEG images
    image_directory = './data/Kinetics/train_jpg/*'
    
    file_paths = glob.glob(image_directory)  # Adjust the number of images to load

    i = -1
    while True:
        i += 1
        start_idx = i*500
        end_idx = (i+1)*500
        batch_paths = file_paths[start_idx:end_idx]
        # Check that the end_idx is not larger than the number of files
        if end_idx > len(glob.glob(image_directory)):
            i = -1
            continue

        # Load images in parallel
        t1 = time.time()
        images = load_samples_parallel(batch_paths)
        # Transform the list into a numpy array
        print(f'Loaded {len(images)} samples in {time.time() - t1} seconds')
    
    
    

if __name__ == '__main__':
    main()


