import numpy
import cv2 
import glob
import os
import sys
import json
from osgeo import gdal
# Get amount of cores
CORES = os.cpu_count() -1 if os.cpu_count() > 1 else 1

FOLDER_PATH = "./data/Kinetics/train_jpg/*"
folder_paths = glob.glob(FOLDER_PATH)
# shuffle
numpy.random.shuffle(folder_paths)
N_VIDEOS = 10000
folder_paths = folder_paths[:N_VIDEOS]
N_FRAMES = 10
# Get 10 videos from each folder
paths = []
for i, folder_path in enumerate(folder_paths):
    # Get all frames
    frame_paths = glob.glob(folder_path + "/*")
    # Shuffle
    numpy.random.shuffle(frame_paths)
    # Get 10 frames
    frame_paths = frame_paths[:N_FRAMES]
    # Append to paths
    paths += frame_paths
# Random shuffle

print("Number of videos:", len(paths))

# Load a video as numpy array
def load_video(path):
    """
    Load a image and return it as numpy array
    """
    # Read image
    img = gdal.Open(path).ReadAsArray()
    # Imt to HWC
    img = numpy.einsum('ijk->jki', img)
    # img = cv2.imread(path)
    # Convert to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def mean_of_img(img):
    """
    Calculate mean of image
    """
    # Calculate mean of image
    mean = numpy.mean(img, axis=(0,1))
    return mean

def var_of_img(img):
    """
    Calculate var of image
    """
    # Calculate std of image
    std = numpy.var(img, axis=(0,1))
    return std

# Given var and mean of each frame, calculate mean and var of dataset
def mean_of_dataset(paths):
    """
    Calculate mean and var of dataset
    """
    # Init
    mean = numpy.zeros((1,3))
    var = numpy.zeros((1,3))
    # Loop over all videos
    for path in paths:
        # Load video
        img = load_video(path)
        # Calculate mean of image
        mean += mean_of_img(img)
        # Calculate std of image
        var += var_of_img(img)
    # Calculate mean of dataset
    mean = mean / len(paths)
    # Calculate var of dataset
    var = var / len(paths)
    return mean, var

# Split paths into 8 equal parts
paths_split = numpy.array_split(paths, CORES)

# Use 8 processes to calculate mean and var of dataset
from multiprocessing import Pool
with Pool(CORES) as p:
    # Calculate mean and var of dataset
    mean_var = p.map(mean_of_dataset, paths_split)

# Calculate mean and var of dataset
mean = numpy.zeros((1,3))
var = numpy.zeros((1,3))
for m, v in mean_var:
    mean += m
    var += v
mean = mean / len(mean_var)
var = var / len(mean_var)
std = numpy.sqrt(var)

# Dump mean and std to json file
with open("mean_std.json", "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)