import os
import glob
import concurrent.futures
# Import decord 
import decord
from decord import VideoReader
from decord import cpu, gpu
import cv2
import random

# Get amount of available CPU cores
cores = os.cpu_count()
print("Number of CPU cores: ", cores)
cores = [cpu(i) for i in range(0, cores)]

PATH_IN = "./data/Kinetics/train_mp4/*"
PATH_OUT = "./data/Kinetics/train_jpg/"
FRAMES_PER_VIDEO = 300
NR_OF_VIDEOS = 5000

# If PATH_OUT does not exist, create it
if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
else:
    pass
    print(len(os.listdir(PATH_OUT)))
    size=0
    for path, dirs, files in os.walk(PATH_OUT):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)

    print("SIZE", size)


# Get all paths to mp4 files using OS
paths = []
for path in glob.glob(PATH_IN):
    paths.append(path)
    
# Shuffle paths randomly
random.shuffle(paths)

# Cut paths to desired length
paths = paths[0:NR_OF_VIDEOS]

print("Number of videos: ", len(paths))

def load_video(path,i):
    """
    Load a video and in path and return all frames as numpy array
    uses cpu core i
    """
    vr = VideoReader(path, ctx=cpu(i))
    # Get entire video
    frames = vr.get_batch(range(0, len(vr))).asnumpy()    
    return frames

def save_frames(frames, path):
    """
    Save all frames of a video in path as jpg
    """
    # Get video name
    name = path.split("/")[-1].split(".")[0]

    # Create directory for video
    # If it does not exist
    if not os.path.exists(PATH_OUT + name):
        os.makedirs(PATH_OUT + name)
    # Save each frame as jpg
    for i, frame in enumerate(frames):
        # Save as jpg
        cv2.imwrite(PATH_OUT + name + "/frame_" + str(i) + ".jpg", frame)

# load and save videos 
def load_and_save(path,i):
    """
    Load a video and save all frames as jpg
    """

    # check if out path already exist, if it does, return
    name = PATH_OUT + path.split("/")[-1].split(".")[0]
    if os.path.isdir(name):
        print("Already exists!")
        return




    frames = load_video(path,i)
    
    # Remove video at path
    os.remove(path)
    
    # if frames is less than FRAMES_PER_VIDEO, skip video
    if len(frames) < FRAMES_PER_VIDEO:
        print("Skipping video: ", path)
        return
    else:
        save_frames(frames, path)
        print("Done with video: ", path)


# # Load and save videos sequentially
# for i, path in enumerate(paths):
#     load_and_save(path, i % len(cores))

# # Parallelize video loading using all available CPU cores, distribute videos evenly among cores 
with concurrent.futures.ProcessPoolExecutor() as executor:
    for i, path in enumerate(paths):
        executor.submit(load_and_save, path, i % len(cores))
