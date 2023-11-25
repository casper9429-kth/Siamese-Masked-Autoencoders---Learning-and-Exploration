import os
import random

NUM_VIDEOS = 10000
VIDEOS_ROOT = "../data/"

def get_filenames(path, num):
    vid_lst = os.listdir(path)
    sublst = random.sample(vid_lst, num)
    for elem in sublst:
        pass

    return sublst


def main():
    subdirs = os.listdir(VIDEOS_ROOT)
    num_per_sub = NUM_VIDEOS/len(subdirs)

    vids_to_extract = []
    for subdir in subdirs:
        vid_files = get_filenames(os.path.join(VIDEOS_ROOT, subdir), num_per_sub)
        vids = [os.path.join(VIDEOS_ROOT, subdir, x) for x in vid_files]
        vids_to_extract.extend(vids)





