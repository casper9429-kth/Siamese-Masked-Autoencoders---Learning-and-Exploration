import os
import random
import math

import numpy as np
import cv2
import scipy.io

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from video_dataset import VideoDataset


class JHMDB(VideoDataset):
    def __init__(self, root_dir="",videos_dir="", video_list_dir="", anns_dir="", num_segments=5, frames_per_segment=1, transform=None, train=False):
        super().__init__()
        self.root_dir = root_dir
        self.videos_dir = videos_dir
        if train:
            self.mode = "1" # training
        else:
            self.mode = "2" # testing
        video_lists = os.listdir(os.path.join(root_dir, video_list_dir))
        for lst in video_lists:
            with open(os.path.join(root_dir, video_list_dir, lst), 'r') as vl:
                for elem in vl:
                    if elem.split(" ")[1][:-1] == self.mode:
                        self.video_list.append(os.path.join("_".join(lst.split("_")[:-2]), elem.split(".")[0]))

        self.anns_dir = anns_dir
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        if transform is not None:
            self.transform = transform

    def get_annotations(self, video, ids=None):
        """
            Overwriting the get_annotations method of the VideoDataset class.
        """
        ann_file = os.path.join(self.root_dir, self.anns_dir, video, 'joint_positions.mat')

        mat = scipy.io.loadmat(ann_file)

        mat["viewpoint"] = str(mat["viewpoint"])

        return mat
    
    # def __get_frames(self, video_dir):
    #     return super().__get_frames(video_dir)
    
    def __getitem__(self, idx):
        video = self.video_list[idx]
        video_dir = os.path.join(self.root_dir, self.videos_dir, video)

        # get frames (and ids of the frames)
        frames, ids = self.get_frames(video_dir)

        # get the annotation files (per frame)
        annotations = self.get_annotations(video, ids)

        return frames, annotations





def main():
    root = "/Users/magnusrubentibbe/Downloads/JHMDB/"
    videos_dir = "Rename_Images/"
    video_list_dir =  "splits/"
    anns_dir = "joint_positions/"

    dataset = JHMDB(root, videos_dir, video_list_dir, anns_dir,train=False)

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    print(len(dl))
    print(dataset.__len__())

    for frames, anns in dl:
        print(frames.shape)
        print(type(anns))


if __name__ == "__main__":
    main()
