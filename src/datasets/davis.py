import os
import random
import math

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from video_dataset import VideoDataset


class DAVIS2017(VideoDataset):
    def __init__(self, root_dir="",videos_dir="", video_list="txt_file", anns_dir="", num_segments=5, frames_per_segment=1, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.videos_dir = videos_dir
        with open(os.path.join(root_dir,video_list), "r") as vl:
            for elem in vl:
                self.video_list.append(elem[:-1])

        self.anns_dir = anns_dir
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        if transform is not None:
            self.transform = transform
        

    def __len__(self):
        return len(self.video_list)
    
    def __get_frames(self, video_dir):
        """
            Private method to get annotations for corresponding videos. Can be overwritten by derived classes.

            video_dir: str (specifies directory of the video frames)
        """
        frame_list = sorted(os.listdir(video_dir))
        len_segment = math.ceil(len(frame_list)/self.num_segments)
        segments = [frame_list[x:x+len_segment] for x in range(0, len(frame_list), len_segment)]
        
        ids = []
        frames = []
        for i, segment in enumerate(segments):
            idxs = random.sample(range(len(segment)), k=self.frames_per_segment)
            seg_frames = []
            ids.extend(list(i*len_segment+np.array(idxs)))
            for elem in [segment[id] for id in idxs]:
                img = cv2.imread(os.path.join(video_dir, elem))
                seg_frames.append(img)
            frames.extend(seg_frames)

        frames = np.stack(frames)
        frames = np.einsum('nhwc->nchw', frames)

        frames = self.transform(torch.tensor(frames))

        return frames, ids
    
    def __get_annotations(self, video, ids=None):
        """
            Private method to get annotations for corresponding videos. Can be overwritten by derived classes.

            video: str (specifies the video)
            ids: list[int] (specifies ids of the selected frames)
        """
        ann_dir = os.path.join(self.root_dir, self.anns_dir, video)
        ann_list = sorted(os.listdir(ann_dir))

        annotations = [ann_list[id] for id in ids]

        return annotations


    def __getitem__(self, idx):
        video = self.video_list[idx]
        video_dir = os.path.join(self.root_dir, self.videos_dir, video)

        # get frames (and ids of the frames)
        frames, ids = self.__get_frames(video_dir)

        # get the annotation files (per frame)
        annotations = self.__get_annotations(video, ids)

        return frames, annotations
        


def main():
    root = "/Users/magnusrubentibbe/Downloads/DAVIS/"
    videos_dir = "JPEGImages/480p/"
    video_list =  "ImageSets/2017/val.txt"
    anns_dir = "Annotations/480p/"

    dataset = DAVIS2017(root, videos_dir, video_list, anns_dir)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    print(dataset.__len__())

    print(len(dl))
    for frames, anns in dl:
        print(type(frames))
        print(len(anns))



if __name__ == "__main__":
    main()

