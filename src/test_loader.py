import glob
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomResizedCrop,RandomHorizontalFlip,ToTensor
import os
from PIL import Image


class TestLoader(Dataset):
    def __init__(self, data_dir = './data/Kinetics/train_jpg_small/*',n_per_video = 8,step_size = 4,target_size = (224,224),scale = (0.5,1),horizontal_flip_prob = 0.5):
        self.data_paths = glob.glob(data_dir)
        self.root = data_dir
        self.n_per_video = n_per_video
        self.step_size = step_size
        self.target_size = target_size
        self.scale = scale
        self.horizontal_flip_prob = horizontal_flip_prob
        self.transform = Compose([ToTensor(),
                                 RandomResizedCrop(size=target_size,scale = scale, antialias=True),
                                  RandomHorizontalFlip(p=horizontal_flip_prob)])

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        frame_list = sorted(os.listdir(self.data_paths[idx]))

        # random sample index of first frame
        id1 = random.randint(0,len(frame_list)-self.n_per_video*self.step_size)

        # get n_per_video-1 frame with step size
        frame_idx = [id1+i*self.step_size for i in range(self.n_per_video)]

        # get frames
        frames = [Image.open(self.data_paths[idx]+'/'+frame_list[i]) for i in frame_idx]
        frames = torch.stack([self.transform(frame) for frame in frames])
        frames1 = frames[0].unsqueeze(0).tile((7,1,1,1))
        frames2 = frames[1:]
        frames = torch.cat([frames1,frames2],dim=0)
        frames = frames.numpy()
        return frames
    

def test():
    dataset = TestLoader()
    dataloader = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=0)
    for batch in dataloader:
        print(batch.shape)
        break

if __name__ == '__main__':
    test()