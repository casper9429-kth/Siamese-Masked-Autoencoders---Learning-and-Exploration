# Datasets

## Pretraining
For pretraining the model, we use the Kinetics-400 dataset, which has these and these properties. bla bla bla,

## Downstream Tasks
The model's performance is tested on three different downstream tasks using different datasets. The tasks are:
- Video object segmentation on [DAVIS](https://davischallenge.org/challenge2017/index.html)
- Semantic part propagation on [VIP](https://sysu-hcp.net/lip/)
- Human pose propagation on [JHMDB](http://jhmdb.is.tue.mpg.de)

For loading the datasets, there is a VideoDataset super-class that the individual dataset classes are derived. For creating a new dataset class, a list of the available videos should be created in the __init__() function. This is then used to get the directories of the frames and annnotations in the __getitem__() function. Each subclass should overwrite the funtions get_frames() and get_annnotations() according to the structure of the dataset.
For sampling the video is split up into *num_segments* segments and we randomly choose *frames_per_segment* frames from that segment, so in total we sample *num_segments x frames_per_segment* frames from each video. The frames are returned as  tensors of size (*num_segments x frames_per_segment*, C, H, W) along with the corresponding annotations.

### [DAVIS](https://davischallenge.org/challenge2017/index.html)


### [VIP](https://sysu-hcp.net/lip/)


### [JHMDB](http://jhmdb.is.tue.mpg.de)
