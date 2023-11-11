# Installation requirements

First, create a new conda environment:

    conda create -n SiamMAE

Then install  the requirements:
- PyTorch 1.7.0+
- torchvision 0.8.1+
- pytorch-image-models 0.3.2 (timm)
- omegaconf
For the Google Cloud machine make sure to install the GPU version of PyTorch! 

    conda install -c pytorch pytorch torchvision

    pip install timm==0.3.2

    pip install omegaconf

In case we will need additional software, please add them to this document!


