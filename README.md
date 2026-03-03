# NGP-SR
This is the official code implementation for NGP-SR 
Project page (here I need to make a link)

# Setup
Python 3 dependencies:
## Pytorch 2.6.0
## Cuda 11.8
## matplotlib
## tqdm
## torchmetrics

# Data Structure
We defined two kinds of dataloader for DTU dataset and Blender/LLFF dataset
（here I need you write a data structure draft, I will change it later)

# Training
For Blender/LLFF dataset
python mainblender.py --train_folder chair --baseres 100 --scale 2  --batch_size 20 --num_mf_layers 1

For DTU dataset
python maindtus.py --baseres 128 --scale 4 --batch_size 3 --resume --gpus "0"
Due to DTU has a huge amount of data, we also implemented a multi-card training
python maindtus.py --baseres 128 --scale 4 --batch_size 3 --resume --gpus "0,1,2,3"


