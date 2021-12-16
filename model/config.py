# referenced & implemented for my own custom dataset relevant for Car Design feature detection.
# https: // debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

import torch

BATCH_SIZE = 2 # + or - depending on GPU
RESIZE_TO = 1024 # 512 # resize for training and transforms 
NUM_EPOCHS = 50 # number of epochs for training 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML directory 
TRAIN_DIR = "/content/drive/MyDrive/CS492I/final/Car_Dataset_2/train"
# validation images and XML directory 
VALID_DIR = "/content/drive/MyDrive/CS492I/final/Car_Dataset_2/test"
# pth saving directory 
PTH_DIR = "/content/drive/MyDrive/CS492I/final/outputs/pth"

# classes: 0 index is for BG 
CLASSES = [
    '_background_',
    'hatchback',
    'headlamp',
    'grill',
    'wheel',
    'opening',
    'bodysurface',
    'rearlamp',
    'dlo',
    'mirror',
    'fender',
    'frontbumper',
    'hood',
    'rearbumper',
    'diffuser',
    'spoiler',
    'aero',
    'wagon',
    'sedan',
    'foglamp',
    'rocker',
    'coupe-convertible',
    'sports',
    'coupe',
    'compact',
    'suv-coupe',
    'suv',
]

NUM_CLASSES = 27 # suv 27th 

# true if visualizing images after transforming
VISUALIZE_TRANSFORMED_IMAGES = False # False

# location to save model and plots 
OUT_DIR = '/content/drive/MyDrive/CS492I/final/outputs'
SAVE_PLOTS_EPOCH = 5 # save loss plots after 2 epochs 
SAVE_MODEL_EPOCH = 50 # save model after 2 epochs 
