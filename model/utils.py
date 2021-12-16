# referenced & implemented for my own custom dataset relevant for Car Design feature detection.
# https: // debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

import torch
import albumentations as A 
import cv2
import numpy as np 

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes ,RESIZE_TO as resize

#### AVERAGE ####
class Average_fn: 
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0 
        
    def send(self, value):
        self.current_total += value 
        self.iterations += 1
        
    @property
    def value(self): 
        if self.iterations == 0:
            return 0 
        else:
            return 1.0 * self.current_total / self.iterations
        
    def reset(self):
        self.current_total = 0.0 
        self.iterations = 0.0 

#### COLLATE FUNCTION ####
def collate_func(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

#### TRAINING & VALUDATION AUGMENTATIONS ####
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),], 
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    # define the validation transforms 
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),], 
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

class SaveBestModel:
    """
    To save the best model
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/best_model_resnet_v3.pth') #best_model #change the name for different models

def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'outputs/last_model{epoch+1}_resnet_v3.pth')

####  SHOW TRANSFORMED IMAGES  ####
def show_transformed_image(train_loader):
    """
    show transformed image from the train_loader 
    show only when if "VISUALIZE_TRANSFORMED_IMAGES = True" in config.py
    """
    # print(len(train_loader))
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                    for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1,2,0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample, 
                                (box[0], box[1]), 
                                (box[2], box[3]),
                                (0, 0, 255), 2)
                cv2.imshow('Transformed image', sample)
                cv2.waitkey(0)
                cv2.destroyAllWindows()

