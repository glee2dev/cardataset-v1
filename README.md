# CS492 Final Project repository
## Vehicle Design Feature Extractor // Team 35 Gyunpyo Lee
This repository is dedicated to CS492I Final project done in 2021. \
Faster R-CNN on custom dataset (car design related) 
still the beta version of the datset, the dataset is going to be developed further. 

![image](https://github.com/glee2dev/cardataset-v1/blob/main/predictions/result_videos/video4_inference_2.gif?raw=true)

Description of the content in this repository is explained in alphabetical order.
If you have any difficulty downloading the dataset, pth file, or using the model feel free to contact via e-mail. 



## Contents
### Car_Dataset
link to the custom dataset, annotations informations & general information related to the dataset


### etc
utility files for data handling


### mAP
mAP result (outputs) of ResNet-50 FPN, VGG-16, MobileNetV3 Large & link to input.zip


### model
python code files used for model training / inference of images & videos


### predictions
prediction results of the model in both images & videos 


### pth
link to pth file created after the model training (ResNet-50 FPN, VGG-16, MobileNetV3 Large)



## Reference
- Train your own object detector with Faster-RCNN & PyTorch // Johannes Schmidt \
https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70

- Custom Object Detection using PyTorch Faster RCNN // Sovit Ranjan Rath \
https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

- TorchVision을 사용한 Faster R-CNN(3) // Hyungjo Byun \
https://hyungjobyun.github.io/machinelearning/FasterRCNN3/

