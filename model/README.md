## General description
- FasterRCNN-Resnet.ipynb | used for training in the Colab, mAP calculation
- train.py | for training (model loading and checkpoint loading need to be modified before training) 
- interence.py inference_video.py | testing the model on both images & videos (resolution / aspect ratio of the video is set for 480 currently) 
- config.py | directory for train / test dataset, class info, batch size, resize, are located in this python file
- model.py | Resnet50 FPN, MobileNetv3 large, VGG16 model 
- util.py | transform, plot saving, best model saving, model saving 
