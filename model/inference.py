import numpy as np 
import cv2 
import torch 
import glob as glob
import time 

from model import get_resnet, get_mobilenet, get_vgg
import matplotlib.pyplot as plt
from config import NUM_CLASSES, DEVICE, CLASSES

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the model and the trained weights 
model = get_resnet(num_classes=NUM_CLASSES).to(DEVICE)
# model = get_mobilenet(num_classes=26).to(DEVICE)
checkpoint = torch.load('/content/drive/MyDrive/CS492I/final/outputs/best_model_resnet_v3.pth', map_location=DEVICE) #load model from location 
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# directory where all the images are present  
DIR_TEST = '/content/drive/MyDrive/CS492I/final/test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# classes: 0 is reserved for background
CLASSES = CLASSES

# threshold
detection_threshold = 0.8

# count total # of images iterated through
frame_count = 0
# keep FPS for each image 
total_fps = 0

for i in range(len(test_images)):
    # get the image file name for saving output 
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    #BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /=255.0
    # color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor 
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = torch.unsqueeze(image, 0)
    start_time = time.time()

    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()

    # for the current fps update
    fps = 1 / (end_time - start_time)
    total_fps += fps

    frame_count += 1
    
    # load all detection to CPU for operations 
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to 'detection_threshold'
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicted class names 
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of the box 
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image, 
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)
            cv2.putText(orig_image, 
                        pred_classes[j],
                        (int(box[0]), int(box[1] -5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 
                        2, lineType=cv2.LINE_AA)
            
        # cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(f"/content/drive/MyDrive/CS492I/final/test_predictions/{image_name}.jpg", orig_image,)
    print(f"Image {i+1} done")
    print('-'*50)


print('TEST PREDICTIONS DONE')
cv2.destroyAllWindows()

avg_fps = total_fps / frame_count
print(f"Average FPS : {avg_fps : .3f}")
