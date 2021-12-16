# referenced & implemented for my own custom dataset relevant for Car Design feature detection.
# https: // debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, PTH_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import get_resnet, get_mobilenet, get_vgg
import utils
from utils import Average_fn, SaveBestModel, save_model
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

import torch 
import matplotlib.pyplot as plt 
import time 

plt.style.use('ggplot')

def train(train_data_loader, model):
    print('Training')
    global train_itr 
    global train_loss_list
    
    # tqdm progress bar 
    progress_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(progress_bar):
        optimizer.zero_grad()
        images, targets = data 
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        # print(loss_value)
        
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        
        losses.backward()
        optimizer.step()
        
        train_itr += 1 
        
        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        # print(train_loss_list)

    return train_loss_list

def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # tqdm progress bar
    progress_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(progress_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)

        val_itr += 1

        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list

if __name__ == '__main__':
    # initialize the model and move to the computation device 

    # train resnet
    model = get_resnet(num_classes = NUM_CLASSES)
    # print(model)
    # train mobilenetv3
    # model = get_mobilenet(num_classes= NUM_CLASSES)
    # train vgg
    # model =get_vgg(num_classes = NUM_CLASSES)

    model = model.to(DEVICE)
    
    # freeze and training 
    checkpoint = torch.load('/content/drive/MyDrive/CS492I/final/outputs/best_model_resnet_v2.pth', map_location=DEVICE) #load model from location 
    model.load_state_dict(checkpoint['model_state_dict'])

    # get the model parameters 
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
       

    # initialize the average class 
    train_loss_hist = Average_fn()
    val_loss_hist = Average_fn()
    

    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values
    # iterations till end and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    

    
    MODEL_NAME = 'cd_model'
    
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_transformed_image
        show_transformed_image(train_loader)

    save_best_model = SaveBestModel()
    

    # may be wrong implementation of the coco eval tool 
    # for epoch in range(NUM_EPOCHS):
    #     print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

    #     train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=10)
    #     lr_scheduler.step()
    #     evaluate(model, valid_loader, device=DEVICE)

    #     if (epoch+1) % SAVE_MODEL_EPOCH == 0:
    #         torch.save(model.state_dict(), f"{PTH_DIR}/model_mAP{epoch+1}.pth")
    #         print('SAVING MODEL COMPLETE \n')
        
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        
        # reset the training and validation loss for the current epoch 
        train_loss_hist.reset()
        val_loss_hist.reset()
        
        # create two subplots, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        
        # start timer and run training and validation 
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch # {epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch # {epoch} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
        # save best model 
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )

        # save model after n epochs
        if (epoch+1) % SAVE_MODEL_EPOCH == 0:
            save_model(epoch, model, optimizer)
            print('SAVING MODEL COMPLETE \n')
        
        # save loss plots after n epochs    
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            print("SAVING PLOTS DONE")
            
        if (epoch+1) == NUM_EPOCHS:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            
            
            torch.save(model.state_dict(), f"{PTH_DIR}/model(epoch+1).pth")
            
        # plt.close('all')
        
        time.sleep(5)
    