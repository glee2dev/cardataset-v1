# referenced & implemented for my own custom dataset relevant for Car Design feature detection.
# https: // debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_resnet(num_classes):
    
    # load faster rcnn pre-trained model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, 
                                                                progress=True, pretrained_backbone=False,
                                                                ) #trainable_backbone_layers=3
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required num of classes 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def get_mobilenet(num_classes):

    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone = torchvision.models.mobilenet_v3_large(pretrained=False, progress=True, pretrained_backbone=False).features
    # backbone = torchvision.models.mobilenet_v2(pretrained=False,
    #                                             progress=True, pretrained_backbone=False, 
    #                                             trainable_backbone_layers).features
    backbone.out_channels = 960 #1280 vgg 

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512), ), 
                                                aspect_ratios=((0.5, 1.0, 2.0), ))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                        box_roi_pool = roi_pooler)

    return model

def get_vgg(num_classes):
    backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
    backbone_out = 512
    backbone.out_channels = backbone_out

    resolution = 7
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= backbone_out*(resolution**2),representation_size=4096) 
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096,26) 

    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                    min_size = 600, max_size = 1000,
                    rpn_anchor_generator=anchor_generator,
                    rpn_pre_nms_top_n_train = 6000, rpn_pre_nms_top_n_test = 6000,
                    rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                    rpn_nms_thresh=0.7,rpn_fg_iou_thresh=0.7,  rpn_bg_iou_thresh=0.3,
                    rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                    box_roi_pool=roi_pooler, box_head = box_head, box_predictor = box_predictor,
                    box_score_thresh=0.6, box_nms_thresh=0.7,box_detections_per_img=300,
                    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                    box_batch_size_per_image=128, box_positive_fraction=0.25
                    )
    
    return model 
