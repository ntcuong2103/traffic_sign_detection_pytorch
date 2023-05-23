from typing import List, Tuple, OrderedDict
import warnings
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator


'''
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
'''
class FasterRCNNResNet50FPN(FasterRCNN):
    def __init__(self, num_classes, **kwargs):
        backbone = resnet_fpn_backbone('resnext50_32x4d', pretrained=True, trainable_layers=5)
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((1.0),) * len(anchor_sizes)

        anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                            aspect_ratios=aspect_ratios)
        super().__init__(backbone, num_classes, 
                         rpn_anchor_generator=anchor_generator, 
                         rpn_pre_nms_top_n_train=4000, rpn_pre_nms_top_n_test=2000,
                         rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                         **kwargs)

if __name__ == "__main__":
    model = FasterRCNNResNet50FPN(num_classes=8)
    pass
    
