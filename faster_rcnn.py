from typing import List, Tuple, OrderedDict
import warnings
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


'''
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
'''
class FasterRCNNResNet50FPN(FasterRCNN):
    def __init__(self, num_classes, **kwargs):
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        super().__init__(backbone, num_classes, **kwargs)

    # def forward(self, images, targets=None):
    #     """
    #     Arguments:
    #         images (list[Tensor]): images to be processed
    #         targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

    #     Returns:
    #         result (list[BoxList] or dict[Tensor]): the output from the model.
    #             During training, it returns a dict[Tensor] which contains the losses.
    #             During testing, it returns list[BoxList] contains additional fields
    #             like `scores`, `labels` and `mask` (for Mask R-CNN models).

    #     """
    #     if self.training and targets is None:
    #         raise ValueError("In training mode, targets should be passed")
    #     if self.training:
    #         assert targets is not None
    #         for target in targets:
    #             boxes = target["boxes"]
    #             if isinstance(boxes, torch.Tensor):
    #                 if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
    #                     raise ValueError("Expected target boxes to be a tensor"
    #                                      "of shape [N, 4], got {:}.".format(
    #                                          boxes.shape))
    #             else:
    #                 raise ValueError("Expected target boxes to be of type "
    #                                  "Tensor, got {:}.".format(type(boxes)))

    #     original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    #     for img in images:
    #         val = img.shape[-2:]
    #         assert len(val) == 2
    #         original_image_sizes.append((val[0], val[1]))

    #     images, targets = self.transform(images, targets)

    #     # Check for degenerate boxes
    #     # TODO: Move this to a function
    #     if targets is not None:
    #         for target_idx, target in enumerate(targets):
    #             boxes = target["boxes"]
    #             degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
    #             if degenerate_boxes.any():
    #                 # print the first degenerate box
    #                 bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
    #                 degen_bb: List[float] = boxes[bb_idx].tolist()
    #                 raise ValueError("All bounding boxes should have positive height and width."
    #                                  " Found invalid box {} for target at index {}."
    #                                  .format(degen_bb, target_idx))

    #     features = self.backbone(images.tensors)
    #     if isinstance(features, torch.Tensor):
    #         features = OrderedDict([('0', features)])
    #     proposals, proposal_losses = self.rpn(images, features, targets)
    #     proposals = [proposal.half() for proposal in proposals]
    #     detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    #     detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    #     losses = {}
    #     losses.update(detector_losses)
    #     losses.update(proposal_losses)

    #     if torch.jit.is_scripting():
    #         if not self._has_warned:
    #             warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
    #             self._has_warned = True
    #         return (losses, detections)
    #     else:
    #         return self.eager_outputs(losses, detections)

if __name__ == "__main__":
    model = FasterRCNNResNet50FPN(num_classes=8)
    pass
    
