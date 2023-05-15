import os
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed


import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from faster_rcnn import FasterRCNNResNet50FPN as Model
from torchmetrics.detection.map import MeanAveragePrecision
import torchvision

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

class DetectionModelTrainer(LightningModule):
    """
    """

    def __init__(
        self,
        dict_file: str = "category_ids.txt",

        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.id2label = {int(line.strip().split('\t')[0]): line.strip().split('\t')[1] 
                              for line in open(dict_file).readlines() }

        # metrics
        self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

        self.model = Model(num_classes=8)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        self.log(
            "train_loss_classifier",
            loss_dict["loss_classifier"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_loss_box_reg",
            loss_dict["loss_box_reg"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "train_loss_objectness",
            loss_dict["loss_objectness"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "train_loss_rpn_box_reg",
            loss_dict["loss_rpn_box_reg"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        
        # total loss
        losses = sum(loss for loss in loss_dict.values())

        self.log('train_loss', losses, prog_bar=True, on_step=True, on_epoch=True)

        # debug: skip calculating loss
        # losses = torch.tensor(0.)
        # losses.requires_grad = True

        return losses
        

    def eval_step(self, batch, batch_idx, prefix: str):

        import random
        if random.random() < 0.10:
            images, targets = batch
            preds = self.model(images)
            self.map.update(preds, targets)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        mAPs = {"val_" + k: v for k, v in self.map.compute().items()}
        self.print(mAPs)
        mAPs_per_class = mAPs.pop("val_map_per_class")
        mARs_per_class = mAPs.pop("val_mar_100_per_class")
        self.log_dict(mAPs, sync_dist=True)

        self.log_dict(
            {
                f"val_map_{label}": value
                for label, value in zip(self.id2label.values(), mAPs_per_class)
            },
            sync_dist=True,
        )
        self.log_dict(
            {
                f"val_mar_100_{label}": value
                for label, value in zip(self.id2label.values(), mARs_per_class)
            },
            sync_dist=True,
        )


        self.map.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
        return [optimizer], [scheduler]

def main() -> None:
    pl.seed_everything()

    model = DetectionModelTrainer()
 
    # if os.path.isfile(args.resume):
    #     print('=> loading checkpoint: {}'.format(args.resume))
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.model.load_state_dict(checkpoint['state_dict'])
    #     print('=> loaded checkpoint: {}'.format(args.resume))
    pass



if __name__ == "__main__":
    main()