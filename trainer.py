from data import TSD_Data
from detection_model_lt import DetectionModelTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    model = DetectionModelTrainer(
        dict_file = "category_ids.txt",
        lr = 1e-4,
        momentum = 0.9,
        weight_decay = 1e-4,        
    ).load_from_checkpoint('checkpoint/zalo-faster-rcnn/lightning_logs/version_40/checkpoints/epoch=0-val_map=0.0000.ckpt')
    # import torch
    # from collections import OrderedDict
    # checkpoint = torch.load('checkpoint/zalo-faster-rcnn/lightning_logs/version_26/checkpoints/epoch=30-val_map=0.3509.ckpt', map_location='cpu')
    # model.model.backbone.load_state_dict(OrderedDict({k.replace('model.backbone.', ''):v for k, v in checkpoint['state_dict'].items() 
    #                                                   if 'model.backbone.' in k}))

    dm = TSD_Data(batch_size=30, workers=15)


    trainer = Trainer(
        checkpoint_callback=True,
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename='{epoch}-{val_map:.4f}', save_top_k=5, monitor='val_map', mode='max'),
        ], 
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        default_root_dir='checkpoint/zalo-faster-rcnn',

        deterministic=False, 
        max_epochs=50, 
        log_every_n_steps=50,

        gpus = [2],
        amp_backend='apex', 
        amp_level='O1', 
        # precision=16,
        # strategy='ddp',        
    )

    if False:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)