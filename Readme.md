# Traffic sign detection using Faster R-CNN

An implementation with [pytorch-lightning](https://lightning.ai/).

## Detection model

Check out [faster_rcnn.py](faster_rcnn.py)

- Faster R-CNN with ResNext50-FPN backbone
- Anchor sizes are set to (16x16, 32x32, 64x64, 128x128)

## Dataloader

Check out [data.py](data.py)

- Load ZaloAI data
- Skip small traffic sign

