
from typing import Optional
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import pytorch_lightning as pl
from albumentations import Compose, RandomCrop, BboxParams
import cv2
import numpy as np
from utils.viz import *
from tqdm import tqdm
import utils.utils as utils

'''
Faster RCNN
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format (pascal voc)

'''

class ImageDetectionDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame = None,
                 mode : str = 'train',
                 image_dir: str = '',
                 transforms: Compose = None):
        """
        Prepare data for image detection.

        Args:
            dataframe: dataframe with image id and bboxes
            image_dir: path to images
            transforms: albumentations
        """
        self.image_dir = image_dir
        self.df = dataframe
        self.image_ids = self.df['image_id'].unique()
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.image_dir}/{image_id}.png', cv2.IMREAD_COLOR).astype(np.float32)

        # normalization.
        image /= 255.0
        target = {}

        MIN_SIZE = 10

        # for train and valid 
        if self.mode != 'test':
            image_data = self.df.loc[self.df['image_id'] == image_id]
            boxes = image_data['bbox'].values
            boxes = [np.fromstring(box.strip('[]'), dtype = int, sep = ', ') for box in boxes]

            labels = image_data['category_id'].values

            # filter small boxes
            selected_boxes = [id for id, box in enumerate(boxes) if (box[2] >= MIN_SIZE or box[3] >= MIN_SIZE)
                              and box[2] + box[0] < image.shape[1] and box[3] + box[1] < image.shape[0]]

            boxes = [boxes[id] for id in selected_boxes]
            # convert [x, y, w, h] to [x1, y1, x2, y2]
            boxes = [(x, y, x+w, y+h) for x, y, w, h in boxes]

            labels = [labels[id] for id in selected_boxes]

            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['image_id'] = torch.tensor([idx])

            if self.transforms:
                image_dict = {
                    'image': image,
                    'bboxes': [list(box) + [label] for box, label in zip(boxes, labels)],
                    'labels': labels
                }
                image_dict = self.transforms(**image_dict)
                image = image_dict['image']

                boxes = [bbox[:4] for bbox in image_dict['bboxes']]
                labels = [bbox[4] for bbox in image_dict['bboxes']]

                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

            if len(boxes) > 0:
                boxes = np.array(boxes)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
                target['area'] = area
                target['iscrowd'] = iscrowd

        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)

class TSD_Data(pl.LightningDataModule):
    def __init__(self,         
        batch_size: int = 10,
        workers: int = 5,
        train_data: str = "za_traffic_2020/traffic_train/images",
        val_data: str = "za_traffic_2020/traffic_train/images",
        test_data: str = "za_traffic_2020/traffic_public_test/images",
        dataframe: str = 'za_traffic_2020/traffic_train/annotation.csv',
        img_size: int = 512,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.img_size = img_size
        self.dataframe = pd.read_csv(dataframe)
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            transforms = Compose([
                            RandomCrop(self.img_size, self.img_size, p=1.0),
                            ], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=None))

            self.train_dataset = ImageDetectionDataset(dataframe=self.dataframe, image_dir=self.train_data, transforms=transforms)
            self.val_dataset = ImageDetectionDataset(dataframe=self.dataframe, image_dir=self.val_data)
        if stage == "test" or stage is None:
            self.test_dataset = ImageDetectionDataset(dataframe=self.dataframe, image_dir=self.val_data)

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, collate_fn=utils.collate_fn,
            persistent_workers=True, 
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader( 
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, collate_fn=utils.collate_fn,
            persistent_workers=True, 
        )
        return val_loader
    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    # transforms = Compose([
    #                 RandomCrop(512, 512, p=1.0),
    #                 ], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=[]))

    # zalo_data = ImageDetectionDataset(dataframe=pd.read_csv('za_traffic_2020/traffic_train/annotation.csv'),
    #                                   image_dir='za_traffic_2020/traffic_train/images',
    #                                   transforms=transforms,
    #                                   )
    # # zalo_data.__getitem__(0)
    # for image, label in tqdm(zalo_data):
    #     pass
    # pass

    dm = TSD_Data()
    dm.setup()
    trainloader = dm.train_dataloader()
    for img, label in trainloader:
        print()
        pass 