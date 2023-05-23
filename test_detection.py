from data import ImageDetectionDataset
from detection_model_lt import DetectionModelTrainer
import pandas as pd
import torch
import torchvision

device = torch.device('cuda', 2)

category_id2name = {line.strip().split('\t')[0]:line.strip().split('\t')[1] 
                   for line in open('category_ids.txt').readlines()}

def apply_nms(orig_prediction, iou_thresh=0.3, min_score = 0.5):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    keep = torch.tensor([idx for idx in keep if orig_prediction['scores'][idx] > min_score], dtype=torch.int64)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

class BoundingBoxAnnotation(object):
    def __init__(self, bbox, category, score) -> None:
        self.bbox = {'xmin' : bbox[0], 'ymin' : bbox[1], 'xmax' : bbox[2], 'ymax':bbox[3]}
        self.category =  category
        self.score = score
from json import JSONEncoder
class CustomEncoder(JSONEncoder):
    def default(self, o):
            return o.__dict__

def run_prediction(model, img, index):
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    print('predicted #boxes: ', len(prediction['labels']))
    
    prediction = apply_nms(prediction, min_score=0.5)
        
    print('predicted #boxes after processing: ', len(prediction['labels']))

    bboxes = [list(box[:2]) + [box[2] - box[0]] + [box[3] - box[1]] for box in prediction['boxes'].cpu().numpy()]

    annotations = []
    for bbox, label, score in zip (prediction['boxes'].cpu().numpy().astype(int).tolist(), 
                                   [category_id2name[str(id)] for id in prediction['labels'].cpu().numpy()], 
                                   prediction['scores'].cpu().numpy().tolist()):
        annotations.append(BoundingBoxAnnotation(bbox, label, score))

    return annotations

    # from utils.viz import visualize
    # visualize(img.permute(1, 2, 0).numpy() * 255.0, bboxes = [list(box[:2]) + [box[2] - box[0]] + [box[3] - box[1]] for box in prediction['boxes'].cpu().numpy()], 
    #           category_ids = prediction['labels'].cpu().numpy().tolist(), 
    #           category_id_to_name = [''] * 10, scores = [score.cpu().numpy() for score in prediction['scores']], index=index)


if __name__ == "__main__":
    model = DetectionModelTrainer(
        dict_file = "category_ids.txt",
        lr = 1e-3,
        momentum = 0.9,
        weight_decay = 1e-4,        
    ).load_from_checkpoint('checkpoint/zalo-faster-rcnn/lightning_logs/version_41/checkpoints/epoch=29-val_map=0.4630.ckpt')

    zalo_data = ImageDetectionDataset(dataframe=pd.read_csv('za_traffic_2020/traffic_train/annotation.csv'),
                                      image_dir='za_traffic_2020/traffic_train/images',
                                      )
    model.cuda(device)
    model.eval()
    import cv2
    import numpy as np
    input_dir = 'input_videos'

    cap = cv2.VideoCapture('input_videos/8269400377657974270.mp4')
 
    index = 0
    # Loop until the end of the video
    annotations_list = []
    while (cap.isOpened()):
        index +=  1
    
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None: break

        if index %10 != 1: 
            continue

        image = frame.astype(np.float32)
        image /= 255.0

        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

        annotations_list.append( {'id' : index, 'objects': run_prediction(model, image, index)}) 

    import json
    with open('annotations.json', 'w') as f:
        f.write(json.dumps([annotations for annotations in annotations_list], cls=CustomEncoder))

    # for index, fn in enumerate(os.listdir(input_dir)):
    #     image = cv2.imread(f'{input_dir}/{fn}', cv2.IMREAD_COLOR).astype(np.float32)

    #     # normalization.
    #     image /= 255.0

    #     image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
    #     run_prediction(model, image, index)

