from data import ImageDetectionDataset
from detection_model_lt import DetectionModelTrainer
import pandas as pd
import torch
import torchvision

device = torch.device('cuda', 2)

def apply_nms(orig_prediction, iou_thresh=0.3, min_score = 0.5):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    keep = torch.tensor([idx for idx in keep if orig_prediction['scores'][idx] > min_score], dtype=torch.int64)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction


def run_prediction(model, img):
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    print('predicted #boxes: ', len(prediction['labels']))
    
    prediction = apply_nms(prediction, min_score=0.5)
        
    print('predicted #boxes after processing: ', len(prediction['labels']))

    from utils.viz import visualize
    visualize(img.permute(1, 2, 0).numpy() * 255.0, bboxes = [list(box[:2]) + [box[2] - box[0]] + [box[3] - box[1]] for box in prediction['boxes'].cpu().numpy()], category_ids = prediction['labels'].cpu().numpy().tolist(), category_id_to_name = [''] * 10, scores = [score.cpu().numpy() for score in prediction['scores']])


if __name__ == "__main__":
    model = DetectionModelTrainer(
        dict_file = "category_ids.txt",
        lr = 1e-3,
        momentum = 0.9,
        weight_decay = 1e-4,        
    ).load_from_checkpoint('checkpoint/zalo-faster-rcnn/lightning_logs/version_15/checkpoints/epoch=4-val_map=0.4510.ckpt')

    zalo_data = ImageDetectionDataset(dataframe=pd.read_csv('za_traffic_2020/traffic_train/annotation.csv'),
                                      image_dir='za_traffic_2020/traffic_train/images',
                                      )
    img, label = zalo_data.__getitem__(4)

    model.cuda(device)
    model.eval()

    run_prediction(model, img)
    pass

