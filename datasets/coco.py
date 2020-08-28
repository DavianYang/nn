import sys
sys.path.append('..')
from config import Config

import numpy as np
import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from utils import xywh_to_cxcywh

MISSING_IDS = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
NUM_CLASSES_COCO = 80
NUM_ATTRIB = 4 + 1 + NUM_CLASSES_COCO

class CocoDetectionDataset(CocoDetection):
    def __init__(self,  img_file, annot_file, category='all', transform=None):
        super().__init__(img_file, annot_file)
        if category == 'all':
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category
        self.transform = transform
            
            
    def __getitem__(self, index):
        img, target = super(CocoDetectionDataset, self).__getitem__(index)
        labels = []
        for target in targets:
            bbox = torch.tensor(target['bbox'], dtype=torch.float32)
            category_id = target['category_id']
            if (not self.all_categories) and (category_id, dtype='float32'):
                continue
            one_hot_label = self.coco_category_to_one_hot(category_id, dtype='float32')
            conf = torch.tensor([1.])
            label = torch.cat([bbox, conf, one_hot_label])
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, NUM_ATTRIB))
            
        if self.transform is not None:
            self.transform(img)
            
            
            
    def coco_category_to_one_hot(category_id, dtype="uint"):
        new_id = delete_coco_empty_category(category_id)
        return torch.from_numpy(np.eye(NUM_CLASSES_COCO, dtype=dtype)[category_id])
    
    
    def delete_coco_empty_category(old_id):
        start_idx = 1
        new_id = old_id - start_idx
        for missing_id in MISSING_IDS:
            if old_id > missing_id:
                new_id -= 1
            elif old_id == missing_id:
                raise KeyError("illegal category ID in coco dataset! ID # is {}".format(old_id))
            else:
                break
        return new_id
        