import numpy as np
import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from nn.utils import xywh_to_cxcywh
from nn.data.transforms import default_transform

MISSING_IDS = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
NUM_CLASSES_COCO = 80
NUM_ATTRIB = 4 + 1 + NUM_CLASSES_COCO

class CocoDetectionDataset(CocoDetection):
    def __init__(self,  img_file, annot_file, img_size, category='all', transform='default'):
        super().__init__(img_file, annot_file)
        self.img_size = img_size
        if transform == 'default':
            self.tf = default_transform(img_size)
        else:
            raise ValueError("input transform can only be 'default or 'random'")
        if category == 'all':
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category
            
            
    def __getitem__(self, index):
        img, targets = super(CocoDetectionDataset, self).__getitem__(index)
        labels = []
        for target in targets:
            bbox = torch.tensor(target['bbox'], dtype=torch.float32)
            category_id = target['category_id']
            if (not self.all_categories) and (category_id != self.category_id):
                continue
            one_hot_label = self.coco_category_to_one_hot(category_id)
            conf = torch.tensor([1.])
            label = torch.cat([bbox, conf, one_hot_label])
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, NUM_ATTRIB))
        transformed_img_tensor, label_tensor = self.tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)
        return transformed_img_tensor, label_tensor, label_tensor.size(0)
    
            
    def coco_category_to_one_hot(self, category_id, dtype="float32"):
        new_id = self.delete_coco_empty_category(category_id)
        return torch.from_numpy(np.eye(NUM_CLASSES_COCO, dtype=dtype)[category_id])
    
    
    def delete_coco_empty_category(self, old_id):
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
        