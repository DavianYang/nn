import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from nn.data.datasets.coco import CocoDetectionDataset

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = './', 
                 annot_dir: str = './', 
                 img_size: int = 224, 
                 batch_size: int = 64,
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1
                 ):
        self.data_dir = data_dir
        self.annot_dir = annot_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
    
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            coco_full = CocoDetectionDataset(img_file=self.data_dir, annot_file=self.annot_dir, img_size=self.img_size)
            length = [int(len(coco_full) * self.train_size), int(len(coco_full) * self.val_size)]
            self.coco_train, self.coco_val = random_split(coco_full, length)
            
    
    def train_dataloader(self):
        return DataLoader(self.coco_train, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    
    def val_dataloader(self):
        return DataLoader(self.coco_val, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)