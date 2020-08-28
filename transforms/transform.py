import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF

def default_transform(img, label_tensor):
    return ComposewithLabel([])


def ComposewithLabel(T.Compose):
    def __call__(self, img, label=None):
        import inspect
        for t in self.transforms:
            num_param = len(inspect.signature(t).parameters)
            if num_param == 2:
                img, label = t(img, label)
            elif num_param == 1:
                img = t(img)
        return img, label
    
    