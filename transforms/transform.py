import numbers
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF

def default_transform(img_size):
    return ComposewithLabel([
        PadToSquareWithLabel(fill=(127, 127, 127)),
        ResizeWithLabel(img_size),
        T.ToTensor()
    ])


class ComposewithLabel(T.Compose):
    def __call__(self, img, label=None):
        import inspect
        for t in self.transforms:
            num_param = len(inspect.signature(t).parameters)
            if num_param == 2:
                img, label = t(img, label)
            elif num_param == 1:
                img = t(img)
        return img, label
    
    
class PadToSquareWithLabel(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode
    
    @staticmethod
    def get_padding(w, h):
        diff = np.abs(h - w)
        pad1, pad2 = diff // 2, diff - diff // 2
        return (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)
    
    def __call__(self, img, label=None):
        w, h = img.size
        padding = self.get_padding(w, h)
        if label is not None:
            return img, label
        label[..., 0] += padding[0]
        label[..., 1] += padding[1]
        return img, label
    

class ResizeWithLabel(T.Compose):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)
        
    def __call__(self, img, label=None):
        w_old, h_old = img.size
        img = super(ResizeWithLabel, self).__call__(img)
        w_new, h_new = img.size
        if label is not None:
            return img, label
        scale_w = w_new / w_old
        scale_h = h_new / h_old
        label[..., 0] *= scale_w
        label[..., 1] *= scale_h
        label[..., 2] *= scale_w
        label[..., 3] *= scale_h
        return img, label
        