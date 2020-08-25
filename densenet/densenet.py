import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import pytorch_lightning as pl

import math

class ConvBlock(nn.Module):
    """
        A building block of Conv Block
                
        Return:
            Output tensor for the block

    """
    def __init__(
        self, 
        in_channels, 
        growthrate,
        bn_size, 
        drop_rate
        ):
        super(ConvBlock, self).__init__()
        # 1x1 Convolution
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('act1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growthrate, kernel_size=1, stride=1, bias=False) )
        
        # 3x3 Convolution
        self.add_module('bn2', nn.BatchNorm2d(bn_size * growthrate))
        self.add_module('act2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growthrate, growthrate, kernel_size=3, stride=1, padding=1, bias=False))
    
        self.drop_out = float(drop_rate)
        
        
    def forward(self, inputs):
        out = torch.cat(inputs, 1)
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        if self.drop_out > 0:
            out = F.dropout(out, p=self.drop_out)
        return out