import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import pytorch_lightning as pl

class ConvBlock(nn.Module):
    """
        A building block of Conv Block
        
        Arguments:
            input_channels: Tensor,
            growth_rate: Float,
            dropout_rate: Float
        
        Return:
            Output tensor for the block

    """
    def __init__(
        self, 
        in_channels, 
        growthrate, 
        dropout_rate):
        super().__init__()
        self.dropout = float(dropout_rate)
        interchannels = 4 * growthrate
        # 1x1 Convolution
        self.conv1x1 = nn.ModuleDict({
            'bn': nn.BatchNorm2d(in_channels),
            'conv': nn.Conv2d(in_channels, interchannels, kernel_size=1, stride=1, bias=False)  
        })
        
        # 3x3 Convolution
        self.conv3x3 = nn.ModuleDict({
            'bn': nn.BatchNorm2d(interchannels),
            'conv': nn.Conv2d(interchannels, growthrate, kernel_size=3, stride=1, padding=1, bias=False)
        })
        
        
    def forward(self, x):
        out = self.conv1x1['conv'](F.relu(self.conv1x1['bn'](x)))
        out = self.conv3x3['conv'](F.relu(self.conv3x3['bn'](out)))
        out = torch.cat((x, out), 1)
        return out
        