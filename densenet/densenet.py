import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import pytorch_lightning as pl

class ConvBlock(nn.Module):
    """
        A building block of Dense Block
        
        Arguments:
            input_channels: Tensor,
            growth_rate: Float,
            dropout_rate: Float
        
        Return:
            Output tensor for the block

    """
    def __init__(
        self, 
        input_channels, 
        growthrate, 
        dropout_rate):
        super().__init__()
        self.dropout = float(dropout_rate)
        interchannels = 4 * growthrate
        # 1x1 Convolution
        self.conv1x1 = nn.ModuleDict({
            'bn1': nn.BatchNorm2d(input_channels),
            'act1': nn.ReLU(inplace=True),
            'conv1': nn.Conv2d(input_channels, interchannels, kernel_size=1, stride=1, bias=False)  
        })
        
        # 3x3 Convolution
        self.conv3x3 = nn.ModuleDict({
            'bn2': nn.BatchNorm2d(interchannels),
            'relu2': nn.ReLU(inplace=True),
            'conv2': nn.Conv2d(interchannels, growthrate, kernel_size=3, stride=1, padding=1, bias=False)
        })
        
        
    def forward(self, x):
        x = self.conv1x1(x)
        
        if self.dropout:
            x = nn.Dropout(p=self.dropout_rate)
        
        x = self.conv3x3(x)
        
        if self.dropout:
            x = nn.Dropout(p=self.dropout_rate)
            
        return x