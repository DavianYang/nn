import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.metrics.functional import accuracy

import math
from collections import OrderedDict

class ConvBlock(nn.Module):
    """
        A building block of Conv Block
                
        Return:
            Output tensor for the block

    """
    def __init__(
        self, 
        in_features: int, 
        growthrate: float,
        bn_size: int, 
        drop_rate: float
        ):
        super(ConvBlock, self).__init__()
        # 1x1 Convolution
        self.add_module('bn1', nn.BatchNorm2d(in_features))
        self.add_module('act1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_features, bn_size * growthrate, kernel_size=1, stride=1, bias=False) )
        
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
    

class TransitionBlock(nn.Module):
    """
        A building block of Transition Block
        
        Return:
            Output tensor for the block

    """
    def __init__(
        self,
        in_features: int,
        out_features: int):
        super().__init__()
        
        self.add_module('bn', nn.BatchNorm2d(in_features))
        self.add_module('act', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        

class DenseBlock(nn.ModuleDict):
    """
        A building block of Transition Block
        
        Return:
            Output tensor for the block
    """
    def __init__(
        self, 
        num_layers: int, 
        in_features: int, 
        bn_size: int, 
        growthrate: float, 
        drop_rate: float):
        super().__init__()
        for i in range(num_layers):
            layer = ConvBlock(
                in_features + i * growthrate,
                growthrate=growthrate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'convblock{i+1}', layer)
            
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    

class DenseNet(nn.Module):
    def __init__(
        self, 
        growthrate=32, 
        block_config=(6, 12, 24, 16), 
        num_init_features=64, bn_size=4, 
        drop_rate=0, 
        num_classes=1000):
        super().__init__()
        
        # 1st Conv
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(num_init_features)),
            ('act0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(
                num_layers=num_layers,
                in_features=num_features,
                bn_size=bn_size,
                growthrate=growthrate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'convblock{i+1}', dense_block)
            num_features = num_features + num_layers * growthrate
            
            if i != len(block_config) - 1:
                transition = TransitionBlock(in_features=num_features, out_features=num_features // 2)
                self.features.add_module(f'transition{i+1}', transition)
                num_features = num_features // 2
        
        # Final BatchNorm
        self.features.add_module('bn5', nn.BatchNorm2d(num_features))
        
        # Linear Layer
        self.classifer = nn.Linear(num_features, num_classes)
        
        self.weight_init()

                
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifer(out)
        return out

class DenseNetLightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module
    ):
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return optim(
            self.parameters(),
            lr=1e-3,
            weight_decay=0
        )
        
    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        acc = accuracy(F.log_softmax(pred, dim=1), target)
        
        return {
            "loss": loss,
            "acc": acc
        }
        
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        
        return {
            "train_epoch_loss": avg_loss,
            "train_epoch_acc": avg_acc    
        }
        
    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        acc = accuracy(F.log_softmax(pred, dim=1), target)
        
        return {
            "val_loss": loss,
            "val_acc": acc
        }
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        
        return {
            "val_epoch_loss": avg_loss,
            "val_epoch_acc": avg_acc
        }
        
    def test_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        acc = accuracy(F.log_softmax(pred, dim=1), target)
        
        return {
            "test_loss": loss,
            "test_acc": acc
        }
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        
        return {
            "test_epoch_loss": avg_loss,
            "test_epoch_acc": avg_acc
        }

if __name__ == "__main__":
    model = DenseNet()
    print(model)