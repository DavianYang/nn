import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class DenseNetLightning(pl.LightningModule):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizers = self.config["optimizers"]
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optim = getattr(torch.optim, self.optimizers["type"])
        return optim(
            self.parameters(),
            lr=self.optimizers["args"]["lr"],
            weight_decay=self.optimizers["args"]["weight_decay"]
        )
        
    def training_step(self, batch, _):
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
        
    def validation_step(self, batch, _):
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