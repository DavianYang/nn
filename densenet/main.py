import sys
sys.path.append('..')
from config import Config
from pytorch_lightning.core.memory import ModelSummary
from .densenet import DenseNet
from .densenetlightning import DenseNetLightning

def main():
    config = Config('configs/densenet_cifar10.json').parse_config()
    num_classes = config["arch"]["num_classes"]
    arch = config["arch"]["type"]["densenet121"]
    model = DenseNet(growthrate=arch["growthrate"], 
                     block_config=arch["block_config"], 
                     num_init_features=arch["num_init_features"], 
                     num_classes=num_classes)
    lighting = DenseNetLightning(model, config["optimizers"])

if __name__ == "__main__":
    main()