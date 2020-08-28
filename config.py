import os
from pathlib import Path
from utils import read_json

class Config:
    def __init__(self, config):
        self.config = Path(config)
        
    def parse_config(self):
        config = read_json(self.config)
        return config
    
    
if __name__ == '__main__':
    config = Config('configs/densenet_cifar10.json').parse_config()
    print(config["arch"]["type"]["densenet121"])