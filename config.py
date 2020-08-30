import os
from pathlib import Path
from utils import read_json

class Config:
    def __init__(self, config):
        self.config = Path(config)
        
    def parse_config(self):
        config = read_json(self.config)
        return config