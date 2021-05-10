#import os
from PIL import Image
from torch.utils import data
#import numpy as np
#from torchvision import transforms as T

class Retina(data.Dataset):
    'define a dataset for Diabetic Retina'
    def __init__(self, pdframe, transforms=None):
        'Initialization'
        self.img_paths = pdframe.path
        self.transforms = transforms
        self.labels = pdframe.labels

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.img_paths)

    def __getitem__(self, index):
        "Generate one sample of data"

        img_path = self.img_paths[index]
        data = Image.open(img_path)
        label = self.labels[index]
        if self.transforms:
            data = self.transforms(data)
        return data, label