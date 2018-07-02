import numpy as np
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

def image_path(root, basename, extension):
    path_string = '{basename}{extension}'.format(basename=basename, extension=extension)
    return os.path.join(root, path_string)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class SemanticSegmentation(Dataset):
    def __init__(self, root, co_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root)]
        self.filenames.sort()
        self.co_transform = co_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        if self.co_transform is not None:
            image,label = self.co_transform(image,label)
        return image, label

    def __len__(self):
        return len(self.filenames)
