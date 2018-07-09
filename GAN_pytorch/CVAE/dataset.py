import os
import glob
from PIL import Image

import torch
import torch.utils.data as data_utils

class FashionDataset(data_utils.Dataset):
    """
    This dataset is for fashion image generation.
    Args:
        root (str): path of root dir of dataset
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        if train:
            self.im_paths = glob.glob(os.path.join(self.root, 'train', 'good') + '/*')
        else:
            self.im_paths = glob.glob(os.path.join(self.root, 'eval', 'good') + '/*')
        self.transform = transform
        self.target_transform = target_transform

        self.labels = []
        for im_path in self.im_paths:
            color = im_path.split('/')[-1][0]
            if color == 'b':
                self.labels.append(0)
            elif color == 'g':
                self.labels.append(1)
            elif color == 'k':
                self.labels.append(2)
            elif color == 'p':
                self.labels.append(3)
            elif color == 'w':
                self.labels.append(4)
            elif color == 'y':
                self.labels.append(5)
            else:
                self.labels.append(6)

    def __getitem__(self, idx):
        img = Image.open(self.im_paths[idx])
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)
