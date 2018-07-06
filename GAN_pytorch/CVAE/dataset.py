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

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.im_paths = glob.glob(os.path.join(self.root) + '/*')
        self.transform = transform
        self.target_transform = target_transform

        self.labels = []
        for im_path in self.im_paths:
            color = im_path.split('/')[-1].split('_')[0]
            if color == 'オレンジ':
                self.labels.append(0)
            elif color == '黄色':
                self.labels.append(1)
            elif color == '灰色':
                self.labels.append(2)
            elif color == '紅' or color == '赤':
                self.labels.append(3)
            elif color == '黒':
                self.labels.append(4)
            elif color == '黒白':
                self.labels.append(5)
            elif color == '紫':
                self.labels.append(6)
            elif color == '水色':
                self.labels.append(7)
            elif color == '青':
                self.labels.append(8)
            elif color == '青緑色':
                self.labels.append(9)
            elif color == '茶色':
                self.labels.append(10)
            elif color == '桃色':
                self.labels.append(11)
            elif color == '白':
                self.labels.append(12)
            elif color == '緑':
                self.labels.append(13)
            elif color == '黄緑':
                self.labels.append(14)
            else:
                print(color)

    def __getitem__(self, idx):
        img = Image.open(self.im_paths[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)
