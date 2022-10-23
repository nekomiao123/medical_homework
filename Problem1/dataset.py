import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from PIL import Image

def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

class LAHeart(Dataset):
    def __init__(self, data_dir="./datas/", split='train', transform=None):
        self.image=[]
        self.label=[]
        self.split = split
        self.address = None
        self.transform = transform
        if split == 'train':
            self.address = data_dir + '/train/'
            for root, _, fnames in sorted(os.walk(self.address)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    img, label = read_h5(path)
                    # print(img.shape(), label.shape())
                    self.image.append(img)
                    self.label.append(label)

        if split == 'test':
            self.address = data_dir + '/test/'
            for root, _, fnames in sorted(os.walk(self.address)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    img, label = read_h5(path)
                    self.image.append(img)
                    self.label.append(label)
    
        print("length of dataset",len(self.image))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]

        sample = {
            'image': image, 
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)  

        return sample

# test the dataset
if __name__ == '__main__':
    train_dataset = LAHeart(split='train')
    test_dataset = LAHeart(split='test')
    print(len(train_dataset))
    print(len(test_dataset))
