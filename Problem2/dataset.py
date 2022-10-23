from cProfile import label
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

class Skin7(Dataset):
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.name = []
        self.label = []
        self.img = []

        if self.train:
            temp = np.array(pd.read_csv('./data/annotation/train.csv'))
        else:
            temp = np.array(pd.read_csv('./data/annotation/test.csv'))

        for i in range(temp.shape[0]):
            self.name.append(temp[i][0])
            self.label.append(temp[i][1])
        self.name = np.array(self.name)
        self.label = np.array(self.label)
        for ind in range(len(self.name)):
            image = cv2.imread('./data/images/'+self.name[ind])
            self.img.append(image)

        print("finish loading data")

    def __getitem__(self, index):
        image = Image.fromarray(self.img[index])
        label = np.array(self.label[index])

        if self.transform:
            img1 = self.transform(image)
            img2 = self.transform(image)

        return [img1, img2], label

    def __len__(self):
        return len(self.label)

# test the dataset
if __name__ == '__main__':
    train_data = Skin7(train=True)
    test_data = Skin7(train=False)
    print(len(train_data))
    print(len(test_data))




