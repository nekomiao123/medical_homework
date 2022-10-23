from cProfile import label
from cv2 import split
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import cv2

class VideoDataSet(Dataset):
    def __init__(self, frame_number=1, split='train', transform=None):
        self.fram_number = frame_number
        self.split = split
        self.path = None
        self.transform = transform
        self.name = []
        self.label = []
        self.image = []
        if self.split == "train":
            for i in range(1,6):
                self.path = './datas/annotation/video_'+ str(i) +'.csv'
                data = np.array(pd.read_csv(self.path))
                for ind in range(data.shape[0]):
                    self.label.append(data[ind][1])
                    path = './datas/' + str(i) + '/' + str(data[ind][0])
                    self.name.append(data[ind][0])
                    image = cv2.imread(path)
                    self.image.append(image)

        if self.split == 'test':
            self.path = './datas/annotation/video_'+ str(41) +'.csv'
            data = np.array(pd.read_csv(self.path))
            for ind in range(data.shape[0]):
                self.label.append(data[ind][1])
                path = './datas/' + str(41) + '/' + str(data[ind][0])
                self.name.append(data[ind][0])
                image = cv2.imread(path)
                self.image.append(image)


    def __getitem__(self, index):
        if self.fram_number == 1:
            label = self.label[index]
            image = Image.fromarray(self.image[index])
            image = self.transform(image)
        elif self.fram_number != 1:
            image = torch.zeros(size=(3,3,224,224))
            label = []
            if index == 0 or index == 1:
                for i in range(3):
                    image[i] = self.transform(Image.fromarray(self.image[index]))
                    label.append(self.label[index])
            else:
                for i in range(3):
                    image[i] = self.transform(Image.fromarray(self.image[index-2+i]))
                    label.append(self.label[index-2+i])
                    # transformlocal = transforms.Totensor()
            # image = torch.FloatTensor(image)
            label = torch.Tensor(label)
            label = label.view(-1)

        # print(image.shape)
        # print(label.shape)
        return image, label

    def __len__(self):
        return len(self.image)

# test the dataset
if __name__ == '__main__':
    Dataset = VideoDataSet(split='test')
    print(len(Dataset))

# print(len(VideoDataLoader()))
# test = VideoDataLoader(frame_number=1, split='test', transform)
# load_data = VideoDataLoader(split = 'test')
# # print(len(load_data))
# for i in range(len(load_data)):
#     load_data[i]
# for i in range(len())

# test = Image.open('./datas/5/0.jpg')
# test = np.array(test)
# lstm=nn.LSTM(250,20,1,bidirectional=False)
# batch1=torch.randn(50,250,250)
# outputs,(h,c)=lstm(batch1)
# shp(batch1)
# shp(outputs) # word_len*batch_size*hidden_size
# shp(h)

# dd = VideoDataLoader()
# print(len(dd))
# revolution = 224
# imagenet_mean = [0.485, 0.456, 0.406]
# imagenet_std = [0.229, 0.224, 0.225]
# test_transform = transforms.Compose([
#         transforms.RandomResizedCrop(160, scale=(0.2, 1.)),
#         transforms.Resize((revolution,revolution)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
#         ])

# frame = VideoDataLoader(frame_number=3, transform=test_transform)
# final = frame[1][1]
# print(final)