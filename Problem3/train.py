import random
import os
import math
import shutil
import time
from unittest import TestLoader
import warnings
from PIL import ImageFilter
import pandas as pd
# from thop import profile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import wandb
import timm

from tqdm import tqdm
import torch.nn.init as init

from dataloader import VideoDataSet
from utils import GaussianBlur
from modellstm import resnet_lstm

from torchvision.models import resnet50, ResNet50_Weights

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
sequence_length = 3
revolution = 224
learning_rate = 3e-4
warm_up_epochs = 10
use_lstm = True

wandb.init(
    project='xm_assignment3',
    entity='nekokiku',
    name='Res50_addtech_lstm',
    save_code=True,
    config={'epochs': epochs, 'learning_rate': learning_rate, 'sequence_length': sequence_length, 'revolution': revolution, 'use_lstm': use_lstm},
)

train_transform = transforms.Compose([
                transforms.RandomResizedCrop(revolution, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([GaussianBlur(sigma1=0.1, sigma2=2.0), ], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
test_transform = transforms.Compose([
        transforms.RandomResizedCrop(250, scale=(0.2, 1.)),
        transforms.Resize((revolution,revolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

def train(model, train_loader, learning_rate, test_dataloader): 
    model.to(device)
    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc = []
    optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay=1e-4)
    loss_CE = torch.nn.CrossEntropyLoss().to(device)

    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * math.pi) + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 3570
        if use_lstm:
            total = total*3
        else:
            total = total
        loss_item = []
        for _,data in tqdm(enumerate(train_loader)):
            image, label = data
            
            image = image.to(device)
            if use_lstm:
                label = label.to(torch.int64).to(device)
                label = label.view(-1)
            else:
                label = label.to(device)

            output = model(image)
            loss = loss_CE(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item.append(loss.item())
            pred = np.array(torch.argmax(output, dim=1).cpu().numpy())
            label = np.array(label.cpu().numpy())
            correct += np.sum(pred == label.data)

        lr_scheduler.step()
        save_dir = './checkpoints/' + wandb.run.name + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), save_dir + 'epoch_{}.pth'.format(epoch))
        print("--epoch:{}/{}".format(epoch,epochs),"loss:{}".format(round(np.mean(loss_item),3)),"acc:{}".format(round(correct/total,3)))
        total_train_acc.append(correct/total)
        total_train_loss.append(np.mean(loss_item))
        testacc, testloss = test(model, test_dataloader)
        total_test_acc.append(testacc)
        total_test_loss.append(testloss)
        wandb.log({'train_acc': correct/total, 'train_loss': np.mean(loss_item), 'test_acc': testacc, 'test_loss': testloss, 'lr': optimizer.param_groups[0]['lr']})

    print(total_train_loss,total_train_acc,total_test_acc,total_test_loss)

def test(model,test_loader):
    model.eval()
    correct = 0
    total = 776
    if use_lstm:
        total = total*3
    else:
        total = total
    loss_item = []
    loss_CE = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for _,data in tqdm(enumerate(test_loader)):
            image, label = data
            image = image.to(device)
            
            if use_lstm:
                label = label.to(torch.int64).to(device)
                label = label.view(-1)
            else:
                label = label.to(device)

            output = model(image)
            loss = loss_CE(output,label)
            pred = torch.argmax(output, dim=1).cpu().numpy()
            correct += np.sum(pred == label.cpu().numpy())
            # print(pred, label.cpu().numpy())
            loss_item.append(loss.item())
    print('Test: Acc {}, Loss {}'.format(round(correct/total,3), round(np.mean(loss_item),3)))
    return (correct/total), np.mean(loss_item)

if __name__ == '__main__':
    if use_lstm:
        model = resnet_lstm()
        frame_number = 3
        print('use lstm')
    else:
        model = timm.create_model('resnet50', pretrained=True, num_classes=7)
        frame_number = 1

    traindataset = VideoDataSet(frame_number=frame_number, split='train', transform = train_transform)
    testdataset = VideoDataSet(frame_number=frame_number, split='test', transform = test_transform)

    train_dataloader = DataLoader(traindataset, batch_size=16 , shuffle=True, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(testdataset, batch_size=16, shuffle=False, drop_last=False, num_workers=4)
    
    train(model, train_dataloader,learning_rate, test_dataloader)

