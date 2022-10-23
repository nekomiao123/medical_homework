from xml.etree.ElementPath import prepare_predicate
import torch
import os
import sys
import wandb
import torch.nn as nn
import pandas as pd
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from model import Network

import math
import timm
import dataset
from losses import NCELoss, SupConLoss

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    revolution = 300
    batch_size = 16
    num_workers = 4
    Max_epoch = 100
    use_nce = True
    warm_up_epochs = 10

    wandb.init(
        project='xm_assignment2',
        entity='nekokiku',
        name='resnet101_addtech_conloss',
        save_code=True,
        config={'epochs': Max_epoch, 'batch_size': batch_size, 'revolution': revolution, 'num_workers': num_workers, 'use_nce': use_nce},
    )

    train_transform = transforms.Compose([
        transforms.Resize((revolution, revolution)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3]),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    trainset = dataset.Skin7(train=True, transform=train_transform)
    testset = dataset.Skin7(train=False, transform=test_transform)

    model = Network(backbone="resnet101", num_classes=7, input_channel=3, pretrained=True)
    # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, pin_memory=True,
                                             num_workers=num_workers)

    nce = SupConLoss()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(Max_epoch - warm_up_epochs) * math.pi) + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=1, gamma=np.power(0.001, 1 / Max_epoch))

    for epoch in range(1, Max_epoch+1):
        # train
        model, train_acc, train_loss = train_one_epoch(
            model, trainloader, optimizer, lr_scheduler, criterion, device, Max_epoch, epoch, nce, use_nce=use_nce)
        print("--epoch:{}/{}".format(epoch, Max_epoch), "train loss:{}".format(round(train_loss, 4)), "train acc:{}".format(round(train_acc*100, 4)))
        # test
        test_accuracy, test_loss = test(model, testloader, epoch, device, criterion)
        print("--epoch:{}/{}".format(epoch, Max_epoch), "test loss:{}".format(round(test_loss, 4)), "test acc:{}".format(round(test_accuracy*100, 4)))
        wandb.log(step=epoch,
            data={'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_accuracy})

    print("done")

def train_one_epoch(model, trainloader, optimizer, lr_scheduler, criterion, device, Max_epoch, epoch, nce, use_nce):

    model = model.to(device)
    model.train()

    loss_list = []
    acc_list = []
    for batch_idx, ([image,image_aug], target) in tqdm(enumerate(trainloader)):
        image, target = image.cuda(), target.cuda()
        image_aug= image_aug.cuda()

        if use_nce:
            bsz = target.size(0)
            concat_image = torch.cat([image, image_aug],dim=0)
            predict, feat = model(concat_image)
            f1, f2 = torch.split(feat, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            precit1, precit2 = torch.split(predict, [bsz, bsz], dim=0)
            loss = criterion(precit1, target) + nce(features, target)
        else:
            precit1, feat = model(image)
            loss = criterion(precit1, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        pred = torch.argmax(precit1, dim=1).cpu().numpy()
        acc_list.append(accuracy_score(target.cpu().numpy(), pred))

    lr_scheduler.step()
    wandb.log(step=epoch,
            data={'lr': lr_scheduler.get_last_lr()[0]})

    if epoch % 100 == 0:
        save_dir = './checkpoints/' + wandb.run.name + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), save_dir + 'epoch_{}.pth'.format(epoch))
        print("-- Epoch {} saved model".format(epoch))

    return model, np.mean(acc_list), np.mean(loss_list)

def test(model, testloader, epoch, device, criterion):
    # model.load_state_dict(torch.load('./weights/epoch990.pth'))
    # model = model.to(device)
    model.eval()
    acc = []
    los_list = []
    with torch.no_grad():
        for _, ([data, data_aug], target) in enumerate(testloader):
            target = target.to(device)
            data = data.to(device)
            output, feature = model(data)
            loss = criterion(output, target)
            target = target.cpu().numpy()
            pred = torch.argmax(output, dim=1).cpu().numpy()
            acc_temp = accuracy_score(target, pred)
            acc.append(acc_temp)
            los_list.append(loss.item())
        accuracy = np.mean(acc)
    return accuracy, np.mean(los_list)


if __name__ == '__main__':
    main()
