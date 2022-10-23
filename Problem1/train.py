import os
import torch
from torch.utils.data import DataLoader
import torchvision
from transforms import RandomCrop, RandomRotFlip, ToTensor
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm
import wandb
from dataset import LAHeart
from model import *
from other import UNet
from lossdice import dice_loss

if __name__ == '__main__':
    max_epoch = 5000
    batch_size = 8
    device = "cuda:0"
    patch_size = (112, 112, 80)

    wandb.init(
        project='xm_assignment1',
        entity='nekokiku',
        name='VNet_Dice',
        save_code=True,
        config={'epochs': max_epoch, 'batch_size': batch_size, 'patch_size': patch_size},
    )

    model = Model()
    train_dst = LAHeart(
        split='train', 
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )

    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=1, 
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=np.power(0.001, 1 / max_epoch))
    BCEloss = torch.nn.BCELoss()

    model.to(device)
    for epoch in range(max_epoch):
        model.train()
        loss_list = []
        dice_loss_list = []
        
        for sample in tqdm(train_loader):
            data = sample['image'].to(device)
            label = sample['label'].to(device)
            output = model(data)
            out_scores = F.softmax(output, dim=1)

            loss_ce = F.cross_entropy(output, label)
            loss_dice = dice_loss(out_scores[:, 1, ...], label == 1)
            # loss = (loss_ce + loss_dice) / 2
            loss = loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            dice_loss_list.append(loss_dice.item())

        lr_scheduler.step()

        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch), "loss:{}".format(round(np.mean(loss_list),2)),"dice loss:{}".format(round(np.mean(dice_loss_list),2)))

        wandb.log(step=epoch + 1,
            data={'epoch': epoch + 1,
                'loss': np.mean(loss_list),
                'dice loss': np.mean(dice_loss_list),
                'lr': lr_scheduler.get_last_lr()[0]
                })

        if epoch % 100 == 99:
            save_dir = './checkpoints/' + wandb.run.name + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), save_dir + 'epoch_{}.pth'.format(epoch + 1))
            print("-- Epoch {}/{} saved model".format(epoch+1,max_epoch))

    print("done")
