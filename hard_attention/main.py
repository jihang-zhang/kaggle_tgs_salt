import numpy as np
import time
import random
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm, tqdm_notebook, tnrange

from metric040 import iou_pytorch
import lovasz_losses
from loaders import make_loader

import torchvision.transforms as transforms
import cv2
import albumentations as aug

from OCNet.attention_model040 import AttentionHyperResUNet
from scheduler import CosineAnnealingLR_with_Restart

import neptune
ctx = neptune.Context()

pixel0_handle = ctx.create_channel("pixel loss < 0.15", neptune.ChannelType.NUMERIC)
pixel1_handle = ctx.create_channel("pixel loss < 0.25", neptune.ChannelType.NUMERIC)
pixel2_handle = ctx.create_channel("pixel loss > 0.25", neptune.ChannelType.NUMERIC)
image_handle  = ctx.create_channel("image loss"       , neptune.ChannelType.NUMERIC)

image_channel_handle = ctx.create_channel("dev image loss", neptune.ChannelType.NUMERIC)
iou_channel_handle = ctx.create_channel("dev iou", neptune.ChannelType.NUMERIC)

# Configurations
SEED = 5327
LOG_INTERVAL = 20
MAX_EPOCHS = 300
BATCH_SIZE = 32
SAVE_PATH = "../output"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the random seed manually for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)

# Load data split
import json
with open('data_split.json') as f:
    cv = json.load(f)
f.close()

train_ids = cv["train"]
train_size = len(train_ids)

val_ids = cv["val"]
val_size = len(val_ids)

# Data augmentation
standardize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def aug_train(p=1):
    return aug.Compose([
        aug.HorizontalFlip(p=0.5),
        aug.OneOf([
            aug.Compose([
                aug.ShiftScaleRotate(rotate_limit=0, p=1),
                aug.RandomSizedCrop((88, 128), 128, 128)
            ]),
            aug.GridDistortion(num_steps=10, distort_limit=np.random.uniform(0, 0.1), p=1),
            aug.ShiftScaleRotate(scale_limit=0, rotate_limit=10, p=1)
        ], p=0.5),
        aug.OneOf([
            aug.RandomBrightness(limit=0.08, p=1),
            aug.RandomContrast(limit=0.08, p=1),
            aug.RandomGamma(gamma_limit=(92, 108), p=1)
        ], p=0.5)
    ], p=p)

upsize = aug.Resize(128, 128, interpolation=cv2.INTER_LINEAR)
downsize = aug.Resize(101, 101, interpolation=cv2.INTER_LINEAR)
augmentor = aug.Compose([upsize, aug_train()])

class SaltTrainSet(Dataset):
    """Salt segmentation dataset."""

    def __init__(self, train_size, train_ids):
        self.train_size = train_size
        self.train_ids = train_ids

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        image = cv2.imread("../input/train/images/{}.png".format(self.train_ids[idx]), 1)
        mask = cv2.imread("../input/train/masks/{}.png".format(self.train_ids[idx]), 0)
        data_dict = {"image": image, "mask": mask}

        augmented = augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        target_onehot = torch.zeros(4, dtype=torch.float)
        pixels = (mask / 255).sum()
        if pixels == 0:
            target = torch.tensor(0).to(DEVICE)
            target_onehot[0] = 1.0
        elif pixels < 2458:
            target = torch.tensor(1).to(DEVICE)
            target_onehot[1] = 1.0
        elif pixels < 4096:
            target = torch.tensor(2).to(DEVICE)
            target_onehot[2] = 1.0
        else:
            target = torch.tensor(3).to(DEVICE)
            target_onehot[3] = 1.0
        target_onehot = target_onehot.to(DEVICE)

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = transforms.functional.to_tensor(np.expand_dims(mask, -1)).to(DEVICE)
        return image, mask, target.long(), target_onehot

class SaltValSet(Dataset):
    """Salt segmentation dataset."""

    def __init__(self, val_size, val_ids):
        self.val_size = val_size
        self.val_ids = val_ids

    def __len__(self):
        return self.val_size

    def __getitem__(self, idx):
        image = cv2.imread("../input/train/images/{}.png".format(self.val_ids[idx]), 1)
        mask = cv2.imread("../input/train/masks/{}.png".format(self.val_ids[idx]), 0)

        target_onehot = torch.zeros(4, dtype=torch.float)
        pixels = (mask / 255).sum()
        if pixels == 0:
            target = torch.tensor(0).to(DEVICE)
            target_onehot[0] = 1.0
        elif pixels < 1530:
            target = torch.tensor(1).to(DEVICE)
            target_onehot[1] = 1.0
        elif pixels < 2550:
            target = torch.tensor(2).to(DEVICE)
            target_onehot[2] = 1.0
        else:
            target = torch.tensor(3).to(DEVICE)
            target_onehot[3] = 1.0
        target_onehot = target_onehot.to(DEVICE)

        data_dict = {"image": image}
        upsized = upsize(**data_dict)
        image = upsized["image"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = transforms.functional.to_tensor(np.expand_dims(mask, -1)).to(DEVICE)
        return image, mask, target.long(), target_onehot

train_ldr = make_loader(SaltTrainSet(train_size, train_ids), BATCH_SIZE)
val_ldr = make_loader(SaltValSet(val_size, val_ids), BATCH_SIZE)

# Define customized loss function
def lovasz(outputs, targets, mode='individual'):
    outputs = outputs.squeeze()
    targets = targets.squeeze()
    return lovasz_losses.lovasz_binary(outputs, targets, mode=mode)

def mixed_loss(logit_pixel, logit_image, mask, target, target_onehot, is_average=True):

    loss_image = F.cross_entropy(logit_image, target)

    loss_pixel_0 = lovasz(logit_pixel[:, 0, :, :], mask, mode='individual')
    loss_pixel_1 = lovasz(logit_pixel[:, 1, :, :], mask, mode='individual')
    loss_pixel_2 = lovasz(logit_pixel[:, 2, :, :], mask, mode='individual')

    loss_pixel_0 = loss_pixel_0 * target_onehot[:, 1:2]
    loss_pixel_1 = loss_pixel_1 * target_onehot[:, 2:3]
    loss_pixel_2 = loss_pixel_2 * target_onehot[:, 3:4]

    if is_average:
        if target_onehot[:, 1].sum() != 0:
            loss_pixel_0 = loss_pixel_0.sum() / target_onehot[:, 1].sum()
        else:
            loss_pixel_0 = loss_pixel_0.sum()
        if target_onehot[:, 2].sum() != 0:
            loss_pixel_1 = loss_pixel_1.sum() / target_onehot[:, 2].sum()
        else:
            loss_pixel_1 = loss_pixel_1.sum()
        if target_onehot[:, 3].sum() != 0:
            loss_pixel_2 = loss_pixel_2.sum() / target_onehot[:, 3].sum()
        else:
            loss_pixel_2 = loss_pixel_2.sum()

    return loss_pixel_0, loss_pixel_1, loss_pixel_2, loss_image

def train(train_ldr):
    net.train()
    total_loss_pixel_0 = 0
    total_loss_pixel_1 = 0
    total_loss_pixel_2 = 0
    total_loss_image   = 0
    start_time = time.time()

    for batch_idx, (image, mask, target, target_onehot) in enumerate(train_ldr):
        optimizer.zero_grad()
        logit_pixel, logit_image = net(image)
        loss_pixel_0, loss_pixel_1, loss_pixel_2, loss_image = criterion(logit_pixel, logit_image, mask, target, target_onehot)
        loss = loss_pixel_0 + loss_pixel_1 + loss_pixel_2 + loss_image
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
        optimizer.step()

        total_loss_pixel_0 += loss_pixel_0.item()
        total_loss_pixel_1 += loss_pixel_1.item()
        total_loss_pixel_2 += loss_pixel_2.item()
        total_loss_image   += loss_image.item()
        if batch_idx % LOG_INTERVAL == 0 and batch_idx > 0:
            cur_loss_pixel_0 = total_loss_pixel_0 / LOG_INTERVAL
            cur_loss_pixel_1 = total_loss_pixel_1 / LOG_INTERVAL
            cur_loss_pixel_2 = total_loss_pixel_2 / LOG_INTERVAL
            cur_loss_image   = total_loss_image   / LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:3d}/{:3d} batches | ms/batch {:5.2f} | pixel loss 1 {:5.4f} | '
                    'pixel loss 2 {:5.4f} | pixel loss 3 {:5.4f} | image loss {:5.4f}'.format(
                epoch, batch_idx, len(train_ldr),
                elapsed * 1000 / LOG_INTERVAL, cur_loss_pixel_0, cur_loss_pixel_1, cur_loss_pixel_2, cur_loss_image))

            pixel0_handle.send(cur_loss_pixel_0)
            pixel1_handle.send(cur_loss_pixel_1)
            pixel2_handle.send(cur_loss_pixel_2)
            image_handle.send(cur_loss_image)

            total_loss_pixel_0 = 0
            total_loss_pixel_1 = 0
            total_loss_pixel_2 = 0
            total_loss_image   = 0
            start_time = time.time()

def evaluate(val_ldr, metric):
    net.eval()
    total_image_loss = 0
    total_iou = 0

    with torch.no_grad():
        for image, mask, target, target_onehot in val_ldr:
            bs = len(image)
            logit_pixel, logit_image = net(image)
            output = torch.zeros_like(logit_pixel[:, 0, :, :])
            for i in range(bs):
                pred_onehot = torch.argmax(logit_image[i])
                if pred_onehot == 0:
                    pass
                elif pred_onehot == 1:
                    output[i] = logit_pixel[i, 0, :, :]
                elif pred_onehot == 2:
                    output[i] = logit_pixel[i, 1, :, :]
                elif pred_onehot == 3:
                    output[i] = logit_pixel[i, 2, :, :]
                else:
                    raise ValueError("Invalid class to predict")

            output = (output > 0) * 255
            output = np.moveaxis(output.cpu().numpy(), 0, -1)
            data_dict = {"image": output, "mask": output}
            downsized = downsize(**data_dict)
            output = downsized["mask"] / 255
            output = torch.from_numpy(np.moveaxis(output, -1, 0)).float().to(DEVICE)

            total_image_loss += F.cross_entropy(logit_image, target) * bs
            total_iou += metric(output, mask).data

    return total_image_loss / val_size, total_iou / val_size

if __name__ == "__main__":
    T_max = 50
    T_mult = 1.0

    net = AttentionHyperResUNet().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = CosineAnnealingLR_with_Restart(optimizer, T_max, T_mult, None, None, False, eta_min=0.0005)
    criterion = mixed_loss

    best_so_far = 0
    dev_image_losses = np.empty(MAX_EPOCHS)
    dev_ious = np.empty(MAX_EPOCHS)

    for epoch in tqdm(range(0, MAX_EPOCHS)):
        scheduler.step()

        epoch_start_time = time.time()
        train(train_ldr)
        dev_image_loss, dev_iou = evaluate(val_ldr, iou_pytorch)
        dev_image_losses[epoch] = dev_image_loss
        dev_ious[epoch] = dev_iou

        image_channel_handle.send(dev_image_loss)
        iou_channel_handle.send(dev_iou)

        print('-' * 89)
        print('end of epoch {:3d} | time: {:5.2f}s | dev image loss {:5.4f} | dev iou {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), dev_image_loss, dev_iou))
        print('-' * 89)

        n_restart, counter = divmod(epoch, T_max)
        if counter == 0:
            best_so_far = 0

        if dev_iou > best_so_far:
            best_so_far = dev_iou
            torch.save(net.state_dict(), os.path.join(SAVE_PATH, "hard_attention_cycle{}".format(n_restart + 1)))
