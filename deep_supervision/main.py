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
from accuracy import accuracy
import lovasz_losses
from loaders import make_loader

import torchvision.transforms as transforms
import cv2
import albumentations as aug

from OCNet.superior040 import Superior
from scheduler import CosineAnnealingLR_with_Restart

import neptune
ctx = neptune.Context()

fuse_handle = ctx.create_channel("train fuse loss", neptune.ChannelType.NUMERIC)
acc_channel_handle = ctx.create_channel("dev accuracy", neptune.ChannelType.NUMERIC)
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

padding = aug.Compose([
    aug.PadIfNeeded(0, 128, border_mode=cv2.BORDER_REFLECT_101),
    aug.PadIfNeeded(128, 0, border_mode=cv2.BORDER_REPLICATE)
])

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

augmentor       = aug.Compose([padding, aug_train()])
downscalor8x8   = aug.Resize(8, 8)
downscalor16x16 = aug.Resize(16, 16)
downscalor32x32 = aug.Resize(32, 32)
downscalor64x64 = aug.Resize(64, 64)

class SaltTrainSet(Dataset):
    """Salt segmentation dataset."""

    def __init__(self, train_size, train_ids):
        self.train_size = train_size
        self.train_ids = train_ids

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        image = cv2.imread("../input/train/images/{}.png".format(self.train_ids[idx]), 1)
        mask  = cv2.imread("../input/train/masks/{}.png".format(self.train_ids[idx]), 0)
        data_dict = {"image": image, "mask": mask}

        augmented = augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        mask_dict = {"image": image, "mask": mask}
        downscaled8x8   = downscalor8x8(**mask_dict)
        downscaled16x16 = downscalor16x16(**mask_dict)
        downscaled32x32 = downscalor32x32(**mask_dict)
        downscaled64x64 = downscalor64x64(**mask_dict)

        mask8x8   = downscaled8x8["mask"]
        mask16x16 = downscaled16x16["mask"]
        mask32x32 = downscaled32x32["mask"]
        mask64x64 = downscaled64x64["mask"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask8x8   = transforms.functional.to_tensor(np.expand_dims(mask8x8, -1)).to(DEVICE)
        mask16x16 = transforms.functional.to_tensor(np.expand_dims(mask16x16, -1)).to(DEVICE)
        mask32x32 = transforms.functional.to_tensor(np.expand_dims(mask32x32, -1)).to(DEVICE)
        mask64x64 = transforms.functional.to_tensor(np.expand_dims(mask64x64, -1)).to(DEVICE)
        mask = transforms.functional.to_tensor(np.expand_dims(mask, -1)).to(DEVICE)
        target = (mask.sum() > 0).float().unsqueeze(0)
        return image, mask8x8, mask16x16, mask32x32, mask64x64, mask, target

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
        image_dict = {"image": image}
        padded = padding(**image_dict)
        image = padded["image"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = transforms.functional.to_tensor(np.expand_dims(mask, -1)).to(DEVICE)
        target = (mask.sum() > 0).float().unsqueeze(0)
        return image, mask, target

train_ldr = make_loader(SaltTrainSet(train_size, train_ids), BATCH_SIZE)
val_ldr = make_loader(SaltValSet(val_size, val_ids), BATCH_SIZE)

# Define customized loss function
def lovasz(outputs, targets, mode='individual'):
    outputs = outputs.squeeze()
    targets = targets.squeeze()
    return lovasz_losses.lovasz_binary(outputs, targets, mode=mode)

def mixed_loss(
        logit,
        logit_pixel_8x8,
        logit_pixel_16x16,
        logit_pixel_32x32,
        logit_pixel_64x64,
        logit_pixel_128x128,
        logit_image,
        truth_pixel_8x8,
        truth_pixel_16x16,
        truth_pixel_32x32,
        truth_pixel_64x64,
        truth_pixel,
        truth_image,
        is_average=True
    ):

    loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image)

    loss_fuse          = lovasz(logit              , truth_pixel      , mode='mean')
    loss_pixel_8x8     = lovasz(logit_pixel_8x8    , truth_pixel_8x8  , mode='individual')
    loss_pixel_16x16   = lovasz(logit_pixel_16x16  , truth_pixel_16x16, mode='individual')
    loss_pixel_32x32   = lovasz(logit_pixel_32x32  , truth_pixel_32x32, mode='individual')
    loss_pixel_64x64   = lovasz(logit_pixel_64x64  , truth_pixel_64x64, mode='individual')
    loss_pixel_128x128 = lovasz(logit_pixel_128x128, truth_pixel      , mode='individual')

    #loss for empty image is weighted 0
    loss_pixel_8x8     = loss_pixel_8x8     * truth_image
    loss_pixel_16x16   = loss_pixel_16x16   * truth_image
    loss_pixel_32x32   = loss_pixel_32x32   * truth_image
    loss_pixel_64x64   = loss_pixel_64x64   * truth_image
    loss_pixel_128x128 = loss_pixel_128x128 * truth_image
    if is_average:
        loss_pixel_8x8     = loss_pixel_8x8.sum()     / truth_image.sum()
        loss_pixel_16x16   = loss_pixel_16x16.sum()   / truth_image.sum()
        loss_pixel_32x32   = loss_pixel_32x32.sum()   / truth_image.sum()
        loss_pixel_64x64   = loss_pixel_64x64.sum()   / truth_image.sum()
        loss_pixel_128x128 = loss_pixel_128x128.sum() / truth_image.sum()

    weight_fuse          = 1.0
    weight_pixel_8x8     = 0.1
    weight_pixel_16x16   = 0.1
    weight_pixel_32x32   = 0.1
    weight_pixel_64x64   = 0.1
    weight_pixel_128x128 = 0.1
    weight_image         = 0.05

    return (
        weight_fuse          * loss_fuse,
        weight_pixel_8x8     * loss_pixel_8x8,
        weight_pixel_16x16   * loss_pixel_16x16,
        weight_pixel_32x32   * loss_pixel_32x32,
        weight_pixel_64x64   * loss_pixel_64x64,
        weight_pixel_128x128 * loss_pixel_128x128,
        weight_image         * loss_image
    )

def train(train_ldr):
    net.train()
    total_loss_fuse  = 0
    start_time = time.time()

    for batch_idx, (image, mask8x8, mask16x16, mask32x32, mask64x64, mask, target) in enumerate(train_ldr):
        optimizer.zero_grad()
        logit, logit_pixel_8x8, logit_pixel_16x16, logit_pixel_32x32, logit_pixel_64x64, logit_pixel_128x128, logit_image = net(image)
        loss_fuse, loss_pixel_8x8, loss_pixel_16x16, loss_pixel_32x32, loss_pixel_64x64, loss_pixel_128x128, loss_image = criterion(
            logit,
            logit_pixel_8x8,
            logit_pixel_16x16,
            logit_pixel_32x32,
            logit_pixel_64x64,
            logit_pixel_128x128,
            logit_image,
            mask8x8,
            mask16x16,
            mask32x32,
            mask64x64,
            mask,
            target
        )
        loss = loss_fuse + loss_pixel_8x8 + loss_pixel_16x16 + loss_pixel_32x32 + loss_pixel_64x64 + loss_pixel_128x128 + loss_image
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
        optimizer.step()

        total_loss_fuse  += loss_fuse.item()
        if batch_idx % LOG_INTERVAL == 0 and batch_idx > 0:
            cur_loss_fuse  = total_loss_fuse / LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:3d}/{:3d} batches | ms/batch {:5.2f} | fuse loss {:5.4f} |'.format(
                epoch, batch_idx, len(train_ldr),
                elapsed * 1000 / LOG_INTERVAL, cur_loss_fuse))

            fuse_handle.send(cur_loss_fuse)

            total_loss_fuse  = 0
            start_time = time.time()

def evaluate(val_ldr, metric):
    net.eval()
    total_iou = 0
    total_acc = 0

    with torch.no_grad():
        for image, mask, target in val_ldr:
            bs = len(image)
            logit, _, _, _, _, _, logit_image = net(image)
            logit = logit[:, :, 14:-13, 14:-13]

            total_acc += accuracy(logit_image, target).data
            total_iou += metric(logit, mask).data

    return (
        total_acc / val_size,
        total_iou / val_size
    )

if __name__ == "__main__":
    T_max = 50
    T_mult = 1.0

    net = Superior().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = CosineAnnealingLR_with_Restart(optimizer, T_max, T_mult, None, None, False, eta_min=0.0005)
    criterion = mixed_loss

    best_so_far = 0
    dev_accs = np.empty(MAX_EPOCHS)
    dev_ious = np.empty(MAX_EPOCHS)

    for epoch in tqdm(range(0, MAX_EPOCHS)):
        scheduler.step()

        epoch_start_time = time.time()
        train(train_ldr)
        dev_acc, dev_iou = evaluate(val_ldr, iou_pytorch)
        dev_accs[epoch] = dev_acc
        dev_ious[epoch] = dev_iou

        acc_channel_handle.send(dev_acc)
        iou_channel_handle.send(dev_iou)

        print('-' * 89)
        print('end of epoch {:3d} | time: {:5.2f}s | dev acc {:5.4f} | dev iou {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), dev_acc, dev_iou))
        print('-' * 89)

        n_restart, counter = divmod(epoch, T_max)
        if counter == 0:
            best_so_far = 0

        if dev_iou > best_so_far:
            best_so_far = dev_iou
            torch.save(net.state_dict(), os.path.join(SAVE_PATH, "deep_supervision_cycle{}".format(n_restart + 1)))
