# This is from Ilya Ezepov's "Fast IOU scoring metric in PyTorch and numpy" kernel on Kaggle

import torch

# PyTroch version

SMOOTH = 1e-6

def iou_pytorch(outputs, labels, size_average=False):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = (outputs > 0.0).long()

    labels = labels.squeeze(1).long()

    intersection = (outputs & labels).float().sum(1).sum(1)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(1).sum(1)         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean() if size_average else thresholded.sum() # Or thresholded.mean() if you are interested in average across the batch
