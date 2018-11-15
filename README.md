# Kaggle TGS Salt Identification Challenge

Jihang Zhang

## Overview

This is the repo that stores the implementations of the models I used in the Kaggle TGS Salt Identification Challenge. The models are to identify the location of salt given patches of seismic images. Details about the data augmentation, architecture of the neural networks and post-processing methods are in https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69060

## Dependencies

You can use https://neptune.ml/ to train the models using cloud computing and all dependencies will be installed automatically. Otherwise, please  make sure the following packages are installed correctly.

* Python 3.6
* NumPy 1.15
* PyTorch 0.4.0 (version 0.4.1 might result in some warnings / bugs which can be easily fixed as well)
* torchvision 0.2.1
* tqdm 4.25
* OpenCV-Python 3.4.3.18
* albumentations 0.1.1
* pretrainedmodels 0.7.0
* neptune-cli

## Contents

deep_supervision/

* python code to create, run and save the deep supervised model

hard_attention/

* python code to create, run and save the multi-class segmentation model

input/

* directory of training and test images and masks

output/

* directory of saving trained models

## Usage

1. Install all dependencies (above)

2. Download dataset from https://www.kaggle.com/c/tgs-salt-identification-challenge/data or type the following command

      ```bash
      $ kaggle competitions download -c tgs-salt-identification-challenge
      ```

3. Unzip the downloaded dataset to input/

4. To train, evaluate and save the deep supervised model, navigate to deep_supervision/, and type the following command in terminal:

      ```bash
      $ python main.py
      ```
      To train, evaluate and save the multi-class segmentation model, navigate to hard_attention/, and type the following command in terminal:

      ```bash
      $ python main.py
      ```

## References

* [OCNet: Object Context Network for Scene Parsing](https://arxiv.org/abs/1809.00916) ([code](https://github.com/PkuRainBow/OCNet))
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Deeply Supervised Salient Object Detection with Short Connections](https://arxiv.org/pdf/1611.04849.pdf)
* [Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

