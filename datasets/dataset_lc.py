import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from torchvision.transforms.functional import InterpolationMode
import random

class LC(Dataset):
    def __init__(self, data_path, transform=None, mode='Training'):
        df = pd.read_csv(os.path.join(data_path, mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list_b = df.iloc[:, 0].tolist()
        self.name_list_n = df.iloc[:, 1].tolist()
        self.label_list_b = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.name_list_b)

    def __getitem__(self, index):
        inout = 1
        point_label = 1

        """Get the images"""
        name_b = self.name_list_b[index]
        name_n = self.name_list_n[index]
        img_b_path = os.path.join(self.data_path, name_b)
        img_n_path = os.path.join(self.data_path, name_n)

        mask_b_name = self.label_list_b[index]
        msk_path_b = os.path.join(self.data_path, mask_b_name)

        img_b = Image.open(img_b_path).convert('RGB')
        img_n = Image.open(img_n_path).convert('RGB')
        mask_b = Image.open(msk_path_b).convert('L')
        name = name_b.split('/')[-1].split(".png")[0]


        sample = {'image_b': img_b, 'image_n': img_n, 'label_b': mask_b,'case_name': name}

        if self.transform:
            sample = self.transform(sample)
        return sample

class Resize_lc(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, sample):
        T1 = transforms.Compose([transforms.Resize(size=(self.img_size,self.img_size))])
        T2 = transforms.Compose([transforms.Resize(size=(self.img_size,self.img_size),interpolation=InterpolationMode.NEAREST)])
        img_b_resize, img_n_resize = T1(sample["image_b"]),  T1(sample["image_n"])
        sample['image_b'], sample['image_n'] = img_b_resize,img_n_resize
        sample['label_b'] = T2(sample['label_b'])
        return sample

class ToTensor_lc(object):

    def __call__(self, sample):
        T1 = transforms.Compose([transforms.ToTensor()])


        sample['image_b'], sample['image_n'] = T1(sample["image_b"]),  T1(sample["image_n"])
        sample['label_b'] = T1(sample['label_b']).squeeze(0)
        return sample