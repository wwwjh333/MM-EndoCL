import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from torchvision.transforms.functional import InterpolationMode
import random
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LC(Dataset):
    def __init__(self, data_path, transform=None, mode='Training'):
        df = pd.read_csv(os.path.join(data_path, mode + '.csv'), encoding='gbk')
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


class GaussianNoise_lc(object):
    def __init__(self, severity=0, modality="nbi", std_list=None, clip=True, seed=1234):
        """
        severity: int, e.g., s = 0..3
        modality: 'nbi' or 'wli'
        std_list: list of std values indexed by severity
                  default is [0.0, 0.03, 0.06, 0.10]
        clip: clamp to [0,1] after adding noise (assumes tensor in [0,1])
        seed: optional int for reproducibility
        """
        self.severity = int(severity)
        self.modality = modality
        self.clip = bool(clip)
        self.seed = seed

        if std_list is None:
            std_list = [0.0, 0.03, 0.06, 0.10]  # adjust if needed
        self.std_list = std_list

        if self.severity < 0 or self.severity >= len(self.std_list):
            raise ValueError(f"severity={self.severity} out of range for std_list (len={len(self.std_list)})")
        self.std = float(self.std_list[self.severity])

    def __call__(self, sample):
        if self.severity <= 0:
            return sample

        key = "image_n" if self.modality == "nbi" else "image_b"
        x = sample[key]  # torch.Tensor [C,H,W]

        # deterministic noise if seed provided
        noise = torch.randn_like(x) * self.std

        x = x + noise
        if self.clip:
            x = torch.clamp(x, 0.0, 1.0)

        sample[key] = x
        return sample


class GaussianBlur_lc(object):
    def __init__(self, severity=0, modality="nbi"):
        """
        severity: int, s = 0,1,2,3,4
        modality: 'nbi' or 'wli'
        """
        self.severity = severity
        self.modality = modality

        # sigma = 0.5 * s
        self.sigma = 3 * severity

        if self.sigma > 0:
            k = 2 * math.ceil(3 * self.sigma) + 1
            self.kernel_size = min(k, 15)  # cap kernel size
        else:
            self.kernel_size = None

    def __call__(self, sample):
        if self.severity <= 0:
            return sample

        if self.modality == "nbi":
            sample["image_n"] = F.gaussian_blur(
                sample["image_n"],
                kernel_size=[self.kernel_size, self.kernel_size],
                sigma=[self.sigma, self.sigma]
            )
            sample["image_n"].save("nbi_blur.png")
        elif self.modality == "wli":
            sample["image_b"] = F.gaussian_blur(
                sample["image_b"],
                kernel_size=[self.kernel_size, self.kernel_size],
                sigma=[self.sigma, self.sigma]
            )
            sample["image_b"].save("wli_blur.png")

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