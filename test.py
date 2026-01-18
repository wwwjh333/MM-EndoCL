import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_lc import LC, Resize_lc, ToTensor_lc

import torchvision.transforms as transforms
from utils import test_single_volume,iou,dice_coeff
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from medpy import metric
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from medpy.metric.binary import hd95
from sklearn.manifold import TSNE
from thop import profile
from thop import clever_format
import time


import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='LC_fusion_dataset1', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--weight_path', type=str,
                    default='...', help='weights to finetune from')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference_lc_fusion(args, model):
    db_test = LC(data_path=args.data_path, mode='Test',
                               transform=transforms.Compose(
                                   [
                                   Resize_lc(args.img_size),ToTensor_lc()
                                   ]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    eiou,edice,ese,esp,emiou,emdice,ehd95 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image,image1, label, case_name = sampled_batch["image_b"].cuda(),sampled_batch["image_n"].cuda(), sampled_batch["label_b"], sampled_batch['case_name']
        bs = image.size(0)
        with torch.no_grad():

            logits = model(image, image1)[0]
            out = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            pred_bin = (out == 1).unsqueeze(1).float()      # [B,1,H,W]
            gt_bin   = (label.cuda() == 1).unsqueeze(1).float()
            hd95_vals, skipped_count = calculate_hd95_2d(pred_bin, gt_bin)
            prediction = out.cpu().detach()
            prediction_iou = prediction.numpy().astype('int32')
            label_iou = label.numpy().astype('int32')

            #se, sp
            tp = ((prediction == 1) & (label == 1)).sum().item()
            tn = ((prediction == 0) & (label == 0)).sum().item()
            fp = ((prediction == 1) & (label == 0)).sum().item()
            fn = ((prediction == 0) & (label == 1)).sum().item()
            dice_val = dice_coeff((prediction==1).float(), (label==1).float()).item()

            se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            ese = ese + se
            esp = esp + sp
            eiou += iou(prediction_iou==1,label_iou==1)
            edice += dice_coeff((prediction==1).float(), (label==1).float()).item()
            ehd95 += hd95_vals
            
    iou_m = eiou/(len(testloader))
    dice_m = edice/(len(testloader))
    se_m = ese/(len(testloader))
    sp_m = esp/(len(testloader))
    hd95_m = ehd95/(len(testloader))
    logging.info("Mean IOU:{},Mean Dice:{},Mean SE:{},Mean SP:{},Mean HD95:{}".format(iou_m,dice_m,se_m,sp_m,hd95_m))
    return "Testing Finished!"


def calculate_hd95_2d(pred_tensor, label_tensor):
    pred_np = pred_tensor.detach().cpu().numpy()
    label_np = label_tensor.detach().cpu().numpy()

    if pred_np.ndim == 4: pred_np = pred_np.squeeze(1)
    if label_np.ndim == 4: label_np = label_np.squeeze(1)

    batch_size = pred_np.shape[0]
    hd95_list = []
    skipped_count = 0
    for i in range(batch_size):
        p = pred_np[i]
        l = label_np[i]
        if np.sum(p) == 0 or np.sum(l) == 0:
            skipped_count += 1
            continue

        try:
            dist = hd95(p, l)
            hd95_list.append(dist)
        except Exception as e:
            print(f"Error: {e}")
            skipped_count += 1

    mean_hd95 = np.mean(hd95_list) if len(hd95_list) > 0 else 0.0
    return mean_hd95, skipped_count

if __name__ == "__main__":

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'demo_dataset': {
            'data_path': './data/demo_dataset',
            'num_classes': 2,
        }
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.data_path = dataset_config[dataset_name]['data_path']


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    from networks.vit_seg_modeling_fusion import VisionTransformer as ViT_seg_fusion
    net = ViT_seg_fusion(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    net.load_state_dict(torch.load(args.weight_path))

    inference_lc_fusion(args, net)