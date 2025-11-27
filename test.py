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
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_tumor import Tumor, Resize_tumor, ToTensor_tumor
from datasets.dataset_brats import Brats,Resize_brats,ToTensor
import torchvision.transforms as transforms
from utils import test_single_volume,iou,dice_coeff
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from medpy import metric
import matplotlib.pyplot as plt
import cv2

from sklearn.manifold import TSNE

import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/devdata/wujunhao_data/data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()



def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

def inference_tumor(args, model, test_save_path=None):
    db_test = args.Dataset(data_path=args.volume_path, mode='Test',
                               transform=transforms.Compose(
                                   [Resize_tumor(args.img_size),ToTensor_tumor()]))
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    eiou,edice,ese,esp = 0,0,0,0
    emiou, emdice = 0, 0
    output_dir = './vis/big_single'
    os.makedirs(output_dir, exist_ok=True)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name,image1 = sampled_batch["image_b"].cuda(), sampled_batch["label_b"], sampled_batch['case_name'],sampled_batch["image_n"].cuda()
        bs = image.size(0)
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(image), dim=1), dim=1)
            prediction = out.cpu().detach()
            prediction_iou = prediction.numpy().astype('int32')
            label_iou = label.numpy().astype('int32')
            #se, sp
            tp = ((prediction == 1) & (label == 1)).sum().item()
            tn = ((prediction == 0) & (label == 0)).sum().item()
            fp = ((prediction == 1) & (label == 0)).sum().item()
            fn = ((prediction == 0) & (label == 1)).sum().item()

            se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            ese = ese + se
            esp = esp + sp
            eiou += iou(prediction_iou == 1, label_iou == 1)
            edice += dice_coeff((prediction == 1).float(), (label == 1).float()).item()
            emiou += (iou(prediction_iou == 1, label_iou == 1) + iou(prediction_iou == 0, label_iou == 0)) / 2
            emdice += (dice_coeff((prediction == 1).float(), (label == 1).float()).item() + dice_coeff(
                (prediction == 0).float(), (label == 0).float()).item()) / 2
            for i in range(bs):
                save_path = os.path.join(output_dir,f'{case_name[i]}.png')
                
               
                pred = prediction[i].cpu().numpy()  # Grayscale maskv
               
                pred = (pred == 1).astype(np.uint8) * 255 

                pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

                # Save the final concatenated image
                cv2.imwrite(save_path, pred)
    iou_m = eiou/len(testloader)
    dice_m = edice/len(testloader)
    se_m = ese/len(testloader)
    sp_m = esp/len(testloader)
    miou = emiou/len(testloader)
    mdice = emdice/len(testloader)
    logging.info("Mean IOU:{},Mean Dice:{},Mean SE:{},Mean SP:{}".format(iou_m,dice_m,se_m,sp_m))
    logging.info("MIOU:{},MDice:{}".format(miou, mdice))
    return "Testing Finished!"

def inference_tumor_fusion(args, model, test_save_path=None):
    db_test = args.Dataset(data_path=args.volume_path, mode='Test',
                               transform=transforms.Compose(
                                   [Resize_tumor(args.img_size),ToTensor_tumor()]))
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    eiou,edice,ese,esp = 0,0,0,0
    emiou,emdice =0,0
    output_dir = './vis/ood_dataset1'
    whi_specific,nbi_specific,whi_share,nbi_share,whi_ori,nbi_ori = [],[],[],[],[],[]
    os.makedirs(output_dir, exist_ok=True)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image,image1, label, case_name,label_path = sampled_batch["image_b"].cuda(),sampled_batch["image_n"].cuda(), sampled_batch["label_b"], sampled_batch['case_name'],sampled_batch['label_path']
        bs = image.size(0)
        with torch.no_grad():
            _,_, _, _, _,x_specific,x1_specific,x_share,x1_share=net(image,image1)
            whi_specific.append(x_specific)
            nbi_specific.append(x1_specific)
            whi_share.append(x_share)
            nbi_share.append(x1_share)
            # whi_ori.append(x_ori)
            # nbi_ori.append(x1_ori)
            out = torch.argmax(torch.softmax(net(image,image1)[0], dim=1), dim=1)
            prediction = out.cpu().detach()
            prediction_iou = prediction.numpy().astype('int32')
            label_iou = label.numpy().astype('int32')

            #se, sp
            tp = ((prediction == 1) & (label == 1)).sum().item()
            tn = ((prediction == 0) & (label == 0)).sum().item()
            fp = ((prediction == 1) & (label == 0)).sum().item()
            fn = ((prediction == 0) & (label == 1)).sum().item()

            se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            ese = ese + se
            esp = esp + sp
            eiou += iou(prediction_iou==1,label_iou==1)
            edice += dice_coeff((prediction==1).float(), (label==1).float()).item()
            emiou += (iou(prediction_iou==1,label_iou==1)+iou(prediction_iou==0,label_iou==0))/2
            emdice += (dice_coeff((prediction==1).float(), (label==1).float()).item()+dice_coeff((prediction==0).float(), (label==0).float()).item())/2
            for i in range(bs):
                save_path = os.path.join(output_dir,f'{case_name[i]}.png')

                pred = prediction[i].cpu().numpy()  # Grayscale maskv

                # Ensure prediction and label are binary masks (0 for black, 255 for white)
                pred = (pred == 1).astype(np.uint8) * 255  # Convert binary mask (0/1) to (0/255)

                # # Convert single-channel prediction and label to 3-channel grayscale images
                pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

                # Save the final concatenated image
                cv2.imwrite(save_path, pred)
                            
    whi_specific = torch.cat(whi_specific, 0).reshape(-1, 196 * 768)
    nbi_specific = torch.cat(nbi_specific, 0).reshape(-1, 196 * 768)
    whi_share = torch.cat(whi_share, 0).reshape(-1, 196 * 768)
    nbi_share = torch.cat(nbi_share, 0).reshape(-1, 196 * 768)
    # whi_ori = torch.cat(whi_ori, 0).reshape(-1, 196 * 768)
    # nbi_ori = torch.cat(nbi_ori, 0).reshape(-1, 196 * 768)


    features = torch.cat([whi_specific, nbi_specific, whi_share, nbi_share], 0).cpu().numpy()
    labels = ['WHI_Specific'] * len(whi_specific) + \
            ['NBI_Specific'] * len(nbi_specific) + \
            ['WHI_Share'] * len(whi_share) + \
            ['NBI_Share'] * len(nbi_share)
    # features = torch.cat([whi_ori, nbi_ori], 0).cpu().numpy()
    # labels = ['WHI'] * len(whi_ori) + \
    #         ['NBI'] * len(nbi_ori) 
    tsne = TSNE(n_components=2, perplexity=2, learning_rate=100, n_iter=1000, verbose=1)
    # tsne = TSNE(n_components=2, perplexity=15, learning_rate=1000, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features)

    # # 极简可视化
    # plt.figure(figsize=(3, 3))
    # sns.scatterplot(
    #     x=features_2d[:, 0],
    #     y=features_2d[:, 1],
    #     hue=labels,
    #     palette='tab10',
    #     s=15,        # 增大点的大小
    #     alpha=0.7,  # 透明度保持适中
    #     linewidth=0 # 去掉点边框
    # )
    # 极简可视化
    # palette = {"WHI_Specific": "#1f77b4",  # Muted blue
    #        "NBI_Specific": "#ff7f0e",  # Warm orange
    #        "WHI_Share": "#2ca02c",  # Balanced green
    #        "NBI_Share": "#d62728"}  # Deep red
    palette = {
        "WHI_Specific": "#9CC2E6",  # 深海钴蓝（主锚点）
        "NBI_Specific": "#A7D08C",  # 极地青蓝（中等对比） 
        "WHI_Share":    "#FED966",  # 冰隙蓝（高明度过渡）
        "NBI_Share":    "#BD8EDE"   # 月面银灰（中性缓冲）
    }

    # Visualization
    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels,
        palette=palette,
        s=20,  # Adjusted point size
        alpha=0.7,  # Transparency
        linewidth=0,  # No border around points
        legend=False
    )

    # Grid and axis styling
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().set_xticklabels([])  # 隐藏 x 轴刻度标签
    plt.gca().set_yticklabels([])  # 隐藏 y 轴刻度标签
    plt.xlabel("")  # 去除 x 轴标签
    plt.ylabel("")  # 去除 y 轴标签
    plt.tick_params(axis='both', which='both', length=0)

    # Legend styling
    # plt.legend(fontsize=12, loc="upper left", frameon=True)

    # Save figure
    plt.savefig('tsne/dataset2_base_tsne.png', dpi=300, bbox_inches='tight')

    iou_m = eiou/len(testloader)
    dice_m = edice/len(testloader)
    se_m = ese/len(testloader)
    sp_m = esp/len(testloader)
    miou = emiou/len(testloader)
    mdice = emdice/len(testloader)
    logging.info("Mean IOU:{},Mean Dice:{},Mean SE:{},Mean SP:{}".format(iou_m,dice_m,se_m,sp_m))
    logging.info("MIOU:{},MDice:{}".format(miou, mdice))
    return "Testing Finished!"

def inference_brats(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, mode='test',
                               transform=transforms.Compose(
                                   [Resize_brats(args.img_size),ToTensor()]))
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    eiou, edice = [0,0,0], [0,0,0]
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label = sampled_batch["flair"].cuda(), sampled_batch["label"]
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(image), dim=1), dim=1)
            prediction = out.cpu().detach()

            prediction_wt = prediction.clone()
            prediction_wt[prediction_wt==3]=1
            prediction_wt[prediction_wt == 2] = 1
            prediction_wt_iou = prediction_wt.numpy().astype('int32')
            label_wt = label.clone()
            label_wt[label_wt==3]=1
            label_wt[label_wt ==2]=1
            label_wt_iou = label_wt.numpy().astype('int32')
            prediction_wt_dice = (prediction_wt == 1).float()
            label_wt_dice = (label_wt == 1).float()
            eiou[0] += iou(prediction_wt_iou == 1, label_wt_iou == 1)
            edice[0] += dice_coeff(prediction_wt_dice, label_wt_dice).item()

            prediction_tc = prediction.clone()
            prediction_tc[prediction_tc == 3] = 1
            prediction_tc_iou = prediction_tc.numpy().astype('int32')
            label_tc = label.clone()
            label_tc[label_tc == 3] = 1
            label_tc_iou = label_tc.numpy().astype('int32')
            prediction_tc_dice = (prediction_tc == 1).float()
            label_tc_dice = (label_tc == 1).float()
            eiou[1] += iou(prediction_tc_iou == 1, label_tc_iou == 1)
            edice[1] += dice_coeff(prediction_tc_dice, label_tc_dice).item()

            prediction_et = prediction.clone()
            prediction_et[prediction_et == 1] = 0
            prediction_et[prediction_et == 3] = 1
            prediction_et_iou = prediction_et.numpy().astype('int32')
            label_et = label.clone()
            label_et[label_et == 1] = 0
            label_et[label_et == 3] = 1
            label_et_iou = label_et.numpy().astype('int32')
            prediction_et_dice = (prediction_et == 1).float()
            label_et_dice = (label_et == 1).float()
            eiou[2] += iou(prediction_et_iou == 1, label_et_iou == 1)
            edice[2] += dice_coeff(prediction_et_dice, label_et_dice).item()

    iou_m = [x / len(testloader) for x in eiou]
    dice_m = [x / len(testloader) for x in edice]
    logging.info("WT: Mean IOU:{},Mean Dice:{}".format(iou_m[0], dice_m[0]))
    logging.info("TC: Mean IOU:{},Mean Dice:{}".format(iou_m[1], dice_m[1]))
    logging.info("ET: Mean IOU:{},Mean Dice:{}".format(iou_m[2], dice_m[2]))
    return "Testing Finished!"

def inference_brats_fusion(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, mode='test',
                           transform=transforms.Compose(
                               [Resize_brats(args.img_size), ToTensor()]))
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    eiou, edice = [0,0,0], [0,0,0]
    eiou1 = 0
    edice1 = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image,image1, label = sampled_batch["t1"].cuda(),sampled_batch["t2"].cuda(), sampled_batch["label"]
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(image,image1)[1], dim=1), dim=1)
            prediction = out.cpu().detach()
            prediction_wt = prediction.clone()
            prediction_wt[prediction_wt == 3] = 1
            prediction_wt[prediction_wt == 2] = 1
            prediction_wt_iou = prediction_wt.numpy().astype('int32')
            label_wt = label.clone()
            label_wt[label_wt == 3] = 1
            label_wt[label_wt == 2] = 1
            label_wt_iou = label_wt.numpy().astype('int32')
            prediction_wt_dice = (prediction_wt == 1).float()
            label_wt_dice = (label_wt == 1).float()
            eiou[0] += iou(prediction_wt_iou == 1, label_wt_iou == 1)
            edice[0] += dice_coeff(prediction_wt_dice, label_wt_dice).item()

            prediction_tc = prediction.clone()
            prediction_tc[prediction_tc == 3] = 1
            prediction_tc_iou = prediction_tc.numpy().astype('int32')
            label_tc = label.clone()
            label_tc[label_tc == 3] = 1
            label_tc_iou = label_tc.numpy().astype('int32')
            prediction_tc_dice = (prediction_tc == 1).float()
            label_tc_dice = (label_tc == 1).float()
            eiou[1] += iou(prediction_tc_iou == 1, label_tc_iou == 1)
            edice[1] += dice_coeff(prediction_tc_dice, label_tc_dice).item()

            prediction_et = prediction.clone()
            prediction_et[prediction_et == 1] = 0
            prediction_et[prediction_et == 3] = 1
            prediction_et_iou = prediction_et.numpy().astype('int32')
            label_et = label.clone()
            label_et[label_et == 1] = 0
            label_et[label_et == 3] = 1
            label_et_iou = label_et.numpy().astype('int32')
            prediction_et_dice = (prediction_et == 1).float()
            label_et_dice = (label_et == 1).float()
            eiou[2] += iou(prediction_et_iou == 1, label_et_iou == 1)
            edice[2] += dice_coeff(prediction_et_dice, label_et_dice).item()

    iou_m = [x / len(testloader) for x in eiou]
    dice_m = [x / len(testloader) for x in edice]
    logging.info("WT: Mean IOU:{},Mean Dice:{}".format(iou_m[0], dice_m[0]))
    logging.info("TC: Mean IOU:{},Mean Dice:{}".format(iou_m[1], dice_m[1]))
    logging.info("ET: Mean IOU:{},Mean Dice:{}".format(iou_m[2], dice_m[2]))
    return "Testing Finished!"

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '/devdata/wujunhao_data/data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'Tumor': {
            'Dataset': Tumor,
            'volume_path': '/home/xinh1/junhao/TransUNet/data_new',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'Tumor_fusion': {
            'Dataset': Tumor,
            'volume_path': '/home/xinh1/junhao/TransUNet/data_small',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'Brats': {
            'Dataset': Brats,
            'volume_path': '/devdata/wujunhao_data/data/brats/annotation_new.json',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'Brats_fusion': {
            'Dataset': Brats,
            'volume_path': '/devdata/wujunhao_data/data/brats/annotation_new.json',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 4,
            'z_spacing': 1,
        },

    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "/home/wujunhao/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    if dataset_name=='Tumor_fusion' or dataset_name=='Brats_fusion':
        from networks.vit_seg_modeling_fusion import VisionTransformer as ViT_seg_fusion
        net = ViT_seg_fusion(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    else:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()


    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    snapshot = '/home/xinh1/junhao/TransUNet/model/TU_Tumor_all224/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs24_224WLI_dataset2/epoch_89.pth'
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    if dataset_name=='Synapse':
        inference(args, net, test_save_path)
    elif dataset_name=='Tumor':
        inference_tumor(args, net, test_save_path)
    elif dataset_name=='Tumor_fusion':
        inference_tumor_fusion(args, net, test_save_path)
    elif dataset_name=='Brats':
        inference_brats(args,net, test_save_path)
    elif dataset_name=='Brats_fusion':
        inference_brats_fusion(args,net, test_save_path)
# python test.py --dataset Tumor_fusion --vit_name R50-ViT-B_16