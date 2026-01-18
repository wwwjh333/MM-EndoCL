import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_lc_fusion

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='.../MM-EndoCL', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='LC_fusion_dataset2', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed, default is 1234, others can be 42,3407')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model R50-ViT-B_16')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


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
    dataset_name = args.dataset
    model_name = "LC_fusion" if "fusion" in dataset_name.lower()
    dataset_config = {
        'LC_fusion_dataset1': {
            'data_path': './data/dataset1',
            'num_classes': 2,
        },
        'LC_fusion_dataset2': {
            'data_path': './data/dataset2',
            'num_classes': 2,
        },
        'LC_fusion_dataset3': {
            'data_path': './data/dataset3',
            'num_classes': 2,
        }
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.data_path = dataset_config[dataset_name]['data_path']
    args.is_pretrain = True
    snapshot_path = f"./log/{dataset_name}/epoch_{args.epochs}_bs_{args.batch_size}_lr_{args.base_lr}_s_{args.seed}"


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    
    from networks.vit_seg_modeling_fusion import VisionTransformer as ViT_seg_fusion
    net = ViT_seg_fusion(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    net.load_from1(weights=np.load(config_vit.pretrained_path))

    trainer = {'LC_fusion':trainer_lc_fusion}

    trainer[model_name](args, net, snapshot_path)