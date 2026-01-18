import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms


def trainer_lc_fusion(args, model, snapshot_path):
    from datasets.dataset_lc import LC, Resize_lc, ToTensor_lc
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')


    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    db_train = LC(data_path=args.data_path, mode='Training',
                               transform=transforms.Compose(
                                   [Resize_lc(args.img_size),ToTensor_lc()]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.epochs
    max_iterations = args.epochs * len(trainloader)  
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=120)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_b_batch, label_b_batch,image_n_batch = sampled_batch['image_b'], sampled_batch['label_b'],sampled_batch['image_n']
            image_b_batch, label_b_batch,image_n_batch = image_b_batch.cuda(), label_b_batch.cuda(),image_n_batch.cuda()
            outputs,fd_loss, da_loss = model(image_b_batch, image_n_batch)
            
            loss_ce = ce_loss(outputs, label_b_batch[:].long())
            loss_dice = dice_loss(outputs, label_b_batch, softmax=True)
            weights_da = 0.0001
            weights_fd = min(0.05, epoch_num / max_epoch * 0.1)
            weights_ce = 0.5
            weights_dice = 0.5
            
            loss = weights_ce * loss_ce + weights_dice * loss_dice + weights_fd * fd_loss + weights_da*da_loss
            optimizer.zero_grad()
            loss.backward() 

            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            iterator.set_postfix({
                "loss": f"{loss.item():.6g}",
                "ce": f"{loss_ce.item():.6g}",
                "dice": f"{loss_dice.item():.6g}",
                "fd": f"{fd_loss:.6g}",
                "da": f"{da_loss:.6g}"
            })


            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, fd: %f, da: %f' % (
                    iter_num, loss.item(), loss_ce.item(), loss_dice.item(), fd_loss,da_loss))


        save_interval = int(max_epoch/6)  
        if  (epoch_num + 1) % save_interval == 0 or (epoch_num + 1) >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            if (epoch_num + 1) >= max_epoch - 1:
                iterator.close()
                break

    writer.close()
    return "Training Finished!"
