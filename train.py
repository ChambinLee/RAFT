from __future__ import print_function, division
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
os.chdir(curPath)
sys.path.append('core')

import argparse
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    with open("logs/train.log","a") as f:
        s = "flow_l: " + str(round(flow_loss.item(), 3)).ljust(10) + \
            "max_fl_pr: " + str(round(torch.max(flow_preds[-1]).item(), 3)).ljust(10) + \
            "max_fl_gt_v: " + str(round(torch.max(flow_gt).item(), 3)) + "\n"
        f.write(s)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def sequence_loss_conf(flow_preds, flow_gt, flow_conf, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    # valid 应该是用于稀疏光流的
    B, H, W = flow_conf[0].shape
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    conf_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()  # (b,h,w), gt光流长度
    valid = (valid >= 0.5) & (mag < max_flow)  # 筛选可用光流，(b,h,w)，bool

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        ################## 计算confidence loss ####################
        flow_dist = torch.sum((flow_preds[i] - flow_gt).abs().permute(0,2,3,1), dim=-1)
        flow_dist_norm = flow_dist/torch.sum(flow_dist)  # 都是正数

        conf_gt = torch.max(flow_dist_norm)*2 - flow_dist_norm  # 依然都是正数
        conf_gt_norm = conf_gt/torch.sum(conf_gt)

        conf = flow_conf[i]  # 现在conf_gt和flow_conf都是（b, h*w）大小的了
        conf_norm = conf/torch.sum(conf)

        conf_loss += i_weight * \
                     (torch.norm(valid[:, None] * (conf_gt_norm - conf_norm),  # 过滤可用光流
                                 p=2)) * 5000  # 置信度误差为conf_gt-conf的二范数
        ##########################################################
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()  # (b,h,w)
    epe = epe.view(-1)[valid.view(-1)]  # (b*h*w,)
    with open("logs/train_downsampling.log","a") as f:
        s = "flow_l: " + str(round(flow_loss.item(), 3)).ljust(10) + \
            "conf_l: " + str(round(conf_loss.item(), 3)).ljust(10) + \
            "avg_fl_pr: " + str(round(torch.mean(valid[:, None] * flow_preds[-1]).item(), 3)).ljust(10) + \
            "max_fl_pr: " + str(round(torch.max(flow_preds[-1]).item(), 3)).ljust(10) + \
            "avg_fl_gt_v: " + str(round(torch.mean(valid[:, None] * flow_gt).item(), 3)).ljust(10) + \
            "max_fl_gt: " + str(round(torch.max(flow_gt).item(), 3)) + "\n"
        f.write(s)
    total_loss = flow_loss + conf_loss

    metrics = {
        'total_loss': total_loss.item(),
        'flow_loss': flow_loss.item(),
        'conf_loss': conf_loss.item(),
        'epe': epe.mean().item(),  # 整体的epe loss，后面是不同错误程度的epe统计
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return total_loss, metrics, flow_loss, conf_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        # 为什么还没开始训练就让我提供模型参数，那怎么在自己的数据集上训练呢？
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000  # 模型每更新5000次保存一次参数
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]  # image, flow: (b,3,h,w), valid: (b,h,w)

            if args.add_noise:  # 对图像加噪声，数据增强？
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            if not args.noconfidence:  # 进行置信度估计
                flow_predictions, flow_confidence = model(image1, image2, iters=args.iters)
                loss, metrics, flow_loss, conf_loss = sequence_loss_conf(flow_predictions, flow, flow_confidence, valid, args.gamma)
            else:  # 原始网络，不进行置信度估计
                flow_predictions = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 梯度截断
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')  # 测试集位置，参数的值可以有多个，以列表形式存储

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--noconfidence', action='store_true', help='not add confidence prediction')

    parser.add_argument('--iters', type=int, default=12)  # GPU循环次数
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)  # 梯度截断
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')  # 指数加权，用于loss计算
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    # args.alternate_corr = True  # JIT的查找表

    torch.manual_seed(1234)
    np.random.seed(1234)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)