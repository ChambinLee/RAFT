import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, confidence

try:
    autocast = torch.cuda.amp.autocast  # 混合精度训练，可以减少显存占用
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:  # use small model
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4  # 添加新参数
            args.corr_radius = 3
        
        else:  # use bigger model
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:  # 这里的dropout用在模型的哪里
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:  #　use efficent correlation implementation
            self.args.alternate_corr = False  # 有什么用？

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # 测试阶段Batch normalization过程和测试不一样

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """
        Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination
        flow: (b, 2, H//8, W//8)
        mask: (b, 8*8*9, H//8, W//8)
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)  # 光流扩大八倍，(4,18,2852)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def upsample_conf(self, conf, mask):
        '''
        conf: (b, H//8, W//8)
        mask: (b, 8*8*9, H//8, W//8), 即上采样权重
        '''
        B, H, W = conf.shape
        mask = mask.view(B, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        conf = conf.view(B, 1, H, W)
        up_conf = F.unfold(conf, [3, 3], padding=1)  # (B, 9, H*W)
        up_conf = up_conf.view(B, 1, 9, 1, 1, H, W)

        up_conf = torch.sum(mask * up_conf, dim=2)
        up_conf = up_conf.permute(0, 1, 4, 2, 5, 3)
        return up_conf.reshape(B, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # step 1：预处理
        image1 = 2 * (image1 / 255.0) - 1.0  # 图像归一化到[-1,1], (b,3,h,w)
        image2 = 2 * (image2 / 255.0) - 1.0  # 图像归一化到[-1,1]

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # step 2：Feature Encoder 提取两图特征（权值共享）
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])  # (b,256,h//8,w//8)

        # step 3：初始化 Correlation Volumes 相关性查找表
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:  # 这两个函数构造的correlation volumes有什么区别？
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # step 4：Context Encoder 提取第一帧图特征
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)  # (1,256,h//8,w//8)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)  # 这啥意思？把特征的前128纬用tanh激活，后128纬用relu激活（用于GRU的两个输入）
            inp = torch.relu(inp)

        # step 5：更新光流
        # 初始化光流的坐标信息，coords0 为初始时刻的坐标，coords1 为当前迭代的坐标，此处两坐标数值相等
        coords0, coords1 = self.initialize_flow(image1)  # 初始化每个像素的坐标，就是网格坐标meshgrid，(1, 2, 55, 128)

        if flow_init is not None:  # 如果使用 warm start，则光流的初始值是非空的，就用初始光流加上原始的meshgrid
            coords1 = coords1 + flow_init  # 得到的coords1中每个像素像素位置存的值为它在第二帧的新位置

        flow_predictions = []
        flow_confidence = []
        for itr in range(iters):
            coords1 = coords1.detach()  # 拷贝，并脱离原来的计算图
            corr = corr_fn(coords1)  # index correlation volume，(b,2,h//8,w//8)

            # 计算光流置信度
            if not self.args.noconfidence:
                conf = confidence(coords1, corr_fn)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)  # update_block

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # step 6：上采样光流和confidence
            if up_mask is None:  # up_mask: (b, 8*8*9, H//8, W//8), 即上采样权重
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                if not self.args.noconfidence:
                    conf_up = self.upsample_conf(conf, up_mask)  # (B, 8 * H, 8 * W)

            flow_predictions.append(flow_up)
            if not self.args.noconfidence:
                flow_confidence.append(conf_up)
        # test，只需要返回最后一层update模块输出的光流和置信度
        if test_mode:
            if not self.args.noconfidence:
                return coords1 - coords0, flow_up, conf_up
            else:
                return coords1 - coords0, flow_up
        # train，需要输出每一层undate模块产生的光流和置信度用于loss计算
        if not self.args.noconfidence:
            return flow_predictions, flow_confidence
        else:
            return flow_predictions

# from utils import flow_viz
# import matplotlib.pyplot as plt
# flow_down = coords1 - coords0
# flow_down = flow_down[0].permute(1,2,0).detach().cpu().numpy()
# flow_down_img = flow_viz.flow_to_image(flow_down)
# plt.imsave('results/1.png', flow_down_img[:, :, [0, 1, 2]] / 255.0)