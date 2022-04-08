import torch
import numpy as np
import copy
import math

'''
    采样方式为每k个点取一个，将一张图分为k份，并保存
'''

K = 4  # 图像降采样份数

def downsampling_Equidistant_4(img):
    '''
    :param
        img: tensor(B,C,H,W)
        k: 分割份数，偶数, 保证是行和列数的因子
    :return: k arrays
    '''
    B, C, H, W = img.shape[:]
    downsampling_list = []
    for j_h in range(2):
        for j_w in range(2):
            idx_h = np.array([i*2+j_h for i in range(H//2)])
            idx_w = np.array([i*2+j_w for i in range(W//2)])
            coords_w, coords_h = np.meshgrid(idx_w, idx_h)
            coords_h, coords_w = np.array(coords_h), np.array(coords_w)
            down_h, down_w = coords_h.shape[0], coords_h.shape[1]  # 下采样后的长和宽
            coords_h = coords_h.reshape(coords_h.shape[0]*coords_h.shape[1])
            coords_w = coords_w.reshape(coords_w.shape[0]*coords_w.shape[1])
            down_img_list = []
            for i in range(B):
                img_i = img[i, ...]  # 第i张图像
                img_i = img_i.permute(1, 2, 0).contiguous()
                down_img_i = img_i[(coords_h, coords_w)]
                down_img_i = down_img_i.reshape(1, down_h, down_w, C)
                down_img_list.append(down_img_i)
            down_img = torch.cat(down_img_list, dim=0)
            down_img = down_img.permute(0,3,1,2)
            downsampling_list.append(down_img)
    return downsampling_list

def combine_flow(flow_list, conf_list, H_patch, W_patch):
    '''
        默认只有一张图片输入
        因为推理前对图像进行了pad，所以需要使用H_patch, W_patch对光流进行裁剪
    '''
    # 裁剪光流和confidence
    B, C, H, W = flow_list[0].shape
    cut_H = (H-H_patch)//2
    cut_W = (W-W_patch)//2
    for i in range(len(flow_list)):
        if cut_H==0:
            flow_list[i] = flow_list[i][::, ::, ::, cut_W:-cut_W]
            conf_list[i] = conf_list[i][::, ::, cut_W:-cut_W]
        elif cut_W==0:
            flow_list[i] = flow_list[i][::, ::, cut_H:-cut_H, ::]
            conf_list[i] = conf_list[i][::, cut_H:-cut_H, ::]
        else:
            flow_list[i] = flow_list[i][::, ::, cut_H:-cut_H, cut_W:-cut_W]
            conf_list[i] = conf_list[i][::, cut_H:-cut_H, cut_W:-cut_W]
    # 将4*4=16个光流按照confidence合并起来
    B, C, H, W = flow_list[0].shape
    device = flow_list[0].device
    # 首先每4个按照confidence合并为一个patch
    conf_list = [torch.cat(conf_list[i*4:(i+1)*4])
                     .permute(1,2,0).view(1,H,W,4).contiguous() for i in range(4)]
    flow_list = [torch.cat(flow_list[i*4:(i+1)*4])
                     .permute(2,3,0,1).contiguous() for i in range(4)]
    conf = torch.cat(conf_list, dim=0)  # (4, H, M, 4)
    # 然后生成一个(4,H,M)的tensor，存储着每四个patch中置信度最高的索引
    _, best_flow_idx = torch.max(conf, dim=-1)  # (4,H,M)
    # 使用meshgrid生成的坐标矩阵
    coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(4, 1, 1, 1)
    coords = coords.permute(0, 2,3,1)
    coords = torch.cat((coords[..., 1].view(4, H, W, 1), coords[..., 0].view(4, H, W, 1)), dim=-1)
    best_flow_idx = best_flow_idx.view(4,H,W,1)  # (4,H,W,1)
    best_coords = torch.cat((coords, best_flow_idx), dim=-1)
    best_coords = best_coords.long()  # 索引需要是long类型

    # 利用坐标索引到最优的光流值
    best_flow_list = []
    for i in range(len(flow_list)):
        flow = flow_list[i]
        best_coord = best_coords[i].view(-1, 3)
        best_flow = flow[best_coord[:, 0], best_coord[:, 1], best_coord[:, 2]]  # (H,W,2)
        best_flow = best_flow.view(H, W, -1)
        best_flow_list.append(best_flow)

    # 最后将这四个光流patch合并为完整光流
    combined_flow = combine_infer_flow_4(best_flow_list)
    return combined_flow



def combine_infer_flow_4(infer_flow_list):
    '''
    将子图合并成原图
    Args:
        infer_flow_list: 子图列表，ndarray

    Returns:
        img_combined： 合并后的整图，ndarray
    '''
    k = 2
    device = infer_flow_list[0].device
    # 光流扩大k倍
    flow_list_temp = []
    dtype = None
    for infer_flow in infer_flow_list:
        infer_flow = infer_flow * k
        dtype = infer_flow.cpu().numpy().dtype
        flow_list_temp.append(infer_flow)
    infer_flow_list = flow_list_temp
    # 初始化合并后的图
    img_combined = np.zeros((infer_flow_list[0].shape[0] * k, infer_flow_list[0].shape[1] * k,
                             infer_flow_list[0].shape[2]), dtype=dtype)  # 必须在这里设置好dtype，否则结果全白
    img_combined = torch.from_numpy(img_combined).cuda(device)

    img_idx = 0
    for pop_c in range(k):
        for pop_r in range(k):
            img_backup = infer_flow_list[img_idx]
            img_w, img_h = img_backup.shape[1], img_backup.shape[0]
            img = copy.deepcopy(img_backup)
            # 需要插入的行和列
            insert_clist = [i for i in range(k)]
            insert_clist.pop(pop_c)  # 除了pop_c列不需要插入
            insert_rlist = [i for i in range(k)]
            insert_rlist.pop(pop_r)  # 除了pop_r行不需要插入
            # 对列进行插入
            for i in range(img_w):
                for insert_r in insert_rlist:
                    img = torch.cat((img[:, :i*k+insert_r, :],
                                 torch.zeros((img.shape[0], 1, img.shape[2])).cuda(device),
                                 img[:, i*k+insert_r:, :]),dim=1)
            # 对行进行插入
            for i in range(img_h):
                for insert_c in insert_clist:
                    img = torch.cat((img[:i*k+insert_c, :, :],
                                 torch.zeros((1, img.shape[1], img.shape[2])).cuda(device),
                                 img[i*k+insert_c:, :, :]),dim=0)
            img_combined += img
            img_idx += 1

    img_combined = img_combined.cpu().numpy()
    return img_combined