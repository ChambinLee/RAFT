import torch
import numpy as np

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

def combine_flow(flow_list, conf_list):
    # 将4*4=16个光流按照confidence合并起来

    # 首先每4个按照confidence合并为一个patch


    # 然后将这四个光流patch合并为完整光流

    pass