import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """
    Wrapper for grid_sample, uses pixel coordinates
    img: 4D correlation volume，(batch*h1*w1, dim, h2, w2)
    coords: (bhw,2r+1,2r+1,2)，对于b张图像，总共bhw个coords，每个coords需要在表中查找 2r+1*2r+1 个值，每个值是uv两维
    """
    H, W = img.shape[-2:]  # 图像大小
    xgrid, ygrid = coords.split([1,1], dim=-1)  # 按最后一纬拆成两个(bhw,2r+1,2r+1,1)的矩阵
    xgrid = 2*xgrid/(W-1) - 1  # 将x坐标归一化到-1~1，归一化中心为网格中心，和align_corners=True保持一致
    ygrid = 2*ygrid/(H-1) - 1  # 将y坐标归一化到-1~1

    grid = torch.cat([xgrid, ygrid], dim=-1)  # x和y分别归一化好之后，再拼接回去(bhw,2r+1,2r+1,2)
    # 将img双线性插值到grid的大小，即将相关性查找表插值到要查找的coords大小。img: (bhw,1,h,w) -> (bhw,1,2r+1,2r+1)
    # 参考https://www.freesion.com/article/4934317912/
    img = F.grid_sample(img, grid, padding_mode="border", align_corners=True)
    # 对于超出画面的光流，赋一个比较大的值，这里选最大值的两倍（注意查询结果越大，相似度越高）
    # 先用padding_mode="border"表示超出的部分使用边界值代替

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def confidence(coords1, corr_fn):
    """
    查询相似度并生成预测的置信度
    coords1: 每一轮update后得到的新位置,coords1 = coords1 + delta_flow,(b,2,h//8,w//8)
    corr: correlation volume 对象
    """
    # 首先进行一次lookup，得到新的coord1中每个特征与窗口邻域特征的相似度
    corr = corr_fn(coords1)  # 每个特征值在四层特征金字塔上查找出4*(2r+1)**2个相似度出来，(b, 4*(2r+1)**2, h//8, w//8)
    # 求窗口内相似度的平均值
    b, _, h, w = coords1.shape
    corr = corr.permute(0, 2, 3, 1)  # (b, h//8, w//8, 4*(2r+1)**2)
    corr_mean = torch.mean(corr, -1)  # （b, h//8, w//8）
    corr_mean = corr_mean.reshape(b, -1)  # 将tensor拉成（b, h//8 * w//8）,便于进行softmax
    # corr_norm = torch.softmax(corr_mean, -1)
    corr_norm = corr_mean/torch.sum(corr_mean)
    return corr_norm.reshape(b, h, w)  # (b, h//8, w//8)，表示每个光流值的置信度
