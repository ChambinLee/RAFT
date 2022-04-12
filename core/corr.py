import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    # 初始化correlation volume
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels  #
        self.radius = radius  # 查找半径
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)  # 对两图特征使用矩阵乘法得到相关性查找表(1,h//8,w//8,1,h//8,w//8)

        batch, h1, w1, dim, h2, w2 = corr.shape  # dim是什么
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)  # 整体相关表
        for i in range(self.num_levels-1):
            # 特征金字塔中其他尺度查找表是在整体相关表最后两纬上求平均得到的
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    # 使用correlation volume查找
    def __call__(self, coords):  # 使得实例化的对象作为函数调用，coords(b,2,h//8,w//8)
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # (b,2,h//8,w//8) -> (b,h//8,w//8,2)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)  # -r, -r+1, ... , r
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)

            # 查找窗 tendor shape为(2r+1,2r+1,2)，窗口大小为(2r+1,2r+1)，窗口上每个位置为其在窗口中的坐标，(0,0)在窗口中心
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  # (b,h//8,w//8,2) -> (b * h//8 * w//8,1,1,2)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)  # (2r+1,2r+1,2) -> (1,2r+1,2r+1,2) 查找窗

            # (b * h//8 * w//8,1,1,2) + (1,2r+1,2r+1,2) -> (b * h//8 * w//8,2r+1,2r+1,2) 可以形象理解为：
            # 对于 b * h//8 * w//8 这么多待查找的点，每一个点需要搜索 (2r+1)*(2r+1) 邻域范围内的其他点，每个点包含 x 和 y 两个坐标值
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)  # ( b,h//8,w//8,(2r+1)**2 )
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        # fmap: (b,256,h//8,w//8)
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        # 特征之间来了个完全的点乘  (b,h//8 * w//8,256)*(b,256,h//8 * w//8) = (b,h//8 * w//8,h//8 * w//8)
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)  # 4D correlation volume
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
