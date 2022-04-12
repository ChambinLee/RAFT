import torch
import numpy as np



def combine_windows_flow_4(infer_flow_list,padder):
    '''
    # 将子图预测的flow合并成原图大小(k=4，总共9个窗口的情况）,使用最简单的方式，将重合位置求均值
    Args:
        infer_flow_list: iters（12）个光流tensor，每个都是tensor(B*9,2,H//2,W//2)
    Returns:
        合并后的光流，tensor(B,2,H,W)
    '''
    if not isinstance(infer_flow_list, list):
        infer_flow_list = [infer_flow_list,]
    combined_flow_list = []
    for infer_flows in infer_flow_list:
        # 删除图像边缘
        pad_infer_flows_list = []
        for i in range(infer_flows.shape[0]):
            pad_infer_flows_list.append(padder.unpad(infer_flows[i]).unsqueeze(0))
        infer_flows = torch.cat(pad_infer_flows_list, dim=0)

        B9, C, H, W = infer_flows.shape
        infer_flows = infer_flows.view(B9//9, 9, C, H, W).contiguous()
        # 窗口大小的一半，这是合并的基本单位
        B, _, C, H, W = infer_flows.shape
        swindow_h = H//2
        swindow_w = W//2

        infer_flows = infer_flows.permute(0, 1, 3, 4, 2).contiguous()

        # 初始化合并后的光流
        device = infer_flows.device
        img_combined = torch.zeros(B, H*2, W*2, C).cuda(device).type(infer_flows.dtype)

        # 对16个小窗口进行赋值
        if True:
            img_combined[::, swindow_h*0:swindow_h*1, swindow_w*0:swindow_w*1, ::] = \
                infer_flows[::, 0, :swindow_h, :swindow_w, ::]
            img_combined[::, swindow_h*0:swindow_h*1, swindow_w*1:swindow_w*2, ::] = \
                (infer_flows[::, 0, :swindow_h, swindow_w:, ::] + infer_flows[::, 1, :swindow_h, :swindow_w, ::]) / 2
            img_combined[::, swindow_h*0:swindow_h*1, swindow_w*2:swindow_w*3, ::] = \
                (infer_flows[::, 1, :swindow_h, swindow_w:, ::] + infer_flows[::, 2, :swindow_h, :swindow_w, ::]) / 2
            img_combined[::, swindow_h*0:swindow_h*1, swindow_w*3:swindow_w*4, ::] = \
                infer_flows[::, 2, :swindow_h, swindow_w:, ::]

            img_combined[::, swindow_h * 1:swindow_h * 2, swindow_w * 0:swindow_w * 1, ::] = \
                (infer_flows[::, 0,swindow_h:, :swindow_w, ::] + infer_flows[::, 3, :swindow_h, :swindow_w, ::]) / 2
            img_combined[::, swindow_h * 1:swindow_h * 2, swindow_w * 1:swindow_w * 2, ::] = \
                (infer_flows[::, 0, swindow_h:, swindow_w:, ::] + infer_flows[::, 1, swindow_h:, :swindow_w, ::] +
                 infer_flows[::, 3, :swindow_h, swindow_w:, ::] + infer_flows[::, 4, :swindow_h, :swindow_w, ::]) / 4
            img_combined[::, swindow_h * 1:swindow_h * 2, swindow_w * 2:swindow_w * 3, ::] = \
                (infer_flows[::, 1, swindow_h:, swindow_w:, ::] + infer_flows[::, 2, swindow_h:, :swindow_w, ::] +
                 infer_flows[::, 4, :swindow_h, swindow_w:, ::] + infer_flows[::, 5, :swindow_h, :swindow_w, ::]) / 4
            img_combined[::, swindow_h * 1:swindow_h * 2, swindow_w * 3:swindow_w * 4, ::] = \
                (infer_flows[::, 2, swindow_h:, swindow_w:, ::] + infer_flows[::, 5, :swindow_h, swindow_w:, ::]) / 2

            img_combined[::, swindow_h * 2:swindow_h * 3, swindow_w * 0:swindow_w * 1, ::] = \
                (infer_flows[::, 3, swindow_h:, :swindow_w, ::] + infer_flows[::, 6, :swindow_h, :swindow_w, ::]) / 2
            img_combined[::, swindow_h * 2:swindow_h * 3, swindow_w * 1:swindow_w * 2, ::] = \
                (infer_flows[::, 3, swindow_h:, swindow_w:, ::] + infer_flows[::, 4, swindow_h:, :swindow_w, ::] +
                 infer_flows[::, 6, :swindow_h, swindow_w:, ::] + infer_flows[::, 7, :swindow_h, :swindow_w, ::]) / 4
            img_combined[::, swindow_h * 2:swindow_h * 3, swindow_w * 2:swindow_w * 3, ::] = \
                (infer_flows[::, 4, swindow_h:, swindow_w:, ::] + infer_flows[::, 5, swindow_h:, :swindow_w, ::] +
                 infer_flows[::, 7, :swindow_h, swindow_w:, ::] + infer_flows[::, 8, :swindow_h, :swindow_w, ::]) / 4
            img_combined[::, swindow_h * 2:swindow_h * 3, swindow_w * 3:swindow_w * 4, ::] = \
                (infer_flows[::, 5, swindow_h:, swindow_w:, ::] + infer_flows[::, 8, :swindow_h, swindow_w:, ::]) / 2

            img_combined[::, swindow_h * 3:swindow_h * 4, swindow_w * 0:swindow_w * 1, ::] = \
                infer_flows[::, 6, swindow_h:, :swindow_w, ::]
            img_combined[::, swindow_h * 3:swindow_h * 4, swindow_w * 1:swindow_w * 2, ::] = \
                (infer_flows[::, 6, swindow_h:, swindow_w:, ::] + infer_flows[::, 7, swindow_h:, :swindow_w, ::]) / 2
            img_combined[::, swindow_h * 3:swindow_h * 4, swindow_w * 2:swindow_w * 3, ::] = \
                (infer_flows[::, 7, swindow_h:, swindow_w:, ::] + infer_flows[::, 8, swindow_h:, :swindow_w, ::]) / 2
            img_combined[::, swindow_h * 3:swindow_h * 4, swindow_w * 3:swindow_w * 4, ::] = \
                infer_flows[::, 8, swindow_h:, swindow_w:, ::]
        combined_flow_list.append(img_combined.permute(0,3,1,2).contiguous())
    return combined_flow_list

def create_windows_4(imgs, padder):
    '''
    将图像按照2*2的方式裁剪成3*3的窗口，步长分别为长和宽的四分之一
    暂时没用unfold改写
    Args:
        img: tensor(N,C,H,W)
    Returns:
        window_list: 九张子图,tensor(N,9,C,H//2,W//2)
    '''
    window_list = []
    N, C, H, W = imgs.shape  # shape of img
    for i in range(3):
        for j in range(3):
            start_row = i * H // 4  # start row of window
            end_row = (i+2) * H // 4  # end row of window
            start_col = j * W // 4  # start col of window
            end_col = (j+2) * W // 4  # end col of window
            window = imgs[::, ::, start_row:end_row, start_col:end_col]
            window = padder.pad(window)[0]  # pad成8的整数倍
            window_list.append(window)
    img_wids = torch.cat(window_list, 0)  # 将所有patch叠在一起  (B*9,C,H,W)
    return img_wids

#
# for i in range(8):
#     for j in range(9):
#         img = img_wids[i,j].permute(1,2,0).cpu().numpy()
#         plt.imsave('results/{0}{1}.png'.format(i,j), img / 255.0)

# import matplotlib.pyplot as plt
# img_combined = img_combined.cpu().numpy()
# for i in range(8):
#     img = img_combined[i]
#     plt.imsave('results/combined{0}.png'.format(i), img / 255.0)