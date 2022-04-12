import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
os.chdir(curPath)

sys.path.append('core')

import argparse
import matplotlib.pyplot as plt
import glob
import numpy as np
import torch
from PIL import Image
import cv2
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz, frame_utils
from utils.utils import InputPadder
from utils.sampling import downsampling_Equidistant_4, combine_flow
from pathlib import Path

import traceback

DEVISE_IDS = [0]
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # 为什么要permute？
    return img[None].cuda()

def EPE(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=2).mean()

def viz(img, flo_pr, flo_gt, img_name):
    img = img[0].permute(1,2,0).cpu().numpy()

    # 计算epe
    epe = EPE(torch.tensor(flo_pr,dtype=torch.float64), torch.tensor(flo_gt,dtype=torch.float64))

    # map flow to rgb image
    flo_pr = flow_viz.flow_to_image(flo_pr)
    flo_gt = flow_viz.flow_to_image(flo_gt)

    # 将epe写到flo_pr上
    font = cv2.FONT_HERSHEY_SIMPLEX
    flo_pr = cv2.putText(flo_pr, "epe: "+str(round(epe.item(), 3)), (20, 60), font, 2, (0, 0, 0), 2)

    img_flo = np.concatenate([img, flo_pr, flo_gt], axis=0)
    plt.imsave('{0}/{1}'.format(args.output_dir, img_name), img_flo[:, :, [0, 1, 2]] / 255.0)


def demo(args):
    # torch.nn.DataParallel包装模型之后会自动将一个batch切成多份交给不同的GPU进行并行计算。
    # 可以通过参数 device_ids=[0, 1, 2]这样指定参与并行计算的显卡数
    model = RAFT(args)
    model = torch.nn.DataParallel(model, device_ids=DEVISE_IDS)  # 加载模型
    model.load_state_dict(torch.load(args.model))  # 加载模型参数
    model.cuda()
    model.eval()  # 测试

    with torch.no_grad():
        frames = glob.glob(os.path.join(args.frames_path, '*.png')) + \
                 glob.glob(os.path.join(args.frames_path, '*.jpg'))
        flows = glob.glob(os.path.join(args.flows_path, '*.flo'))
        frames = sorted(frames)  # 这个函数好啊，直接对glob的图像进行排序
        flows = sorted(flows)
        for imfile1, imfile2, flofile in tqdm(zip(frames[:-1], frames[1:], flows), total=len(flows)):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            flo_gt = frame_utils.read_gen(flofile)

            # 将图像下采样成4份
            down1_list = downsampling_Equidistant_4(image1)
            down2_list = downsampling_Equidistant_4(image2)
            flow_up_list = []  # 总共4*4=16个光流patch
            conf_up_list = []
            for i in range(len(down1_list)):
                for j in range(len(down2_list)):
                    # 默认是有confidence的，就不加if了
                    patch1, patch2 = down1_list[i], down2_list[j]
                    _, _, H_patch, W_patch = patch1.shape
                    # 将图像pad成8的整数倍，横纵都向上pad，因为提特征后分辨率缩小八倍
                    padder = InputPadder(patch1.shape)
                    patch1, patch2 = padder.pad(patch1, patch2)
                    # inference
                    if not args.noconfidence:
                        _, flow_up, conf_up = model(patch1, patch2, iters=20, test_mode=True)
                    else:
                        _, conf_up = model(patch1, patch2, iters=20, test_mode=True)
                    # viz(patch1,
                    #     flow_up.permute(2,3,1,0).view(flow_up.shape[2], flow_up.shape[3], flow_up.shape[1]).cpu().numpy(),
                    #     "{0}_{1}{2}.png".format(str(Path(imfile1).stem),i,j))
                    flow_up_list.append(flow_up)
                    conf_up_list.append(conf_up)
            # 合并光流patch
            flow = combine_flow(flow_up_list, conf_up_list, H_patch, W_patch)

            img_name = str(Path(imfile1).name)
            viz(image1, flow, flo_gt, img_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help="dir to store results")
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--frames_path', help="frames for evaluation")
    parser.add_argument('--flows_path', help="flows for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    args.noconfidence = False

    demo(args)
