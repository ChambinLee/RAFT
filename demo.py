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

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from pathlib import Path

import traceback

DEVISE_IDS = [0]
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # 为什么要permute？
    return img[None].cuda()


def viz(img, flo, img_name):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    plt.imsave('results/{}'.format(img_name), img_flo[:, :, [0, 1, 2]]/255.0)


def demo(args):
    # torch.nn.DataParallel包装模型之后会自动将一个batch切成多份交给不同的GPU进行并行计算。
    # 可以通过参数 device_ids=[0, 1, 2]这样指定参与并行计算的显卡数
    model = RAFT(args)
    model = torch.nn.DataParallel(model, device_ids=DEVISE_IDS)  # 加载模型
    model.load_state_dict(torch.load(args.model))  # 加载模型参数
    model.cuda()
    model.eval()  # 测试

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)  # 这个函数好啊，直接对glob的图像进行排序
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # 将图像pad成8的整数倍，横纵都向上pad，因为提特征后分辨率缩小八倍
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            try:
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)  # 调用RAFT类的forward函数
                print("successed", imfile1, imfile2)
            except:
                print("error", imfile1, imfile2)
                traceback.print_exc()
                return
            img_name = str(Path(imfile1).name)
            viz(image1, flow_up, img_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
