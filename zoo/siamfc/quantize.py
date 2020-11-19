# -*- coding: utf-8 -*-
"""
Time    : 11/19/20 3:36 PM
Author  : Zhao Mingxin
Email   : zhaomingxin17@semi.ac.cn
File    : quantize.py
Description:
"""

from zoo.siamfc.siamfc import SiamFCTracker
from utils.loader import CalibrationData
from torch.utils.data import DataLoader
from utils import pipeline


def quantize_siamfc(bit_width=8):
    net = SiamFCTracker('./siamfc_lite.pth', 0)
    model = net.model.features

    data = CalibrationData(folder='/media/zhaomingxin/Document1/Dataset/ImageNet/calib',
                           im_size=(255, 255), mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])

    loader = DataLoader(data, num_workers=8, batch_size=100, shuffle=False)

    q = pipeline.Quantize(bit_width=bit_width, loader=loader,
                          use_gpu=True, q_method='Naive')

    q.apply(model)


def generate_params(bit_width=8):
    from utils.pipeline import post_processing
    from utils.sim_tool import Simulation
    net = SiamFCTracker('./siamfc_lite.pth', 0)
    model = net.model.features
    post_processing(model=model, pth_path='./result/pth/quantized.pth',
                    info_path='./report/info.csv', bit_width=bit_width)
    sim = Simulation()
    sim.convert(model)


if __name__ == "__main__":
    # quantize_siamfc()
    generate_params()
