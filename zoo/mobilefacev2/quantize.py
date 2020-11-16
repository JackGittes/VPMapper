import torch
from torch.utils.data import dataloader
import utils.pipeline as pipeline
from utils.loader import CalibrationData
from zoo.mobilefacev2.mbv2 import MBV2Face


def search_quantization(params_path, calibration_path=None):
    data_set = CalibrationData(calibration_path)
    loader = dataloader.DataLoader(data_set, num_workers=8,
                                   batch_size=100, shuffle=False)

    net = MBV2Face()
    net.load_state_dict(torch.load(params_path))

    q = pipeline.Quantize(bit_width=8, loader=loader,
                          use_gpu=True, q_method='Naive')
    q.apply(net)


if __name__ == '__main__':
    calibration_root = ''
    search_quantization(calibration_root)
