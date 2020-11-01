import torch


if __name__ == '__main__':
    from utils.loader import CalibrationData
    from torch.utils.data import DataLoader
    data_set = CalibrationData(folder='H:\Dataset\Face\calib')
    loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    from utils import pipeline

    from model.mbv2 import MBV2Face

    net = MBV2Face()
    net.load_state_dict(torch.load('./model/039.pth')['state_dict'])
    q = pipeline.Quantize(bit_width=8, loader=loader,
                          use_gpu=True, q_method='MSE')
    q.apply(net)
