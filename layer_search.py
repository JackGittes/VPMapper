import torch


if __name__ == '__main__':
    from utils.loader import CalibrationData
    from torch.utils.data import DataLoader
    data_set = CalibrationData(folder="H:/Dataset/Ship_Data/Ship_Four_CLS/calib", im_size=(32, 32))
    loader = DataLoader(data_set, num_workers=8, batch_size=4000, shuffle=False)
    from utils import pipeline

    from model.mbv2 import MBV2Face
    from model.shipnet import ShipNet

    # net = MBV2Face()
    # net.load_state_dict(torch.load('./model/039.pth')['state_dict'])
    # q = pipeline.Quantize(bit_width=8, loader=loader,
    #                       use_gpu=True, q_method='MSE')
    # q.apply(net)

    net = ShipNet()
    net.load_state_dict(torch.load('./model/model_029.pth'))
    q = pipeline.Quantize(bit_width=8, loader=loader,
                          use_gpu=True, q_method='MSE')
    q.apply(net)

    # root = "H:/Dataset/Ship_Data/Ship_Four_CLS/Ship_Four_CLS/train"
    # dst = "H:/Dataset/Ship_Data/Ship_Four_CLS/calib"
    # import os
    # import random
    # import shutil
    # dir_list = os.listdir(root)
    #
    # idx = 0
    # for item in dir_list:
    #     sub_path = os.path.join(root, item)
    #     sub_files = os.listdir(sub_path)
    #     random.shuffle(sub_files)
    #     cnt = 0
    #     for _item in sub_files:
    #         shutil.copy(os.path.join(sub_path, _item), os.path.join(dst, "{:>03d}.jpg".format(idx)))
    #         idx += 1
    #         cnt += 1
    #         if cnt >= 1000:
    #             break
