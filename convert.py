import torch
from model.mobilenet import MobileNet_v1
from model.mobilenet import Conv2dTF


def Convert():
    net = MobileNet_v1()
    dummy_input = torch.rand((1, 3, 224, 224))
    net(dummy_input)
    for m in net.modules():
        if isinstance(m, )