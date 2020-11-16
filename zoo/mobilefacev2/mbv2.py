"""
    Author: Zhao Mingxin
    Date:   2020/09/08
    Description: MobileNetv2 backbone for face verification.
"""
import torch
import torch.nn as nn
from zoo.mobilefacev2.mobilenetv2_q import MobileNetV2, InvertedResidual


class MBV2Face(nn.Module):
    def __init__(self):
        super(MBV2Face, self).__init__()
        _net = MobileNetV2(width_mult=1)  # Make the backbone
        self.backbone = _net.features[:18]
        self.expand = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=1, bias=False,
                      groups=512)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.expand(x)
        x = x.view(-1, 512)
        return x


if __name__ == "__main__":
    net = MBV2Face()
    net.load_state_dict(torch.load('./params.pth')['state_dict'])

    from utils.layer import QAddition
    for name, m in net.named_modules():
        if isinstance(m, InvertedResidual):
            m.turn_on_add()

    for name, m in net.named_modules():
        if isinstance(m, QAddition):
            print(name)

    print(net)
    from torchsummary import summary
    summary(net, (3, 128, 128), device='cpu')

