# -*- coding: utf-8 -*-
"""
Time    : 9/20/20 1:57 PM
Author  : Zhao Mingxin
Email   : zhaomingxin17@semi.ac.cn
File    : litenet.py
Description:
"""

import torch.nn as nn


class LiteNet(nn.Module):
    def __init__(self):
        super(LiteNet, self).__init__()
        self.features = nn.Sequential(
            # First Layer of AlexNet #
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            # Second Layer #
            nn.MaxPool2d(3, 2),

            # Third Layer #
            nn.Conv2d(96, 96, kernel_size=5, stride=1, groups=96),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Fourth Layer #
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Fifth Layer #
            nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Sixth Layer #
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=384),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Output Layer #
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=384),
            nn.ReLU(384),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=1, stride=1)
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )

    def forward(self, x):
        return self.features(x)


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    net = LiteNet()
    torch.save(net.state_dict(), './net.pth')
    summary(net, (3, 255, 255), device='cpu')
