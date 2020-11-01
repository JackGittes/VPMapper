"""
    Author: Zhao Mingxin
    Date:   2020/09/08
    Description: MobileNetv2 backbone for face verification.
"""
import torch
import torch.nn as nn
from model.backbone.mobilenetv2 import MobileNetV2


def fuse_bn(c_layer, b_layer):
    assert isinstance(c_layer, nn.Conv2d)
    assert isinstance(b_layer, nn.BatchNorm2d)

    mu_, var_, gamma, beta = b_layer.running_mean, b_layer.running_var, b_layer.weight, b_layer.bias
    eps_ = b_layer.eps
    std_ = torch.sqrt(var_ + eps_)
    fused_w = c_layer.weight * gamma[:, None, None, None] / std_[:, None, None, None]

    if not c_layer.bias:
        fused_b = beta - mu_ * gamma / std_
    else:
        fused_b = c_layer.bias * gamma[:, None, None, None] / std_[:, None, None, None] + beta - mu_ * gamma / std_

    c_layer.weight = nn.Parameter(fused_w)
    c_layer.bias = nn.Parameter(fused_b)


class MBV2Face(nn.Module):
    def __init__(self):
        super(MBV2Face, self).__init__()
        _net = MobileNetV2(width_mult=1)
        # _net.features[14].conv[3].stride = (1, 1)
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


def inference_wrapper(mb_net):
    fuse_bn(mb_net.backbone[0][0], mb_net.backbone[0][1])
    mb_net.backbone[0] = nn.Sequential(
        mb_net.backbone[0][0],
        nn.ReLU6(inplace=True),
    )
    fuse_bn(mb_net.backbone[1].conv[0], mb_net.backbone[1].conv[1])
    fuse_bn(mb_net.backbone[1].conv[3], mb_net.backbone[1].conv[4])
    mb_net.backbone[1].conv = nn.Sequential(
        mb_net.backbone[1].conv[0],
        nn.ReLU6(inplace=True),
        mb_net.backbone[1].conv[3]
    )
    for i in range(2, 18):
        fuse_bn(mb_net.backbone[i].conv[0], mb_net.backbone[i].conv[1])
        fuse_bn(mb_net.backbone[i].conv[3], mb_net.backbone[i].conv[4])
        fuse_bn(mb_net.backbone[i].conv[6], mb_net.backbone[i].conv[7])
        mb_net.backbone[i].conv = nn.Sequential(
            mb_net.backbone[i].conv[0],
            nn.ReLU6(inplace=True),
            mb_net.backbone[i].conv[3],
            nn.ReLU6(inplace=True),
            mb_net.backbone[i].conv[6],
        )
    fuse_bn(mb_net.expand[0], mb_net.expand[1])
    fuse_bn(mb_net.expand[3], mb_net.expand[4])
    mb_net.expand = nn.Sequential(
        mb_net.expand[0],
        nn.ReLU(inplace=True),
        mb_net.expand[3],
        nn.ReLU(inplace=True),
        mb_net.expand[6],
    )


if __name__ == "__main__":
    net = MBV2Face()
    net.load_state_dict(torch.load('./039.pth')['state_dict'])
    inference_wrapper(net)
    print(net)
    from torchsummary import summary
    summary(net, (3, 128, 128), device='cpu')

