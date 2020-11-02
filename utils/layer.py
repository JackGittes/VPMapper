"""
    Author: Zhao Mingxin
    Date:   2020/10/31
    Description: Quantized Layers for Post-training Quantization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as func


class QuantizeLayer(nn.Module):
    def __init__(self, bit_width, s):
        super(QuantizeLayer, self).__init__()
        self.bit_width = bit_width
        self.s = nn.Parameter(s)
        self.up_bound = 2**(bit_width-1)-1
        self.low_bound = -2**(bit_width-1)

    def forward(self, x):
        tmp = torch.clamp(x * self.s, self.low_bound, self.up_bound)
        return torch.round(tmp) / self.s


class QConv2d(nn.Module):
    def __init__(self, conv, bit_width, sx=1.0, sw=1.0):
        super(QConv2d, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight)

        if conv.bias is not None and torch.numel(conv.bias) > 0:
            self.bias = nn.Parameter(conv.bias)
        else:
            self.bias = None

        self.w_quantizer = QuantizeLayer(bit_width, torch.tensor(sw))
        self.x_quantizer = QuantizeLayer(bit_width, torch.tensor(sx))

        self.q_inference = False
        self.quantized = False

    def forward(self, x):
        if self.q_inference:
            q_weight = self.w_quantizer(self.weight)
            q_x = self.x_quantizer(x)
        else:
            q_weight = self.weight
            q_x = x
        if self.bias is not None:
            return func.conv2d(q_x, q_weight, bias=None,
                               stride=self.stride,
                               padding=self.padding,
                               groups=self.groups)
        else:
            return func.conv2d(q_x, q_weight, bias=self.bias,
                               stride=self.stride,
                               padding=self.padding,
                               groups=self.groups)


def search_replace_convolution2d(model, bit_width):
    """
    Recursively search the model tree using Depth-First-Search method,
    and convert all convolution and linear layers to QConv2d.
    :param model: the network model.
    :param bit_width: omitted.
    :return: None
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(model, child_name, QConv2d(child, bit_width))
        elif isinstance(child, nn.Linear):
            """ Linear layer should be converted to Conv2d layer."""
            in_ch = child.weight.size(1)
            out_ch = child.weight.size(0)
            if child.bias is not None:  # bias or not
                _layer = nn.Conv2d(in_ch, out_ch, 1, 1, bias=True)
                _layer.bias.data = child.bias.data
            else:
                _layer = nn.Conv2d(in_ch, out_ch, 1, 1, bias=False)
            _layer.weight.data = child.weight.data
            setattr(model, child_name, QConv2d(_layer, bit_width))
        else:
            """ Recursively search the tree from left child to right child. """
            search_replace_convolution2d(child, bit_width)
