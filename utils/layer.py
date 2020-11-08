"""
    Author: Zhao Mingxin
    Date:   2020/10/31
    Description: Quantized Layers for Post-training Quantization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from .base import pure_quantize


class QuantizeLayer(nn.Module):
    """
    self.true_quantize = True means it will not de-quantize the output.
    self.true_quantize = False means the output will be de-quantized to
    floating-point numbers.
    """
    def __init__(self, bit_width, s):
        super(QuantizeLayer, self).__init__()
        self.bit_width = bit_width
        self.s = nn.Parameter(s)
        self.up_bound = 2**(bit_width-1)-1
        self.low_bound = -2**(bit_width-1)

        self.true_quantize = False

    def forward(self, x):
        if not self.true_quantize:
            tmp = torch.clamp(x * self.s, self.low_bound, self.up_bound)
            return torch.round(tmp) / self.s
        else:
            return torch.round(torch.clamp(x * self.s,
                                           self.low_bound, self.up_bound))


class QTemplate(nn.Module):
    """
    The q_inference and q_inference_with_output properties are utilized as
    indicators to change the forward behavior of QConv2d.

    Although two indicators combine to produce four states (2x2=4), we merely
    allow three of these states as below:

    (1) q_inference = True / q_inference_with_output = False
        In this state, the integer quantization simulation is adopted, which means
        we first quantize weights and input using QuantizeLayer, then
        de-quantize them with same scaling factors.

        We use this state in optimization stage to find the optimal quantization
        parameters.

    (2) q_inference = False / q_inference_with_output = True
        In this state, the full integer inference is turned on, where we quantize
        weight, input, and bias, with corresponding scaling factors. We DO NOT
        de-quantize them such that the output is also quantized. The multiplier
        and shift are used to re-scale the output to a small numeric range.

        More details can be found in the 8-bit quantization paper:

    (3) q_inference = False / q_inference_with_output = False
        In this state, the true floating-point inference is active.

    (4) q_inference = True / q_inference_with_output = True
        Meaningless and NOT ALLOWED.
    """
    def __init__(self):
        super(QTemplate, self).__init__()
        self.q_inference = True  # Only for quantization simulation.
        self.q_inference_with_output = False  # With multiplier and shift.

        # Flag, do not change.
        self.quantized = False
        self.reset_quantization()

        self.saturated = True

    def forward(self, *x):
        pass

    def use_full_quantization(self):
        self.q_inference_with_output = True
        self.q_inference = False

    def use_quantization_simulation(self):
        self.q_inference_with_output = False
        self.q_inference = True

    def reset_quantization(self):
        self.q_inference = False
        self.q_inference_with_output = False


class QConv2dv1(nn.Module):
    def __init__(self, conv, bit_width, sx=1.0, sw=1.0, mul=1.0, shift=0.0):
        super(QConv2d, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        self.bit_width = bit_width

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

        self.mul = nn.Parameter(torch.tensor(mul))
        self.shift = nn.Parameter(torch.tensor(shift))

        self.q_inference = True  # Only for quantization simulation.
        self.q_inference_with_output = False  # With multiplier and shift.

        # Flag, do not change.
        self.quantized = False
        self.reset_quantization()

        self.saturated = True

    def forward(self, x):
        assert not (self.q_inference and self.q_inference_with_output), "Ambiguous " \
                                                                 "configuration."
        if self.q_inference:
            self.w_quantizer.true_quantize = False
            self.x_quantizer.true_quantize = False
            q_weight = self.w_quantizer(self.weight)
            q_x = self.x_quantizer(x)
            tmp_bias = self.bias

        elif self.q_inference_with_output:
            self.w_quantizer.true_quantize = True
            q_weight = self.w_quantizer(self.weight)
            q_x = x
            tmp_bias = pure_quantize(self.bias,
                                     self.w_quantizer.s * self.x_quantizer.s,
                                     self.bit_width * 2)

        else:  # No quantization, no simulation.
            q_weight = self.weight
            q_x = x
            tmp_bias = self.bias

        conv_res = func.conv2d(q_x, q_weight, bias=tmp_bias,
                               stride=self.stride,
                               padding=self.padding,
                               groups=self.groups)
        if self.q_inference_with_output:
            if self.saturated:
                tmp_res = torch.floor(conv_res * self.mul / (2 ** self.shift))
                tmp_res[tmp_res > (2**(self.bit_width - 1)-1)] = 2**(self.bit_width - 1)-1
                tmp_res[tmp_res < -2 ** (self.bit_width - 1)] = -2 ** (self.bit_width - 1)
            else:
                tmp_res = conv_res
            return tmp_res
        else:
            return conv_res

    def use_full_quantization(self):
        self.q_inference_with_output = True
        self.q_inference = False

    def use_quantization_simulation(self):
        self.q_inference_with_output = False
        self.q_inference = True

    def reset_quantization(self):
        self.q_inference = False
        self.q_inference_with_output = False


class QConv2d(QTemplate):
    """
    The q_inference and q_inference_with_output properties are utilized as
    indicators to change the forward behavior of QConv2d.

    Although two indicators combine to produce four states (2x2=4), we merely
    allow three of these states as below:

    (1) q_inference = True / q_inference_with_output = False
        In this state, the integer quantization simulation is adopted, which means
        we first quantize weights and input using QuantizeLayer, then
        de-quantize them with same scaling factors.

        We use this state in optimization stage to find the optimal quantization
        parameters.

    (2) q_inference = False / q_inference_with_output = True
        In this state, the full integer inference is turned on, where we quantize
        weight, input, and bias, with corresponding scaling factors. We DO NOT
        de-quantize them such that the output is also quantized. The multiplier
        and shift are used to re-scale the output to a small numeric range.

        More details can be found in the 8-bit quantization paper:

    (3) q_inference = False / q_inference_with_output = False
        In this state, the true floating-point inference is active.

    (4) q_inference = True / q_inference_with_output = True
        Meaningless and NOT ALLOWED.
    """
    def __init__(self, conv, bit_width, sx=1.0, sw=1.0, mul=1.0, shift=0.0):
        super(QConv2d, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        self.bit_width = bit_width

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

        self.mul = nn.Parameter(torch.tensor(mul))
        self.shift = nn.Parameter(torch.tensor(shift))

    def forward(self, x):
        assert not (self.q_inference and self.q_inference_with_output), "Ambiguous " \
                                                                 "configuration."
        if self.q_inference:
            self.w_quantizer.true_quantize = False
            self.x_quantizer.true_quantize = False
            q_weight = self.w_quantizer(self.weight)
            q_x = self.x_quantizer(x)
            tmp_bias = self.bias

        elif self.q_inference_with_output:
            self.w_quantizer.true_quantize = True
            q_weight = self.w_quantizer(self.weight)
            q_x = x
            tmp_bias = pure_quantize(self.bias,
                                     self.w_quantizer.s * self.x_quantizer.s,
                                     self.bit_width * 2)

        else:  # No quantization, no simulation.
            q_weight = self.weight
            q_x = x
            tmp_bias = self.bias

        conv_res = func.conv2d(q_x, q_weight, bias=tmp_bias,
                               stride=self.stride,
                               padding=self.padding,
                               groups=self.groups)
        if self.q_inference_with_output:
            tmp_res = torch.floor(conv_res * self.mul / (2 ** self.shift))
            if self.saturated:
                tmp_res[tmp_res > (2**(self.bit_width - 1) - 1)] = 2 ** (self.bit_width - 1) - 1
                tmp_res[tmp_res < -2 ** (self.bit_width - 1)] = -2 ** (self.bit_width - 1)
            return tmp_res
        else:
            return conv_res


class QAvgPooling(nn.Module):
    def __init__(self, p_layer):
        super(QAvgPooling, self).__init__()
        assert isinstance(p_layer, nn.AvgPool2d)
        self.q_inference = False

        self.stride = p_layer.stride
        self.kernel_size = p_layer.kernel_size
        self.padding = p_layer.padding

    def forward(self, x):
        if self.q_inference:
            # Use torch.floor not torch.round to keep consistent with
            # hardware features.
            return torch.floor(func.avg_pool2d(x, self.kernel_size,
                                               self.stride,
                                               self.padding))
        else:
            return func.avg_pool2d(x, self.kernel_size,
                                   self.stride,
                                   self.padding)

    def __repr__(self):
        """
        This __repr__ method is an identifier for its layer type.
        As implemented in utils.sim_tool.generate, pooling layers
        are converted to MATLAB code by their __repr__ methods. We
        thus define its __repr__ as "AvgPool2dQ" such that it can
        be converted to AvgPooling2d.
        :return:
        """
        return "AvgPool2dQ"


class QAddition(QTemplate):
    """
    The addition layer is a bi-operand layer, which takes two operands as inputs
    and adds them. Therefore, it can also be represented an quantization operator
    in a given network. We denote the addition layer as QAddition layer.

    We use the same strategy as the convolution layer to quantize addition layer.

    First, we collect the input lhs and rhs (i.e. x1, x2) to get the proper scaling
    factors. After that, we re-scale the output based on the cascaded layer's input
    scaling factor.
    """
    def __init__(self, bit_width):
        super(QAddition, self).__init__()
        self.bit_width = bit_width

        self.x_quantizer = QuantizeLayer(bit_width, torch.tensor(1.0))
        self.w_quantizer = QuantizeLayer(bit_width, torch.tensor(1.0))

        self.mul_lhs = nn.Parameter(torch.tensor(1.0))
        self.shift_lhs = nn.Parameter(torch.tensor(0.0))

        self.mul_rhs = nn.Parameter(torch.tensor(1.0))
        self.shift_rhs = nn.Parameter(torch.tensor(0.0))

        self.forward_1 = True

    def forward(self, x1, x2):
        if self.q_inference:
            self.x_quantizer.true_quantize = False
            self.w_quantizer.true_quantize = False
            x1 = self.x_quantizer(x1)
            x2 = self.w_quantizer(x2)
            return x1 + x2
        elif self.q_inference_with_output:
            # out = torch.floor(x1 * self.mul_lhs / (2 ** self.shift_lhs)) + \
            #     torch.floor(x2 * self.mul_rhs / (2 ** self.shift_rhs))
            if self.forward_1:
                out = torch.round((x1 * self.mul_lhs + x2 * self.mul_rhs) / 2 ** self.shift_lhs)
            else:
                out = torch.round(x1 * self.mul_lhs / (2 ** self.shift_lhs) + \
                        x2 * self.mul_rhs / (2 ** self.shift_rhs))

                out = torch.round(x1 * self.mul_lhs / (2 ** self.shift_lhs)) + \
                      torch.round(x2 * self.mul_rhs / (2 ** self.shift_rhs))
            # out = torch.round((x1 * self.mul_lhs + x2 * self.mul_rhs) / 2 ** self.shift_lhs)
            # out = x1 * self.mul_lhs / (2 ** self.shift_lhs) + \
            #     x2 * self.mul_rhs / (2 ** self.shift_rhs)
            # if self.saturated:
            #     out[out > (2 ** (self.bit_width - 1) - 1)] = 2**(self.bit_width - 1) - 1
            #     out[out < -2 ** (self.bit_width - 1)] = -2 ** (self.bit_width - 1)
            return out
        else:
            return x1 + x2


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


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
            _layer.weight.data = child.weight.data.view(out_ch, in_ch, 1, 1)
            composed_layer = nn.Sequential(Reshape((-1, in_ch, 1, 1)),
                                           QConv2d(_layer, bit_width))
            setattr(model, child_name, composed_layer)
        elif isinstance(child, nn.AvgPool2d):
            setattr(model, child_name, QAvgPooling(child))  # Replace AvgPooling
        else:
            """ Recursively search the tree from left child to right child. """
            search_replace_convolution2d(child, bit_width)


def search_turn_off_relu6(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            search_turn_off_relu6(child)
