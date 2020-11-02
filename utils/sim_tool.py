import torch.nn as nn
from .pipeline import search_replace_convolution2d, mark_layer
from .absorb_bn import search_absorb_bn
from .layer import QConv2d


class Simulation(object):
    def __init__(self, bit_width=8, mac_width=32, author='Zhao Mingxin'):
        self.bit_width = bit_width
        self.mac_width = mac_width
        self.name = author

    def convert(self, model):
        assert isinstance(model, nn.Module)
        marked = False
        for m in model.modules():
            if isinstance(m, QConv2d):
                marked = True
                break
        if not marked:
            search_absorb_bn(model)
            search_replace_convolution2d(model, self.bit_width)
            mark_layer(model, 'root')

        print("=============================================================")
        print("Start converting quantized layers to MATLAB code.")

        pre_code = [""]
        pre_code[0] += hint()
        pre_code[0] += author_info(self.name)
        pre_code[0] += time_stamp()
        pre_code[0] += main_func_def()

        pre_code[0] += arithmetic(self.mac_width)
        generate(model, pre_code, [1])
        pre_code[0] += "\nend\n"

        pre_code[0] += aux_func_def(self.bit_width)
        with open('./template.m', 'w') as fp:
            fp.writelines(pre_code[0])

        print("MATLAB code converted. Check and correct the generated code.")
        print("=============================================================")


def hint():
    info = "%{ \n" \
           "\tThis is an AUTO-GENERATED network template. \n" \
           "\tThere is no guarantee for correctness. To start a \n" \
           "\tsimulation, check the topology with original network and \n" \
           "\tadd necessary operations in the below code.\n" \
           "%} \n"
    return info


def author_info(author_name):
    info = "% Author:  {}\n".format(author_name)
    return info


def time_stamp():
    from datetime import datetime
    info = "% Date:\t" + datetime.today().strftime('%Y-%m-%d') + "\n"
    return info


def main_func_def():
    res = "\nfunction im = Network(im, nn, net)\n"
    return res


def aux_func_def(bit_width):
    max_v = 2 ** (bit_width-1) - 1
    min_v = - 2 ** (bit_width - 1)
    res = "\nfunction res = cast_int(im, mul, sft) \n" \
          "\t im = im * mul;\n" \
          "\t im = bitshift(im, -sft);\n" \
          "\t im(im > {}) = {};\n" \
          "\t im(im < {}) = {};\n" \
          "\t res = im;\n" \
          "end\n".format(max_v, max_v, min_v, min_v)
    return res


def arithmetic(mac_width):
    res = "\tf = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... \n\t" \
          "'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength', {}, " \
          "... \n\t 'ProductFractionLength', 0, 'SumWordLength', {}, 'SumFractionLength', 0); \n\t" \
          "t = numerictype('WordLength', {}, 'FractionLength', 0); \n".format(mac_width,
                                                                                mac_width,
                                                                                mac_width)
    return res


def conv2d(weight_name, stride, padding, im_name='im')->str:
    assert padding in ['same', 'valid']
    _stride = "[{}, {}]".format(str(stride[0]), str(stride[1]))
    padding = ["'SAME'", "'VALID'"][padding == 'valid']
    return "im = nn.Conv2d({}, {}, t, f, {}, {});".format(im_name,
                                                          weight_name,
                                                          _stride,
                                                          padding)


def depth_wise_conv(weight_name, stride, padding)->str:
    """
    Generator for DepthwiseConv2d in FAST-CNN.
    Prototype: DepthwiseConv2d(im,ker,t,f,stride,padding_method)
    :return:
    """
    assert padding in ['same', 'valid']
    _stride = "[{}, {}]".format(str(stride[0]), str(stride[1]))
    padding = ["'SAME'", "'VALID'"][padding == 'valid']
    return "im = nn.DepthwiseConv2d(im, {}, t, f, {}, {});".format(weight_name,
                                                                  _stride,
                                                                  padding)


def point_wise_conv(weight_name)->str:
    """
    Generator for PointwiseConv2d in FAST-CNN.
    Prototype: PointwiseConv2d(im,ker,t,f)
    :return:
    """
    return "im = nn.PointwiseConv2d(im, {}, t, f);".format(weight_name)


def add_bias(bias_name)->str:
    return "im = nn.AddBias(im, {}, t, f);".format(bias_name)


def relu(im_name='im')->str:
    """
    Generator for DepthwiseConv2d in FAST-CNN.
    Prototype: DepthwiseConv2d(obj,im,ker,t,f,stride,padding_method)
    :param im_name:
    :return:
    """
    return "im = nn.ReLU({});".format(im_name)


def mul_shift(mul, shift)->str:
    tmp = "im = im * {}; \n".format(mul)
    tmp += "im = bitshift(im, -{}); \n".format(shift)
    return tmp


def generate(model, pre_code, cnt)->None:
    for m in model.children():
        if isinstance(m, QConv2d):
            pre_code[0] += "\n% --- Layer: {}\n".format(m.name)
            stride = (int(m.stride[0]), int(m.stride[1]))
            if m.padding[0] > 0 or m.padding[1] > 0:
                padding = 'same'
            else:
                padding = 'valid'
            if m.groups > 1:
                pre_code[0] += ("\t" + depth_wise_conv("net{{{}}}.Weight".format(cnt[0]),
                                                       stride, padding))
            elif m.weight.size(2) == 1 and m.weight.size(3):
                pre_code[0] += ("\t" + point_wise_conv("net{{{}}}.Weight".format(cnt[0])))
            else:
                pre_code[0] += ("\t" + conv2d("net{{{}}}.Weight".format(cnt[0]),
                                              stride, padding))
            pre_code[0] += '\n'

            if m.bias is not None:
                pre_code[0] += ("\t" + add_bias("net{{{}}}.Bias".format(cnt[0])))
                pre_code[0] += '\n'

            pre_code[0] += ("\t" + "im = cast_int(im, net{{{}}}.Mul, "
                                   "net{{{}}}.Shift);\n".format(cnt[0], cnt[0]))

            cnt[0] += 1
        if isinstance(m, nn.ReLU) or isinstance(m, nn.ReLU6):
            pre_code[0] += ("\t" + relu())
            pre_code[0] += '\n'

        else:
            generate(m, pre_code, cnt)
