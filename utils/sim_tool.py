import torch.nn as nn
from .pipeline import search_replace_convolution2d, mark_layer
from .absorb_bn import search_absorb_bn
from .layer import QConv2d, QAvgPooling, QAddition
from .helper import check_tuple


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
        pre_code[0] += input_normalize()

        generate(model, pre_code, [1])
        pre_code[0] += "\n\tstat={};  % Collect desired intermediate results in stat.\n" \
                       "end\n"

        pre_code[0] += aux_func_def(self.bit_width)

        pre_code[0] = pre_code[0].replace("\t", "    ")  # Code style: 1 Tab = 4 Spaces.
        with open('./template.m', 'w') as fp:
            fp.writelines(pre_code[0])

        print("MATLAB code converted. Check and correct the generated code.")
        print("=============================================================")


def hint()->str:
    info = "%{ \n" \
           "\t-------- THIS IS AN AUTO-GENERATED NETWORK TEMPLATE --------\n" \
           "\tNOTE: There is no guarantee for correctness. To start a \n" \
           "\tsimulation, check the topology with the original network and \n" \
           "\tadd necessary operations in the below code.\n\n" \
           "\t(1). The shortcut connection is ignored in conversion.\n" \
           "\t(2). Numeric type cast may be incorrect.\n" \
           "\t------------------------------------------------------------ \n" \
           "%} \n"
    return info


def author_info(author_name)->str:
    info = "% Author: {}\n".format(author_name)
    return info


def time_stamp()->str:
    from datetime import datetime
    info = "% Date: " + datetime.today().strftime('%Y-%m-%d') + "\n"
    return info


def main_func_def()->str:
    res = "\nfunction [im, stat] = template(nn, net, im)\n"
    return res


def aux_func_def(bit_width)->str:
    max_v_b = 2 ** (bit_width-1) - 1
    min_v_b = - 2 ** (bit_width - 1)

    max_v_bb = 2 ** (2*bit_width-1) - 1
    min_v_bb = - 2 ** (2*bit_width - 1)
    res = "\nfunction res = cast_int(im, mul, sft) \n" \
          "%------ Uncomment to use intermediate results cast.------\n" \
          "%\tim(im < {}) = {};\n" \
          "%\tim(im > {}) = {};\n" \
          "%-------------------- Comment end. ----------------------\n" \
          "\tim = im * mul;\n" \
          "\tim = bitshift(im, -sft);\n" \
          "\tim(im > {}) = {};\n" \
          "\tim(im < {}) = {};\n" \
          "\tres = im;\n" \
          "end\n".format(min_v_bb, min_v_bb, max_v_bb, max_v_bb,
                         max_v_b, max_v_b, min_v_b, min_v_b)
    return res


def arithmetic(mac_width)->str:
    res = "\tf = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... \n\t" \
          "'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength', {}, " \
          "... \n\t 'ProductFractionLength', 0, 'SumWordLength', {}, 'SumFractionLength', 0); \n\t" \
          "t = numerictype('WordLength', {}, 'FractionLength', 0); \n".format(mac_width,
                                                                              mac_width,
                                                                              mac_width)
    return res


def input_normalize()->str:
    res = "\n% --- WARNING: Input is adjusted to [-128, 127].\n" \
          "% --- If your pre-processing is not like this,\n" \
          "% --- change it to what you used.\n" \
          "\tim = fi(single(im)-128, t, f);\n"
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


def pool2d(k_size, stride, p_type, padding)->str:
    _k_size = '[{}, {}]'.format(str(k_size[0]), str(k_size[1]))
    _stride = '[{}, {}]'.format(str(stride[0]), str(stride[1]))
    _type = ["'MAX'", "'AVG'"][p_type != 'MaxPool2d']
    if padding[0] >= 1 or padding[1] >= 1:
        _padding = "'SAME'"
    else:
        _padding = "'VALID'"
    return "im = nn.Pooling(im, t, f, {}, {}, {}, {});".format(_k_size,
                                                               _type,
                                                               _stride,
                                                               _padding)


def zero_pad2d(padding):
    assert isinstance(padding, tuple), "Padding info must be a tuple."
    if len(padding) == 4:
        return "im = nn.ZeroPad2d(im, [{}, {}, {}, {}]);".format(padding[0],
                                                                 padding[1],
                                                                 padding[2],
                                                                 padding[3])
    elif len(padding) == 2:
        return "im = nn.ZeroPad2d(im, [{}, {}]);".format(padding[0],
                                                         padding[1])


def add_bias(bias_name)->str:
    return "im = nn.AddBias(im, {}, t, f);".format(bias_name)


def add(m1, s1, m2, s2)->str:
    return "im = nn.Add(im, im, {}, {}, {}, {});".format(m1, s1, m2, s2)


def relu(im_name='im')->str:
    """
    Generator for ReLU in FAST-CNN.
    Prototype: im = ReLU(im)
    :param im_name:
    :return:
    """
    return "im = nn.ReLU({});".format(im_name)


def generate(model, pre_code, cnt)->None:
    """
    Generate matlab simulation code recursively.
    As the string object is immutable in Python, we use List to wrap the
    code string to make it mutable during recursive calls.
    :param model: the model should be replaced with QConv2d and marked its name.
    :param pre_code: prefix of the code.
    :param cnt: use this counter to record which params should be access.
    :return:
    """
    for m in model.children():
        if isinstance(m, QConv2d):
            pre_code[0] += "\n% --- Layer: {}\n".format(m.name)
            stride = check_tuple(m.stride)
            padding = check_tuple(m.padding)
            if padding[0] > 0 or padding[1] > 0:
                pre_code[0] += ("\t" + zero_pad2d(padding) + "\n")
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

        if isinstance(m, QAddition):
            pre_code[0] += "\n% --- Element-wise Addition: {}\n".format(m.name)
            pre_code[0] += ("\t" + add("net{{{}}}.Mul_L", "net{{{}}}.Shift_L",
                                       "net{{{}}}.Mul_R", "net{{{}}}.Shift_R").format(cnt[0], cnt[0],
                                                                                      cnt[0], cnt[0]))
            pre_code[0] += '\n'
            cnt[0] += 1

        if isinstance(m, nn.ReLU) or isinstance(m, nn.ReLU6):
            pre_code[0] += ("\t" + relu())
            pre_code[0] += '\n'

        if isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d) \
                or isinstance(m, QAvgPooling):
            pre_code[0] += "\n% --- Layer: {}\n".format(m.name)
            kernel_size = check_tuple(m.kernel_size)
            stride = check_tuple(m.stride)
            padding = check_tuple(m.padding)
            pre_code[0] += ("\t" + pool2d(kernel_size, stride, m.__repr__()[:9], padding))
            pre_code[0] += "\n"
            cnt[0] += 1
        if isinstance(m, nn.ZeroPad2d):
            padding = m.padding
            padding = check_tuple(padding)
            pre_code[0] += ("\t" + zero_pad2d(padding))
            pre_code[0] += '\n'

        else:
            generate(m, pre_code, cnt)
