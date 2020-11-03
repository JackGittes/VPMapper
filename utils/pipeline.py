import time
import torch
import torch.nn as nn
import pandas as pd
from .layer import QConv2d, search_replace_convolution2d
from .absorb_bn import search_absorb_bn
from .optimizer import mse_minimize_quantize, naive_scaling_quantize


class Quantize(object):
    def __init__(self, bit_width, loader=None, per_layer=True, q_method='MSE', use_gpu=False):
        assert bit_width in [8, 4], "bit width should be 8 or 4."
        assert isinstance(per_layer, bool), "per_layer should be a boolean."
        assert q_method in ['MSE', 'Naive', 'Heuristic']

        self.bit_width = bit_width
        self.per_layer = per_layer
        self.q_method = q_method

        self.max_value = 2**(bit_width-1) - 1
        self.min_value = -2**(bit_width-1)

        self.loader = loader
        self.use_gpu = use_gpu

    def apply(self, model):
        assert isinstance(model, torch.nn.Module)

        model.eval()
        print("================= Search all BN layers and remove. ============")
        search_absorb_bn(model)

        print("=========== Search all convolution layers and replace. ========")
        search_replace_convolution2d(model, self.bit_width)
        mark_layer(model, 'root')

        print("=========== Perform layer-wise statistics. ====================")
        for m in model.modules():
            if isinstance(m, QConv2d):
                QConv2d.quantized = False

        if self.use_gpu:
            model = model.cuda()

        cnt = 0
        res_report = []
        total_time = 0.0
        for m in model.modules():
            if isinstance(m, QConv2d) and not m.quantized:
                start_time = time.time()

                cached = []
                print("==> Start quantizing layer: {} ".format(m.name))

                def save_hook(_, _input, _output):
                    cached.append(_input[0].detach().cpu())
                    cached.append(_output.detach().cpu())

                handle = m.register_forward_hook(save_hook)
                for im in self.loader:
                    if self.use_gpu:
                        im = im.cuda()
                    model(im)
                handle.remove()  # save_hook must be removed.

                s1, s2 = self.quantize(cached[0], m.weight.detach().cpu(), cached[1], m)
                print("S1: {:>2.2f}".format(float(s1)), " S2: {:>2.2f}".format(float(s2)))
                m.x_quantizer.s.data = s1
                m.w_quantizer.s.data = s2
                m.quantized = True
                res_report.append([cnt, m.name, s1.item(), s2.item()])
                cnt += 1

                end_time = time.time()
                print("Time: {:>4.2f} s".format(end_time - start_time))
                total_time += (end_time - start_time)
        csv_writer(res_report)
        torch.save(model.state_dict(), './result/pth/quantized.pth')

        print("Total Time: {:>4.2f}".format(total_time))
        print("\n==> Check quantization report and correct parameters.")

    def quantize(self, x, w, o, layer):
        if self.q_method == 'MSE':
            return self.mse_quantize(x, w, o, layer)
        elif self.q_method == 'Naive':
            return self.naive_quantize(x, w, o, layer)
        else:
            raise RuntimeError("Unknown Quantization Method.")

    def mse_quantize(self, x, w, o, layer):
        return mse_minimize_quantize(x, w, o, layer, self.max_value, self.min_value)

    def naive_quantize(self, x, w, _, __):
        return naive_scaling_quantize(x, w, None, None, self.max_value, self.min_value)


def mark_layer(model, name):
    cnt = 0
    for child_name, child in model.named_children():
        if isinstance(child, QConv2d):
            child.name = name + '.' + str(cnt)
            cnt += 1
        else:
            mark_layer(child, name + '.' + child_name)


def csv_writer(res):
    assert isinstance(res, list)
    info_len = len(res)
    extended_res = []
    for _idx in range(info_len):
        tmp = res[_idx]
        tmp.append(res[(_idx+1) % info_len][2])
        tmp.append((_idx+1) % info_len)
        extended_res.append(tmp)
    extended_res[info_len-1][4] = 0.0
    df = pd.DataFrame(res, columns=["SRC", "Name", "S1", "S2", "S3", "DST"])
    df.to_csv('./report/info.csv', index=False)


def post_processing(model, pth_path, info_path, bit_width):
    import numpy as np
    import scipy.io as sio
    from .optimizer import mul_shift_calculator

    assert isinstance(model, nn.Module)

    df = pd.read_csv(info_path)
    q_info = dict()
    for idx, row in df.iterrows():
        q_info[row['Name']] = [row['S1'], row['S2'], row['S3']]

    model.eval()
    search_absorb_bn(model)
    search_replace_convolution2d(model, bit_width)
    mark_layer(model, 'root')

    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    print('==> Params loaded.')
    mat_file = []
    for m in model.modules():
        if isinstance(m, QConv2d):
            scaling = q_info[m.name]
            mat_params = dict()
            mul, shift = mul_shift_calculator(scaling[0], scaling[1], scaling[2])
            q_weight = pure_quantize(m.weight, scaling[1], bit_width)

            mat_params['Name'] = m.name
            q_weight = q_weight.permute(3, 2, 1, 0)
            mat_params['Weight'] = q_weight.detach().cpu().numpy()

            if m.bias is not None:
                q_bias = pure_quantize(m.bias, scaling[0] * scaling[1],
                                       bit_width * 2)
                q_bias = q_bias.detach().cpu().numpy()
                bias_abs = torch.abs(m.bias * scaling[0] * scaling[1])
                if bias_abs.max() > 2**(2*bit_width - 1) - 1:
                    print("==> Layer: {} Bias overflow.".format(m.name))
                mat_params['Bias'] = q_bias

            mat_params['Stride'] = np.asarray(m.stride)
            mat_params['Groups'] = float(m.groups)
            mat_params['Padding'] = np.asarray(m.padding)
            mat_params['Mul'] = mul
            mat_params['Shift'] = shift

            mat_file.append(mat_params)
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            mat_params = dict()
            mat_params['Name'] = "Pooling"
            mat_params['Stride'] = np.asarray(m.stride)
            mat_params['Kernel'] = np.asarray(m.kernel_size)
            mat_params['Type'] = m.__repr__()[:9]

            mat_file.append(mat_params)
    sio.savemat('./result/mat/quantized.mat', {'Net': mat_file})


def pure_quantize(t, s, bit_width):
    max_v = 2 ** (bit_width - 1) - 1
    min_v = -2 ** (bit_width - 1)
    return torch.round(torch.clamp(t * s, min_v, max_v))
