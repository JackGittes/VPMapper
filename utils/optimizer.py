import torch
import numpy as np
import torch.nn as nn
from .layer import QConv2d


@torch.no_grad()
def mse_minimize_quantize(x, w, o, layer, max_v, min_v):
    """
    The algorithm utilized to determine optimal scaling factors for x and w is proposed in paper:

    M. Zhao, K. Ning, S. Yu, L. Liu and N. Wu, "Quantizing Oriented Object Detection Network via
    Outlier-Aware Quantization and IoU Approximation," in IEEE Signal Processing Letters,
    doi: 10.1109/LSP.2020.3031490.

    Some details are slightly different from the original paper, where we search not only the scaling
    factors for weight but also for input features. Hence, the search space is two-dimensional, indicating
    a more time-consuming procedure as well as a more accurate approximation.

    :param x: input feature map of a given convolution layer, should be a tensor.
    :param w: weight of a given convolution layer, should be a tensor.
    :param o: output feature map of a given convolution layer, should be a tensor.
    :param layer: the convolution layer that need to be quantized.
    :param max_v: the max numeric value of a certain bit-width, e.g. 127 of 8-bit signed integer.
    :param min_v: the same as max_v.
    :return: scaling factors for input (x) and weight (w).
    """
    assert isinstance(layer, QConv2d)
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(o, torch.Tensor)

    beta = 0.85
    alpha = 5
    tolerance = 1.5
    sample_points = 40

    criterion = nn.MSELoss()

    "Estimate the optimal scaling factor range."
    std_w = torch.sqrt(w.var()).item()
    std_x = torch.sqrt(x.var()).item()

    sx0, sw0 = naive_scaling_quantize(x, w, None, None, max_v, min_v)
    sx0 = sx0.item()
    sw0 = sw0.item()

    sx_min, sw_min = sx0 * beta, sw0 * beta

    sx_max = max((sx0 * tolerance, max_v / (alpha * std_x)))
    sw_max = max((sw0 * tolerance, max_v / (alpha * std_w)))

    " Uniformly sampling in the scaling factor range. "
    s_x_list = [(sx_min + (_item + 1) * (sx_max - sx_min) / sample_points) for _item in range(sample_points)]
    s_w_list = [(sw_min + (_item + 1) * (sw_max - sw_min) / sample_points) for _item in range(sample_points)]

    if layer.weight.device != torch.device('cpu'):
        x = x.to(layer.weight.device)
        o = o.to(layer.weight.device)

    err_array = []

    layer.x_quantizer.s.data = torch.tensor(sx0)
    layer.w_quantizer.s.data = torch.tensor(sw0)

    layer.q_inference = True

    err0 = criterion(layer(x), o).detach().item()
    print("Initial MSE error: {:>4.4f}".format(float(err0)))

    print("Searching ...")

    with torch.no_grad():
        for _s_x in s_x_list:
            for _s_w in s_w_list:
                layer.x_quantizer.s.data = torch.tensor(_s_x)
                layer.w_quantizer.s.data = torch.tensor(_s_w)
                err = criterion(layer(x), o)
                err_ = err.detach().item()
                err_array.append([_s_x, _s_w, float(err_)])

    err_array = np.asarray(err_array).astype(np.float32)
    idx = err_array[:, 2].argmin()

    if float(err_array[idx, 2]) > float(err0):
        err_final = err0
        s1, s2 = sx0, sw0
    else:
        err_final = err_array[idx, 2]
        s1, s2 = err_array[idx, 0], err_array[idx, 1]

    print("Searched optimal MSE error: {:>4.4f}".format(err_final))
    return torch.tensor(float(s1)), torch.tensor(float(s2))


def naive_scaling_quantize(x, w, _, __, max_v, min_v) -> \
        (torch.Tensor, torch.Tensor):
    assert isinstance(x, torch.Tensor)
    assert isinstance(w, torch.Tensor)

    def find_scale(_x):
        max_in, min_in = _x.max(), _x.min()
        if torch.abs(max_in) > torch.abs(min_in):
            s1 = max_v / max_in
        else:
            s1 = min_v / min_in
        return s1

    return find_scale(x), find_scale(w)


def mul_shift_calculator(s1, s2, s3):
    assert isinstance(s1, float) and isinstance(s2, float) \
           and isinstance(s3, float)
    if s3 == 0:
        s3 = 10e-7  # eps for safe division.
    re_scale = 1 / (s1 * s2 / s3)
    shift_list = list(range(10, 23))

    m_s_list = []
    for shift in shift_list:
        mul = np.round((2 ** shift) * re_scale)
        err = np.abs(mul / (2 ** shift) - re_scale)
        m_s_list.append([mul, shift, err])

    mul_range_safe = False
    final_list = []
    for item in m_s_list:
        if 100 <= item[0] <= 512:
            mul_range_safe = True
            final_list.append(item)
    if mul_range_safe is False:
        print("Multiplier is out of numeric range.")
        err_res = np.asarray(m_s_list).astype(np.float32)
    else:
        err_res = np.asarray(final_list).astype(np.float32)
    idx = err_res[:, 2].argmin()
    mul, shift = float(err_res[idx][0]), float(err_res[idx][1])
    return mul, shift
