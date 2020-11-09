import torch
import numpy as np


def mul_shift_calculator(s1, s2, s3):
    assert isinstance(s1, float) and isinstance(s2, float) \
           and isinstance(s3, float)
    if s3 == 0:
        s3 = 10e-7  # eps for safe division.
    re_scale = 1 / (s1 * s2 / s3)
    shift_list = list(range(1, 23))

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


def scaling_align_calculator(s3_1, s3_2):

    shift_range = list(range(1, 23))
    m_s_list = []
    for shift in shift_range:
        mul_l = np.round((2 ** shift) * s3_1)
        mul_r = np.round((2 ** shift) * s3_2)
        err = np.abs(mul_l / (2 ** shift) - s3_1) + np.abs(mul_r / (2 ** shift) - s3_2)
        m_s_list.append([mul_l, mul_r, shift, err])

    mul_range_safe = False
    final_list = []
    for item in m_s_list:
        if 8192 <= item[0] <= 32767 and 8192 <= item[1] <= 32767:
            mul_range_safe = True
            final_list.append(item)

    if mul_range_safe is False:
        print("Multiplier is out of numeric range.")
        err_res = np.asarray(m_s_list).astype(np.float32)
    else:
        err_res = np.asarray(final_list).astype(np.float32)

    idx = err_res[:, 3].argmin()
    mul_l, mul_r, shift = float(err_res[idx][0]), float(err_res[idx][1]), float(err_res[idx][2])
    return mul_l, mul_r, shift


def pure_quantize(t, s, bit_width):
    max_v = 2 ** (bit_width - 1) - 1
    min_v = -2 ** (bit_width - 1)
    return torch.round(torch.clamp(t * s, min_v, max_v))
