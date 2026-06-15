"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = La.reshape(1)
        Lb = Lb.reshape(1)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return float(out[0]) if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _prepare_channel_llr(llr_ch, N):
    """比特倒序编码器对应的信道 LLR 预处理"""
    br = bit_reversal_permutation(N)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return llr_ch[np.argsort(br)]


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _decide_bit(llr_val, bit_pos, info_indices, frozen_val):
    if bit_pos in info_indices:
        return 0 if llr_val >= 0 else 1
    return frozen_val


def _sc_decode_core(y_llr, info_indices, frozen_val=0):
    """基于因子图遍历的 SC 译码核心（非递归状态机）"""
    N = len(y_llr)
    n = int(math.log2(N))
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_val.flatten()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _decide_bit(
                    right_llr[0], right_bit_pos, info_set, frozen_val
                )
                bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                right_llr_val
            )
        elif not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _decide_bit(
                    left_llr[0], left_bit_pos, info_set, frozen_val
                )
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: True/1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    info_indices = np.where(~frozen_bits)[0]
    y_llr = _prepare_channel_llr(llr_ch, N)
    u_hat = _sc_decode_core(y_llr, info_indices, frozen_val=0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        bits = bin(phi)[2:].zfill(n)
        for layer in range(n):
            if bits[n - 1 - layer] == "0":
                layers.append(layer)
            else:
                break
        llr_layer_vec.append(layers)

        b_layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                b_layers.append(layer)
        bit_layer_vec.append(b_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（状态机实现）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    info_indices = np.where(frozen_bits == 0)[0]
    y_llr = _prepare_channel_llr(llr_ch, N)
    return _sc_decode_core(y_llr, info_indices, frozen_val=0)
