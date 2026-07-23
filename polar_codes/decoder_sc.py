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
    result = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return float(result[0]) if scalar else result


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
    return int(not np.isnan(x).any())


def _up(position):
    p0 = position[0] - 1
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp[0]


def _get_left_llr(up_llr):
    half = up_llr.size // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = left_bit.size
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
    )


def _get_left_bit(left_llr, frozen_bits, pos):
    if frozen_bits[pos]:
        return 0
    return 0 if left_llr >= 0 else 1


def _get_right_bit(right_llr, frozen_bits, pos):
    if frozen_bits[pos]:
        return 0
    return 0 if right_llr > 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与高效实现等价，保留接口）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        i = 0
        while (phi >> i) & 1:
            i += 1
        while i < n:
            layers.append(i)
            i += 1
        llr_layer_vec.append(layers)

        layers = []
        i = 0
        p = phi + 1
        while i < n and not ((p >> i) & 1):
            layers.append(i)
            i += 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（树形遍历，O(N log N)）。
    信道 LLR 在叶节点按比特倒序放置，与 G_N = B_N F^{⊗n} 编码一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch[bit_reversal_permutation(N)]

    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start : start + span] = up_bit_val.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_pos = start + half
                bit_val = _get_right_bit(right_llr[0], frozen_bits, right_pos)
                bit_matrix[position[0] + 1, right_pos] = bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_val
        elif _all_num(left_llr) == 0:
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_val
        elif position[0] == position[2] - 1:
            left_pos = start
            bit_val = _get_left_bit(left_llr[0], frozen_bits, left_pos)
            bit_matrix[position[0] + 1, left_pos] = bit_val
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)
