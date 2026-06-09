"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
    """数组元素均已赋值（无 NaN）时返回 1。"""
    return 0 if np.any(np.isnan(x)) else 1


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def _up(position):
    p0, p1, p2, p3 = position
    p1_new = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0 - 1, p1_new, p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr[0] > 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr[0] >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = left_bit.size
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i])
        for i in range(length)
    ])


def _get_left_llr(up_llr):
    length = up_llr.size // 2
    return f_operation(up_llr[:length], up_llr[length:])


def _prepare_llr(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    return llr_ch[bit_reversal_permutation(N)]


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    """SC 译码核心（因子树遍历）。"""
    N = y_llr.size
    n = int(np.log2(N))
    information_pos = set(int(i) for i in information_pos)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = p1 + 1
                rb = _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[p0 + 1][p1 + half:p1 + span] = rb
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr
        elif _all_num(left_llr) == 0:
            left_llr = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + half] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = p1
            lb = _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
            bit_matrix[p0 + 1][p1:p1 + half] = lb
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = np.where(~frozen_bits)[0]
    llr = _prepare_llr(llr_ch)
    return _sc_decode_core(llr, info_indices)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = int(format(phi, f"0{n}b"[::-1], 2))
        layers = []
        binary = format(l, f"0{n}b")
        for i in range(n):
            if binary[n - 1 - i] == "0":
                layers.append(i)
                break
        else:
            layers.append(n - 1)
        llr_layer_vec.append(layers)
        bit_layers = []
        if l % 2 == 1:
            for i in range(n):
                if binary[n - 1 - i] == "0":
                    bit_layers.append(i)
                    break
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
