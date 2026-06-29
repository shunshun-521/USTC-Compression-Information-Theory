"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效树遍历实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(format(i, f'0{n}b')[::-1], 2)


def _prepare_llr(llr_ch, N):
    """极化码编码含比特倒序，信道 LLR 需同步倒序后送入 SC 树"""
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _all_num(x):
    return int(np.all(~np.isnan(x)))


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
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_bit(right_llr, info_set, frozen_bit, right_bit_pos):
    if right_bit_pos in info_set:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return g_operation(up_llr[:length], up_llr[length:], left_bit)


def _get_left_bit(left_llr, info_set, frozen_bit, left_bit_pos):
    if left_bit_pos in info_set:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _sc_tree_decode(y_llr, info_set, frozen_bit=0):
    """非递归树遍历 SC 译码核心"""
    N = y_llr.size
    n = int(math.log2(N))
    info_set = set(info_set)
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        left_llr = llr_matrix[p0 + 1][p1:p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1:p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2:p1 + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit = _get_right_bit(right_llr, info_set, frozen_bit, right_bit_pos)
                bit_matrix[p0 + 1][p1 + span // 2:p1 + span] = right_bit
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + span // 2:p1 + span] = right_llr
        elif not _all_num(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + span // 2] = left_llr
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            left_bit = _get_left_bit(left_llr, info_set, frozen_bit, left_bit_pos)
            bit_matrix[p0 + 1][p1:p1 + span // 2] = left_bit
        else:
            position = _leftdown(position)

    return np.array(bit_matrix[n], dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（包装树遍历实现，保持接口一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    info_set = np.where(~frozen_bits)[0]
    y_llr = _prepare_llr(llr, N)
    return _sc_tree_decode(y_llr, info_set, frozen_bit=0)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（兼容接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layers = []
        bit_layers = []
        temp = l
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            temp //= 2
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    info_set = np.where(~frozen_bits)[0]
    y_llr = _prepare_llr(llr_ch, N)
    return _sc_tree_decode(y_llr, info_set, frozen_bit=0)
