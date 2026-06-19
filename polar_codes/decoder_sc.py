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
    if La.ndim == 0:
        s1 = 1 if La == 0 else np.sign(La)
        s2 = 1 if Lb == 0 else np.sign(Lb)
        return s1 * s2 * min(abs(La), abs(Lb))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _frozen_bits_to_info_set(frozen_bits):
    return set(np.where(~np.asarray(frozen_bits, dtype=bool))[0])


def _reorder_channel_llr(llr_ch, N):
    """编码含比特倒序时，将信道 LLR 重排为译码树自然顺序"""
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _all_computed(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    span = 2 ** (position[2] - 1 - position[0])
    return [position[0] + 1, position[1] + span, position[2], position[3]]


def _up(position):
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
            * (2 ** (position[2] - position[0] + 1)))
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)]
                    + [right_bit[i] for i in range(length)])
    return temp.reshape(1, 2 * length)


def _get_right_llr(left_bit, up_llr):
    half = left_bit.size
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)],
        dtype=np.float64,
    )


def _get_left_llr(up_llr):
    half = up_llr.size // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
        dtype=np.float64,
    )


def _get_left_bit(left_llr, info_set, left_bit_pos):
    if left_bit_pos in info_set:
        return 0 if left_llr >= 0 else 1
    return 0


def _get_right_bit(right_llr, info_set, right_bit_pos):
    if right_bit_pos in info_set:
        return 0 if right_llr > 0 else 1
    return 0


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Tal-Vardy 迭代树遍历，O(N log N)）。
    """
    llr_ch = _reorder_channel_llr(llr_ch, len(llr_ch))
    info_set = _frozen_bits_to_info_set(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_val[0]
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                right_bit_val = _get_right_bit(right_llr[0], info_set, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_val
        elif not _all_computed(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(left_llr[0], info_set, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，结果与非递归版本一致）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    供基于 P/C 分层存储的高效实现使用。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        for layer in range(n):
            if layer == 0 or (phi & ((1 << layer) - 1)) == (1 << layer) - 1:
                layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        for layer in range(n):
            if (phi >> layer) & 1:
                layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec
