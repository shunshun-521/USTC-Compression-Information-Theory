"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _as_frozen_mask(frozen_bits):
    frozen = np.asarray(frozen_bits)
    if frozen.dtype != bool:
        frozen = frozen.astype(bool)
    return frozen


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def _up(position):
    p0, p1, p2, p3 = position
    p1 = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0 - 1, p1, p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_bit(right_llr, information_pos, frozen_val, pos):
    if pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _get_left_bit(left_llr, information_pos, frozen_val, pos):
    if pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_llr(left_bit, up_llr):
    length = left_bit.size
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _get_left_llr(up_llr):
    half = up_llr.size // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _sc_tree_decode(y_llr, information_pos, frozen_val=0):
    """
    基于因子树遍历的非递归 SC 译码（参考实现）。
    输入 LLR 需已按比特倒序重排以匹配编码器输出。
    """
    N = y_llr.size
    n = int(np.log2(N))
    information_pos = set(int(i) for i in information_pos)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        span = 2 ** (position[2] - position[0])
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2 : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2 : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + span] = up_bit.copy()
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + span // 2
                        right_bit_val = _get_right_bit(
                            right_llr[0], information_pos, frozen_val, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + span // 2 : position[1] + span
                        ] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + span // 2 : position[1] + span
                        ] = right_llr
                    else:
                        if not _all_filled(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + span // 2] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_left_bit(
                                    left_llr[0], information_pos, frozen_val, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1] + span // 2
                                ] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（保留作参考）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _as_frozen_mask(frozen_bits)
    info_pos = np.where(~frozen_bits)[0]
    br = bit_reversal_permutation(len(llr))
    return _sc_tree_decode(llr[br], info_pos, frozen_val=0)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        trailing_ones = 0
        temp = phi
        while temp & 1:
            trailing_ones += 1
            temp >>= 1
        llr_layer_vec.append(list(range(trailing_ones, n)))
        bit_layers = []
        temp = phi
        layer = 0
        while (temp & 1) and layer < n:
            bit_layers.append(layer)
            temp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 经比特倒序重排后送入因子树译码器。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _as_frozen_mask(frozen_bits)
    N = len(llr_ch)
    info_pos = np.where(~frozen_bits)[0]
    br = bit_reversal_permutation(N)
    return _sc_tree_decode(llr_ch[br], info_pos, frozen_val=0)
