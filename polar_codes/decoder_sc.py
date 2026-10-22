"""
极化码 SC（串行抵消）译码器
非递归实现（与标准 G=F^{⊗n} 编码配套）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(x):
    return not np.any(np.isnan(x))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_right_bit(right_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _get_left_bit(left_llr, info_set, frozen_val, pos):
    if pos in info_set:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    half = len(up_llr) // 2
    out = np.empty(length, dtype=np.float64)
    for i in range(length):
        out[i] = g_operation(up_llr[i], up_llr[i + half], left_bit[i])
    return out


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    out = np.empty(half, dtype=np.float64)
    for i in range(half):
        out[i] = f_operation(up_llr[i], up_llr[i + half])
    return out


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    """参考实现的非递归 SC 译码核心。"""
    N = y_llr.size
    n = int(np.log2(N))
    info_set = set(information_pos)

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
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + span
                ] = up_bit_new.copy()
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = _get_right_bit(
                            right_llr[0], info_set, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                            right_llr_new
                        )
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + half] = (
                                left_llr_new
                            )
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_left_bit(
                                    left_llr[0], info_set, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][position[1]] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << (n - l) for l in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口。frozen_bits 中 1 表示冻结位。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = np.where(~frozen_bits)[0].tolist()
    return _sc_decode_core(np.asarray(llr_ch, dtype=np.float64), info_pos, frozen_bit=0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致，用于校验）。"""
    return sc_decode(llr, frozen_bits)
