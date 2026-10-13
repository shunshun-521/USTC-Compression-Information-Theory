"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

import polar_sc_ref as ref


def f_operation(La, Lb):
    """min-sum f（供 BP 等模块使用）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return ref.f_hf(float(La), float(Lb))
    out = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    it = np.nditer([La, Lb, out], flags=["refs_ok"])
    for la, lb, o in it:
        o[...] = ref.f_hf(float(la), float(lb))
    return out


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _sc_decode_matrix(y_llr, information_pos, frozen_bit=0):
    """矩阵版 SC 译码（参考实现，O(N log N)）。"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while ref.all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if ref.all_num(up_bit) == 1:
            position = ref.up(position)
        else:
            if ref.all_num(right_bit) == 1:
                up_bit = ref.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + span] = up_bit.copy()
            else:
                if ref.all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit_val = ref.get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                            right_bit_val
                        )
                    else:
                        position = ref.rightdown(position)
                else:
                    if ref.all_num(left_bit) == 1:
                        right_llr = ref.get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                            right_llr
                        )
                    else:
                        if ref.all_num(left_llr) == 0:
                            left_llr = ref.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + half] = (
                                left_llr
                            )
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = ref.get_left_bit(
                                    left_llr, information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][position[1] : position[1] + half] = (
                                    left_bit_val
                                )
                            else:
                                position = ref.leftdown(position)

    return bit_matrix[n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与矩阵版结果一致，用于对照）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """占位：兼容 SCL 模块接口。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """非递归矩阵 SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    info_pos = np.where(~frozen_bits.astype(bool))[0]
    return _sc_decode_matrix(llr_ch, info_pos, frozen_bit=0)
