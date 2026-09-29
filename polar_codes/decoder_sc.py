"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（树遍历状态机，与 F^{\otimes n} 编码一致）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（硬件友好型）。"""
    La = float(La)
    Lb = float(Lb)
    s1 = 1 if La >= 0 else -1
    s2 = 1 if Lb >= 0 else -1
    return s1 * s2 * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * int(u_hat)) * La + Lb


def _all_filled(x):
    return not np.any(np.isnan(x))


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
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.ravel()


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _sc_tree_decode(y_llr, information_pos, frozen_value):
    """参考 PolarCodesPython 的 SC 树遍历译码。"""
    n = int(np.log2(len(y_llr)))
    n_layers = n + 1
    llr_matrix = np.ones((n_layers, len(y_llr)), dtype=np.float64) * np.nan
    bit_matrix = np.ones((n_layers, len(y_llr)), dtype=np.float64) * np.nan
    llr_matrix[0] = y_llr
    position = [0, 0, n, len(y_llr)]

    info_set = set(information_pos)

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][
            position[1]: position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1]: position[1] + 2 ** (position[2] - position[0])
        ]
        span = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1][position[1]: position[1] + span]
        left_bit = bit_matrix[position[0] + 1][position[1]: position[1] + span]
        right_llr = llr_matrix[position[0] + 1][position[1] + span: position[1] + 2 * span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span: position[1] + 2 * span]

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1]: position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_new
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        if right_bit_pos in info_set:
                            right_bit_val = 0 if right_llr[0] >= 0 else 1
                        else:
                            right_bit_val = frozen_value
                        bit_matrix[position[0] + 1][position[1] + span] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + span: position[1] + 2 * span] = (
                            right_llr_new
                        )
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]: position[1] + span] = (
                                left_llr_new
                            )
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                if left_bit_pos in info_set:
                                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                                else:
                                    left_bit_val = frozen_value
                                bit_matrix[position[0] + 1][position[1]] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0].tolist()
    return _sc_tree_decode(llr_ch, information_pos, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价接口，调用树遍历实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(
            [layer for layer in range(n) if ((phi >> layer) & 1) == 0]
        )
        if phi == N - 1:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layer_vec.append(
                [layer for layer in range(n) if ((phi >> layer) & 1) == 1]
            )
    return lambda_offset, llr_layer_vec, bit_layer_vec
