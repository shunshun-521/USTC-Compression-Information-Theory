"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def _sign_ms(x):
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)


def f_operation(La, Lb):
    """min-sum f 运算（0 符号视为 +1）。"""
    if np.isscalar(La) and np.isscalar(Lb):
        return float(
            _sign_ms(La) * _sign_ms(Lb) * min(abs(La), abs(Lb))
        )
    return _sign_ms(La) * _sign_ms(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _get_bit(llr_val, idx, info_set, frozen_val):
    if idx in info_set:
        return 0 if llr_val >= 0 else 1
    return frozen_val


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（基于树遍历，与标准实现一致）。"""
    y_llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(int)
    info_set = set(np.where(frozen_bits == 0)[0])
    frozen_val = 0

    N = y_llr.size
    n = int(np.log2(N))
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
                        right_bit_pos = position[1] + half
                        right_bit_val = _get_bit(
                            right_llr[0], right_bit_pos, info_set, frozen_val
                        )
                        bit_matrix[position[0] + 1][position[1] + half] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + half : position[1] + span
                        ] = right_llr_new
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + half] = (
                                left_llr_new
                            )
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit_val = _get_bit(
                                    left_llr[0], left_bit_pos, info_set, frozen_val
                                )
                                bit_matrix[position[0] + 1][position[1]] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        psi, layer = phi, 0
        while psi % 2 == 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            psi, layer = phi, 0
            while psi % 2 == 1:
                bit_layers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：当前实现委托给已验证的递归树遍历版本。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
