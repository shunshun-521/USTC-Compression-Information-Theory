"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _all_filled(arr):
    for i in range(arr.size):
        if np.isnan(arr[i]):
            return False
    return True


def _up_position(position):
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


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
        dtype=np.float64,
    )


def _get_right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)],
        dtype=np.float64,
    )


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.ravel()


def _get_left_bit(left_llr, info_pos, frozen_bit, bit_pos):
    if bit_pos in info_pos:
        if left_llr >= 0:
            return 0
        return 1
    return frozen_bit


def _get_right_bit(right_llr, info_pos, frozen_bit, bit_pos):
    if bit_pos in info_pos:
        if right_llr > 0:
            return 0
        return 1
    return frozen_bit


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        temp = phi
        while temp % 2 == 1:
            layers.append(int(math.log2(temp & -temp)))
            temp //= 2
        llr_layer_vec.append(layers)
        bit_layers = []
        temp = phi + 1
        while temp % 2 == 0:
            bit_layers.append(int(math.log2(temp & -temp)))
            temp //= 2
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（树遍历）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    information_pos = list(np.where(~frozen_bits)[0])
    frozen_bit = 0

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while _all_filled(bit_matrix[n]) == 0:
        span = 1 << (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + span // 2:position[1] + span
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + span // 2:position[1] + span
        ]

        if _all_filled(up_bit):
            position = _up_position(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit_new.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr, information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] + span // 2:position[1] + span
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = (
                right_llr_new
            )
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr, information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = (
                    left_bit_val
                )
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(np.int8)
