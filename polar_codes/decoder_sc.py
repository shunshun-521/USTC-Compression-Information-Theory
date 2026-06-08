"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（因子图矩阵遍历）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    s1 = np.sign(La).copy()
    s2 = np.sign(Lb).copy()
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return out[0] if out.size == 1 else out


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    u_hat = np.atleast_1d(np.asarray(u_hat, dtype=np.float64))
    out = (1.0 - 2.0 * u_hat) * La + Lb
    return out[0] if out.size == 1 else out


def _all_ready(x):
    return not np.any(np.isnan(x))


def _left_down(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _right_down(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _decide_bit(llr_val, bit_pos, info_set):
    if bit_pos in info_set:
        return 0 if llr_val >= 0 else 1
    return 0


def sc_decode_nonrecursive(llr_ch, frozen_bits, info_indices=None):
    """非递归 SC 译码（与 x = u @ G_N 编码配套）。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    y_llr = np.asarray(llr_ch, dtype=np.float64)

    if info_indices is None:
        info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_ready(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[
            position[0] + 1, position[1] + half : position[1] + span
        ]
        right_bit = bit_matrix[
            position[0] + 1, position[1] + half : position[1] + span
        ]

        if _all_ready(up_bit):
            position = _up(position)
        elif _all_ready(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = up_bit_val.copy()
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1, position[1] + half] = _decide_bit(
                    right_llr[0], right_bit_pos, info_set
                )
            else:
                position = _right_down(position)
        elif _all_ready(left_bit):
            right_llr_val = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                    for i in range(half)
                ]
            )
            llr_matrix[
                position[0] + 1, position[1] + half : position[1] + span
            ] = right_llr_val
        elif not _all_ready(left_llr):
            left_llr_val = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)]
            )
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1, position[1]] = _decide_bit(
                    left_llr[0], left_bit_pos, info_set
                )
            else:
                position = _left_down(position)

    return np.nan_to_num(bit_matrix[n], nan=0).astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=np.int8)

    def decode_block(llr_blk, offset):
        length = len(llr_blk)
        if length == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_blk[0] >= 0 else 1
            return

        half = length // 2
        llr_left = f_operation(llr_blk[:half], llr_blk[half:])
        decode_block(llr_left, offset)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_blk[:half], llr_blk[half:], u_left)
        decode_block(llr_right, offset + half)

    decode_block(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        for layer in range(n):
            if (psi & 1) == 0:
                layers_llr.append(layer)
                psi >>= 1
            else:
                break
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        psi = phi + 1
        for layer in range(n):
            if (psi & 1) == 1:
                layers_bit.append(layer)
                psi >>= 1
            else:
                break
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口。"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
