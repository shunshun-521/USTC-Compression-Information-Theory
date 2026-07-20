"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    def decode_block(llr_block, depth, offset):
        if depth == 0:
            if frozen_bits[offset]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_block[0] >= 0 else 1], dtype=int)

        half = len(llr_block) // 2
        llr_left = f_operation(llr_block[:half], llr_block[half:])
        u_left = decode_block(llr_left, depth - 1, offset)
        llr_right = g_operation(llr_block[:half], llr_block[half:], u_left)
        u_right = decode_block(llr_right, depth - 1, offset + half)
        return np.concatenate([u_left, u_right])

    return decode_block(llr, n, 0)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于因子图深度优先遍历）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = set(np.where(~frozen_bits)[0])
    frozen_bit = 0

    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        block_len = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + block_len]
        up_bit = bit_matrix[position[0], position[1]:position[1] + block_len]
        half = block_len // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + block_len]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + block_len]

        if _all_filled(up_bit):
            # move up
            position[0] -= 1
            position[1] = int(np.floor(position[1] / (2 ** (position[2] - position[0]))) *
                             (2 ** (position[2] - position[0])))
        elif _all_filled(right_bit):
            new_up = np.concatenate([(left_bit + right_bit) % 2, right_bit])
            bit_matrix[position[0], position[1]:position[1] + block_len] = new_up
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                if right_bit_pos in information_pos:
                    bit_val = 0 if right_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[position[0] + 1, right_bit_pos] = bit_val
            else:
                position[0] += 1
                position[1] += half
        elif _all_filled(left_bit):
            right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
            llr_matrix[position[0] + 1, position[1] + half:position[1] + block_len] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in information_pos:
                    bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[position[0] + 1, left_bit_pos] = bit_val
            else:
                position[0] += 1

    u_hat = bit_matrix[n].astype(int)
    u_hat[np.isnan(bit_matrix[n])] = 0
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        temp = phi
        layer = 0
        while temp % 2 == 1 and layer < n:
            llr_layers.append(layer)
            temp //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        temp = phi
        layer = 0
        while temp % 2 == 1 and layer < n:
            bit_layers.append(layer)
            temp //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
