"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效树遍历实现）
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


def _all_computed(arr):
    return not np.any(np.isnan(arr))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - pos[0] - 1), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * len(left_bit))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
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
    """预计算非递归 SC 译码辅助向量（层索引列表）"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (layer - 1))

    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        p, layer = phi, 0
        while p % 2 == 0 and layer < n:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        p, layer = phi, 0
        while p % 2 == 1 and layer < n:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（树遍历，O(N log N)）。
    信道 LLR 置于第 0 层，译码结果在第 n 层。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    info_pos = set(np.where(frozen_bits == 0)[0])
    frozen_val = 0

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + span]
        up_bit = bit_matrix[position[0], position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
            continue

        if _all_computed(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1]:position[1] + span] = up_bit_new.flatten()
            continue

        if _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                if right_bit_pos in info_pos:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                else:
                    right_bit_val = frozen_val
                bit_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
            continue

        if _all_computed(left_bit):
            right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
            llr_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_llr_new
            continue

        if not _all_computed(left_llr):
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr_new
            continue

        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            if left_bit_pos in info_pos:
                left_bit_val = 0 if left_llr[0] >= 0 else 1
            else:
                left_bit_val = frozen_val
            bit_matrix[position[0] + 1, position[1]:position[1] + half] = left_bit_val
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)
