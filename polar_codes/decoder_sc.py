"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（树遍历高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _frozen_mask_to_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    信道 LLR 直接输入，无需比特倒序。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _frozen_mask_to_bool(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_node(node_llr, stage, bit_offset):
        length = 1 << (n - stage)
        if length == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if node_llr[0] >= 0 else 1
            return

        half = length // 2
        llr_left = f_operation(node_llr[:half], node_llr[half:])
        decode_node(llr_left, stage + 1, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(node_llr[:half], node_llr[half:], u_left)
        decode_node(llr_right, stage + 1, bit_offset + half)

    decode_node(llr, 0, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层更新列表）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        t = phi
        for layer in range(n):
            if (t >> layer) & 1 == 0:
                llr_layers.append(layer)
            else:
                break
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        for layer in range(n):
            if (phi + 1) & (1 << layer):
                bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _all_known(bits):
    """检查切片内比特是否均已判决"""
    return not np.any(np.isnan(bits))


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（树遍历实现，O(N log N)）。
    信道 LLR 直接输入。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _frozen_mask_to_bool(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0, :] = llr_ch

    position = [0, 0, n, N]

    def up(pos):
        pos[0] -= 1
        block = 1 << (pos[2] - pos[0] + 1)
        pos[1] = int(np.floor(pos[1] / block) * block)

    def leftdown(pos):
        pos[0] += 1

    def rightdown(pos):
        pos[0] += 1
        pos[1] += 1 << (pos[2] - 1 - pos[0] + 1)

    def get_up_bit(left_bit, right_bit):
        length = len(left_bit)
        out = np.empty(2 * length, dtype=np.float64)
        out[:length] = (left_bit + right_bit) % 2
        out[length:] = right_bit
        return out

    while not _all_known(bit_matrix[n]):
        row, col, depth, _ = position
        span = 1 << (depth - row)
        up_llr = llr_matrix[row, col : col + span]
        up_bit = bit_matrix[row, col : col + span]
        half = span // 2
        left_llr = llr_matrix[row + 1, col : col + half]
        left_bit = bit_matrix[row + 1, col : col + half]
        right_llr = llr_matrix[row + 1, col + half : col + span]
        right_bit = bit_matrix[row + 1, col + half : col + span]

        if _all_known(up_bit):
            up(position)
            continue

        if _all_known(right_bit):
            bit_matrix[row, col : col + span] = get_up_bit(left_bit, right_bit)
            continue

        if _all_known(right_llr):
            if row == depth - 1:
                right_pos = col + half
                if frozen_bits[right_pos]:
                    bit_matrix[row + 1, right_pos] = 0
                else:
                    bit_matrix[row + 1, right_pos] = 0 if right_llr[0] >= 0 else 1
            else:
                rightdown(position)
            continue

        if _all_known(left_bit):
            llr_matrix[row + 1, col + half : col + span] = g_operation(
                up_llr[:half], up_llr[half:], left_bit
            )
            continue

        if not _all_known(left_llr):
            llr_matrix[row + 1, col : col + half] = f_operation(up_llr[:half], up_llr[half:])
            continue

        if row == depth - 1:
            left_pos = col
            if frozen_bits[left_pos]:
                bit_matrix[row + 1, left_pos] = 0
            else:
                bit_matrix[row + 1, left_pos] = 0 if left_llr[0] >= 0 else 1
        else:
            leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_fast(llr_ch, frozen_bits):
    """与 sc_decode 相同的高效树遍历实现"""
    return sc_decode(llr_ch, frozen_bits)
