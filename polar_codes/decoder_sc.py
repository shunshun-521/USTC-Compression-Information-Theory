"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_computed(arr):
    return not np.any(np.isnan(arr))


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)],
        dtype=np.float64,
    )


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0.0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层矩阵遍历）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch

    position = [0, 0, n, N]

    def up(pos):
        p0 = pos[0] - 1
        p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
        return [p0, p1, pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        start = position[1]

        up_llr = llr_matrix[position[0], start:start + span]
        up_bit = bit_matrix[position[0], start:start + span]
        left_llr = llr_matrix[position[0] + 1, start:start + half]
        left_bit = bit_matrix[position[0] + 1, start:start + half]
        right_llr = llr_matrix[position[0] + 1, start + half:start + span]
        right_bit = bit_matrix[position[0] + 1, start + half:start + span]

        if _all_computed(up_bit):
            position = up(position)
            continue

        if _all_computed(right_bit):
            combined = np.zeros(span, dtype=int)
            combined[:half] = (left_bit.astype(int) + right_bit.astype(int)) % 2
            combined[half:] = right_bit.astype(int)
            bit_matrix[position[0], start:start + span] = combined
            continue

        if _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_pos = start + half
                if frozen_bits[right_pos]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1, right_pos] = val
            else:
                position = rightdown(position)
            continue

        if _all_computed(left_bit):
            new_right = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half:start + span] = new_right
            continue

        if not _all_computed(left_llr):
            new_left = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start:start + half] = new_left
            continue

        if position[0] == position[2] - 1:
            left_pos = start
            if frozen_bits[left_pos]:
                val = 0
            else:
                val = 0 if left_llr[0] >= 0 else 1
            bit_matrix[position[0] + 1, left_pos] = val
        else:
            position = leftdown(position)

    return bit_matrix[n].astype(int)
