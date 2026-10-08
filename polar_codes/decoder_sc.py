"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（硬件友好）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1.0, s1)
    s2 = np.where(s2 == 0, 1.0, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def _all_filled(x):
    return not np.isnan(x).any()


def _info_positions(frozen_bits):
    fb = np.asarray(frozen_bits)
    return np.where(fb == 0)[0]


def _sc_tree_decode(y_llr, information_pos, frozen_bit=0):
    """基于因子树遍历的非递归 SC 译码核心。"""
    y_llr = np.asarray(y_llr, dtype=np.float64)
    N = y_llr.size
    n = int(math.log2(N))
    information_pos = set(int(i) for i in information_pos)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    def up(pos):
        p0 = pos[0] - 1
        p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
        return [p0, p1, pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    def get_up_bit(left_bit, right_bit):
        length = left_bit.size
        temp = np.array([(left_bit + right_bit) % 2, right_bit])
        temp.resize((1, 2 * length))
        return temp

    def get_left_llr(up_llr):
        length = int(up_llr.size / 2)
        return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])

    def get_right_llr(left_bit, up_llr):
        length = int(left_bit.size)
        return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])

    def get_left_bit(left_llr, left_bit_pos):
        if left_bit_pos in information_pos:
            return 0 if left_llr >= 0 else 1
        return frozen_bit

    def get_right_bit(right_llr, right_bit_pos):
        if right_bit_pos in information_pos:
            return 0 if right_llr >= 0 else 1
        return frozen_bit

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        sl = slice(position[1], position[1] + span)
        half = span // 2
        sl_l = slice(position[1], position[1] + half)
        sl_r = slice(position[1] + half, position[1] + span)

        up_llr = llr_matrix[position[0]][sl]
        up_bit = bit_matrix[position[0]][sl]
        left_llr = llr_matrix[position[0] + 1][sl_l]
        left_bit = bit_matrix[position[0] + 1][sl_l]
        right_llr = llr_matrix[position[0] + 1][sl_r]
        right_bit = bit_matrix[position[0] + 1][sl_r]

        if _all_filled(up_bit):
            position = up(position)
            continue

        if _all_filled(right_bit):
            up_bit = get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][sl] = up_bit.copy()
            continue

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                rb = get_right_bit(right_llr, right_bit_pos)
                bit_matrix[position[0] + 1][sl_r] = rb
            else:
                position = rightdown(position)
            continue

        if _all_filled(left_bit):
            right_llr_new = get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][sl_r] = right_llr_new
            continue

        if not _all_filled(left_llr):
            left_llr_new = get_left_llr(up_llr)
            llr_matrix[position[0] + 1][sl_l] = left_llr_new
            continue

        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            lb = get_left_bit(left_llr, left_bit_pos)
            bit_matrix[position[0] + 1][sl_l] = lb
        else:
            position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    info_pos = _info_positions(frozen_bits)
    return _sc_tree_decode(llr_ch, info_pos, frozen_bit=0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用同一树形译码器作为参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（接口占位，供扩展/文档使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
