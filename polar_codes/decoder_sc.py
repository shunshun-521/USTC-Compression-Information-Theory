"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（Hard-Friendly），支持标量。
    """
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return s1 * s2 * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * int(u_hat)) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _up_position(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1, position[1] + 2 ** (position[2] - 1 - position[0]), position[2], position[3]]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然序 LLR，与比特倒序编码配套时需先倒序 LLR）。"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = np.array([f_operation(llr_node[i], llr_node[i + half]) for i in range(half)])
        u_left = np.zeros(half, dtype=int)
        for i in range(half):
            u_left[i] = decode_node(llr_left[i:i + 1], depth - 1, bit_offset + i)
        llr_right = np.array([
            g_operation(llr_node[i], llr_node[i + half], u_left[i]) for i in range(half)
        ])
        for i in range(half):
            decode_node(llr_right[i:i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算辅助信息（供 SCL 等模块使用）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = decode_order[phi]
        psi = l
        layer = 0
        while psi & 1:
            layer += 1
            psi >>= 1
        llr_layer_vec.append(list(range(layer, n)))
        if l == N - 1:
            bit_layer_vec.append([])
        else:
            psi = l + 1
            layer = 0
            layers = []
            while psi & 1:
                layers.append(layer)
                layer += 1
                psi >>= 1
            bit_layer_vec.append(layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(y_llr, frozen_bits):
    """树遍历式 SC 译码核心（输入 LLR 已按比特倒序排列）。"""
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if _all_filled(up_bit):
            position = _up_position(position)
        else:
            if _all_filled(right_bit):
                up_bit_new = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        if right_bit_pos in info_pos:
                            right_bit_val = 0 if right_llr[0] > 0 else 1
                        else:
                            right_bit_val = 0
                        bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit.astype(int), up_llr)
                        llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr_new
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                if left_bit_pos in info_pos:
                                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                                else:
                                    left_bit_val = 0
                                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    与比特倒序编码配套：先对信道 LLR 做比特倒序置换，再执行树遍历译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_decode_core(llr_ch[br], frozen_bits)
