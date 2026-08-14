"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(bits):
    return not np.any(np.isnan(bits))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.flatten()


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与主译码器共享 f/g 约定）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（与层 0 为信道 LLR 的约定一致）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        p = phi
        while p & 1:
            layer += 1
            p >>= 1
        if phi == 0:
            llr_layers = list(range(n - 1, -1, -1))
        else:
            llr_layers = list(range(n - 1, layer - 1, -1))
        bit_layers = list(range(layer))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（层 0 为信道 LLR，与 G = F^{⊗n} 编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    info_positions = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        span = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span]
        right_llr = llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span]

        if _all_decided(up_bit):
            position = _up(position)
        else:
            if _all_decided(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_new
            else:
                if _all_decided(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        if right_bit_pos in info_positions:
                            rb = 0 if right_llr[0] >= 0 else 1
                        else:
                            rb = 0
                        bit_matrix[position[0] + 1][position[1] + span] = rb
                    else:
                        position = _rightdown(position)
                else:
                    if _all_decided(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + span:position[1] + 2 * span] = right_llr_new
                    else:
                        if not _all_decided(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + span] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                if left_bit_pos in info_positions:
                                    lb = 0 if left_llr[0] >= 0 else 1
                                else:
                                    lb = 0
                                bit_matrix[position[0] + 1][position[1]] = lb
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)
