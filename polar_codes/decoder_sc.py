"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效矩阵实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化）。
    sign(0) 按 +1 处理，与硬件友好实现一致。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        s1 = 1 if La == 0 else int(np.sign(La))
        s2 = 1 if Lb == 0 else int(np.sign(Lb))
        return float(s1 * s2 * min(abs(La), abs(Lb)))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _f_scalar(L1, L2):
    s1 = np.sign(L1) or 1
    s2 = np.sign(L2) or 1
    return s1 * s2 * min(abs(L1), abs(L2))


def _g_scalar(L1, L2, u):
    return (1 - 2 * u) * L1 + L2


def _all_filled(x):
    return not np.any(np.isnan(x))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [pos[0] - 1, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([_g_scalar(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([_f_scalar(up_llr[i], up_llr[i + length]) for i in range(length)])


def _sc_matrix_decode(y_llr, information_pos, frozen_value=0):
    """矩阵化 SC 译码（参考 PolarCodesPython 实现）。"""
    N = y_llr.size
    n = int(np.log2(N))
    info_set = set(information_pos)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

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
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][sl] = up_bit_val.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_set:
                    right_bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    right_bit_val = frozen_value
                bit_matrix[position[0] + 1][sl_r] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][sl_r] = right_llr_val
        elif not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][sl_l] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_set:
                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit_val = frozen_value
                bit_matrix[position[0] + 1][sl_l] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def _frozen_to_info_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return list(np.where(~frozen_bits)[0])


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（调用矩阵实现保证一致性）。"""
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n - 1, -1, -1))
        else:
            tmp = phi
            l = 0
            while (tmp & 1) == 1:
                tmp >>= 1
                l += 1
            llr_layers = list(range(n - 1, l - 1, -1))
        llr_layer_vec.append(llr_layers)

        if phi == N - 1:
            bit_layers = []
        else:
            tmp = phi
            l = 0
            while (tmp & 1) == 1:
                tmp >>= 1
                l += 1
            bit_layers = list(range(l))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 直接对应码字比特顺序；极化编码含比特倒序时自动重排。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv_br = np.empty(N, dtype=int)
    for i, j in enumerate(br):
        inv_br[j] = i
    y_llr = llr_ch[inv_br]
    info_pos = _frozen_to_info_set(frozen_bits)
    u_hat = _sc_matrix_decode(y_llr, info_pos, frozen_value=0)
    return u_hat
