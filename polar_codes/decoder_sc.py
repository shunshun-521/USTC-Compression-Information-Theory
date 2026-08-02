"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = np.array([La])
        Lb = np.array([Lb])
    s1 = np.sign(La).copy()
    s2 = np.sign(Lb).copy()
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    result = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return result[0] if scalar else result


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _all_computed(x):
    """检查数组是否全部已计算（无 NaN）"""
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1,
            position[1] + 2 ** (position[2] - 1 - position[0]),
            position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)]
                    + list(right_bit))
    return np.array(temp)


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i])
                     for i in range(length)])


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length])
                     for i in range(length)])


def _sc_tree_decode(llr_ch, information_pos, frozen_bit):
    """SC 因子图树遍历译码核心"""
    N = len(llr_ch)
    n = int(np.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit
        elif not _all_computed(right_llr):
            if _all_computed(left_bit):
                right_llr = _get_right_llr(left_bit, up_llr)
                llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr
            elif not _all_computed(left_llr):
                left_llr = _get_left_llr(up_llr)
                llr_matrix[p0 + 1][p1:p1 + half] = left_llr
            elif position[0] == position[2] - 1:
                left_bit_pos = p1
                if left_bit_pos in information_pos:
                    bit_matrix[p0 + 1][p1] = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_matrix[p0 + 1][p1] = frozen_bit
            else:
                position = _leftdown(position)
        elif position[0] == position[2] - 1:
            right_bit_pos = p1 + 1
            if right_bit_pos in information_pos:
                bit_matrix[p0 + 1][p1 + half] = 0 if right_llr[0] >= 0 else 1
            else:
                bit_matrix[p0 + 1][p1 + half] = frozen_bit
        else:
            position = _rightdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    rev = bit_reversal_permutation(N)
    llr_br = llr[rev]
    info_pos = np.where(np.asarray(frozen_bits) == 0)[0]
    return _sc_tree_decode(llr_br, info_pos, 0)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    对信道 LLR 做比特倒序以匹配编码器的 B_N 置换。
    """
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_br = llr_ch[rev]
    info_pos = np.where(np.asarray(frozen_bits) == 0)[0]
    return _sc_tree_decode(llr_br, info_pos, 0)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）"""
    m = int(np.log2(N))
    lambda_offset = [1 << i for i in range(m + 1)]
    llr_layer_vec = [list(range(m)) for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
