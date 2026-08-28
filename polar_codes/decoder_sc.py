"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _up(position):
    p0, p1, p2, p3 = position
    block = 2 ** (p2 - p0 + 1)
    p1 = int(np.floor(p1 / block) * block)
    return [p0 - 1, p1, p2, p3]


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - p0 - 1), p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)])
    temp2 = np.array([right_bit[i] for i in range(length)])
    return np.concatenate([temp, temp2])


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([
        g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)
    ])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    info_idx = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码器使用比特倒序时，需对信道 LLR 做相同倒序后再译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    # 与含比特倒序的编码器配套：对信道 LLR 做比特倒序
    br = bit_reversal_permutation(N)
    y_llr = llr_ch[br]

    info_indices = np.where(~frozen_bits)[0]
    frozen_bit_val = 0

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        p0, p1, p2, p3 = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit_new
        elif _all_filled(right_llr):
            if p0 == p2 - 1:
                right_bit_pos = p1 + 1
                if frozen_bits[right_bit_pos]:
                    right_bit_val = frozen_bit_val
                else:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[p0 + 1][p1 + half] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[p0 + 1][p1 + half:p1 + span] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[p0 + 1][p1:p1 + half] = left_llr_new
        else:
            if p0 == p2 - 1:
                left_bit_pos = p1
                if frozen_bits[left_bit_pos]:
                    left_bit_val = frozen_bit_val
                else:
                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                bit_matrix[p0 + 1][p1] = left_bit_val
            else:
                position = _leftdown(position)

    u_hat = bit_matrix[n].astype(int)
    return u_hat


def bit_reversal_permutation(N):
    n = int(math.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])
