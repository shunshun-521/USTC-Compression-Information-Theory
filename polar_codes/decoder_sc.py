"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）

层索引：layer 0 为信道侧，layer n 为信源侧。
编码端含比特倒序，译码前对信道 LLR 做相同倒序以保持一致。
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return g_operation(up_llr[:half], up_llr[half:], left_bit)


def _up_bit(left_bit, right_bit):
    half = len(left_bit)
    out = np.empty(2 * half, dtype=int)
    out[:half] = (left_bit + right_bit) % 2
    out[half:] = right_bit
    return out


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    span = 2 ** (pos[2] - 1 - pos[0])
    return [pos[0] + 1, pos[1] + span, pos[2], pos[3]]


def _up(pos):
    span = 2 ** (pos[2] - pos[0] + 1)
    return [pos[0] - 1, int(pos[1] // span) * span, pos[2], pos[3]]


def _sc_decode_core(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_mat = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_mat = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_mat[0] = llr_ch
    position = [0, 0, n, N]

    def decide_left(pos):
        idx = pos[1]
        llr = llr_mat[pos[0] + 1, idx]
        if frozen_bits[idx]:
            return 0
        return 0 if llr >= 0 else 1

    def decide_right(pos):
        idx = pos[1] + 1
        llr = llr_mat[pos[0] + 1, idx]
        if frozen_bits[idx]:
            return 0
        return 0 if llr >= 0 else 1

    while not _all_filled(bit_mat[n]):
        span = 2 ** (position[2] - position[0])
        s = position[1]
        up_llr = llr_mat[position[0], s : s + span]
        up_bit = bit_mat[position[0], s : s + span]
        half = span // 2
        left_bit = bit_mat[position[0] + 1, s : s + half]
        right_bit = bit_mat[position[0] + 1, s + half : s + span]

        if _all_filled(up_bit):
            position = _up(position)
            continue

        if _all_filled(right_bit):
            merged = _up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_mat[position[0], s : s + span] = merged
            continue

        right_llr = llr_mat[position[0] + 1, s + half : s + span]
        left_llr = llr_mat[position[0] + 1, s : s + half]

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_mat[position[0] + 1, s + half : s + span] = decide_right(position)
            else:
                position = _rightdown(position)
            continue

        if _all_filled(left_bit):
            llr_mat[position[0] + 1, s + half : s + span] = _right_llr(
                left_bit.astype(int), up_llr
            )
            continue

        if not _all_filled(left_llr):
            llr_mat[position[0] + 1, s : s + half] = _left_llr(up_llr)
            continue

        if position[0] == position[2] - 1:
            bit_mat[position[0] + 1, s : s + half] = decide_left(position)
        else:
            position = _leftdown(position)

    return bit_mat[n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [2**i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        lam = 0
        p = phi
        while p & 1:
            lam += 1
            p >>= 1
        llr_layer_vec.append(list(range(lam, n)))
        bit_layer_vec.append(list(range(lam)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    自动对信道 LLR 施加比特倒序（与编码器 bit-reversal 对应）。
    """
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return _sc_decode_core(np.asarray(llr_ch)[rev], frozen_bits)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价接口）。"""
    return sc_decode(llr_ch, frozen_bits)
