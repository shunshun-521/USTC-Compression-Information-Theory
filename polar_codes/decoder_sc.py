"""
极化码 SC（串行抵消）译码器
基于分层 LLR 的非递归实现（自然顺序相位）
"""
import math
import numpy as np
from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum f 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_minsum(La, Lb):
    return f_operation(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：right = Lb - (2*u-1)*La"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb - (2 * u_hat - 1) * La


def _compute_left_alpha(llr):
    half = len(llr) // 2
    return f_operation(llr[:half], llr[half:])


def _compute_right_alpha(llr, left_beta):
    half = len(llr) // 2
    left = llr[:half]
    right = llr[half:]
    return right - (2 * left_beta - 1) * left


def _compute_encoding_step(level, n, source, result):
    result = result.copy()
    step = 1 << (n - level - 1)
    groups = 1 << level
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]
    return result


def _position_bits(position, n):
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bits[n - 1 - i] = (position >> i) & 1
    return bits


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return None, list(range(N)), None


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    info_mask = ~frozen_bits

    intermediate_llr = [llr_ch.copy()]
    length = N // 2
    while length > 0:
        intermediate_llr.append(np.zeros(length, dtype=np.float64))
        length //= 2

    intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
    current_state = np.zeros(n, dtype=np.int8)
    previous_state = np.ones(n, dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)

    for position in range(N):
        current_state = _position_bits(position, n)

        for i in range(1, n + 1):
            if current_state[i - 1] == previous_state[i - 1]:
                continue
            llr = intermediate_llr[i - 1]
            if current_state[i - 1] == 0:
                intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (n - i))
                left_bits = intermediate_bits[i][start:end]
                intermediate_bits[i][start:end] = left_bits
                intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

        if info_mask[position]:
            decision = 1 if intermediate_llr[-1][0] < 0 else 0
        else:
            decision = 0
        u_hat[position] = decision
        intermediate_bits[-1][position] = decision

        for i in range(n - 1, -1, -1):
            intermediate_bits[i] = _compute_encoding_step(
                i, n, intermediate_bits[i + 1], intermediate_bits[i]
            )

        previous_state = current_state.copy()

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考）"""
    return sc_decode(llr, frozen_bits)


def sc_decode_permuted(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
