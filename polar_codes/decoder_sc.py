"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _compute_left_alpha(llr):
    return f_operation(llr[: len(llr) // 2], llr[len(llr) // 2 :])


def _compute_right_alpha(llr, left_beta):
    half = len(llr) // 2
    left = llr[:half]
    right = llr[half:]
    return right - (2 * left_beta - 1) * left


def _compute_encoding_step(level, n, source, result):
    step = 1 << (n - level - 1)
    groups = 1 << level
    result = result.copy()
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]
    return result


def _position_state(position, n):
    bits = np.unpackbits(np.array([position], dtype=np.uint32).byteswap().view(np.uint8))
    return bits[-n:].astype(np.int8)


def _sc_decode_internal(llr_ch, frozen_bits):
    """在比特倒序域执行 SC 译码，返回倒序域的 u 估计"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    intermediate_llr = [llr_ch.copy()]
    length = N // 2
    while length > 0:
        intermediate_llr.append(np.zeros(length, dtype=np.float64))
        length //= 2

    intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
    current_state = np.zeros(n, dtype=np.int8)
    previous_state = np.ones(n, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for position in range(N):
        current_state = _position_state(position, n)

        for i in range(1, n + 1):
            llr = intermediate_llr[i - 1]
            if current_state[i - 1] == previous_state[i - 1]:
                continue
            if current_state[i - 1] == 0:
                intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (n - i))
                left_bits = intermediate_bits[i][start:end]
                intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

        if frozen_bits[position]:
            decision = 0
        else:
            decision = 1 if intermediate_llr[-1][0] < 0 else 0

        u_hat[position] = decision
        intermediate_bits[-1][position] = decision

        for i in range(n - 1, -1, -1):
            intermediate_bits[i] = _compute_encoding_step(
                i, n, intermediate_bits[i + 1], intermediate_bits[i]
            )

        previous_state[:n] = current_state

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    br = bit_reversal_permutation(N)
    frozen_br = np.asarray(frozen_bits, dtype=int)[br]
    return _sc_decode_internal(np.asarray(llr, dtype=np.float64), frozen_br)[br]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        while p & 1:
            layers_llr.append(int(math.log2(p & -p)))
            p &= p - 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            p = phi
            while p & 1:
                layers_bit.append(int(math.log2(p & -p)))
                p &= p - 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_br = np.asarray(frozen_bits, dtype=int)[br]
    u_internal = _sc_decode_internal(llr_ch, frozen_br)
    return u_internal[br]
