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
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(btm, top, bit):
    return btm + top if bit == 0 else btm - top


def _channel_llr_to_decoder(llr_ch, N):
    inv = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    N = len(llr)
    n = int(math.log2(N))
    llr_in = _channel_llr_to_decoder(llr, N)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_in

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - half, s], int(B[j - half, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 2 ** s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** (layer - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec, br


_SC_CACHE = {}


def _get_sc_tables(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr_in = _channel_llr_to_decoder(llr_ch, N)
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])

    _, llr_layer_vec, bit_layer_vec, _ = _get_sc_tables(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_in
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed_index(phi, n)

        for s in llr_layer_vec[phi]:
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - half, s], int(B[j - half, s + 1])
                    )

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            u_hat[l] = bit
            B[l, n] = bit

        if l >= N / 2:
            for s in bit_layer_vec[phi]:
                block = 2 ** s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                        B[j, s - 1] = B[j, s]

    return u_hat
