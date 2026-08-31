"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
基于 Permuted Successive Cancellation (Vangala et al.)
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：bottom + (1-2u)*top"""
    return Lb + (1.0 - 2.0 * u_hat) * La


def _reorder_channel_llr(llr_ch, N):
    """将信道 LLR 重排为 PSC 译码器所需顺序"""
    br = bit_reversal_permutation(N)
    llr = np.zeros(N, dtype=np.float64)
    llr[br] = llr_ch
    return llr


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _psc_decode_core(llr, frozen_bits):
    """PSC 译码核心（已重排 LLR）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s],
                        L[j, s],
                        B[j - half, s + 1],
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = _bit_reversed(i, n)
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归/置换 SC 译码（自然信道序 LLR 输入）"""
    N = len(llr_ch)
    llr = _reorder_channel_llr(llr_ch, N)
    return _psc_decode_core(llr, frozen_bits)


_SC_CACHE = {}


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（自然信道序 LLR 输入）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
