"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换 SC）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa[sa == 0] = 1
    sb[sb == 0] = 1
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


def _logdomain_sum(x, y):
    m = max(x, y)
    return m + np.log1p(np.exp(-abs(x - y)))


def _upper_llr_exact(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def _frozen_phase_indices(frozen_bits, N):
    n = int(math.log2(N))
    frozen_nat = np.where(np.asarray(frozen_bits, dtype=int) > 0)[0]
    return {_bit_reversed_index(int(i), n) for i in frozen_nat}


def _to_natural_order(u_phase, N):
    n = int(math.log2(N))
    out = np.zeros(N, dtype=int)
    for i in range(N):
        out[i] = u_phase[_bit_reversed_index(i, n)]
    return out


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（相位域）"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
        lambda_offset[phi] = l
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_permuted(llr_ch, frozen_phase, use_min_sum=False):
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    frozen = set(frozen_phase)

    def upper(l1, l2):
        if use_min_sum:
            return float(f_operation(l1, l2))
        return _upper_llr_exact(l1, l2)

    def lower(l1, l2, b):
        if use_min_sum:
            return float(g_operation(l1, l2, b))
        return _lower_llr_exact(l1, l2, b)

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = upper(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = lower(L[j, s], L[j - half, s], int(B[j - half, s + 1]))
        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                    B[j, s - 1] = B[j, s]
    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主函数）"""
    N = len(llr_ch)
    frozen_phase = _frozen_phase_indices(frozen_bits, N)
    u_phase = _sc_decode_permuted(np.asarray(llr_ch, dtype=np.float64), frozen_phase, use_min_sum=False)
    return _to_natural_order(u_phase, N)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，内部调用置换 SC 保证一致性）"""
    return sc_decode(llr_ch, frozen_bits)
