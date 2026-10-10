"""
极化码 SC（串行抵消）译码器
Permuted SCD（与 encoder 的 F^{⊗ n} 编码一致）
"""
import math
import numpy as np


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


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


def _frozen_index_set(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(fb)[0])


def _sc_core(llr_ch, frozen_set, N, use_minsum=False):
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    def f_llr(a, b):
        if use_minsum:
            return float(f_operation(a, b))
        return _upper_llr_exact(a, b)

    def g_llr(a, b, bit):
        if use_minsum:
            return float(g_operation(a, b, bit))
        return _lower_llr_exact(a, b, bit)

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            branch = block >> 1
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_llr(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_llr(L[j, s], L[j - branch, s], B[j - branch, s + 1])
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 1 << s
                branch = block >> 1
                for j in range(l, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                        B[j, s - 1] = B[j, s]
    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits, use_minsum=False):
    """非递归 Permuted SC 译码（默认精确 log-domain f/g；可选 min-sum）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_set = _frozen_index_set(frozen_bits)
    return _sc_core(llr_ch, frozen_set, N, use_minsum=use_minsum)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（调用与主译码器相同的 Permuted SCD，便于对照）"""
    return sc_decode(llr_ch, frozen_bits, use_minsum=False)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（层活跃深度）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = []
    off = 0
    for layer in range(n + 1):
        lambda_offset.append(off)
        off += 1 << (n - layer)
    return lambda_offset, llr_layer_vec, bit_layer_vec
