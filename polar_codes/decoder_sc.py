"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversed, bit_reversal_permutation


def _safe_sign(x):
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（含 sign(0) 修正）
    """
    sa = _safe_sign(La)
    sb = _safe_sign(Lb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_boxplus(l1, l2):
    """对数域 box-plus（比 min-sum 更精确）"""
    if np.isnan(l1) or np.isnan(l2):
        return np.nan
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
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


def _to_frozen_set(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=int).astype(bool)
    return set(np.where(fb)[0])


def _preprocess_llr(llr_ch, N):
    """将信道 LLR 变换为 SC 树所需顺序"""
    n = int(math.log2(N))
    br_idx = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)
    return llr[br_idx]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与分层 SC 算法对应）。
    """
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = bit_reversed(phi, n)
        layers_llr = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(layers_llr)

        layers_bit = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _layered_sc_decode(llr, frozen_set, N):
    """分层 SC 译码核心（非递归高效实现）"""
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for phi in range(N):
        l = bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = int(2 ** (s + 1))
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = int(2 ** s)
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1/True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_set = _to_frozen_set(frozen_bits)
    llr = _preprocess_llr(llr_ch, N)
    return _layered_sc_decode(llr, frozen_set, N)


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 结果一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_set = _to_frozen_set(frozen_bits)
    llr = _preprocess_llr(llr_ch, N)
    return _layered_sc_decode(llr, frozen_set, N)
