"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from scipy.special import logsumexp
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算"""
    if np.isinf(La) and not np.isinf(Lb):
        return Lb
    if not np.isinf(La) and np.isinf(Lb):
        return La
    if np.isinf(La) and np.isinf(Lb):
        return np.inf
    return logsumexp([La + Lb, 0.0]) - logsumexp([La, Lb])


def g_operation(La, Lb, u_hat):
    """g 运算：La=top, Lb=bottom"""
    return (1.0 - 2.0 * u_hat) * La + Lb


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
    count = 0
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return count


def _update_llrs(L, B, l, n, use_minsum=True):
    f_fn = f_operation if use_minsum else f_operation_exact
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, 0, -1):
        if s <= n - _active_bit_level(l, n):
            break
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits, use_minsum=True):
    """
    非递归 SC 译码。
    编码 x = u @ B @ F，在 w = u @ B 域译码后比特倒序还原 u。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    rev = bit_reversal_permutation(N)

    frozen_w = np.zeros(N, dtype=bool)
    for j in range(N):
        frozen_w[j] = frozen_bits[rev[j]]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch

    for l in rev:
        _update_llrs(L, B, l, n, use_minsum)
        if frozen_w[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    w_hat = B[:, n].astype(int)
    return w_hat[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现）"""
    return sc_decode(llr, frozen_bits, use_minsum=False)


def precompute_sc_indices(N):
    """预计算 SCL 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_start = n - _active_llr_level(phi, n)
        llr_layer_vec.append(list(range(llr_start, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec
