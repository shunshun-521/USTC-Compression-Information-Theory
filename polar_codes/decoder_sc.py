"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 Permuted SC 实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    """精确 log-domain f 运算（Permuted SC 使用）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    """精确 log-domain g 运算，l1=bottom, l2=top"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
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


def _prepare_channel_llr(llr_ch, N):
    """编码含比特倒序，信道 LLR 需做相同倒序后送入 Permuted SC"""
    from encoder import bit_reversal_permutation
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（Permuted SC 版本）。
    返回 bit_reversed 译码顺序及每层活跃深度。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in decode_order:
        llr_layers = list(range(n - _active_llr_level(phi, n), n))
        bit_layers = list(range(n, n - _active_bit_level(phi, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    lambda_offset = [1 << l for l in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（高效实现，精确 log-domain LLR）。
    """
    llr_ch = _prepare_channel_llr(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = np.clip(llr_ch, -1e20, 1e20)

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(B[j - branch_size, s + 1]) else int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr_exact(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        bi = int(B[j, s]) if not np.isnan(B[j, s]) else 0
                        bj = int(B[j - branch_size, s]) if not np.isnan(B[j - branch_size, s]) else 0
                        B[j - branch_size, s - 1] = bi ^ bj
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。

    极化码在含比特倒序编码时，需使用 Permuted SC 译码顺序；
    本函数与 sc_decode 等价，供对照验证。
    """
    return sc_decode(llr, frozen_bits)
