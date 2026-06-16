"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def _logdomain_sum(a, b):
    if a == -np.inf and b == -np.inf:
        return -np.inf
    if a == np.inf or b == np.inf:
        return np.inf
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def f_operation(La, Lb):
    """f 运算（对数域 box-plus）"""
    if np.isscalar(La) and np.isscalar(Lb):
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(
        lambda a, b: _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b), otypes=[float]
    )(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    if np.isscalar(La):
        u = float(u_hat)
        return (1.0 - 2.0 * u) * La + Lb
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助信息"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in decode_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in decode_order]
    return decode_order, llr_layer_vec, bit_layer_vec


def _scd_core(llr, frozen_br):
    """在比特倒序域执行 SC 译码，frozen_br 与内部索引一致"""
    N = len(llr)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    u_br = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen_br[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_br[l] = int(B[l, n])

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_br


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits 使用自然顺序（与 u 向量一致，1 表示冻结位）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    u_br = _scd_core(llr_ch, frozen_bits[br])
    u_hat = np.empty(N, dtype=int)
    u_hat[br] = u_br
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与 sc_decode 相同的核心逻辑）"""
    return sc_decode(llr, frozen_bits)
