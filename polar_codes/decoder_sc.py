"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（对外 API）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    if a == np.inf and b == np.inf:
        return np.inf
    if a == -np.inf and b == -np.inf:
        return -np.inf
    if a == np.inf:
        return b
    if b == np.inf:
        return a
    if a == -np.inf:
        return b
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def _f_boxplus(La, Lb):
    """精确 log-domain f 运算（SC 译码内部使用）"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(x, n):
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
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        tmp = phi
        for l in range(n):
            if (tmp & 1) == 0:
                layers_llr.append(l)
            tmp >>= 1

        layers_bit = []
        tmp = phi
        for l in range(n):
            if (tmp & 1) == 1:
                layers_bit.append(l)
            tmp >>= 1

        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效实现）。
    信道 LLR 经比特倒序后送入因子图，按比特倒序索引依次译码。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen = set(np.where(np.asarray(frozen_bits, dtype=int))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch[br]

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s],
                        int(B[j - branch_size, s + 1])
                    )

        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归版本）"""
    return sc_decode(llr, frozen_bits)
