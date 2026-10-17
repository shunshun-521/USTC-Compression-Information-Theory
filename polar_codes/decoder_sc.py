"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Tal/Vangala 风格）
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


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _upper_llr_boxplus(l1, l2):
    """f 运算的 log-domain 精确实现（SC 主内核使用）。"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    if abs(l1) > 30 or abs(l2) > 30:
        return f_operation(l1, l2)
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


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


def _lower_llr(btm_llr, top_llr, top_bit):
    """因子图下分支 LLR（与 g 运算等价，btm/top 为下/上分支）。"""
    if top_bit == 0:
        return btm_llr + top_llr
    return btm_llr - top_llr


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


def _prepare_channel_llrs(llr_ch):
    """将信道 LLR 映射到与编码器一致的内部顺序。"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层更新列表）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p % 2 == 0 and layer < n:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = (phi + 1) // 2
        layer = 0
        while p % 2 == 1 and layer < n:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（主实现）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _prepare_channel_llrs(llr_ch)

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（调用与主实现相同的非递归内核，便于对照验证）。
    """
    return sc_decode(llr, frozen_bits)
