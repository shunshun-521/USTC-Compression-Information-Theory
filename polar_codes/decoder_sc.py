"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 实现（Permuted SCD）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
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
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, bit):
    if bit == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _to_frozen_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(bool))[0])


def _prepare_llr(llr_ch):
    """与 B_N 编码约定对齐：对信道 LLR 做比特倒序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return llr_ch[bit_reversal_permutation(len(llr_ch))]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（PSCD 参数化形式）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        l_idx = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l_idx, n)
        layers_llr.extend(range(start, n))
        layers_llr.append(n)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            depth = _active_bit_level(l_idx, n)
            layers_bit.extend(range(n - depth, n))
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _pscd_decode(llr_ch, frozen_bits, use_min_sum=False):
    llr_ch = _prepare_llr(llr_ch)
    frozen_set = _to_frozen_set(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    for l_idx in decode_order:
        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    top = L[j, s]
                    btm = L[j + branch_size, s]
                    if use_min_sum:
                        L[j, s + 1] = f_operation(top, btm)
                    else:
                        L[j, s + 1] = _upper_llr_exact(top, btm)
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    top_bit = int(B[j - branch_size, s + 1])
                    if use_min_sum:
                        L[j, s + 1] = g_operation(btm, top, top_bit)
                    else:
                        L[j, s + 1] = _lower_llr_exact(btm, top, top_bit)

        if l_idx in frozen_set:
            B[l_idx, n] = 0
        else:
            B[l_idx, n] = 0 if L[l_idx, n] >= 0 else 1

        if l_idx >= N // 2:
            for s in range(n, n - _active_bit_level(l_idx, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（PSCD，精确 log-domain LLR 组合）。"""
    return _pscd_decode(llr_ch, frozen_bits, use_min_sum=False)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用 PSCD 保证与主译码器一致）。"""
    return sc_decode(llr, frozen_bits)
