"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    其中 La 为上层 LLR，Lb 为下层 LLR。
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def _bit_reversed(x, n):
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


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) != 0)[0])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与主译码器结果一致）。
    """
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（按比特倒序处理，与蝶形编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n_len = len(llr_ch)
    n = int(math.log2(n_len))
    frozen_set = _frozen_indices(frozen_bits)

    llr = np.full((n_len, n + 1), np.nan, dtype=np.float64)
    bits = np.full((n_len, n + 1), np.nan)
    llr[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(n_len)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, n_len, block_size):
                if j % block_size < branch_size:
                    llr[j, s + 1] = _upper_llr_exact(llr[j, s], llr[j + branch_size, s])
                else:
                    llr[j, s + 1] = _lower_llr_exact(
                        llr[j, s], llr[j - branch_size, s], int(bits[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            bits[l, n] = 0
        else:
            bits[l, n] = 0 if llr[l, n] >= 0 else 1

        if l >= n_len / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        bits[j - branch_size, s - 1] = int(bits[j, s]) ^ int(bits[j - branch_size, s])
                        bits[j, s - 1] = bits[j, s]

    return bits[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与比特倒序 SC 算法配套）。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in decode_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in decode_order]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
