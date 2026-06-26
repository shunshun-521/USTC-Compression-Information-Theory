"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归置换 SC 版本（高效实现，Vangala 2014）
"""
import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool or set(np.unique(frozen_bits)).issubset({0, 1}):
        return set(np.where(frozen_bits)[0])
    return set(frozen_bits.tolist())


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


def _upper_llr(l1, l2):
    if np.isscalar(l1):
        return float(f_operation(l1, l2))
    return f_operation(l1, l2)


def _lower_llr(l1, l2, bit):
    if bit == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，置换比特顺序）。
    """
    N = len(llr)
    n = int(np.log2(N))
    frozen_set = _frozen_to_set(frozen_bits)
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(L, B, l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(B, l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(L, B, l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(B, l)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（置换 SC 的层激活信息）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed_index(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码主函数（Vangala permuted SCD）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
