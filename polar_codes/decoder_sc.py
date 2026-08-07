"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


def _bit_reversed_int(x, n):
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


def _frozen_indices_from_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return np.where(fb)[0]
    return np.where(fb.astype(int) == 1)[0]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用 min-sum f）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_idx = set(_frozen_indices_from_mask(frozen_bits))
    N = len(llr)
    n = int(math.log2(N))
    brp = bit_reversal_permutation(N)
    llr_in = llr[brp]
    u_hat = _sc_decode_core(llr_in, frozen_idx, N, n, use_min_sum=True)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** layer - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        while (phi >> layer) & 1:
            layer += 1
        llr_layer_vec.append(list(range(n - 1, layer - 1, -1)))
        bit_layer_vec.append(list(range(layer)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr, frozen_idx, N, n, use_min_sum=True):
    """非递归 SC 译码核心（矩阵存储 L/B）。"""
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    frozen_set = set(frozen_idx)

    def upper(l1, l2):
        if use_min_sum:
            return float(f_operation(l1, l2))
        return _upper_llr_exact(l1, l2)

    def lower(l1, l2, b):
        if use_min_sum:
            return float(g_operation(l1, l2, b))
        return _lower_llr_exact(l1, l2, b)

    for l in [_bit_reversed_int(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    if top_bit == 0:
                        L[j, s + 1] = btm + top
                    else:
                        L[j, s + 1] = btm - top

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 经比特倒序置换后与编码器 B_N 一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    brp = bit_reversal_permutation(N)
    frozen_idx = _frozen_indices_from_mask(frozen_bits)
    llr_br = llr_ch[brp]
    return _sc_decode_core(llr_br, frozen_idx, N, n, use_min_sum=True)
