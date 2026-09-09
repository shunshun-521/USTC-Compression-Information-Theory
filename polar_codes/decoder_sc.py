"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 PSCD 实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def bit_reversed(x, n):
    """比特倒序（单索引）"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def active_llr_level(i, n):
    """找到二进制展开中第一个 1 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """找到二进制展开中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用 PSCD）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回：lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(math.log2(N))
    lambda_offset = [1 << l for l in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSCD 译码主函数（Permuted Successive Cancellation Decoder）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l_idx = bit_reversed(i, n)

        for s in range(n - active_llr_level(l_idx, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l_idx in frozen_set:
            B[l_idx, n] = 0
        else:
            B[l_idx, n] = 0 if L[l_idx, n] >= 0 else 1
        u_hat[l_idx] = B[l_idx, n]

        if l_idx < N // 2:
            continue

        for s in range(n, n - active_bit_level(l_idx, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
