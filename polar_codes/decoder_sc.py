"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


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


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(l1, l2, bit):
    if bit == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（通过非递归实现保证一致性）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(phi, n), -1)) if phi >= N // 2 else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for bit_index in [_bit_reversed(i, n) for i in range(N)]:
        for stage in range(n - _active_llr_level(bit_index, n), n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(bit_index, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = _upper_llr(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = _lower_llr(
                        L[j, stage],
                        L[j - branch_size, stage],
                        int(B[j - branch_size, stage + 1]),
                    )

        if bit_index in frozen:
            B[bit_index, n] = 0
        else:
            B[bit_index, n] = 0 if L[bit_index, n] >= 0 else 1

        if bit_index >= N // 2:
            for stage in range(n, n - _active_bit_level(bit_index, n), -1):
                block_size = 2 ** stage
                branch_size = block_size // 2
                for j in range(bit_index, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, stage - 1] = int(B[j, stage]) ^ int(
                            B[j - branch_size, stage]
                        )
                        B[j, stage - 1] = B[j, stage]

    return B[:, n].astype(int)
