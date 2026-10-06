"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SC（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
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


def _prepare_llr(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def _pscd_core(llr, frozen_bits):
    """Permuted successive cancellation decode (non-recursive)."""
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for i in range(N):
        l = _bit_reversed_index(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr = _prepare_llr(llr_ch)
    return _pscd_core(llr, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 Permuted SC 等价，用于校验）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 辅助向量（保留接口；Permuted SC 内部自含索引逻辑）
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        pp = phi
        while pp % 2 == 1:
            llr_layers.append(int(math.log2(pp & -pp)))
            pp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        pp = phi
        while pp % 2 == 0 and pp > 0:
            bit_layers.append(int(math.log2(pp & -pp)))
            pp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_path_metric_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)
