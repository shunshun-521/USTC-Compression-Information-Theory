"""
极化码 SC（串行抵消）译码器
非递归实现（与 polarcodes.SCD 结构一致），f 运算为 min-sum
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La,Lb,u) = (1-2u)La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """自 MSB 起第一个 0 之前已扫描的层数（与 polarcodes 一致）"""
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
    """自 MSB 起第一个 0 的层计数（比特回传用）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码，返回估计源序列 u_hat"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits.astype(bool))[0])

    for i in range(N):
        l = br[i]
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """与 sc_decode 相同（保留接口）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """SCL 使用的层列表（近似）"""
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [
        list(range(n - _active_llr_level(br[i], n), n)) for i in range(N)
    ]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(br[i], n), -1)) if br[i] >= N // 2 else []
        for i in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec
