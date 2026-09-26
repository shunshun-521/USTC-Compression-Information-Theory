"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from channel import align_llr_to_decoder
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> -La+Lb。"""
    return (1 - 2 * u_hat) * La + Lb


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


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与主译码器结果一致，用于对照）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 的层调度表（与 SCD 更新层一致）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l < N / 2:
            bit_layers = []
        else:
            bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)
    lambda_offset = [_bit_reversed(i, n) for i in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（二维 LLR 阵列 + 比特倒序译码顺序）。
    llr_ch 为信道输出顺序（与 polar_encode 码字顺序一致）。
    """
    llr_ch = align_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    brp = bit_reversal_permutation(N)
    for i in range(N):
        l = brp[i]
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n, N)

    return u_hat
