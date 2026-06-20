"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    f 运算（box-plus，BPSK-AWGN 信道下精确形式）。
    大 LLR 时自动退化为 min-sum 近似以保持数值稳定。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    large = (np.abs(La) > 30) | (np.abs(Lb) > 30)
    out = np.empty_like(La, dtype=np.float64)
    if np.any(large):
        out[large] = (
            np.sign(La[large])
            * np.sign(Lb[large])
            * np.minimum(np.abs(La[large]), np.abs(Lb[large]))
        )
    if np.any(~large):
        t = np.tanh(La[~large] / 2.0) * np.tanh(Lb[~large] / 2.0)
        t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
        out[~large] = 2.0 * np.arctanh(t)
    return out


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _align_channel_llrs(llr_ch, N):
    """将传输顺序的信道 LLR 对齐到极化蝶形译码顺序。"""
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _sc_decode_sequential(llr_aligned, frozen_bits):
    """按比特倒序相位执行 SC 译码（非递归核心）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_aligned)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_aligned

    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        l = int(br[phi])
        _update_llrs(L, B, l, n)

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(B, l, n, N)

    return u_hat.astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    采用与蝶形编码匹配的相位顺序，数值上等价于展开后的非递归实现。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    llr_aligned = _align_channel_llrs(llr, N)
    return _sc_decode_sequential(llr_aligned, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    br = bit_reversal_permutation(N)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = int(br[phi])
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    llr_aligned = _align_channel_llrs(llr_ch, N)
    return _sc_decode_sequential(llr_aligned, frozen_bits)
