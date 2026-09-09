"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 2014）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：La 为上半分支，Lb 为下半分支。
    u=0: La+Lb；u=1: Lb-La
    """
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La, dtype=np.float64)
    mask0 = u_hat == 0
    out[mask0] = La[mask0] + Lb[mask0]
    out[~mask0] = Lb[~mask0] - La[~mask0]
    if np.isscalar(La) and np.isscalar(Lb) and np.isscalar(u_hat):
        return float(out)
    return out


def _to_frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        return fb.astype(bool)
    return fb


def _bit_reversed(i, n):
    return int(bit_reversal_permutation(1 << n)[i])


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


def _hard_decision(y):
    return 0 if y >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（Arikan 自然序分块递归，min-sum f 运算）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _to_frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_block, frozen_block, offset):
        n = len(llr_block)
        if n == 1:
            if frozen_block[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = _hard_decision(llr_block[0])
            return

        half = n // 2
        llr_left = f_operation(llr_block[:half], llr_block[half:])
        decode_block(llr_left, frozen_block[:half], offset)

        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_block[:half], llr_block[half:], u_left)
        decode_block(llr_right, frozen_block[half:], offset + half)

    decode_block(llr, frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Vangala 2014 置换 SC 译码器）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _to_frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    if top_bit == 0:
                        L[j, s + 1] = btm_llr + top_llr
                    else:
                        L[j, s + 1] = btm_llr - top_llr

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec = [
        list(range(n - _active_llr_level(_bit_reversed(phi, n), n), n))
        for phi in range(N)
    ]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(_bit_reversed(phi, n), n), -1))
        for phi in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec
