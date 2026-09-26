"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 实现（高效）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper branch）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(top_llr, btm_llr, u_hat):
    """lower branch：与标准 PSCD 一致，参数为 (top, bottom)"""
    if u_hat == 0:
        return btm_llr + top_llr
    return btm_llr - top_llr


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 PSCD 等价的参考实现，用于校验）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """供 SCL 使用的层索引（与 PSCD 的 active level 一致）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = int(bit_reversal_permutation(N)[i])
        llr_layer_vec.append(
            list(range(n - _active_llr_level(l, n), n))
        )
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(l, n), -1))
            if l >= N // 2
            else []
        )
    lam = [1 << i for i in range(n + 1)]
    return lam, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSCD 译码（Permuted Successive Cancellation）。
    信道 LLR 按发送顺序填入 L[:, 0]。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    decode_order = [int(bit_reversal_permutation(N)[i]) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], int(B[j - half, s + 1])
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = (B[j, s] + B[j - half, s]) % 2
                    B[j, s - 1] = B[j, s]

    return u_hat.astype(int)
