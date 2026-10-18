"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> Lb-La"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与极化变换因子图一致）。
    frozen_bits: True/1 表示冻结位
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = np.zeros(N, dtype=np.int8)

    def sc_block(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return np.array([decisions[i]], dtype=np.int8)

        half = len(indices) // 2
        left_llr = np.array(
            [f_operation(llrs[i], llrs[i + half]) for i in range(half)], dtype=np.float64
        )
        enc_left = sc_block(left_llr, indices[:half])
        right_llr = np.array(
            [g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)],
            dtype=np.float64,
        )
        enc_right = sc_block(right_llr, indices[half:])
        return np.array(
            [enc_left[i] ^ enc_right[i] for i in range(half)] + list(enc_right), dtype=np.int8
        )

    sc_block(llr, list(range(N)))
    return decisions


def _active_llr_level(i, n):
    """polarcodes: 从 MSB 起连续 0 的个数 + 1（上限 n）"""
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
    """polarcodes: 从 MSB 起连续 1 的个数 + 1（上限 n）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（参考 polarcodes.SCD）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    def update_llrs(l):
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block_size = 2 ** (s + 1)
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
                    L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        stop = n - _active_bit_level(l, n)
        for s in range(n, stop, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=np.int8)
    for i in range(N):
        l = rev[i]
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat


def precompute_sc_indices(N):
    """保留接口：非递归实现使用 polarcodes 风格，无需预计算。"""
    n = int(math.log2(N))
    return [1] * (n + 1), [[] for _ in range(N)], [[] for _ in range(N)]
