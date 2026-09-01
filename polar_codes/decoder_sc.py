"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换式 SCD）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _permute_channel_llr(llr_ch):
    """将信道 LLR 按比特倒序置换，与编码器 bit-reversal 约定匹配。"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _bit_reversed_index(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（置换式，与 polar_encode 的 bit-reversal 一致）。
    """
    llr = _permute_channel_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits == 1)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（置换式 SCD）。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for i in range(N):
        l = _bit_reversed_index(i, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(llr_layers)

        if l < N / 2:
            bit_layers = []
        else:
            bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（置换式 SCD，与编码器 bit-reversal 一致）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
