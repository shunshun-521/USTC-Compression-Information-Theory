"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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
    return (1 - 2 * u_hat) * La + Lb


def _is_frozen(frozen_bits, idx):
    return bool(frozen_bits[idx])


def _active_llr_level(i, n):
    """二进制表示中自高位起第一个 1 之前 0 的个数（参考 SCD 实现）"""
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
    """二进制表示中自高位起第一个 0 之前 1 的个数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _upper_llr(l1, l2):
    return float(f_operation(l1, l2))


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组（自然信道顺序，对应编码后的码字位置）
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位（置 0）
    返回：
        u_hat: 长度 N 的估计源序列
    """
    N = len(llr)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    # 将信道 LLR 映射到译码树叶子（与比特倒序编码对应）
    llr_tree = llr[br].copy()

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_tree

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    decode_order = [int(bit_reversal_permutation(N)[i]) for i in range(N)]
    for l in decode_order:
        update_llrs(l)
        if _is_frozen(frozen_bits, l):
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * N
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        p = phi
        layer = 0
        while p % 2 == 1:
            llr_layer_vec[phi].append(layer)
            p //= 2
            layer += 1
        llr_layer_vec[phi].append(layer)

        p = phi
        layer = 0
        while p % 2 == 0 and p > 0:
            bit_layer_vec[phi].append(layer)
            p //= 2
            layer += 1

        lambda_offset[phi] = phi >> 1

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于分层 L/B 数组，与递归版本算法一致）。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（float64）
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（0/1 int 数组）
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch[br]

    decode_order = [int(br[i]) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if _is_frozen(frozen_bits, l):
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
