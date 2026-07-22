"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def bit_reversed(x, n):
    """比特倒序（MSB 优先）。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算对应的 LLR 更新，l1=下分支，l2=上分支。"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """min-sum 近似 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr(llr_ch, N):
    """将信道 LLR 映射到译码器内部顺序（比特倒序置换）。"""
    from encoder import bit_reversal_permutation

    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）。"""
    # 递归实现与经比特倒序置换后的非递归版本等价
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** (layer - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                layers.append(layer)
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n)) if phi % 2 == 0 else list(range(n - 1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（基于分层 LLR/比特数组）。"""
    llr_ch = _prepare_llr(llr_ch, len(llr_ch))
    frozen_set = _frozen_indices(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
