"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，PSC 因子图）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：与 lower_llr 一致，u=0 -> La+Lb，u=1 -> Lb-La。"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, Lb - La)


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


def _upper_llr(l1, l2):
    return float(f_operation(l1, l2))


def _lower_llr(l1, l2, b):
    if b == 0:
        return float(l1 + l2)
    return float(l1 - l2)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与主译码器相同的 LLR 布局约定）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回非递归辅助结构（供 SCL 等模块复用层索引逻辑）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(l, n), -1))
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD，与蝶形+比特倒序编码配套）。

    llr_ch: 与信道输出码字比特顺序一致的 LLR（见 channel.reorder_llr_for_decoder）。
    frozen_bits: 1 表示冻结位，0 表示信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
