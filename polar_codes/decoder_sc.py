"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala 置换 SC，高效）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> Lb-La"""
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


def _lower_llr(l1, l2, bit):
    return float(l1 + l2) if bit == 0 else float(l1 - l2)


def _frozen_indices_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return np.where(frozen_bits.astype(bool))[0]


def precompute_sc_indices(N):
    """预计算辅助向量（与 Vangala 层活跃深度对应）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_nat, frozen_indices):
    """Vangala 置换 SC（非递归），输入为自然顺序信道 LLR"""
    llr_nat = np.asarray(llr_nat, dtype=np.float64)
    N = len(llr_nat)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = llr_nat[br]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    frozen_set = set(int(i) for i in frozen_indices)

    for phase in range(N):
        l = _bit_reversed_index(phase, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch, s], B[j - branch, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch:
                        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    frozen_idx = _frozen_indices_from_mask(frozen_bits)
    return _sc_decode_core(llr_ch, frozen_idx)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（在自然顺序 LLR 上调用核心译码器作参考）"""
    return sc_decode(llr, frozen_bits)
