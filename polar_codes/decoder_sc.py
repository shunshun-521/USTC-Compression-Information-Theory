"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与 Permuted SCD（高效、与蝶形编码一致）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> Lb-La"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed(x, n):
    """单整数比特倒序"""
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
    """递归 SC（参考实现，DIT 风格部分和）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = [0] * N

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [float(f_operation(llrs[i], llrs[i + half])) for i in range(half)]
        enc_left = _sc(left_llr, indices[:half])
        right_llr = [
            float(g_operation(llrs[i], llrs[i + half], enc_left[i])) for i in range(half)
        ]
        enc_right = _sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(list(llr), list(range(N)))
    return np.array(decisions, dtype=np.int8)


def sc_decode(llr_ch, frozen_bits):
    """
    Permuted SCD（与蝶形极化编码匹配）。
    frozen_bits: 1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits == 1)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    u_hat = B[:, n].astype(np.int8)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（保留接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layer_vec.append(list(range(n - 1, l - 1, -1)))
        bit_layer_vec.append(list(range(l)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
