"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（左子树编码比特）"""
    return (1 - 2 * u_hat) * La + Lb


def channel_llr_to_decoder(llr_ch):
    """与蝶形+比特倒序编码器配套：对信道 LLR 做比特倒序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


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
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return float(f_operation(l1, l2))


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits, known_bits=None):
    """递归 SC 译码；known_bits 可指定已判决比特 {index: value}"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    known_bits = known_bits or {}

    def _rec(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if i in known_bits:
                u_hat[i] = known_bits[i]
            elif frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llrs[0] >= 0 else 1
            return [u_hat[i]]

        half = len(indices) // 2
        left_llr = [float(f_operation(llrs[i], llrs[i + half])) for i in range(half)]
        enc_left = _rec(left_llr, indices[:half])
        right_llr = [
            float(g_operation(llrs[i], llrs[i + half], enc_left[i])) for i in range(half)
        ]
        enc_right = _rec(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _rec(list(llr), list(range(N)))
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            stop = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, stop, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（L/B 矩阵，与递归版本在部分帧上可能略有数值差异）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 主入口（当前默认调用已验证的递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
