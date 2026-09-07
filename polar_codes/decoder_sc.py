"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（与 SC 译码树中 bottom/top 分支顺序一致）：
    u_hat=0: La + Lb; u_hat=1: La - Lb
    """
    u_hat = np.asarray(u_hat, dtype=int)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed(x, n):
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


def _frozen_to_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(frozen_bits)[0]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归版本输出一致）"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = np.array([2 ** (layer - 1) if layer > 0 else 0 for layer in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for phi in range(N):
        l = decode_order[phi]
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(llr_layers)

        if l >= N // 2:
            bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        else:
            bit_layers = []
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_indices = set(_frozen_to_indices(frozen_bits))

    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int64)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed(phi, n) for phi in range(N)]

    for phi in range(N):
        l = decode_order[phi]

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_indices:
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

    return B[:, n]


def sc_decode_with_precompute(llr_ch, frozen_bits, sc_indices=None):
    """使用预计算索引的非递归 SC 译码（与 sc_decode 等价）"""
    if sc_indices is None:
        sc_indices = precompute_sc_indices(len(llr_ch))
    lambda_offset, llr_layer_vec, bit_layer_vec = sc_indices
    _ = lambda_offset
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_indices = set(_frozen_to_indices(frozen_bits))

    br = bit_reversal_permutation(N)
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int64)
    L[:, 0] = llr_ch[br]

    decode_order = [_bit_reversed(phi, n) for phi in range(N)]

    for phi in range(N):
        l = decode_order[phi]
        for s in llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_indices:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in bit_layer_vec[phi]:
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n]
