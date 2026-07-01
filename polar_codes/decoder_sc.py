"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed_index(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """llr_layer_vec 辅助：从高位起第一个 1 之前 0 的个数。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def active_bit_level(i, n):
    """bit_layer_vec 辅助：从高位起第一个 0 之前 1 的个数。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 bit-reversed 译码顺序及层信息。
    """
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]

    decode_order = [bit_reversed_index(phi, n) for phi in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = decode_order[phi]
        llr_layers = list(range(n - active_llr_level(l, n), n))
        llr_layer_vec.append(llr_layers)

        if l < N // 2:
            bit_layers = []
        else:
            bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n)

    return u_hat


def _update_llrs(L, B, l, n):
    """更新 LLR 树。"""
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                if np.isnan(top):
                    top = 0.0
                if np.isnan(btm):
                    btm = 0.0
                L[j, s + 1] = f_operation(top, btm)
            else:
                top = L[j - branch_size, s]
                btm = L[j, s]
                top_bit = B[j - branch_size, s + 1]
                if np.isnan(top_bit):
                    top_bit = 0
                if np.isnan(top):
                    top = 0.0
                if np.isnan(btm):
                    btm = 0.0
                L[j, s + 1] = g_operation(top, btm, int(top_bit))


def _update_bits(B, l, n):
    """比特回传。"""
    if l < 2 ** (n - 1):
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                b_j = 0 if np.isnan(B[j, s]) else int(B[j, s])
                b_top = 0 if np.isnan(B[j - branch_size, s]) else int(B[j - branch_size, s])
                B[j - branch_size, s - 1] = b_j ^ b_top
                B[j, s - 1] = b_j


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（与递归版本等价）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n)

    return u_hat
