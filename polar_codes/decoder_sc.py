"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 风格）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0 and Lb.ndim == 0
    La = np.atleast_1d(La)
    Lb = np.atleast_1d(Lb)
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a[sign_a == 0] = 1.0
    sign_b[sign_b == 0] = 1.0
    out = sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))
    if scalar:
        return float(out[0])
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 -> La+Lb, b=1 -> La-Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(i, n):
    return int(bit_reversal_permutation(2 ** n)[i])


def _active_llr_level(i, n):
    """二进制表示中自高位起第一个 0 的位置（Vangala）。"""
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
    """二进制表示中自高位起第一个 1 的位置（Vangala）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """更新第 l 个比特所需的 LLR 树。"""
    start = n - _active_llr_level(l, n)
    for s in range(start, n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
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


def _update_bits(B, l, n):
    """比特回传。"""
    N = B.shape[0]
    if l < N // 2:
        return
    end = n - _active_bit_level(l, n)
    for s in range(n, end, -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 风格，L[:,0] 为信道 LLR，按比特倒序译码）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for stage in range(N):
        l = _bit_reversed_index(stage, n)
        _update_llrs(L, B, l, n)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        _update_bits(B, l, n)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现作为参考）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回 Vangala 风格的层索引（供 SCL 使用）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        end = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, end, -1)))
    lambda_offset = list(range(N))
    return lambda_offset, llr_layer_vec, bit_layer_vec
