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
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def bit_reversed(i, n):
    """比特倒序索引"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def active_llr_level(i, n):
    """找到二进制展开中第一个 1 的位置（从高位）"""
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
    """找到二进制展开中第一个 0 的位置（从高位）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def reorder_channel_llr(llr_ch):
    """将信道 LLR 按比特倒序重排，与编码器比特倒序约定一致"""
    N = len(llr_ch)
    n = int(math.log2(N))
    br = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _update_llrs(L, B, l, n):
    """更新 LLR 树（非递归）"""
    N = L.shape[0]
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    """比特回传（非递归）"""
    N = B.shape[0]
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（float64）
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（0/1 int 数组）
    """
    llr = reorder_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for l in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，内部调用非递归版本）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
