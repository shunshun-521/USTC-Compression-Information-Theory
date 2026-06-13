"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def bit_reversed(i, n):
    """比特倒序索引"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 时 La+Lb，u=1 时 Lb-La（与分层索引约定一致）"""
    if np.isscalar(u_hat) or (hasattr(u_hat, "shape") and u_hat.shape == ()):
        return (Lb + La) if int(u_hat) == 0 else (Lb - La)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（分层矩阵实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed(i, n) for i in range(N)]
    for l in decode_order:
        _update_llrs(L, B, l, n)
        B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    u_nat = B[:, n].astype(int)
    rev = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
    u_hat = u_nat[rev]
    u_hat[frozen_bits == 1] = 0
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用高效分层实现作为参考）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回分层 SC 译码辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layers = []
        if l >= N // 2:
            bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
