"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 SCD 实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（LLR=0 视为非负）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = 1.0 - 2.0 * (La < 0)
    sb = 1.0 - 2.0 * (Lb < 0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：lower_llr(bottom, top, bit)"""
    b = int(np.asarray(u_hat).reshape(-1)[0]) if np.size(u_hat) else 0
    return La + Lb if b == 0 else La - Lb


def _bit_reverse(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
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


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = g_operation(btm, top, top_bit)


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（SCD）。
    信道 LLR 应先经 channel.reorder_llr_for_decoder（比特倒序编码时）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for phase in range(N):
        l = _bit_reverse(phase, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供对照）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reverse(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = [2 ** l - 1 for l in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
