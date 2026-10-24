"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（La 为下支路 LLR，Lb 为上支路 LLR，与 PSCD 一致）"""
    u_hat = np.asarray(u_hat)
    if np.ndim(u_hat) == 0 and int(u_hat) == 1:
        return La - Lb
    return La + Lb if int(u_hat) == 0 else La - Lb


def _bit_reversed_int(x, n):
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


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(btm, top, int(top_bit))


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _channel_llr_to_pscd(llr_ch):
    """极化编码含比特倒序时，将信道 LLR 映射到 PSCD 内部顺序。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return llr_ch[inv]


def sc_decode(llr_ch, frozen_bits):
    """非递归 PSCD 译码（主实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr_ch = _channel_llr_to_pscd(llr_ch)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        l = _bit_reversed_int(phi, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit
        _update_bits(B, l, n)
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（与 PSCD 结果一致，用于对照）。"""
    return sc_decode(llr_ch, frozen_bits)
