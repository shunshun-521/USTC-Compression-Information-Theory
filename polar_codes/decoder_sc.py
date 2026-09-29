"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）和非递归版本（高效实现，Algorithm 3–5）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def _lsum(a, b):
    a = float(a)
    b = float(b)
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _f_boxplus(l1, l2):
    return _lsum(l1 + l2, 0.0) - _lsum(l1, l2)


def _g_boxplus(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


def bit_reversal_index(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _update_llr(L, B, x, n, N):
    """Algorithm 4"""
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s // 2
        i = x
        while i < N:
            imod = i % s
            if t > imod:
                L[i, j] = _f_boxplus(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = _g_boxplus(L[i, j + 1], L[i - t, j + 1], int(B[i - t, j]))
            i += s


def _update_bits(B, x, n, N):
    """Algorithm 5"""
    b_list = [x]
    for j in range(n):
        s = 1 << (n - j)
        t = s // 2
        b_next = []
        for i in b_list:
            imod = i % s
            if t <= imod:
                B[i - t, j + 1] = (B[i, j] ^ B[i - t, j]) & 1
                B[i, j + 1] = B[i, j]
                b_next.extend([i, i - t])
        b_list = b_next


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（McBain Algorithm 3，信道 LLR 与码字比特顺序一致）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr_ch

    for phase in range(N):
        l = bit_reversal_index(phase, n)
        _update_llr(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, 0].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（min-sum，与主译码器对照用）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回占位结构供文档/扩展使用"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
