"""
极化码 SC（串行抵消）译码器
非递归 Permuted SCD + 递归包装
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_bit):
    """g 运算（对数域精确形式）"""
    if u_bit == 0:
        return La + Lb
    return La - Lb


def bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << (i - 1) if i > 0 else 0 for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed_int(i, n)
        pm = l
        llr_layers = []
        while pm % 2 == 1 and len(llr_layers) < n:
            llr_layers.append(len(llr_layers))
            pm //= 2
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        pm = l
        while pm % 2 == 0 and pm > 0:
            bit_layers.append(int(round(math.log2(pm & -pm))))
            pm //= 2
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llr(L, B, x, n):
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s // 2
        for i in range(x, L.shape[0], s):
            mod = i % s
            if t > mod:
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, x, n):
    blocks = [x]
    for j in range(n):
        s = 1 << (n - j)
        t = s // 2
        nxt = []
        for i in blocks:
            if t <= i % s:
                B[i - t, j + 1] = int(B[i, j]) ^ int(B[i - t, j])
                B[i, j + 1] = B[i, j]
                nxt.extend([i, i - t])
        blocks = nxt


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    与 encoder.polar_encode（含比特倒序）配套：先对信道 LLR 做比特倒序。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = llr_ch[br]

    frozen = _frozen_mask(frozen_bits)
    frozen_set = set(np.where(frozen)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr

    for i in range(N):
        l = bit_reversed_int(i, n)
        _update_llr(L, B, l, n)
        if l in frozen_set:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, 0].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用非递归实现）"""
    return sc_decode(llr, frozen_bits)
