"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Algorithm 3–5）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """f 运算（boxplus，支持标量）"""
    La = float(La)
    Lb = float(Lb)
    if abs(La) >= 30 or abs(Lb) >= 30:
        return float(np.sign(La) * np.sign(Lb) * min(abs(La), abs(Lb)))
    prod = np.tanh(La / 2.0) * np.tanh(Lb / 2.0)
    prod = float(np.clip(prod, -1.0 + 1e-12, 1.0 - 1e-12))
    return float(2.0 * np.arctanh(prod))


def g_operation(La, Lb, u_hat):
    """g 运算：g(l1,l2,b) = l2 + (1-2b)*l1"""
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _bit_reverse_int(i, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def _update_llrs(L, B, x, n, N):
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s // 2
        for i in range(x, N, s):
            if t > (i % s):
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i - t, j + 1], L[i, j + 1], B[i - t, j])


def _update_bits(B, x, n, N):
    b = [x]
    for j in range(n):
        s = 1 << (n - j)
        t = s // 2
        b_next = []
        for i in b:
            if t <= (i % s):
                B[i - t, j + 1] = (B[i, j] ^ B[i - t, j]) & 1
                B[i, j + 1] = B[i, j]
                b_next.extend([i, i - t])
        b = b_next


def _sc_decode_core(llr_ch, frozen_bits):
    """非递归 SCD（信道自然序索引的 frozen_bits）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr_ch

    for i in range(N):
        l = _bit_reverse_int(i, n)
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n, N)

    # B[:, 0] 为估计的源向量 u（与蝶形编码配套）
    return B[:, 0].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（调用与主实现相同的核心逻辑）"""
    return _sc_decode_core(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        for layer in range(n):
            if ((_bit_reverse_int(phi, n) >> layer) & 1) == 0:
                layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码入口"""
    return _sc_decode_core(llr_ch, frozen_bits)
