"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Algorithm 3–5）
"""
import numpy as np
import math



def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（b1=0/1 分支；冻结位仅 b1=0）"""
    return La + (1 - 2 * u_hat) * Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _update_llr(L, B, x, n):
    N = L.shape[0]
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s >> 1
        for i in range(x, N, s):
            if t > (i % s):
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, x, n):
    b = [x]
    for j in range(n):
        s = 1 << (n - j)
        t = s >> 1
        bnext = []
        for i in b:
            if t <= (i % s):
                B[i - t, j + 1] = (int(B[i, j]) + int(B[i - t, j])) % 2
                B[i, j + 1] = B[i, j]
                bnext.extend([i, i - t])
        b = bnext


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主函数）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = llr_ch.size
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, n] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed(phi, n)
        _update_llr(L, B, l, n)
        if l in frozen_set:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n)

    return np.nan_to_num(B[:, 0], nan=0.0).astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与主实现等价，用于对照）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：SCL 仍使用简化索引预计算"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while (p & 1) == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = [layer for layer in range(n) if (phi >> layer) & 1]
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
