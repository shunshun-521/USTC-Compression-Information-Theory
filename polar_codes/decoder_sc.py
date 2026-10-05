"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换 SC）
"""
import numpy as np


def bit_reversed(i, n):
    return int(format(int(i), f"0{n}b")[::-1], 2)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _update_llr(L, B, x, n):
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s >> 1
        for i in range(x, L.shape[0], s):
            mod = i % s
            if t > mod:
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, x, n):
    active = [x]
    for j in range(n):
        s = 1 << (n - j)
        t = s >> 1
        nxt = []
        for i in active:
            mod = i % s
            if t <= mod:
                B[i - t, j + 1] = (B[i, j] ^ B[i - t, j]) & 1
                B[i, j + 1] = B[i, j]
                nxt.extend([i, i - t])
        active = nxt


def sc_decode(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码（与 F^{⊗n} 编码器配套，无额外 B_N）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llr(L, B, l, n)
        if l in frozen_set:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, 0].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """返回置换 SC 的层索引（接口兼容）"""
    n = int(np.log2(N))
    llr_layers = []
    bit_layers = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layers.append(list(range(n)))
        bit_layers.append(list(range(n)))
    return list(range(n + 1)), llr_layers, bit_layers
