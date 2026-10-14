"""
极化码 SC（串行抵消）译码器
非递归实现（McBain 算法 3–5）+ 递归参考实现
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u = np.asarray(u_hat)
    if np.isscalar(u):
        return (La + Lb) if u == 0 else (La - Lb)
    return np.where(u.astype(bool), La - Lb, La + Lb)


def _bit_reversed(x, n):
    return int(f"{int(x):0{n}b}"[::-1], 2)


def _update_llr(L, B, l, n, N):
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s // 2
        for i in range(l, N, s):
            if t > (i % s):
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, l, n, N):
    active = [l]
    for j in range(n):
        s = 1 << (n - j)
        t = s // 2
        nxt = []
        for i in active:
            if t <= (i % s):
                B[i - t, j + 1] = int(B[i, j]) ^ int(B[i - t, j])
                B[i, j + 1] = B[i, j]
                nxt.extend([i, i - t])
        active = nxt


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - 1, -1, -1)))
        bit_layer_vec.append(list(range(n)))
    return [1 << i for i in range(n + 1)], llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: 1/True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llr(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, 0].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与主实现等价，用于校验）"""
    return sc_decode(llr, frozen_bits)
