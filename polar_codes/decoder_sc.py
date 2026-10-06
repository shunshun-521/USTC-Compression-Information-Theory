"""
极化码 SC（串行抵消）译码器
Permuted SCD（log-domain），与 encoder.polar_encode 配套
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 使用）"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def _bit_reversed(x, n):
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


def _llr_layer_update(L, C, lam, N):
    h = 1 << (lam - 1)
    for j in range(0, N, h << 1):
        for jj in range(h):
            a = j + jj
            b = a + h
            L[lam - 1, a] = f_operation(L[lam, a], L[lam, b])
            L[lam - 1, b] = g_operation(L[lam, a], L[lam, b], C[lam, a])


def _bit_layer_update(C, lam, phi, N):
    h = 1 << (lam - 1)
    base = (phi >> lam) << lam
    for jj in range(h):
        a = base + jj
        b = a + h
        C[lam, a] = (C[lam - 1, a] + C[lam - 1, b]) % 2
        C[lam, b] = C[lam - 1, b]


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（与 Permuted SCD 层更新对应）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(1, n + 1))
        else:
            llr_layers = []
            t = phi + 1
            while t % 2 == 0:
                llr_layers.append(int(math.log2(t & -t)))
                t >>= 1
            llr_layers = sorted(set(llr_layers))
        bit_layers = []
        t = phi
        l = 0
        while t % 2 == 1:
            bit_layers.append(l + 1)
            t >>= 1
            l += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_core(llr_ch, frozen_indices):
    """Permuted successive-cancellation 译码，返回长度 N 的 u_hat"""
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    frozen = set(int(i) for i in frozen_indices)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（调用 Permuted SCD 作为参考）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口。
    frozen_bits: 长度 N，1 表示冻结位（置 0），0 表示信息位。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_idx = np.where(frozen_bits == 1)[0]
    return _scd_core(np.asarray(llr_ch, dtype=np.float64), frozen_idx)
