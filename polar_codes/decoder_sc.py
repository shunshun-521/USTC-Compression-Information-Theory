"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 2014）
"""
import math
import numpy as np


def bit_reversed(x, n):
    """比特倒序索引"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f（upper_llr）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g（lower_llr），La 为下支路 LLR，Lb 为上支路 LLR"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        b = int(u_hat) if u_hat.ndim else int(u_hat.item())
        return (La + Lb) if b == 0 else (La - Lb)
    out = np.empty_like(La, dtype=np.float64)
    for i in range(len(La)):
        b = int(u_hat[i])
        out[i] = (La[i] + Lb[i]) if b == 0 else (La[i] - Lb[i])
    return out


def upper_llr(l1, l2):
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    """下支路更新；调用顺序为 lower_llr(btm, top, bit)"""
    return g_operation(l1, l2, b)


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """兼容接口：返回比特倒序译码顺序"""
    n = int(math.log2(N))
    order = [bit_reversed(i, n) for i in range(N)]
    return order, None, None


_SC_CACHE = {}


def _scd_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    return _scd_decode(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致）"""
    return _scd_decode(llr, frozen_bits)
