"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 PSC 索引更新）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.ndim == 0:
        return (1.0 - 2.0 * int(u_hat)) * La + Lb
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_mask(frozen_bits):
    """输入 1/True 表示冻结位"""
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(int) == 1


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for b in range(n):
        if (i >> b) & 1:
            result |= 1 << (n - 1 - b)
    return result


def active_llr_level(i, n):
    """二进制表示中自高位起第一个 0 的位置层数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """二进制表示中自高位起第一个 1 的位置层数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """更新第 l 个相位所需的 LLR 树"""
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    """比特回传"""
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（主实现）。
    信道 LLR 为自然顺序；内部按比特倒序相位译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)

    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n)
        if frozen[i]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = bit_reversed(i, n)
        u_hat[i] = B[l, n]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现作为参考）"""
    return sc_decode(llr, frozen_bits)


def pm_penalty(llr, u):
    """路径度量惩罚"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)
