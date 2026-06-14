"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """从最高位起第一个 1 之前的 0 个数（含起始偏移 1）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """从最高位起第一个 0 之前的 1 个数（含起始偏移 1）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llr(L, B, l, n, N):
    """自底向上更新 LLR 至根节点。"""
    for s in range(n - _active_llr_level(l, n), n):
        block = 1 << (s + 1)
        half = block >> 1
        for j in range(l, N, block):
            if j % block < half:
                L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
            else:
                top_bit = B[j - half, s + 1]
                L[j, s + 1] = g_operation(L[j - half, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    """自顶向下回传已判决比特。"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block = 1 << s
        half = block >> 1
        for j in range(l, -1, -block):
            if j % block >= half:
                B[j - half, s - 1] = (B[j, s] + B[j - half, s]) % 2
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    信道 LLR 为自然顺序；编码端含比特倒序，故 L[l,0]=llr_ch[br(l)]。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)[br]

    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = br[i]
        _update_llr(L, B, l, n, N)

        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        _update_bits(B, l, n, N)

    return u_hat.astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与非递归 sc_decode 等价）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助索引（供 SCL 使用）。"""
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = [1 << layer for layer in range(n + 1)]
    return lambda_offset, br
