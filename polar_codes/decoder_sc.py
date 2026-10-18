"""
极化码 SC（串行抵消）译码器
Permuted SC（Vangala），与含比特倒序的 polar_encode 配套
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：与标准 SC 一致"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    result = 0
    for b in range(n):
        if (i >> b) & 1:
            result |= 1 << (n - 1 - b)
    return result


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


def _frozen_set_from_mask(frozen_bits):
    """约定：frozen_bits[i] 为 1/True 表示冻结位"""
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        fb = fb.astype(bool)
    return set(np.where(fb)[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（用于无比特倒序的编码，主要作参考）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = [None] * N

    def _f(a, b):
        return float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))

    def _g(a, b, u):
        return b - a if u else b + a

    def sc_sub(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]
        half = len(indices) // 2
        left_llr = [_f(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = sc_sub(left_llr, indices[:half])
        right_llr = [_g(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)]
        enc_right = sc_sub(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    sc_sub(list(llr), list(range(N)))
    return np.array(decisions, dtype=int)


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block = 1 << (s + 1)
        half = block >> 1
        for j in range(l, N, block):
            if j % block < half:
                L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
            else:
                L[j, s + 1] = g_operation(L[j - half, s], L[j, s], B[j - half, s + 1])


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block = 1 << s
        half = block >> 1
        for j in range(l, -1, -block):
            if j % block >= half:
                B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC。
    信道 LLR 为码字自然顺序（与 polar_encode 输出一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    rev = bit_reversal_permutation(N)
    frozen_set = _frozen_set_from_mask(frozen_bits)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch[rev]

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
