"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：与 log-domain 中 b±a 形式等价"""
    return (Lb + La) if u_hat == 0 else (Lb - La)


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


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（Arikan / dit 形式，输入为信道顺序 LLR）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llrs = llr_ch[rev].tolist()
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    decisions = [0] * N

    def _f(a, b):
        return float(f_operation(a, b))

    def _g(a, b, u):
        return float(g_operation(a, b, u))

    def rec(llrs_local, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs_local[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [_f(llrs_local[i], llrs_local[i + half]) for i in range(half)]
        enc_left = rec(left_llr, indices[:half])
        right_llr = [_g(llrs_local[i], llrs_local[i + half], enc_left[i]) for i in range(half)]
        enc_right = rec(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    rec(llrs, list(range(N)))
    return np.array(decisions, dtype=int)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码索引（与逐比特倒序译码顺序一致）"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    llr_ch：信道输出顺序的 LLR（与编码后的码字比特顺序一致）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    rev = bit_reversal_permutation(N)
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch[rev]

    u_hat = np.zeros(N, dtype=int)

    for phase in range(N):
        l = _bit_reversed(phase, n)
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top = L[j, s]
                    btm = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(top, btm)
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top, btm, top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        end = n - _active_bit_level(l, n)
        for s in range(n, end, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
