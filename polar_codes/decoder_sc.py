"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n 风格）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def hard_decision(y):
    return 0 if y >= 0 else 1


def logdomain_sum(a, b):
    """log(exp(a) + exp(b)) 的稳定实现。"""
    if a == -np.inf and b == -np.inf:
        return -np.inf
    if a == np.inf or b == np.inf:
        return np.inf
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def upper_llr(l1, l2):
    """f 运算（对数域精确 boxplus）。"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算。"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _permute_channel_llr(llr_ch, N):
    """编码端含比特倒序时，将信道 LLR 映射到译码树自然顺序。"""
    br = bit_reversal_permutation(N)
    ibr = np.argsort(br)
    return llr_ch[ibr]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _permute_channel_llr(np.asarray(llr, dtype=np.float64), len(llr))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, fbits, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if fbits[0]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = hard_decision(llr_node[0])
            return
        half = n // 2
        llr_left = np.array([upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)])
        decode_node(llr_left, fbits[:half], offset)
        u_left = u_hat[offset : offset + half]
        llr_right = np.array(
            [lower_llr(llr_node[i], llr_node[i + half], u_left[i]) for i in range(half)]
        )
        decode_node(llr_right, fbits[half:], offset + half)

    decode_node(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = bit_reversed(phi, n)
        start_llr = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_llr, n)))

        start_bit = n - active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, start_bit, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr = _permute_channel_llr(llr_ch, N)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed(phi, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = hard_decision(L[l, n])
            B[l, n] = u_hat[l]

        if l < N / 2:
            continue
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
