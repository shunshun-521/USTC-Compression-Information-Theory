"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 L/B 数组）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def channel_llr_to_decoder(llr_ch):
    """
    将信道输出顺序的 LLR 映射为 SC 译码树所需的顺序（与比特倒序编码配套）。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    if np.isinf(l1):
        return l2
    if np.isinf(l2):
        return l1
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def hard_decision(y):
    return 0 if y >= 0 else 1


def lower_llr(l1, l2, b):
    if b != 0 and b != 1:
        return np.nan
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def bit_reversed_index(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（递归参考实现用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits, llr_is_channel_order=False):
    """递归 SC 译码（参考）。默认 llr 已为译码树顺序；可设 llr_is_channel_order=True。"""
    llr = np.asarray(llr, dtype=np.float64)
    if llr_is_channel_order:
        llr = channel_llr_to_decoder(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 的层索引（与分层 SCD 一致，供文档/扩展使用）"""
    n = int(math.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - active_llr_level(bit_reversed_index(phi, n), n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(bit_reversed_index(phi, n), n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_decode_core(llr, frozen_bits):
    """非递归 SC 核心（log-domain f/g）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = hard_decision(L[l, n])

        if l < N / 2:
            continue
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（信道自然序 LLR 输入）"""
    llr_dec = channel_llr_to_decoder(llr_ch)
    return _scd_decode_core(llr_dec, frozen_bits)
