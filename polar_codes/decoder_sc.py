"""
极化码 SC（串行抵消）译码器
非递归置换 SC（Vangala），与 encoder.polar_encode 配套。
"""
import numpy as np

from encoder import bit_reversal_permutation
from utils import logdomain_sum


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _decoder_llr_order(llr_ch, N):
    """将信道 LLR 重排为 SC 因子图顺序（与含 B_N 的编码配套）。"""
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_block(llr_node, fbits):
        n = len(llr_node)
        if n == 1:
            if fbits[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)
        half = n // 2
        llr_u = np.array([upper_llr(a, b) for a, b in zip(llr_node[:half], llr_node[half:])])
        u_left = decode_block(llr_u, fbits[:half])
        llr_u_prime = np.array(
            [lower_llr(a, b, ui) for a, b, ui in zip(llr_node[:half], llr_node[half:], u_left)]
        )
        u_right = decode_block(llr_u_prime, fbits[half:])
        return np.concatenate([u_left, u_right])

    return decode_block(llr, frozen_bits.astype(bool))


def precompute_sc_indices(N):
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(
            [] if l < N // 2 else list(range(n, n - active_bit_level(l, n), -1))
        )
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])
    llr = _decoder_llr_order(llr_ch, N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
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
        B[l, n] = 0 if l in frozen_set else (0 if L[l, n] >= 0 else 1)
        update_bits(l)

    return B[:, n].astype(int)
