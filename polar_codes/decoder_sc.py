"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，对数域 f/g）
"""
import math
import numpy as np

from encoder import bit_reversed_index


def hard_decision(llr):
    return 0 if llr >= 0 else 1


def logdomain_sum(a, b):
    if np.isinf(a) and a < 0:
        return b
    if np.isinf(b) and b < 0:
        return a
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """对外接口：min-sum 近似（向量化）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


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
    """保留接口：返回比特倒序译码顺序"""
    n = int(math.log2(N))
    order = [bit_reversed_index(i, n) for i in range(N)]
    return order, None, None


class _SCDState:
    def __init__(self, N, n, llr_ch, frozen_set):
        self.N = N
        self.n = n
        self.frozen_set = frozen_set
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    state = _SCDState(N, n, llr_ch, frozen_set)
    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = bit_reversed_index(i, n)
        state.update_llrs(l)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = hard_decision(state.L[l, n])
        u_hat[l] = int(state.B[l, n])
        state.update_bits(l)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（min-sum，自然序 LLR）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat
