"""
极化码 SC（串行抵消）译码器
非递归实现（Arikan 因子图，比特倒序相位访问）
"""
import math
import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper branch）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（lower branch）：La=top, Lb=bottom"""
    if u_hat == 0:
        return La + Lb
    return Lb - La


def active_llr_level(i, n):
    """自高位起第一个 1 之前 0 的个数 + 1"""
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
    """自高位起第一个 0 之前 1 的个数 + 1"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class _SCDState:
    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top = self.L[j, s]
                    btm = self.L[j + branch_size, s]
                    self.L[j, s + 1] = f_operation(top, btm)
                else:
                    btm = self.L[j, s]
                    top = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(top, btm, top_bit)

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        self.B[j, s] ^ self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    state = _SCDState(N, n, llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)
        state.update_llrs(l)
        if frozen_bits[l]:
            state.B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if state.L[l, n] >= 0 else 1
            state.B[l, n] = bit
            u_hat[l] = bit
        state.update_bits(l)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    empty = [[] for _ in range(N)]
    return [1 << i for i in range(n + 1)], empty, empty


def _get_sc_cache(N):
    return precompute_sc_indices(N)
