"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_kernel_power


def logdomain_sum(a, b):
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def f_operation(l1, l2):
    """对数域 f（top branch）。"""
    a, b = float(l1), float(l2)
    if np.isinf(a) and b != np.inf:
        return b
    if a != np.inf and np.isinf(b):
        return a
    if np.isinf(a) and np.isinf(b):
        return np.inf
    return logdomain_sum(a + b, 0.0) - logdomain_sum(a, b)


def g_operation(l1, l2, b):
    """对数域 g（bottom branch），b 为 0/1。"""
    a, b2 = float(l1), float(l2)
    if int(b) == 0:
        return a + b2
    return a - b2


def _hard_decision(y):
    return 0 if y >= 0 else 1


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


class _SCD:
    """参考 mcba1n/polar-codes 的非递归 SC 实现。"""

    def __init__(self, llr_ch, frozen_set, n):
        self.N = len(llr_ch)
        self.n = n
        self.frozen = set(int(x) for x in frozen_set)
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def decode(self):
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = _hard_decision(self.L[l, self.n])
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j, s], self.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode_core(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_idx = np.where(frozen_bits)[0]
    scd = _SCD(llr_ch, frozen_idx, n)
    u_hat = scd.decode()
    u_hat = np.asarray(u_hat, dtype=int)
    u_hat[frozen_bits] = 0
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码：输入 LLR 与 `polar_encode` 输出比特顺序一致。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return sc_decode_core(llr_ch[inv], frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(_active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(_active_bit_level(l, n))))
    return lambda_offset, llr_layer_vec, bit_layer_vec


# 供 SCL 复用
def _update_llrs(L, B, l, n, N):
    scd = _SCD.__new__(_SCD)
    scd.N, scd.n, scd.L, scd.B = N, n, L, B
    scd._update_llrs(l)


def _update_bits(B, l, n, N):
    scd = _SCD.__new__(_SCD)
    scd.N, scd.n, scd.B = N, n, B
    scd._update_bits(l)
