"""
极化码 SC（串行抵消）译码器
Permuted SCD（Vangala et al. 2014），与蝶形编码 u*F^{\\otimes n} 一致
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 风格，供 SCL/BP 复用）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed(i, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    if abs(l1) > 30 or abs(l2) > 30:
        return float(f_operation(l1, l2))
    return _logdomain_diff(l1 + l2, 0.0) - _logdomain_diff(l1, l2)


def _lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    """MSB 起第一个 1 的位置（与 polarcodes 一致）"""
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
    """MSB 起第一个 0 的位置"""
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
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n - active_bit_level(phi, n), n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PermutedSCD:
    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return _PermutedSCD(len(llr_ch), frozen_bits).decode(llr_ch)


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
