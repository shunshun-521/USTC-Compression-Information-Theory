"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    return max(x, y) + np.log1p(np.exp(-abs(x - y)))


def _upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l2 == np.inf and l1 != np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


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


class _SCD:
    def __init__(self, N, n, frozen_indices, llr):
        self.N = N
        self.n = n
        self.frozen = np.asarray(frozen_indices, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(self, l):
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

    def decode(self):
        for l in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = _hard_decision(self.L[l, self.n])
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（min-sum，仅作参考）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_mask = np.asarray(frozen_bits, dtype=bool)
    frozen_idx = np.where(frozen_mask)[0]
    return _SCD(N, n, frozen_idx, llr_ch).decode()
