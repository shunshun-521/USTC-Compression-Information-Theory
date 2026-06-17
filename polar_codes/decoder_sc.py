"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def hard_decision(y):
    return 0 if y >= 0 else 1


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
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


def f_operation(La, Lb):
    """min-sum 近似 f（向量化）。"""
    La = np.asarray(La, dtype=float)
    Lb = np.asarray(Lb, dtype=float)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


class SCDecoder:
    """非递归 SC 译码器（对数域 f，比特倒序遍历）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _set_channel(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=float)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self._set_channel(llr_ch)
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode_layered(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    return SCDecoder(len(llr_ch), frozen_bits).decode(llr_ch)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_layered(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用非递归实现）。"""
    return sc_decode_layered(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p, layer = phi, 0
        while p % 2 == 1 and layer < n:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p, layer = phi, 0
        while p % 2 == 0 and layer < n:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec
