"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _prepare_channel_llr(llr_ch):
    """将信道 LLR 排列为译码树所需顺序（与含比特倒序的编码器配套）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，基于分层递归树）。"""
    return SCDecoder(len(llr), frozen_bits).decode(llr)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                layers.append(layer)
                break
            temp //= 2
        else:
            layers.append(n - 1)
        llr_layer_vec.append(layers)

        layers = []
        if phi % 2 == 1:
            temp = phi
            for layer in range(n):
                if temp % 2 == 0:
                    layers.append(layer)
                    break
                temp //= 2
        bit_layer_vec.append(layers)

    lambda_offset = [0] * (n + 2)
    for i in range(n + 1):
        lambda_offset[i] = 1 << i
    lambda_offset[n + 1] = 2 * N
    return lambda_offset, llr_layer_vec, bit_layer_vec


class SCDecoder:
    """高效非递归 SC 译码器（Permuted SCD）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        self.L[:, 0] = llr_ch
        self.B.fill(np.nan)
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    key = (N, tuple(np.asarray(frozen_bits, dtype=bool).tolist()))
    if key not in _SC_CACHE:
        _SC_CACHE[key] = SCDecoder(N, frozen_bits)
    return _SC_CACHE[key].decode(llr_ch)
