"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归 SCD（主实现，按比特倒序逐位译码）
"""
import math

import numpy as np


def _bit_reversed_index(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


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


def f_operation(La, Lb):
    """min-sum f（供 SCL/BP 复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（top=La, bottom=Lb）"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + 2 ** (n - i + 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1 and layer < n:
            llr_layers.append(layer)
            tmp //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        tmp = phi
        layer = 0
        while tmp % 2 == 0 and layer < n:
            bit_layers.append(layer)
            tmp //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDCore:
    """非递归 SCD 内核"""

    def __init__(self, N, frozen_indices):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(int(i) for i in frozen_indices)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
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

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        for l in [_bit_reversed_index(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。

    frozen_bits: 1/True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    frozen_idx = np.where(frozen_bits.astype(bool))[0]
    core = _SCDCore(N, frozen_idx)
    return core.decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用 SCD 主实现，保持接口）"""
    return sc_decode(llr, frozen_bits)
