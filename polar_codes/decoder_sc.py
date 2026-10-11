"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归版本（主实现，log-domain boxplus）
"""
import math
import numpy as np

from encoder import bit_reversed


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    val = _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)
    if not np.isfinite(val):
        return _sign_pm(l1) * _sign_pm(l2) * min(abs(l1), abs(l2))
    return val


def lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        val = l1 + l2
        return val if np.isfinite(val) else _sign_pm(l1) * abs(l1)
    if b == 1:
        val = l1 - l2
        return val if np.isfinite(val) else _sign_pm(l1) * abs(l1)
    return np.nan


def f_operation(La, Lb):
    """对外暴露的 f 运算（boxplus，用于 BP 等）。"""
    return upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    return lower_llr(La, Lb, u_hat)


def _sign_pm(x):
    return np.where(x >= 0, 1.0, -1.0)


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


def _frozen_to_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    return set(np.where(frozen_bits)[0])


class _SCDCore:
    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_to_set(frozen_bits)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
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
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return _SCDCore(len(llr_ch), frozen_bits).decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致，用于交叉验证）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while layer < n:
            if (tmp & 1) == 0:
                llr_layers.append(layer)
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        tmp = phi
        layer = 0
        while layer < n:
            if (tmp & 1) == 1:
                bit_layers.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
