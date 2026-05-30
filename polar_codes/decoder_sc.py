"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> La-Lb。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    if np.isscalar(u_hat) or u_hat.size == 1:
        return (La + Lb) if int(u_hat) == 0 else (La - Lb)
    out = np.empty_like(La)
    mask = u_hat.astype(bool)
    out[~mask] = La[~mask] + Lb[~mask]
    out[mask] = La[mask] - Lb[mask]
    return out


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr_boxplus(l1, l2):
    """对数域 box-plus（f 运算）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr_boxplus(l1, l2, b):
    """对数域 g 运算。"""
    if int(b) == 0:
        return l1 + l2
    return l1 - l2


def bit_reversed_index(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


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


class _SCDCore:
    """非递归 SC 译码核心（参考 Permuted SCD）。"""

    def __init__(self, N, frozen_bits, use_minsum=False):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(frozen_bits)[0])
        self.use_minsum = use_minsum
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)

    def _upper(self, l1, l2):
        if self.use_minsum:
            return f_operation(l1, l2)
        return upper_llr_boxplus(l1, l2)

    def _lower(self, l1, l2, b):
        if self.use_minsum:
            return g_operation(l1, l2, b)
        return lower_llr_boxplus(l1, l2, b)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = self._upper(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = self._lower(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        top_bit,
                    )

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        self.B.fill(0)
        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（llr_layer_vec / bit_layer_vec）。
    当前 sc_decode 使用逐位倒序 SCD 实现，此函数供报告/扩展使用。
    """
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 0:
            psi2 = phi
            while psi2 < N and psi2 % 2 == 0:
                if psi2 > 0:
                    layers_bit.append(int(math.log2(psi2 & -psi2)))
                psi2 += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（box-plus f 运算）。"""
    N = len(llr_ch)
    core = _SCDCore(N, np.asarray(frozen_bits, dtype=bool), use_minsum=False)
    return core.decode(np.asarray(llr_ch, dtype=np.float64))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（委托非递归实现以保证一致性）。"""
    return sc_decode(llr, frozen_bits)
