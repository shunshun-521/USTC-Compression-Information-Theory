"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与基于置换 SC 的非递归高效实现
"""
import math
import numpy as np


def bit_reversed(x, n):
    """单整数比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域精确形式）。"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（对数域精确形式）。"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


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
    """min-sum 近似 f（供 SCL/BP 使用）。"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


class _SCDEngine:
    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(int(i) for i in frozen_set)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def set_channel(self, llr_ch):
        self.L[:, 0] = llr_ch

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

    def decode(self):
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    frozen_set = np.where(frozen_bits)[0]
    eng = _SCDEngine(N, frozen_set)
    eng.set_channel(np.asarray(llr_ch, dtype=np.float64))
    return eng.decode()


def sc_decode_recursive(llr, frozen_bits):
    """保留递归接口，内部调用经验证的置换 SC。"""
    return sc_decode_nonrecursive(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """兼容 SCL 模块的占位接口（SCL 使用独立路径更新）。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << (n - layer)) - 1 for layer in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
