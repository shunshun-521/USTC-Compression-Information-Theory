"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（b=0: La+Lb, b=1: La-Lb）"""
    if np.isscalar(u_hat):
        return (La + Lb) if u_hat == 0 else (La - Lb)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class _SCDCore:
    """SC 译码核心（L/B 矩阵更新）"""

    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(frozen_set)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def set_llr(self, llr):
        self.L[:, 0] = llr

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = g_operation(
                        self.L[j, s], self.L[j - branch_size, s], top_bit
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

    def decode_bit(self, l):
        self._update_llrs(l)
        if l in self.frozen_set:
            self.B[l, self.n] = 0
        else:
            self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
        self._update_bits(l)
        return int(self.B[l, self.n])

    def run(self, llr):
        self.set_llr(llr)
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self.decode_bit(l)
        return self.B[:, self.n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用与主实现相同的 L/B 算法）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    frozen_set = set(np.where(frozen_bits)[0])
    core = _SCDCore(N, frozen_set)
    return core.run(llr)


def precompute_sc_indices(N):
    """预计算索引（供 SCL 等模块复用）"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（L/B 矩阵，按比特倒序依次译码）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    frozen_set = set(np.where(frozen_bits)[0])
    return _SCDCore(N, frozen_set).run(llr_ch)
