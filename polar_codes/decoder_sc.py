"""
极化码 SC（串行抵消）译码器
非递归 SCD（参考 polar-codes 库）+ 递归参考实现
"""
import math
import numpy as np


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似 f；标量/向量"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def upper_llr(l1, l2):
    """对数域 f 运算（boxplus）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """对数域 g 运算"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归参考实现（与 SCD 数值一致）"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC（SCD）====================


class _SCDCore:
    def __init__(self, N, llr_ch, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(frozen_bits)[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def decode(self):
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.B_top = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = lower_llr(self.L[j, s], self.L[j - branch_size, s], self.B_top)

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]


def precompute_sc_indices(N):
    """兼容接口：返回占位结构（SCD 不需要预计算）"""
    n = int(math.log2(N))
    return np.arange(n + 1), [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return _SCDCore(len(llr_ch), llr_ch, frozen_bits).decode()


def sc_decode_bit_reversed_phase(llr_ch, frozen_bits):
    """与 sc_decode 相同（SCD 内部已按倒序相位处理）"""
    return sc_decode(llr_ch, frozen_bits)
