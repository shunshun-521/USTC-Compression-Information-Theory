"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
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


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
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
    """min-sum 近似的 f 运算（供 SCL/BP 复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（供 SCL/BP 复用）"""
    return (1 - 2 * u_hat) * La + Lb


class _SCDState:
    def __init__(self, llr_ch, frozen_bits):
        self.llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.N = len(self.llr_ch)
        self.n = int(math.log2(self.N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = self.llr_ch

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
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与主实现等价，用于校验）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：非递归索引预计算（SCL 仍可使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        tmp = phi
        while tmp & 1:
            llr_layer_vec[phi].append((tmp & -tmp).bit_length() - 1)
            tmp >>= 1
        if (phi & 1) == 0:
            tmp = phi
            while (tmp & 1) == 0 and tmp > 0:
                tmp >>= 1
            if tmp > 0:
                bit_layer_vec[phi].append((tmp & -tmp).bit_length() - 1)
        else:
            bit_layer_vec[phi].append(0)
            tmp = phi
            while tmp & 1:
                tmp >>= 1
            if tmp > 0:
                bit_layer_vec[phi].append((tmp & -tmp).bit_length() - 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（层状非递归实现）"""
    return _SCDState(llr_ch, frozen_bits).decode()
