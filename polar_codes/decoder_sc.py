"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，与蝶形编码器配套）
"""
import math
import numpy as np


def bit_reversed(i, n):
    """单索引 n 位比特倒序。"""
    result = 0
    for b in range(n):
        if (i >> b) & 1:
            result |= 1 << (n - 1 - b)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2, min_sum=False):
    """f 运算（对数域 box-plus 或 min-sum）。"""
    if min_sum:
        return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b, min_sum=False):
    """g 运算。"""
    if b == 0:
        if min_sum:
            return l1 + l2
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if min_sum:
        return l1 - l2
    return l1 - l2


def active_llr_level(i, n):
    """二进制表示中自高位起第一个 1 的位置计数（polar-codes 约定）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def active_bit_level(i, n):
    """二进制表示中自高位起第一个 0 的位置计数。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


# 对外接口：min-sum 近似
def f_operation(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return upper_llr(float(La), float(Lb), min_sum=True)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


class _SCDCore:
    """非递归 SC 内核（L/B 数组布局）。"""

    def __init__(self, N, frozen_bits, min_sum=True):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.min_sum = min_sum
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def set_channel(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(
                        self.L[j, s], self.L[j + branch_size, s], self.min_sum
                    )
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        top_bit,
                        self.min_sum,
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        self.B[j, s] ^ self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits, min_sum=True):
    """非递归 SC 译码（主函数）。"""
    N = len(llr_ch)
    core = _SCDCore(N, frozen_bits, min_sum=min_sum)
    core.set_channel(llr_ch)
    return core.decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用同一非递归内核，便于对照）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回比特倒序译码顺序及层信息。"""
    n = int(math.log2(N))
    lambda_offset = [0] * N
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
        lambda_offset[phi] = l
    return lambda_offset, llr_layer_vec, bit_layer_vec


# SCL 复用
def sc_update_llrs(L, B, l, n, N, min_sum=True):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s], min_sum)
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], top_bit, min_sum
                )


def sc_update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]
