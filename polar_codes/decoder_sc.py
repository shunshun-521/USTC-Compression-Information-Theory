"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）、非递归 permuted SCD（高效实现，与编码 u@G 配套）
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


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
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
    """
    min-sum 近似的 f 运算（SCL/BP 使用）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用 permuted SCD，保持接口一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：permuted SCD 在运行时按需更新，无需预计算表。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


class _PermutedSCD:
    """Permuted successive cancellation decoder（log-domain f/g）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(frozen_bits)[0])

    def decode(self, likelihoods):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, 0] = likelihoods

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(L, B, l)
            if l in self.frozen_set:
                B[l, self.n] = 0
            else:
                B[l, self.n] = 0 if L[l, self.n] >= 0 else 1
            self._update_bits(B, l)

        return B[:, self.n].astype(int)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（permuted SCD）。
    llr_ch: 与信道输出 x 逐位对应的 LLR（长度 N）
    frozen_bits: True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    decoder = _PermutedSCD(N, frozen_bits)
    return decoder.decode(llr_ch)


def channel_llr_to_decoder(llr_ch, N):
    """编码为 u@G 时，信道 LLR 无需重排。"""
    return np.asarray(llr_ch, dtype=np.float64)
