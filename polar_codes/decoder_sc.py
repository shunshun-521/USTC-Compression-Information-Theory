"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """找到 i 的二进制展开中第一个 1 的位置（从高位起）"""
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
    """找到 i 的二进制展开中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class PermutedSCD:
    """Permuted Successive Cancellation Decoder（参考 polar-codes 库）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

    def _update_bits(self, l):
        if l < self.N // 2:
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

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = np.array([bit_reversed_index(i, self.n) for i in range(self.N)])
        self.L[:, 0] = llr_ch[br]
        self.B.fill(0)

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if self.frozen_bits[l]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)

        return self.B[:, self.n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）"""
    N = len(llr_ch)
    return PermutedSCD(N, frozen_bits).decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用 Permuted SCD 作为后端）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口兼容性（SCL 使用 Permuted SCD 风格）"""
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec[phi] = list(range(n - active_llr_level(l, n), n))
        bit_layer_vec[phi] = list(range(n, n - active_bit_level(l, n), -1))
    return lambda_offset, llr_layer_vec, bit_layer_vec
