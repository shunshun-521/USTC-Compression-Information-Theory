"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def _sign_pm(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
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


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，连续半块分裂）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_block(llr_block, offset):
        m = len(llr_block)
        if m == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_block[0] >= 0 else 1
            return
        half = m // 2
        decode_block(f_operation(llr_block[:half], llr_block[half:]), offset)
        decode_block(
            g_operation(llr_block[:half], llr_block[half:], u_hat[offset : offset + half]),
            offset + half,
        )

    decode_block(llr, 0)
    return u_hat


class _SCDState:
    """非递归 SC 内核（比特倒序相位更新）"""

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
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s], self.L[j, s], top_bit
                    )

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
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(l)
            if self.frozen_bits[l]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = _hard_decision(self.L[l, self.n])
            self._update_bits(l)
        return self.B[:, self.n].copy()


def precompute_sc_indices(N):
    """保留接口：非递归实现使用 SCD 内核，仍导出索引供 SCL 参考"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n) + 1, n + 1)))
        bit_layer_vec.append(list(range(1, active_bit_level(l, n) + 1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（自然信道序 LLR 经比特倒序重排后译码）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = np.array([bit_reversed(i, n) for i in range(N)], dtype=np.int64)
    llr_ordered = np.asarray(llr_ch, dtype=np.float64)[rev]
    return _SCDState(N, frozen_bits).decode(llr_ordered)
