"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from decoder_utils_internal import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用精确 boxplus）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = np.array(
            [upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)]
        )
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset: bit_offset + half]
        llr_right = np.array(
            [
                lower_llr(llr_node[i], llr_node[i + half], u_left[i])
                for i in range(half)
            ]
        )
        decode_node(llr_right, bit_offset + half)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (i - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        for l in range(n):
            if ((phi >> l) & 1) == 0:
                layers_llr = list(range(l, n))
                break
        else:
            layers_llr = list(range(n))
        llr_layer_vec.append(layers_llr)

        layers_bit = [l for l in range(n) if ((phi >> l) & 1) == 1]
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCD:
    """非递归 SC 译码内核（Permuted SC 调度）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    def decode(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(L, B, l)
            if l in self.frozen_set:
                B[l, self.n] = 0
            else:
                B[l, self.n] = 0 if L[l, self.n] >= 0 else 1
            self._update_bits(B, l)

        u_hat = B[:, self.n].astype(int)
        return u_hat

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    N = len(llr_ch)
    return _SCD(N, frozen_bits).decode(llr_ch)
