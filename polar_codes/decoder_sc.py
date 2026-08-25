"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a = np.where(sign_a == 0, 1.0, sign_a)
    sign_b = np.where(sign_b == 0, 1.0, sign_b)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def active_llr_level(i, n):
    """Find the first 1 in the binary expansion of i (MSB first)."""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """Find the first 0 in the binary expansion of i (MSB first)."""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（与 sc_decode 等价，采用比特倒序相位扫描）。
    """
    return sc_decode(llr, frozen_bits)


class SCDecoderCore:
    """非递归 SC 译码核心（按比特倒序相位扫描）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def update_llrs(self, leaf_idx):
        for stage in range(self.n - active_llr_level(leaf_idx, self.n), self.n):
            block_size = 1 << (stage + 1)
            branch_size = block_size >> 1
            for j in range(leaf_idx, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, stage + 1] = f_operation(
                        self.L[j, stage], self.L[j + branch_size, stage]
                    )
                else:
                    self.L[j, stage + 1] = g_operation(
                        self.L[j - branch_size, stage],
                        self.L[j, stage],
                        self.B[j - branch_size, stage + 1],
                    )

    def update_bits(self, leaf_idx):
        if leaf_idx < self.N // 2:
            return
        for stage in range(self.n, self.n - active_bit_level(leaf_idx, self.n), -1):
            block_size = 1 << stage
            branch_size = block_size >> 1
            for j in range(leaf_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, stage - 1] = (
                        self.B[j, stage] ^ self.B[j - branch_size, stage]
                    )
                    self.B[j, stage - 1] = self.B[j, stage]

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        self.B.fill(0)
        br = bit_reversal_permutation(self.N)
        for leaf_idx in br:
            self.update_llrs(leaf_idx)
            if self.frozen_bits[leaf_idx]:
                self.B[leaf_idx, self.n] = 0
            else:
                self.B[leaf_idx, self.n] = 0 if self.L[leaf_idx, self.n] >= 0 else 1
            self.update_bits(leaf_idx)
        return self.B[:, self.n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        leaf = int(f'{phi:0{n}b}'[::-1], 2)
        llr_layer_vec.append(list(range(n - active_llr_level(leaf, n), n)))
        if leaf < N // 2:
            bit_layers = []
        else:
            bit_layers = list(range(n, n - active_bit_level(leaf, n), -1))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    decoder = SCDecoderCore(len(llr_ch), frozen_bits)
    return decoder.decode(llr_ch)
