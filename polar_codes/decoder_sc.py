"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，SCD 风格）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """SCD: 找 i 的二进制表示中第一个 1 的位置（从高位起）。"""
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
    """SCD: 找 i 的二进制表示中第一个 0 的位置（从高位起）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    使用中间阶段部分和 u_up 进行 g 运算（与 Sionna/标准因子图一致）。
    """
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(llr_vec, frozen_vec):
        n = len(llr_vec)
        if n == 1:
            if frozen_vec[0]:
                bit = 0
            else:
                bit = 0 if llr_vec[0] >= 0 else 1
            return np.array([bit], dtype=int), np.array([bit], dtype=int)

        half = n // 2
        llr_left_in = f_operation(llr_vec[:half], llr_vec[half:])
        u_hat_left, u_up_left = decode(llr_left_in, frozen_vec[:half])
        llr_right_in = g_operation(llr_vec[:half], llr_vec[half:], u_up_left)
        u_hat_right, u_up_right = decode(llr_right_in, frozen_vec[half:])
        u_hat = np.concatenate([u_hat_left, u_hat_right])
        u_up = np.concatenate([u_up_left ^ u_up_right, u_up_right])
        return u_hat, u_up

    u_hat, _ = decode(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat


# ==================== 非递归 SC 译码（高效实现，SCD 风格）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        start_bit = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, start_bit, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCD:
    """SCD 风格非递归 SC 译码器内部状态。"""

    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(frozen_set)
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s], self.L[j, s], top_bit
                    )

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = self.B[j, s] ^ self.B[j - branch_size, s]
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self.update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（SCD 风格）。
    """
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    scd = _SCD(N, frozen_set)
    scd.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
    return scd.decode()
