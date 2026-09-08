"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala 置换 SC，高效实现）
"""
import math
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域精确 box-plus）。"""
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(btm_llr, top_llr, top_bit):
    """g 运算（bottom, top, bit）。"""
    if top_bit == 0:
        return btm_llr + top_llr
    if top_bit == 1:
        return btm_llr - top_llr
    return np.nan


def hard_decision(y):
    return 0 if y >= 0 else 1


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            return 0 if frozen_bits[idx] else (0 if llr_node[0] >= 0 else 1)
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = [decode_node(llr_left[i : i + 1], bit_offset + i) for i in range(half)]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = [
            decode_node(llr_right[i : i + 1], bit_offset + half + i) for i in range(half)
        ]
        return u_left + u_right

    return np.array(decode_node(llr, 0), dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        while tmp % 2 == 1:
            llr_layers.append(int(math.log2(tmp & -tmp)) - 1)
            tmp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        tmp = phi
        while tmp % 2 == 0 and tmp < N:
            bit_layers.append(int(math.log2(tmp & -tmp)) if tmp > 0 else 0)
            tmp >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDEngine:
    """Vangala 置换 SC 译码引擎（与 reference SCD 一致）。"""

    def __init__(self, N, n, llr_ch, frozen_indices):
        self.N = N
        self.n = n
        self.frozen = frozen_indices
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def decode(self):
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Vangala 置换 SC 译码（高效实现）。
    frozen_bits: True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_indices = np.where(frozen_bits)[0]
    engine = _SCDEngine(N, n, llr_ch, frozen_indices)
    return engine.decode()
