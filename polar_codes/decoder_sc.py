"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    """将 n 位索引 i 做比特倒序。"""
    result = 0
    for bit in range(n):
        if (i >> bit) & 1:
            result |= 1 << (n - 1 - bit)
    return result


def _active_llr_level(i, n):
    """llr 更新起始层（首个 1 位之前含该位的层数）。"""
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
    """比特回传起始层。"""
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
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode(llr_right, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode(llr, frozen_bits)


class _SCDCore:
    """非递归 SC 译码核心（层存储）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, self.N, block):
                if j % block < half:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + half, s])
                else:
                    top_bit = self.B[j - half, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j - half, s], self.L[j, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    self.B[j - half, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - half, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(l)
            if self.frozen_bits[phi]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)

        br = bit_reversal_permutation(self.N)
        u_hat = np.zeros(self.N, dtype=int)
        for phi in range(self.N):
            u_hat[phi] = self.B[br[phi], self.n]
        return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(
            list(range(n - _active_llr_level(l, n), n))
        )
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(l, n), -1))
        )
    return br, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _SCDCore(len(llr_ch), frozen_bits).decode(llr_ch)
