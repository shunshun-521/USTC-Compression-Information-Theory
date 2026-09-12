"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SC）
"""
import numpy as np

from encoder import bit_reversed, bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位起）"""
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
    """找到 i 的二进制表示中第一个 0 的位置（从高位起）"""
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
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
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
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（兼容接口）。
    实际译码使用 active_llr_level / active_bit_level。
    """
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 1 << (i - 1)
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, br


class SCDecoder:
    """Permuted SC 译码器（Vangala et al. 风格）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        self.L[:, 0] = llr_ch[br]
        self.B[:] = 0

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            self._update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)

        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        self.B[j, s] ^ self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码主函数"""
    N = len(llr_ch)
    decoder = SCDecoder(N, frozen_bits)
    return decoder.decode(llr_ch)
