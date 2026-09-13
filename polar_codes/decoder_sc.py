"""
极化码 SC（串行抵消）译码器
Vangala 2014 置换 SC 实现，min-sum 近似
"""
import numpy as np
import math

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def active_llr_level(i, n):
    """找到索引 i 二进制表示中第一个 1 的位置（从高位起）"""
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
    """找到索引 i 二进制表示中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class _SCDState:
    """SC 译码内部状态"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        top_bit,
                    )

    def update_bits(self, l):
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
    """
    非递归 SC 译码（Vangala 置换顺序）。
    frozen_bits: 1/True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    state = _SCDState(N, n)
    state.L[:, 0] = llr_ch

    for phi in range(N):
        l = bit_reversed(phi, n)
        state.update_llrs(l)
        if frozen_bits[l]:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    return state.B[:, n].copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_node, offset, block_size):
        if block_size == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = block_size // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_left, offset, half)

        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, offset + half, half)

    decode_block(np.asarray(llr, dtype=np.float64), 0, N)
    return u_hat


def precompute_sc_indices(N):
    """兼容 SCL 译码器的辅助向量（Vangala 风格）"""
    n = int(math.log2(N))
    decode_order = [bit_reversed(phi, n) for phi in range(N)]
    return decode_order, n, N
