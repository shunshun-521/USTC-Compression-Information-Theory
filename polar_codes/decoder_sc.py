"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从 MSB 起）"""
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
    """找到 i 的二进制表示中第一个 0 的位置（从 MSB 起）"""
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
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(l_blk, offset, length):
        if length == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] else (0 if l_blk[0] >= 0 else 1)
            return

        half = length // 2
        l_left = f_operation(l_blk[:half], l_blk[half:])
        decode_block(l_left, offset, half)
        u_left = u_hat[offset : offset + half]
        l_right = g_operation(l_blk[:half], l_blk[half:], u_left)
        decode_block(l_right, offset + half, half)

    decode_block(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供参考）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec


class _SCDState:
    """非递归 SC 译码内部状态"""

    def __init__(self, N, frozen_bits, llr_ch):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
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

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    按比特倒序相位顺序译码，信道 LLR 按码字自然顺序输入。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    state = _SCDState(N, frozen_bits, llr_ch)

    for i in range(N):
        l = bit_reversed(i, n)
        state.update_llrs(l)
        if frozen_bits[l]:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    return state.B[:, n].astype(int)
