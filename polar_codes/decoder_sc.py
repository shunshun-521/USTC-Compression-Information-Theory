"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与 Vangala 置换非递归实现（主用）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量 u_hat 为左分支已译比特）"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """块递归 SC（LLR 已为译码器顺序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_blk, offset):
        n = len(llr_blk)
        if n == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_blk[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_blk[:half], llr_blk[half:])
        decode_block(llr_left, offset)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_blk[:half], llr_blk[half:], u_left)
        decode_block(llr_right, offset + half)

    decode_block(llr, 0)
    return u_hat


class _SCDState:
    def __init__(self, N, n, llr_ch, frozen_bits):
        self.N = N
        self.n = n
        self.frozen_bits = frozen_bits
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    ) % 2
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 置换顺序，信道 LLR 为自然顺序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    state = _SCDState(N, n, llr_ch, frozen_bits)

    for i in range(N):
        l = _bit_reversed(i, n)
        state.update_llrs(l)
        if frozen_bits[l]:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    return state.B[:, n].astype(int)


def precompute_sc_indices(N):
    """保留接口：SCL 仍使用层向量（与置换 SC 兼容的简化索引）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if phi == N - 1:
            bit_layer_vec.append([])
        else:
            depth = _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n - depth, n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
