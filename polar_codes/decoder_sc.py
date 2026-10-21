"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 -> La+Lb, b=1 -> -La+Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    y = 0
    for i in range(n):
        if x & (1 << i):
            y |= 1 << (n - 1 - i)
    return y


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
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然比特顺序，信道 LLR 与编码比特一一对应）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)
    n = int(math.log2(N))

    def decode_rec(llr_vec, depth, offset):
        half = len(llr_vec) // 2
        if half == 0:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_vec[0] >= 0 else 1
            return
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])
        decode_rec(llr_left, depth - 1, offset)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_vec[:half], llr_vec[half:], u_left)
        decode_rec(llr_right, depth - 1, offset + half)

    decode_rec(np.asarray(llr, dtype=np.float64), n, 0)
    return u_hat


# ==================== 非递归 Permuted SC 译码 ====================


def precompute_sc_indices(N):
    """预计算每个自然序相位对应的 LLR/比特层（用于文档与调试）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = list(range(N))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDState:
    def __init__(self, N):
        self.N = N
        self.n = int(math.log2(N))
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    # lower_llr(btm, top, bit) — 与 polar-codes 参考实现一致
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], int(top_bit)
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（与 encoder.polar_encode 的蝶形输出一致）。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    state = _SCDState(N)
    state.L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        state.update_llrs(l)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    return state.B[:, n].astype(int)
