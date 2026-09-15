"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSC 版本（高效实现）
"""
import math
import numpy as np
from construction import phi, phi_inv
from encoder import bit_reversal_permutation


def bit_reversed_index(x, n):
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
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def hard_decision(y):
    return 0 if y >= 0 else 1


def f_operation(La, Lb):
    """min-sum 近似 f（供 BP 等模块复用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def f_operation_exact(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * phi_inv(
        1.0 - (1.0 - phi(np.abs(La))) * (1.0 - phi(np.abs(Lb)))
    )


class _SCDState:
    __slots__ = ("N", "n", "L", "B", "frozen_set")

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]


def _prepare_channel_llr(llr_ch):
    """
    编码器输出含比特倒序，PSC 译码器期望 butterfly 顺序的 LLR。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr_ch))
    return llr_ch[rev]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（PSC）。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    state = _SCDState(N, frozen_bits)
    state.L[:, 0] = _prepare_channel_llr(llr_ch)

    for i in range(N):
        l = bit_reversed_index(i, n)
        state.update_llrs(l)
        if l in state.frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = hard_decision(state.L[l, n])
        state.update_bits(l)

    return state.B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托 PSC 实现，保持接口一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
