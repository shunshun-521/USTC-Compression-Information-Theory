"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

from polar_utils import (
    bit_reversed,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    hard_decision,
)


def f_operation(La, Lb):
    """min-sum f（供 SCL/BP 使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 域，与 SC 对数域 lower_llr 等价）"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0:
        return lower_llr(La, Lb, int(u_hat))
    out = np.empty_like(La, dtype=np.float64)
    for i in range(out.size):
        out[i] = lower_llr(La[i], Lb[i], int(u_hat[i]))
    return out


def precompute_sc_indices(N):
    """SCL 使用的层索引（与 SC 比特处理顺序一致）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return None, llr_layer_vec, bit_layer_vec


class _SCDState:
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)

    def set_channel(self, llr):
        self.L[:, 0] = llr

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(btm_llr, top_llr, int(top_bit))

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits == 1)[0])

    state = _SCDState(N, n)
    state.set_channel(llr_ch)

    for i in range(N):
        l = bit_reversed(i, n)
        state.update_llrs(l)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = hard_decision(state.L[l, n])
        state.update_bits(l)

    return state.B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
