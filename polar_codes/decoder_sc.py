"""
极化码 SC（串行抵消）译码器 — Permuted SCD（非递归）
"""
import numpy as np

from encoder import bit_reversed
from polar_ops import upper_llr, lower_llr, active_llr_level, active_bit_level, f_operation, g_operation


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（连续分段，用于交叉验证）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


class _SCDState:
    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch


def _update_llrs(state):
    my_n = state.n
    N = state.N

    def update_for_l(l):
        for s in range(my_n - active_llr_level(l, my_n), my_n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = state.L[j, s]
                    btm_llr = state.L[j + branch_size, s]
                    state.L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = state.L[j, s]
                    top_llr = state.L[j - branch_size, s]
                    top_bit = state.B[j - branch_size, s + 1]
                    state.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    return update_for_l


def _update_bits(state, l):
    my_n = state.n
    if l < state.N / 2:
        return
    for s in range(my_n, my_n - active_bit_level(l, my_n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                state.B[j - branch_size, s - 1] = int(state.B[j, s]) ^ int(
                    state.B[j - branch_size, s]
                )
                state.B[j, s - 1] = state.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    Permuted 非递归 SC 译码。
    frozen_bits[i]=True 表示索引 i 为冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    state = _SCDState(N, n, llr_ch)
    update_llrs = _update_llrs(state)
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        _update_bits(state, l)

    return state.B[:, n].astype(int)


def precompute_sc_indices(N):
    """兼容接口：返回 Permuted SCD 的层索引描述。"""
    n = int(np.log2(N))
    llr_layer_vec = [list(range(n - active_llr_level(bit_reversed(phi, n), n), n)) for phi in range(N)]
    bit_layer_vec = [
        list(range(n, n - active_bit_level(bit_reversed(phi, n), n), -1)) if phi >= N / 2 else []
        for phi in range(N)
    ]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
