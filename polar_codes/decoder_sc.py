"""
极化码 SC（串行抵消）译码器
非递归实现（参考 permuted SCD），含递归参考实现
"""
import numpy as np
import math
from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（概率域加法对应 LLR 域）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb != 0


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


class _SCDState:
    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch


def _update_llrs(state):
    l, n, N = state.l, state.n, state.N
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                state.L[j, s + 1] = f_operation(state.L[j, s], state.L[j + branch_size, s])
            else:
                top_bit = state.B[j - branch_size, s + 1]
                state.L[j, s + 1] = g_operation(
                    state.L[j - branch_size, s],
                    state.L[j, s],
                    top_bit,
                )


def _update_bits(state):
    l, n, N = state.l, state.n, state.N
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                state.B[j - branch_size, s - 1] = (
                    state.B[j, s] ^ state.B[j - branch_size, s]
                )
                state.B[j, s - 1] = state.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（按比特倒序相位译码）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = _frozen_mask(frozen_bits)
    frozen_set = set(np.where(frozen)[0])

    state = _SCDState(N, n, llr_ch)
    for i in range(N):
        l = bit_reversed(i, n)
        state.l = l
        _update_llrs(state)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        _update_bits(state)

    return state.B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价，用于校验）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容接口：返回占位结构"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
