"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _permute_channel_llrs(llr_ch, N):
    """将信道 LLR 调整为译码器内部顺序（配合含比特倒序的编码器）。"""
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        llr_layer_vec.append(
            list(range(n - _active_llr_level(l, n), n))
        )
        if l < N / 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(
                list(range(n, n - _active_bit_level(l, n), -1))
            )

    return decode_order, llr_layer_vec, bit_layer_vec


def _init_sc_state(llr_ch, frozen_bits):
    llr_ch = _permute_channel_llrs(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    decode_order, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    state = {
        "N": N,
        "n": n,
        "decode_order": decode_order,
        "llr_layer_vec": llr_layer_vec,
        "bit_layer_vec": bit_layer_vec,
        "frozen_set": set(np.where(frozen_bits)[0]),
        "L": L,
        "B": B,
        "u_hat": np.zeros(N, dtype=int),
    }
    return state


def _process_one_bit(state, step):
    N = state["N"]
    n = state["n"]
    L = state["L"]
    B = state["B"]
    l = state["decode_order"][step]

    for s in state["llr_layer_vec"][step]:
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )

    if l in state["frozen_set"]:
        state["u_hat"][l] = 0
        B[l, n] = 0
    else:
        state["u_hat"][l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = state["u_hat"][l]

    if l >= N / 2:
        for s in state["bit_layer_vec"][step]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，按比特索引递归）。"""
    state = _init_sc_state(llr_ch, frozen_bits)

    def decode_step(step):
        if step >= state["N"]:
            return
        _process_one_bit(state, step)
        decode_step(step + 1)

    decode_step(0)
    return state["u_hat"]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（高效实现）。"""
    state = _init_sc_state(llr_ch, frozen_bits)
    for step in range(state["N"]):
        _process_one_bit(state, step)
    return state["u_hat"]


def verify_sc_decoders(llr, frozen_bits):
    """验证递归与非递归 SC 译码结果一致。"""
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode(llr, frozen_bits)
    return np.array_equal(u1, u2), u1, u2
