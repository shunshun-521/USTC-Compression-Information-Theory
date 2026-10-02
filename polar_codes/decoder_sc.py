"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from scd_core import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    scd_decode,
    upper_llr,
)


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    return scd_decode(llr_ch, frozen_set, len(llr_ch))


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, frozen_slice, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_slice[0] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = np.array(
            [upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)]
        )
        decode_node(llr_left, frozen_slice[:half], bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = np.array(
            [
                lower_llr(llr_node[i + half], llr_node[i], u_left[i])
                for i in range(half)
            ]
        )
        decode_node(llr_right, frozen_slice[half:], bit_offset + half)

    decode_node(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return [1 << i for i in range(n + 1)], llr_layer_vec, bit_layer_vec
