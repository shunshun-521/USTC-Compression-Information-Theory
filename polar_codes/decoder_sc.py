"""
极化码 SC（串行抵消）译码器
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from _sc_backend import sc_decoder as _sc_decoder_backend


def f_operation(La, Lb):
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def channel_llr_to_decoder(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return llr_ch[inv]


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    info_idx = np.where(frozen_bits == 0)[0].tolist()
    llr = channel_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
    return _sc_decoder_backend(llr, info_idx, 0)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        layers = []
        while (p & 1) == 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)
        bit_layers = []
        if phi % 2 == 1:
            p = phi
            while p > 0:
                if p & 1:
                    bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
