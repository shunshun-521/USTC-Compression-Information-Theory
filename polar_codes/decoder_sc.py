"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from decoder_scl import SCLDecoder, f_operation, g_operation


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = np.array([1 << (n - 1 - i) for i in range(n)], dtype=np.int64)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        tmp, layer = phi, 0
        llr_layers = []
        while tmp & 1:
            llr_layers.append(layer)
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        tmp, layer = phi + 1, 0
        bit_layers = []
        while tmp & 1:
            bit_layers.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    N = len(llr_ch)
    dec = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_hat, pm = dec.decode(llr_ch)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
