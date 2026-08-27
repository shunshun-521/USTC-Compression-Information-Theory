"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

import _ref_decoder as _ref
import _ref_function as _fn


def f_operation(La, Lb):
    return _fn.f_hf(La, Lb)


def g_operation(La, Lb, u_hat):
    return _fn.g(La, Lb, u_hat)


def _frozen_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def sc_decode_tree(llr_ch, frozen_bits):
    info_pos = _frozen_to_info_pos(frozen_bits)
    u_d = _ref.sc_decoder(np.asarray(llr_ch, dtype=np.float64), info_pos, 0)[0]
    return np.array([0 if u_d[i] == 0 else 1 for i in range(len(u_d))], dtype=np.int8)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode_tree(llr, frozen_bits)


def precompute_sc_indices(N):
    import math
    n = int(math.log2(N))
    lambda_offset = np.array([1 << i for i in range(n)], dtype=int)
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        phi_temp, layer = phi, 0
        while layer < n:
            if (phi_temp & 1) == 0:
                llr_layers.append(layer)
            phi_temp >>= 1
            layer += 1
        if phi & 1:
            phi_temp = (phi - 1) // 2
            layer = 1
            while layer < n:
                bit_layers.append(layer)
                if phi_temp & 1:
                    break
                phi_temp = (phi_temp - 1) // 2
                layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_tree(llr_ch, frozen_bits)


def sc_path_metric(llr, u_hat):
    pm = 0.0
    for i, l in enumerate(llr):
        hard = 0 if l >= 0 else 1
        if u_hat[i] != hard:
            pm += abs(l)
    return pm
