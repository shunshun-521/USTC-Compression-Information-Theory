"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def _sign_llr(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    return (
        _sign_llr(La)
        * _sign_llr(Lb)
        * np.minimum(np.abs(La), np.abs(Lb))
    )


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while p & 1:
            layers_llr.append(layer)
            layer += 1
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        for layer in range(n):
            if p & 1:
                break
            layers_bit.append(layer)
            p >>= 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _polar_decode_sc(llr_ch, frozen_bits):
    n = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        frozen1 = frozen_bits[:half]
        frozen2 = frozen_bits[half:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc(x_llr1, frozen1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc(x_llr2, frozen2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_i = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(
            np.float64
        )
        u_hat_up = np.concatenate([u_hat1_up_i, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_bits[0]:
        u_hat = np.array([0.0])
    else:
        u_hat = np.array([1.0 if llr_ch[0] < 0 else 0.0])
    return u_hat, u_hat


def _prepare_llr(llr_ch):
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_recursive(llr, frozen_bits):
    llr_work = _prepare_llr(llr)
    u_hat, _ = _polar_decode_sc(llr_work, frozen_bits)
    return u_hat.astype(int)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_channel(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
