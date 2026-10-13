"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from encoder import bit_reversal_permutation
from scd_core import SCD


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u = int(u_hat)
    return La + Lb if u == 0 else Lb - La


def _frozen_for_scd(frozen_bits, N):
    """GA/编码使用自然索引；SCD 内核使用比特倒序索引。"""
    rev = bit_reversal_permutation(N)
    return np.asarray(frozen_bits, dtype=bool)[rev]


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_idx = np.where(_frozen_for_scd(frozen_bits, N))[0]
    u_br = SCD(N, n, llr_ch, frozen_idx).decode()
    rev = bit_reversal_permutation(N)
    return u_br[rev]


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


# 供 SCL 复用
_SCDCore = SCD
_bit_reverse_int = None


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        psi = phi
        for layer in range(n):
            if (psi & 1) == 0:
                llr_layers.append(layer)
            psi >>= 1
        psi = phi + 1
        for layer in range(n):
            if psi & 1:
                bit_layers.append(layer)
            psi >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
