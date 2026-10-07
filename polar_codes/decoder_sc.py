"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from decoder_sc_core import f_boxplus, g_boxplus, sc_tree_decode
from scd_vendor import SCD


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    return sc_tree_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [(2 ** layer) - 1 for layer in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [list(range(n)) for _ in range(N)]


class _PC:
    def __init__(self, N, n, frozen_idx, llr):
        self.N = N
        self.n = n
        self.frozen = frozen_idx
        self.likelihoods = llr


def sc_decode(llr_ch, frozen_bits):
    """Permuted SCD 译码（与标准极化蝶形编码配套）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_idx = np.where(frozen_bits)[0]
    pc = _PC(N, n, frozen_idx, llr_ch)
    return SCD(pc).decode()


def sc_decode_auto(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
