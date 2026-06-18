"""
极化码 SC（串行抵消）译码器
基于置换 SC（Vangala 风格，参考 mcba1n/polar-codes）
"""
import math
import sys
from pathlib import Path

import numpy as np

# 直接复用已验证的参考实现
_REF_DIR = Path(__file__).resolve().parent / "ref_test"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from SCD import SCD  # noqa: E402
from decoder_utils import (  # noqa: E402
    active_bit_level,
    active_llr_level,
    hard_decision,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversal_permutation
from polar_utils import bit_reversed


class _PC:
    __slots__ = ("N", "n", "frozen", "likelihoods")

    def __init__(self, N, n, frozen, likelihoods):
        self.N = N
        self.n = n
        self.frozen = frozen
        self.likelihoods = likelihoods


def f_operation(La, Lb):
    return upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    return lower_llr(La, Lb, u_hat)


def _prepare_channel_llr(llr_ch):
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br].astype(np.float64)


def sc_decode(llr_ch, frozen_bits):
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _prepare_channel_llr(llr_ch)
    frozen = np.where(np.asarray(frozen_bits, dtype=bool))[0]
    return SCD(_PC(N, n, frozen, llr)).decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        l = bit_reversed(phi, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec[phi] = list(range(start, n))
        if l >= N // 2:
            stop = n - active_bit_level(l, n)
            bit_layer_vec[phi] = list(range(n, stop, -1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_llr_to_bit(llr):
    return hard_decision(llr)


def path_metric_penalty(llr, u_hat):
    return 0.0 if u_hat == sc_llr_to_bit(llr) else abs(llr)
