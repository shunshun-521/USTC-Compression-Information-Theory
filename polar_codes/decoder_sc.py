"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from decoder_core import sc_decoder_impl
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """占位接口（惰性 SC 不需要预计算表）。"""
    n = int(np.log2(N))
    return [1 << i for i in range(n + 1)], [], []


def _to_if_information(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return (~frozen_bits).astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（惰性 LLR + 信道 LLR 比特倒序对齐）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr = llr_ch[rev].astype(np.float32)
    return sc_decoder_impl(llr, _to_if_information(frozen_bits)).astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """参考接口：与 sc_decode 相同实现。"""
    return sc_decode(llr_ch, frozen_bits)
