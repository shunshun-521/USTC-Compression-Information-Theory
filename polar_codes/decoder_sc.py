"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from decoder_core import sc_decode_core
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or (isinstance(u_hat, np.ndarray) and u_hat.ndim == 0):
        b = int(u_hat)
        return La + Lb if b == 0 else La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _prepare_llr_and_mask(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_perm = llr_ch[br].astype(np.float32)
    if_info = (1 - frozen_bits).astype(np.int8)
    return llr_perm, if_info


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码。
    frozen_bits: 自然顺序，1=冻结
    """
    llr_perm, if_info = _prepare_llr_and_mask(llr_ch, frozen_bits)
    u_hat = sc_decode_core(llr_perm, if_info)
    return u_hat.astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """SCL 兼容占位（惰性核心不使用预计算表）"""
    n = int(np.log2(N))
    return [[] for _ in range(N)], [[] for _ in range(N)]
