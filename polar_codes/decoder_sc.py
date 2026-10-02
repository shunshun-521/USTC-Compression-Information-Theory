"""
极化码 SC（串行抵消）译码器
基于 Permuted SCD（与 vendor.SCD 一致）
"""
import numpy as np

from vendor.PolarCode import PolarCode
from vendor.SCD import SCD


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """兼容接口"""
    return None, None, None


def _frozen_indices_from_mask(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=np.int_)
    return np.where(fb == 1)[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_idx = _frozen_indices_from_mask(frozen_bits)
    K = N - len(frozen_idx)

    pc = PolarCode(N, K)
    pc.frozen = frozen_idx.astype(np.int64)
    pc.frozen_lookup = pc.get_lut(pc.frozen)
    pc.likelihoods = llr_ch.copy()
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归接口（与 sc_decode 等价）"""
    return sc_decode(llr, frozen_bits)
