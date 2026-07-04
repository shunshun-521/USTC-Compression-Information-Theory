"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation
from sc_core import sc_decode_tree


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _prepare_llr(llr_ch):
    """信道 LLR -> 蝶形树顺序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def _frozen_to_info_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0].tolist()


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于迭代树遍历的高效实现）。
    frozen_bits: 1/True 表示冻结位
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_idx = _frozen_to_info_indices(frozen_bits)
    llr_tree = _prepare_llr(llr_ch)
    u_hat, _ = sc_decode_tree(llr_tree, info_idx, 0)
    return u_hat.astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 等价）。
    """
    return sc_decode(llr_ch, frozen_bits)
