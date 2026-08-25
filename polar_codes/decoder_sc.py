"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from scd_core import (
    bit_reversal_permutation,
    scd_decode,
    active_llr_level,
    active_bit_level,
)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr_for_scd(llr_ch):
    """
    发送端对码字做比特倒序置换，接收 LLR 需映射回 SCD 输入顺序。
    llr_for_c[j] = llr_ch[br(j)]
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，调用统一 SCD 核心）"""
    llr = _prepare_llr_for_scd(llr_ch)
    return scd_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = []
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = list(range(active_llr_level(phi, n) - 1, n))
        layers_bit = list(range(n, n - active_bit_level(phi, n), -1))
        lambda_offset.append(phi + 1)
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（基于 permuted SCD）"""
    llr = _prepare_llr_for_scd(llr_ch)
    return scd_decode(llr, frozen_bits)
