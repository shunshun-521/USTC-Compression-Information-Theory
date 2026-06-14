"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation
from sc_core import (
    f_operation,
    g_operation,
    _frozen_to_info_set,
    _sc_decode_core,
)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（树形遍历高效实现）。"""
    info_set = _frozen_to_info_set(frozen_bits)
    u_hat, _, _ = _sc_decode_core(np.asarray(llr_ch, dtype=np.float64), info_set)
    return u_hat.astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用同一核心逻辑）。"""
    return sc_decode_nonrecursive(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while (psi & 1) == 1 and layer < n:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        if layer < n:
            llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            psi = phi
            layer = 0
            while (psi & 1) == 0 and layer < n:
                bit_layers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码入口。
    编码器含比特倒序置换，信道 LLR 需做相同倒序。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return sc_decode_nonrecursive(llr_ch[br], frozen_bits)
