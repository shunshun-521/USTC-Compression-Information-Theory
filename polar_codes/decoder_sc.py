"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from sc_core import (
    f_hf as f_operation,
    g as g_operation,
    preprocess_llr_for_polar_encode,
    sc_tree_decode,
)


def phi(x):
    """保留 API：GA phi 在 construction 模块"""
    from construction import phi as _phi
    return _phi(x)


def _frozen_to_info(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    return np.where(~frozen_bits)[0]


def _preprocess_channel_llr(llr):
    return preprocess_llr_for_polar_encode(llr)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，基于树遍历因子图）。
    """
    llr = _preprocess_channel_llr(llr)
    info_idx = _frozen_to_info(frozen_bits)
    return sc_tree_decode(llr, info_idx)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    用于按相位更新的 LLR/比特层索引（layer-0 为信道 LLR）。
    """
    n = int(np.log2(N))
    assert 2 ** n == N

    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while (p & 1) == 1:
            layers_llr.append(layer)
            p >>= 1
            layer += 1
        layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        layer = 0
        while (p & 1) == 0 and p > 0:
            layers_bit.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（树遍历实现，O(N log N)）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
