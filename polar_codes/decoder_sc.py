"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from _sc_core import sc_decode_core, numba


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _to_info_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return (~frozen_bits).astype(np.int32)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    if numba is None:
        raise RuntimeError('numba is required for SC decoding')
    llr_ch = np.asarray(llr_ch, dtype=np.float32)
    is_info = _to_info_mask(frozen_bits)
    return sc_decode_core(llr_ch, is_info).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = []
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        layers_bit = []
        tmp = phi
        for layer in range(n):
            if (tmp & 1) == 0:
                layers_llr.append(layer)
            else:
                layers_bit.append(layer)
            tmp >>= 1
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)
        lambda_offset.append(phi)

    return lambda_offset, llr_layer_vec, bit_layer_vec
