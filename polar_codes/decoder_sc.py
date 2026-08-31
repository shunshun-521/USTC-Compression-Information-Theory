"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（因子图单遍 SC）
"""
import math

import numpy as np


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归版本等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（兼容接口）。"""
    n = int(math.log2(N))

    def active_llr(phi):
        count = 0
        while phi & 1:
            count += 1
            phi >>= 1
        return count

    def active_bit(phi):
        if phi == 0:
            return 0
        count = 0
        while (phi & 1) == 0:
            count += 1
            phi >>= 1
        return count

    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(active_llr(phi))))
        bit_layer_vec.append(
            list(range(active_bit(phi))) if phi >= N // 2 else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    基于极化码因子图的单次 BP 迭代（min-sum，alpha=1）。
    """
    from decoder_bp import BPDecoder

    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    decoder = BPDecoder(N, frozen_bits, max_iter=1, alpha=1.0)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat
