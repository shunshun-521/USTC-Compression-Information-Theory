"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from polar_ops import f_boxplus, f_operation, g_operation  # noqa: F401 — re-exported API
from decoder_scl import SCLDecoder, _SCLDecoderCore


# ==================== 递归 SC 译码（参考实现）====================


def _sc_node_recursive(llr, frozen_bits, u_hat, base, length):
    if length == 1:
        idx = base
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr[0] >= 0 else 1
        return np.array([u_hat[idx]], dtype=np.uint8)
    half = length // 2
    beta_upper = _sc_node_recursive(
        f_boxplus(llr[:half], llr[half:]), frozen_bits, u_hat, base, half,
    )
    beta_lower = _sc_node_recursive(
        g_operation(llr[:half], llr[half:], beta_upper), frozen_bits, u_hat, base + half, half,
    )
    return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（精确 f，列表核 + boxplus）"""
    import polar_ops
    old_f = polar_ops.f_operation
    polar_ops.f_operation = polar_ops.f_boxplus
    try:
        core = _SCLDecoderCore(np.asarray(frozen_bits, dtype=bool), 1)
        _, u_hat = core.decode(llr)[0]
        return u_hat.astype(int)
    finally:
        polar_ops.f_operation = old_f


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layer_vec.append(list(range(n - 1, -1, -1)))
        else:
            layers = []
            p = phi
            layer = 0
            while (p & 1) == 1:
                layers.append(layer)
                p >>= 1
                layer += 1
            if layer < n:
                layers.append(layer)
            llr_layer_vec.append(sorted(set(layers), reverse=True))
        bit_layers = []
        p = phi
        layer = 0
        while (p & 1) == 1:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（SCL 列表大小为 1，min-sum f）"""
    u_hat, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    return u_hat
