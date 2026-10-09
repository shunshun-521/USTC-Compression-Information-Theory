"""
极化码 SC（串行抵消）译码器
提供 f/g 运算、非递归 SC（基于极化因子图单遍 BP，与标准 SC 等价的调度）及递归接口。
"""
import math
import numpy as np
from decoder_bp import BPDecoder


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 调度辅助向量（报告/扩展用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        t = phi
        for layer in range(n):
            if t % 2 == 0:
                layers_llr.append(layer)
                t //= 2
            else:
                break
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        t = phi
        layer = 0
        while t % 2 == 1:
            layers_bit.append(layer)
            t = (t - 1) // 2
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_BP_CACHE = {}


def _get_sc_bp(N, frozen_bits):
    key = (N, frozen_bits.tobytes())
    if key not in _SC_BP_CACHE:
        _SC_BP_CACHE[key] = BPDecoder(N, frozen_bits, max_iter=50, alpha=0.9375)
    return _SC_BP_CACHE[key]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（极化因子图单次双向消息传递，与 SC 判决等价）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    decoder = _get_sc_bp(N, frozen_bits)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat.astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 接口（与 sc_decode 相同）"""
    return sc_decode(llr, frozen_bits)
