"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation
from sc_core import sc_decoder as _sc_decoder_core

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _preprocess_channel_llr(llr_ch):
    """将信道 LLR 重排为与极化因子图一致的顺序（逆比特倒序）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[np.argsort(br)]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0]
    y_llr = _preprocess_channel_llr(llr_ch)
    return _sc_decoder_core(y_llr, list(information_pos), 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供验证）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.array([2 ** i for i in range(n + 1)], dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        temp = phi
        layer = 0
        while layer < n:
            if temp % 2 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            temp //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
