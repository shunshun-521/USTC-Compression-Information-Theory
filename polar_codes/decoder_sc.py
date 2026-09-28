"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from vendor_decoder_utils import upper_llr
from vendor_scd import SCD

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _permute_channel_llr(llr_ch):
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[rev] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    llr = _permute_channel_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = np.array(
            [upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)]
        )
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算调度表（与 PSCD 相位更新层一致，供文档/扩展使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    from vendor_decoder_utils import active_bit_level, active_llr_level
    from vendor_utils import bit_reversed

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        if phi % 2 == 0:
            bit_layer_vec.append([0])
        else:
            layers = []
            p = phi
            layer = 0
            while (p & 1) == 0:
                layers.append(layer)
                p >>= 1
                layer += 1
            bit_layer_vec.append(layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = _permute_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(int(i) for i in np.where(frozen_bits)[0])
    return SCD(N, n, llr_ch, frozen_set).decode()
