"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from decoder_scl import scl_decode


def f_operation(La, Lb):
    """精确 log 域 f 运算（与 SCL 一致）"""
    a = np.asarray(La, dtype=np.float64)
    b = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb, dtype=np.float64) + (1.0 - 2.0 * u) * np.asarray(La, dtype=np.float64)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（由 SCL L=1 实现）"""
    paths = scl_decode(llr, np.asarray(frozen_bits, dtype=bool), list_size=1)
    return paths[0][1].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（供文档/扩展使用）"""
    import math

    n = int(math.log2(N))
    lambda_offset = [1 << (n - layer) for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        tmp = phi
        layer = 0
        while tmp % 2 == 1 and layer < n:
            layer += 1
            tmp >>= 1
        llr_layer_vec.append(list(range(layer, n)))
        tmp = phi + 1
        bl = 0
        while tmp % 2 == 0 and bl < n:
            bl += 1
            tmp >>= 1
        bit_layer_vec.append(list(range(bl)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：SCL 列表大小为 1 的 SC 译码"""
    return sc_decode_recursive(llr_ch, frozen_bits)
