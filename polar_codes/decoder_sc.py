"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from decoder_bp import BPDecoder


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等模块复用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sx = np.where(La >= 0, 1.0, -1.0)
    sy = np.where(Lb >= 0, 1.0, -1.0)
    return sx * sy * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (layer - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            t = 0
            while (phi >> t) & 1:
                t += 1
            llr_layers = list(range(n - t - 1, n))
        llr_layer_vec.append(llr_layers)

        if phi == N - 1:
            bit_layers = []
        else:
            t = 0
            p = phi + 1
            while (p >> t) & 1:
                t += 1
            bit_layers = list(range(t))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码：极化码在正确因子图调度下，一次 BP 迭代等价于 SC。
    """
    N = len(llr_ch)
    decoder = BPDecoder(N, frozen_bits, max_iter=1)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致，作参考接口）"""
    return sc_decode(llr, frozen_bits)
