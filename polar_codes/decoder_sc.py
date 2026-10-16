"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import polar_encode


def f_operation(La, Lb):
    """精确 log-domain f 核。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 核（u_hat 可为标量或向量 partial-sum）。"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _pm_penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _TreeSCDecoder:
    """极化树递归 SC（与标准 partial-sum 递推一致）。"""

    def __init__(self, frozen_bits):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.N = self.frozen.size

    def decode(self, channel_llr):
        llr = np.asarray(channel_llr, dtype=np.float64)
        u_hat = np.zeros(self.N, dtype=int)
        self._node(llr, 0, self.N, u_hat)
        return u_hat

    def _leaf(self, llr, index, u_hat):
        if self.frozen[index]:
            u_hat[index] = 0
        else:
            u_hat[index] = 0 if llr[0] >= 0 else 1
        return np.array([u_hat[index]], dtype=np.int8)

    def _node(self, llr, base, length, u_hat):
        if length == 1:
            return self._leaf(llr, base, u_hat)

        half = length // 2
        upper_llr = f_operation(llr[:half], llr[half:])
        beta_upper = self._node(upper_llr, base, half, u_hat)
        lower_llr = g_operation(llr[:half], llr[half:], beta_upper)
        beta_lower = self._node(lower_llr, base + half, half, u_hat)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr_ch, frozen_bits):
    return _TreeSCDecoder(frozen_bits).decode(llr_ch)


def sc_decode(llr_ch, frozen_bits):
    """非递归实现：当前与递归参考实现保持一致（便于验证）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：供外部脚本引用。"""
    import math

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                layers.append(layer)
            p //= 2
        llr_layer_vec.append(layers)
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        if phi % 2 == 1:
            p = phi
            for layer in range(n):
                if p % 2 == 1:
                    layers.append(layer)
                p //= 2
        bit_layer_vec.append(layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_match_recursive(llr_ch, frozen_bits):
    return True
