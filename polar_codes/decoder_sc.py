"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    精确 log-domain f 运算（box-plus）：
    f(a,b) = ln((1 + e^(a+b)) / (e^a + e^b))
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


class _RecursiveSCDecoder:
    """递归 SC 译码器（与 SCL L=1 等价的树形实现）。"""

    def __init__(self, frozen_bits):
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.N = self.frozen_bits.size
        self.u_hat = np.zeros(self.N, dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.u_hat[:] = 0
        self._node(llr_ch, 0, self.N)
        return self.u_hat.copy()

    def _leaf(self, llr, index):
        if self.frozen_bits[index]:
            self.u_hat[index] = 0
        else:
            self.u_hat[index] = 0 if llr[0] >= 0 else 1
        return np.array([self.u_hat[index]], dtype=np.int8)

    def _node(self, llr, base, length):
        if length == 1:
            return self._leaf(llr, base)

        half = length // 2
        beta_upper = self._node(f_operation(llr[:half], llr[half:]), base, half)
        llr_right = g_operation(llr[:half], llr[half:], beta_upper)
        beta_lower = self._node(llr_right, base + half, half)
        return np.concatenate([(beta_upper ^ beta_lower), beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return _RecursiveSCDecoder(frozen_bits).decode(llr)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit = list(range(n))
        else:
            psi = phi
            while psi % 2 == 1:
                layers_bit.append(int(math.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（通过 SCL L=1 实现，与递归版本等价）。"""
    from decoder_scl import SCLDecoder

    u_hat, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1).decode(llr_ch)
    return u_hat
