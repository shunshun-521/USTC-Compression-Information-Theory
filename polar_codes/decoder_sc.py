"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import prepare_decoder_llr


def f_operation(La, Lb):
    """精确 log-domain box-plus f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, beta) = Lb + (1 - 2*beta) * La"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCDecoderState:
    """树形 SC 译码状态（与极化变换的 partial-sum 结构一致）。"""

    def __init__(self, frozen_bits):
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.N = self.frozen_bits.size
        self.u_hat = np.zeros(self.N, dtype=int)

    def _leaf(self, llr, index):
        bit = 0 if self.frozen_bits[index] or llr[0] >= 0 else 1
        self.u_hat[index] = bit
        return np.array([bit], dtype=int)

    def _node(self, llr, base, length):
        if length == 1:
            return self._leaf(llr, base)

        half = length // 2
        upper = f_operation(llr[:half], llr[half:])
        beta_upper = self._node(upper, base, half)
        lower = g_operation(llr[:half], llr[half:], beta_upper)
        beta_lower = self._node(lower, base + half, half)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    state = _SCDecoderState(frozen_bits)
    state._node(np.asarray(llr, dtype=np.float64), 0, state.N)
    return state.u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while layer < n:
            if psi % 2 == 0:
                layers_llr.append(n - 1 - layer)
            psi //= 2
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        psi = phi
        layer = 0
        while psi > 0 and psi % 2 == 1:
            layers_bit.append(layer)
            psi //= 2
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数。"""
    llr_ch = prepare_decoder_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return sc_decode_recursive(llr_ch, frozen_bits)
