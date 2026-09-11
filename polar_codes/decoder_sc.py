"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _f_exact(a, b):
    """精确 log-domain f 运算（支持向量化）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.asarray(np.logaddexp(0.0, a + b) - np.logaddexp(a, b), dtype=np.float64)


def _g_exact(a, b, u):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    return np.asarray(b + (1.0 - 2.0 * u) * a, dtype=np.float64)


class _SCDecoder:
    """递归 SC 译码器内核。"""

    def __init__(self, frozen_bits):
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.n = len(self.frozen_bits)
        self.u_hat = np.zeros(self.n, dtype=int)

    def decode(self, llr):
        llr = np.asarray(llr, dtype=np.float64)
        self.u_hat[:] = 0
        self._node(llr, 0, self.n)
        return self.u_hat.copy()

    def _leaf(self, llr, index):
        if self.frozen_bits[index]:
            self.u_hat[index] = 0
        else:
            self.u_hat[index] = 0 if llr[0] >= 0 else 1
        return np.array([self.u_hat[index]], dtype=int)

    def _node(self, llr, base, length):
        if length == 1:
            return self._leaf(llr, base)

        half = length // 2
        upper = _f_exact(llr[:half], llr[half:])
        beta_upper = self._node(upper, base, half)
        lower = _g_exact(llr[:half], llr[half:], beta_upper)
        beta_lower = self._node(lower, base + half, half)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数。"""
    return _SCDecoder(frozen_bits).decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n, 0, -1)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
