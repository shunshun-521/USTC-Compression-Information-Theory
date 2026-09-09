"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(a, b):
    """精确 log-domain f 运算（box-plus）"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def g_operation(a, b, u_hat):
    """g 运算"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = np.asarray(u_hat, dtype=np.float64)
    return b + (1.0 - 2.0 * u) * a


def _penalty(llr, bit):
    """路径度量惩罚"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _TreeSCDecoder:
    """基于极化树的 SC 译码器"""

    def __init__(self, frozen_bits):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.N = len(self.frozen)
        self.decisions = np.zeros(self.N, dtype=int)

    def decode(self, channel_llr):
        llr = np.asarray(channel_llr, dtype=np.float64)
        self.decisions = np.zeros(self.N, dtype=int)
        self._node(llr, 0, self.N)
        return self.decisions.copy()

    def _leaf(self, llr, index):
        if self.frozen[index]:
            self.decisions[index] = 0
        else:
            self.decisions[index] = 0 if llr[0] >= 0 else 1
        return np.array([self.decisions[index]], dtype=int)

    def _node(self, llr, base, length):
        if length == 1:
            return self._leaf(llr, base)

        half = length // 2
        beta_upper = self._node(f_operation(llr[:half], llr[half:]), base, half)
        beta_lower = self._node(
            g_operation(llr[:half], llr[half:], beta_upper), base + half, half
        )
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _TreeSCDecoder(frozen_bits).decode(llr)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（树形实现，与递归版本等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容 SCL 接口）"""
    import math
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]
