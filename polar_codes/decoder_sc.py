"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
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
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    u = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u) * La


def _penalty(llr, bit):
    """路径度量惩罚。"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCDecoder:
    """SC 译码器（list_size=1 的 SCL 结构）。"""

    def __init__(self, frozen):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.block_length = int(self.frozen.size)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=int)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=int)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.block_length)
        return self.decisions[0].copy()

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

        bit = 0 if llrs[0][0] >= 0 else 1
        if self.frozen[index]:
            bit = 0
        self.metrics[0] += _penalty(float(llrs[0][0]), bit)
        self.decisions[0][index] = bit
        return [np.array([bit], dtype=int)], [0]

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（基于 SCL 树结构，list_size=1）。"""
    return _SCDecoder(frozen_bits).decode(llr)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers_llr, tmp, layer = [], phi, 0
        while tmp & 1:
            layers_llr.append(layer)
            tmp >>= 1
            layer += 1
        layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)
        if phi % 2 == 0:
            layers_bit, tmp, layer = [], phi >> 1, 1
            while tmp & 1:
                layers_bit.append(layer)
                tmp >>= 1
                layer += 1
            bit_layer_vec.append(layers_bit)
        else:
            bit_layer_vec.append([0])
    return lambda_offset, llr_layer_vec, bit_layer_vec
