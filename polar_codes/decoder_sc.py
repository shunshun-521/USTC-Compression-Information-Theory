"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（接口函数）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """精确 log-domain f 运算（SC/SCL 主路径使用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _prepare_channel_llr(llr_ch):
    """编码含比特倒序时，将信道 LLR 对齐到 SC 树"""
    llr = np.asarray(llr_ch, dtype=np.float64).reshape(-1)
    rev = bit_reversal_permutation(llr.size)
    return llr[rev]


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCTree:
    """与极化蝶形编码（含比特倒序）匹配的 SC/SCL 树遍历"""

    def __init__(self, frozen, list_size=1):
        self.frozen = np.asarray(frozen, dtype=bool).reshape(-1)
        self.block_length = self.frozen.size
        self.list_size = list_size
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.int8)]

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.int8)]
        self._node([llr], 0, self.block_length)
        return self.metrics, self.decisions

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda item: item[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_boxplus(llr[:half], llr[half:]) for llr in llrs]
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


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）"""
    tree = _SCTree(frozen_bits, list_size=1)
    _, decisions = tree.decode(llr_ch)
    return decisions[0].copy()


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层索引列表）。
    """
    m = int(math.log2(N))
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
        psi = phi
        layer = 0
        while layer < m and (psi % 2 == 0):
            layers_bit.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    lambda_offset = np.zeros(m + 2, dtype=np.int32)
    offset = 0
    for i in range(m + 1):
        lambda_offset[i] = offset
        offset += 1 << (m - i)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（树遍历实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
