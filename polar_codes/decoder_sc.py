"""
极化码 SC（串行抵消）译码器
提供递归参考实现与基于极化树的 SC/SCL 核心（与编码器 F^{⊗n} 约定一致）
"""
import numpy as np


def f_operation(La, Lb):
    """log-domain 精确 f 运算（SC/SCL 主路径）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算（BP 等场景）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, beta):
    """g 运算：beta 为左子树部分和（蝶形域比特）"""
    beta = np.asarray(beta, dtype=np.float64)
    return np.asarray(Lb + (1.0 - 2.0 * beta) * La, dtype=np.float64)


def _pm_penalty(llr, bit):
    """路径度量惩罚（对数域）"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _PolarTreeDecoder:
    """极化码树形 SC / SCL 译码器"""

    def __init__(self, frozen_bits, list_size):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = int(list_size)
        self.n = int(self.frozen.size)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.n, dtype=np.int8)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.n, dtype=np.int8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        codewords, _ = self._node([llr], 0, self.n)
        paths = list(zip(self.metrics, self.decisions, codewords, strict=True))
        return sorted(paths, key=lambda item: item[0])

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((self.metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit))
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
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
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        lower_llrs = []
        for p in range(len(map_upper)):
            parent = map_upper[p]
            a = llrs[parent][:half]
            b = llrs[parent][half:]
            lower_llrs.append(g_operation(a, b, beta_upper[p]))

        beta_lower, map_lower = self._node(lower_llrs, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


def scl_decode_paths(channel_llr, frozen_bits, list_size):
    """返回按路径度量排序的 (metric, u_hat, beta_codeword) 列表"""
    return _PolarTreeDecoder(frozen_bits, list_size).decode(channel_llr)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（保留接口；内部调用树形 SC）"""
    paths = scl_decode_paths(llr, frozen_bits, 1)
    return paths[0][1].copy()


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（文档/扩展用）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        bit_layers = []
        tmp = phi
        layer = 0
        while tmp & 1:
            bit_layers.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

        llr_layers = []
        tmp = phi
        for l in range(n):
            if (tmp & 1) == 0:
                llr_layers.append(l)
            tmp >>= 1
        llr_layer_vec.append(llr_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
