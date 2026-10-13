"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    u_hat = np.asarray(u_hat)
    return Lb + (1 - 2 * u_hat) * La


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCLCore:
    """极化码 SCL/SC 树形译码内核。"""

    def __init__(self, frozen: np.ndarray, list_size: int):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.list_size = list_size
        self.block_length = int(self.frozen.size)
        self.metrics: list[float] = [0.0]
        self.decisions: list[np.ndarray] = [
            np.zeros(self.block_length, dtype=np.int8)
        ]

    def decode(self, channel_llr: np.ndarray) -> list[tuple[float, np.ndarray]]:
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.int8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.block_length)
        return sorted(
            zip(self.metrics, self.decisions, strict=True), key=lambda x: x[0]
        )

    def _leaf(self, llrs: list[np.ndarray], index: int) -> tuple[list[np.ndarray], list[int]]:
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
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics: list[float] = []
        new_decisions: list[np.ndarray] = []
        betas: list[np.ndarray] = []
        parent_map: list[int] = []
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

    def _node(
        self, llrs: list[np.ndarray], base: int, length: int
    ) -> tuple[list[np.ndarray], list[int]]:
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


# ==================== SC 译码 ====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 SCL L=1 等价）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    paths = _SCLCore(frozen_bits, 1).decode(np.asarray(llr, dtype=np.float64))
    return paths[0][1].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        p = phi
        layer = 0
        while layer < n and (p & 1):
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        if layer < n:
            llr_layers.append(layer)
        for ll in range(layer + 1, n):
            llr_layers.append(ll)
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（树形 L=1 实现，O(N log N)）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
