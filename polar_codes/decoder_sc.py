"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归（列表 L=1 的 SCL 实现，与 commpy 算法一致）
"""
import numpy as np


def f_operation(La, Lb):
    """精确 log 域 f（box-plus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb + (1.0 - 2.0 * u_hat) * La, dtype=np.float64)


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def channel_llr_to_decoder(llr_ch):
    """信道 LLR 顺序与编码输出一致，无需重排"""
    return np.asarray(llr_ch, dtype=np.float64)


class _SCRecursive:
    """递归 SC（参考实现）"""

    def __init__(self, frozen_bits):
        self.frozen = np.asarray(frozen_bits, dtype=bool)

    def decode(self, llr):
        llr = np.asarray(llr, dtype=np.float64)
        u = np.zeros(self.frozen.size, dtype=np.int8)

        def node(L, base, length):
            if length == 1:
                idx = base
                if self.frozen[idx]:
                    u[idx] = 0
                else:
                    u[idx] = 0 if L[0] >= 0 else 1
                return
            half = length // 2
            upper = f_operation(L[:half], L[half:])
            node(upper, base, half)
            lower = g_operation(L[:half], L[half:], u[base : base + half])
            node(lower, base + half, half)

        node(llr, 0, len(llr))
        return u


class _SCLCore:
    """SCL/SC 核心（L=1 时为 SC）"""

    def __init__(self, frozen_bits, list_size=1):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.N = self.frozen.size
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]

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
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
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

        lower = []
        for p in range(len(map_upper)):
            a = llrs[map_upper[p]][:half]
            b = llrs[map_upper[p]][half:]
            lower.append(g_operation(a, b, beta_upper[p]))

        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map

    def decode(self, llr_ch):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]
        llr = np.asarray(llr_ch, dtype=np.float64)
        self._node([llr], 0, self.N)
        return self.decisions[0], self.metrics[0]


def sc_decode_recursive(llr_ch, frozen_bits):
    llr = channel_llr_to_decoder(llr_ch)
    return _SCRecursive(frozen_bits).decode(llr).astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（SCL L=1）"""
    u_hat, _ = _SCLCore(frozen_bits, list_size=1).decode(channel_llr_to_decoder(llr_ch))
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """保留接口：非递归索引预计算（供扩展用）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
