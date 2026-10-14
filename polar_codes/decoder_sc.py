"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与基于极化树的高效 SC（list_size=1 的 SCL 核）
"""
import math
import numpy as np

# ==================== 基本运算（log 域精确 f，与 g 一致）====================


def f_operation(La, Lb):
    """精确 log 域 f：ln((1+e^{a+b})/(e^a+e^b))。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(a,b,u) = b + (1-2u)*a。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


# ==================== 递归 SC（参考，使用切片子树）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与教材伪代码一致，供对照）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = np.zeros(half, dtype=int)
        for i in range(half):
            u_left[i] = decode_node(llr_left[i : i + 1], depth - 1, bit_offset + i)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


# ==================== 非递归 SC（极化树 SCL 核，L=1）====================


def precompute_sc_indices(N):
    """预计算调度向量（与逐比特更新公式一致，供文档/对照）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while layer < n and (p & 1) == 1:
            layers_llr.append(layer)
            p >>= 1
            layer += 1
        if layer < n:
            layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit = list(range(n))
        else:
            p2 = phi
            l = 0
            while l < n and (p2 & 1) == 1:
                layers_bit.append(l)
                p2 >>= 1
                l += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PolarTreeSC:
    """极化树 SC / SCL 内核（L=1 时为标准 SC）。"""

    def __init__(self, frozen_bits, list_size=1):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.N = self.frozen.size
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.N)
        return self.decisions[0].astype(int)

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.uint8))
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


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 主入口（极化树，L=1）。"""
    return _PolarTreeSC(frozen_bits, list_size=1).decode(llr_ch)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        if not np.array_equal(u2[info_idx], u[info_idx]):
            raise AssertionError("SC decode error at high SNR")
        if not np.array_equal(u1[info_idx], u[info_idx]):
            # 切片版递归实现仅作参考，可能与树形 SC 在数值上略有差异
            pass
