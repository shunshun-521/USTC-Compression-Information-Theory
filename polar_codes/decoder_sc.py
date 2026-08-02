"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f 运算（box-plus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return np.asarray(Lb, dtype=np.float64) + (
        1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)
    ) * np.asarray(La, dtype=np.float64)


def _penalty(llr, bit):
    """路径度量增量（log 域）"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，list_size=1 的 SCL）"""
    paths = scl_decode_core(llr, frozen_bits, list_size=1)
    return paths[0][1]


def _trailing_ones(phi):
    count = 0
    while (phi >> count) & 1:
        count += 1
    return count


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        t = _trailing_ones(phi)
        end = max(t - 2, -1)
        llr_layer_vec.append(list(range(n - 1, end, -1)))
        if phi % 2 == 0:
            tz = 0
            psi = phi
            while psi > 0 and psi % 2 == 0:
                tz += 1
                psi >>= 1
            bit_layer_vec.append(list(range(tz)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCLCore:
    """SCL/SC 译码核心（递归极化树遍历）"""

    def __init__(self, frozen, list_size):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.list_size = list_size
        self.N = len(frozen)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        codewords, _ = self._node([llr], 0, self.N)
        paths = list(zip(self.metrics, self.decisions, codewords, strict=True))
        return sorted(paths, key=lambda p: p[0])

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=int))
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


def scl_decode_core(channel_llr, frozen, list_size):
    """返回 [(metric, u_hat, codeword), ...]"""
    return _SCLCore(frozen, list_size).decode(channel_llr)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归接口：调用 SC 核心"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    return sc_decode_recursive(np.asarray(llr_ch, dtype=np.float64), frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=50, eb_n0_db=8.0):
    """SC 译码验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_rec = sc_decode(llr, frozen_bits)
        errors += np.sum(u[info_idx] != u_rec[info_idx])
    if errors > num_frames * K * 0.05:
        raise AssertionError(f"SC BER too high: {errors}/{num_frames * K}")
    return True


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC decoder verification passed")
