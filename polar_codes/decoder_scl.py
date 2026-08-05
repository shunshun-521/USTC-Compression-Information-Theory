"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    reg = 0
    width = crc_length
    mask = (1 << width) - 1
    top_bit = 1 << (width - 1)
    for bit in info_bits:
        reg ^= (int(bit) << (width - 1))
        for _ in range(width):
            if reg & top_bit:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _penalty(llr, bit):
    """路径度量惩罚。"""
    if bit == 0:
        return 0.0 if llr >= 0 else abs(llr)
    return 0.0 if llr < 0 else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径通过索引映射避免大量复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=int)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]
        self._node([llr_ch], 0, self.N)

        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda item: item[0])

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen)[0]
            crc_ok = [
                p for p in paths
                if crc_check(p[1][info_idx], self.crc_length)
            ]
            if crc_ok:
                paths = crc_ok

        best = paths[0]
        return best[1], best[0]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            betas = [np.array([0], dtype=int) for _ in llrs]
            return betas, list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda item: item[0])

        kept = candidates[:self.list_size]
        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
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


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    info = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)

    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)
    assert np.array_equal(u_sc, u)
    print("SCL L=1 matches SC.")
